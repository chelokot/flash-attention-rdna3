"""Shared pieces for the RDNA3 FlashAttention kernels.

Autotune plumbing, the miscompile blacklist, config grids, and the inner
key/value loop device functions that the forward, backward, decode, and
variable-length kernels all build on.

RDNA3 notes: 32-lane WMMA fragments, a 64 KB per-workgroup LDS budget, and no
async-copy pipelining (unlike CUDA cp.async), which favours num_stages=1. The
inner loop is split by callers into an unmasked region (full tiles below the
causal diagonal / before the key boundary) and a masked region (diagonal band
and ragged tail); the unmasked region carries no tl.where and no boundary loads.
"""

import functools

import triton
import triton.language as tl
from triton.testing import do_bench

LOG2E = tl.constexpr(1.44269504088896)  # log2(e); folded into the scale so softmax uses exp2

# Rank configs over more repetitions than the Triton default: on a consumer
# RX 7900 XTX the per-config timing is noisy enough that a single warmup can
# lock in a config ~30% off the best, so the extra reps pay for themselves in
# stable selection.
_autotune_bench = functools.partial(do_bench, warmup=40, rep=120)


# (BLOCK_M, BLOCK_N, num_warps) geometries that miscompile in the ROCm Triton
# WMMA backend on gfx1100: they run fast enough to win autotuning but return
# large localized errors for some dtypes. Confirmed still broken on Triton 3.5.1
# / ROCm 6.4 by bench/config_sweep.py; kept out of the search space so
# autotuning can only pick a numerically correct config.
_MISCOMPILED_ON_GFX1100 = {(64, 64, 4), (128, 128, 8)}


def _fwd_configs():
    configs = []
    for block_m in (64, 128):
        for block_n in (32, 64, 128):
            for num_warps in (2, 4, 8):
                if (block_m, block_n, num_warps) in _MISCOMPILED_ON_GFX1100:
                    continue
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n},
                        num_warps=num_warps,
                        num_stages=1,
                    )
                )
    return configs

def _bwd_configs():
    # The dK/dV and dQ kernels carry more live fp32 accumulators than the
    # forward (two head-dim tiles plus the score tile), so the same blacklist
    # applies and small tiles dominate. Reuse the validated forward geometries.
    return _fwd_configs()

def _split_configs():
    # Decode is memory-bound on the KV load, so favour small key tiles and few
    # warps; the miscompile blacklist still applies to the WMMA in the partial.
    configs = []
    for block_n in (32, 64, 128):
        for num_warps in (1, 2, 4):
            configs.append(triton.Config({"BLOCK_N": block_n}, num_warps=num_warps, num_stages=1))
    return configs

@triton.jit
def _attention_inner(
    acc, l_i, m_i, q,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_d,
    start_n, end_n, seqlen_k, seqlen_q,
    BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    MASKED: tl.constexpr, IS_CAUSAL: tl.constexpr, PRE_LOAD_V: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    bias_base=0, stride_bm=0, stride_bn=0, HAS_BIAS: tl.constexpr = False,
    alibi_slope=0.0, HAS_ALIBI: tl.constexpr = False,
):
    """Accumulate one contiguous band of key blocks into the online softmax.

    ``q`` is pre-scaled by ``softmax_scale * log2(e)`` so scores land directly in
    the log2 domain and ``exp2`` replaces ``exp``. When ``MASKED`` is false the
    band is fully inside the valid, causal-kept region and no masking is emitted.
    ``PRE_LOAD_V`` loads V before the QK dot to overlap its latency, at the cost
    of holding it live across the softmax (more register pressure).
    """
    for start in range(start_n, end_n, BLOCK_N):
        start = tl.multiple_of(start, BLOCK_N)
        offs_n = start + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if MASKED:
            k = tl.load(k_ptrs, mask=offs_n[None, :] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs)
        if PRE_LOAD_V:
            if MASKED:
                v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs)

        qk = tl.dot(q, k)
        if HAS_SOFTCAP:
            # Logit cap softcap*tanh(s/softcap), applied in the log2 domain (q is
            # pre-scaled by scale*log2(e)): cap = softcap*log2(e), qk = cap*tanh(qk/cap).
            cap = softcap * LOG2E
            qk = cap * (2.0 * tl.sigmoid(2.0 * qk / cap) - 1.0)
        if HAS_BIAS:
            bias_ptrs = bias_base + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
            if MASKED:
                bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0)
            else:
                bias = tl.load(bias_ptrs)
            qk += bias.to(tl.float32) * LOG2E  # additive logit bias, into the log2 domain
        if HAS_ALIBI:
            # ALiBi: slope * (key_pos - query_pos); query i sits at i + (seqlen_k - seqlen_q).
            alibi = (offs_n[None, :] - offs_m[:, None] - (seqlen_k - seqlen_q)).to(tl.float32)
            qk += alibi_slope * alibi * LOG2E

        if MASKED:
            if IS_CAUSAL:
                # Bottom-right alignment: query i sits at absolute position
                # i + (seqlen_k - seqlen_q), so it attends keys j <= that.
                keep = (offs_m[:, None] + (seqlen_k - seqlen_q) >= offs_n[None, :]) & (offs_n[None, :] < seqlen_k)
            else:
                keep = offs_n[None, :] < seqlen_k
            if WINDOW_LEFT >= 0:
                keep = keep & (offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                keep = keep & (offs_n[None, :] <= offs_m[:, None] + WINDOW_RIGHT)
            qk = tl.where(keep, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        if MASKED:
            # A tile fully outside a row's kept region leaves m_new at -inf; reset
            # it so the row contributes nothing instead of exp2(-inf - -inf) = NaN.
            # The online softmax result is invariant to the running max, so this is
            # exact. Only reachable with a window/boundary that can empty a tile.
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            if MASKED:
                v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    return acc, l_i, m_i

@triton.jit
def _bwd_dkdv_inner(
    dk, dv, k, v,
    q_base, do_base, lse_base, delta_base,
    stride_qm, stride_qd, stride_dom, stride_dod, stride_lm, stride_dem,
    offs_n, offs_d, start_m, end_m, seqlen_q, seqlen_k, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
    MASKED: tl.constexpr, IS_CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    bias_base=0, stride_bm=0, stride_bn=0, HAS_BIAS: tl.constexpr = False,
    alibi_slope=0.0, HAS_ALIBI: tl.constexpr = False,
):
    """Fold one query band into dK, dV for a fixed key block.

    When ``MASKED`` is false the band is a full set of query rows strictly below
    the causal diagonal for an in-bounds key block, so neither the causal nor the
    boundary comparison is emitted.
    """
    for m in range(start_m, end_m, BLOCK_M):
        m = tl.multiple_of(m, BLOCK_M)
        offs_m = m + tl.arange(0, BLOCK_M)
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        if MASKED:
            m_mask = offs_m[:, None] < seqlen_q
            q = tl.load(q_ptrs, mask=m_mask, other=0.0)
            do = tl.load(do_ptrs, mask=m_mask, other=0.0)
            lse = tl.load(lse_base + offs_m * stride_lm, mask=offs_m < seqlen_q, other=0.0)
            delta = tl.load(delta_base + offs_m * stride_dem, mask=offs_m < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
            lse = tl.load(lse_base + offs_m * stride_lm)
            delta = tl.load(delta_base + offs_m * stride_dem)

        qkT = tl.dot(k, tl.trans(q)) * qk_scale
        if HAS_SOFTCAP:
            cap = softcap * LOG2E
            softcap_t = 2.0 * tl.sigmoid(2.0 * qkT / cap) - 1.0
            qkT = cap * softcap_t
        if HAS_BIAS:
            biasT_ptrs = bias_base + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
            if MASKED:
                biasT = tl.load(biasT_ptrs, mask=(offs_m[None, :] < seqlen_q) & (offs_n[:, None] < seqlen_k), other=0.0)
            else:
                biasT = tl.load(biasT_ptrs)
            qkT += biasT.to(tl.float32) * LOG2E
        if HAS_ALIBI:
            alibiT = (offs_n[:, None] - offs_m[None, :] - (seqlen_k - seqlen_q)).to(tl.float32)
            qkT += alibi_slope * alibiT * LOG2E
        pT = tl.exp2(qkT - lse[None, :] * LOG2E)
        if MASKED:
            keep = (offs_n[:, None] < seqlen_k) & (offs_m[None, :] < seqlen_q)
            if IS_CAUSAL:
                keep = keep & (offs_m[None, :] + (seqlen_k - seqlen_q) >= offs_n[:, None])
            if WINDOW_LEFT >= 0:
                keep = keep & (offs_n[:, None] >= offs_m[None, :] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                keep = keep & (offs_n[:, None] <= offs_m[None, :] + WINDOW_RIGHT)
            pT = tl.where(keep, pT, 0.0)

        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        if HAS_SOFTCAP:
            dsT = dsT * (1.0 - softcap_t * softcap_t)  # chain through softcap*tanh(s/softcap)
        dk += tl.dot(dsT.to(q.dtype), q)

    return dk, dv

@triton.jit
def _bwd_dq_inner(
    dq, q, do, lse, delta,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_d, start_n, end_n, seqlen_q, seqlen_k, qk_scale,
    BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    MASKED: tl.constexpr, IS_CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    bias_base=0, stride_bm=0, stride_bn=0, HAS_BIAS: tl.constexpr = False,
    alibi_slope=0.0, HAS_ALIBI: tl.constexpr = False,
):
    """Fold one key band into dQ for a fixed query block."""
    for n in range(start_n, end_n, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        offs_n = n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if MASKED:
            n_mask = offs_n[:, None] < seqlen_k
            k = tl.load(k_ptrs, mask=n_mask, other=0.0)
            v = tl.load(v_ptrs, mask=n_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if HAS_SOFTCAP:
            cap = softcap * LOG2E
            softcap_t = 2.0 * tl.sigmoid(2.0 * qk / cap) - 1.0
            qk = cap * softcap_t
        if HAS_BIAS:
            bias_ptrs = bias_base + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
            if MASKED:
                bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0)
            else:
                bias = tl.load(bias_ptrs)
            qk += bias.to(tl.float32) * LOG2E
        if HAS_ALIBI:
            alibi = (offs_n[None, :] - offs_m[:, None] - (seqlen_k - seqlen_q)).to(tl.float32)
            qk += alibi_slope * alibi * LOG2E
        p = tl.exp2(qk - lse[:, None] * LOG2E)
        if MASKED:
            keep = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            if IS_CAUSAL:
                keep = keep & (offs_m[:, None] + (seqlen_k - seqlen_q) >= offs_n[None, :])
            if WINDOW_LEFT >= 0:
                keep = keep & (offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                keep = keep & (offs_n[None, :] <= offs_m[:, None] + WINDOW_RIGHT)
            p = tl.where(keep, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if HAS_SOFTCAP:
            ds = ds * (1.0 - softcap_t * softcap_t)
        dq += tl.dot(ds.to(k.dtype), k)

    return dq
