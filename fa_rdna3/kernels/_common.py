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
# large localized errors (often non-deterministic) for some dtypes. Confirmed on
# Triton 3.5.1 / ROCm 6.4 by bench/config_sweep.py. (128, 64, 8) tipped over once
# the kernel grew the optional score-mod arguments (bias/alibi/dropout) — the
# WMMA codegen for that geometry is fragile enough that extra scalar args change
# register allocation and break it. Kept out of the search space so autotuning
# can only pick a numerically correct config.
_MISCOMPILED_ON_GFX1100 = {(64, 64, 4), (128, 128, 8), (128, 64, 8)}
# With bf16 Q scaled after the WMMA, this geometry corrupts both output and LSE
# non-deterministically. Nearby geometries are bitwise stable on the same input.
_MISCOMPILED_BF16_POST_SCALE = {(64, 32, 4)}
# The fp32 backward codegen never returns from these geometries on gfx1100.
_NONRETURNING_FP32_DKDV = {(64, 128, 4), (64, 128, 8), (128, 64, 4)}
_NONRETURNING_FP32_DQ = {(64, 64, 2)}
_FP32_BACKWARD_GEOMETRY = (64, 32, 4)


# The exhaustive grid remains in bench/config_sweep.py. Production keeps every
# geometry observed as a winner or close runner-up, which halves cold compile
# work and disk-cache growth without removing a measured best configuration.
def _configs_from_geometries(geometries):
    return [
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_N": block_n},
            num_warps=num_warps,
            num_stages=1,
        )
        for block_m, block_n, num_warps in geometries
    ]


def _fwd_configs():
    return _configs_from_geometries((
        (64, 32, 2), (64, 32, 4), (64, 32, 8),
        (64, 64, 8),
        (64, 128, 4), (64, 128, 8),
        (128, 32, 8),
    ))


def _bwd_dkdv_configs():
    return _configs_from_geometries((
        (64, 32, 2), (64, 32, 4), (64, 32, 8),
        (64, 64, 2), (64, 64, 8),
        (64, 128, 4), (64, 128, 8),
        (128, 64, 4),
    ))


def _bwd_dq_configs():
    return _configs_from_geometries((
        (64, 32, 2), (64, 32, 4), (64, 32, 8),
        (64, 64, 2), (64, 64, 8),
        (128, 32, 4), (128, 32, 8),
    ))


# A tile's fp32 accumulator and K/V staging scale with tile_rows * head_dim. At
# head_dim 128 even the 128x128 tile is only 16K elements, but at head_dim 512 it
# is 64K — far past the register file / 64 KB LDS, so those configs either fail to
# fit or are pathologically slow to compile-and-benchmark during autotuning. Prune
# oversized tiles per head_dim: head_dim <= 256 is untouched (the common path),
# large head_dim keeps only the tiles that actually fit.
_TILE_HEAD_DIM_BUDGET = 32768


def _prune_configs_by_head_dim(configs, named_args, **kwargs):
    head_dim = kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM"))
    post_scale_q = kwargs.get("POST_SCALE_Q", named_args.get("POST_SCALE_Q", False))
    kept = configs
    if post_scale_q:
        kept = [config for config in kept if (
            config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps
        ) not in _MISCOMPILED_BF16_POST_SCALE]
    if head_dim is None or head_dim <= 256:
        return kept
    kept = [config for config in kept
            if config.kwargs["BLOCK_M"] * head_dim <= _TILE_HEAD_DIM_BUDGET
            and config.kwargs["BLOCK_N"] * head_dim <= _TILE_HEAD_DIM_BUDGET]
    return kept or configs


def _prune_backward_configs(configs, named_args, **kwargs):
    query = named_args.get("q_ptr")
    if query is not None and query.element_size() == 4:
        return [config for config in configs if (
            config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps
        ) == _FP32_BACKWARD_GEOMETRY]
    return _prune_configs_by_head_dim(configs, named_args, **kwargs)


def _prune_bwd_dkdv_configs(configs, named_args, **kwargs):
    return _prune_backward_configs(configs, named_args, **kwargs)


def _prune_bwd_dq_configs(configs, named_args, **kwargs):
    return _prune_backward_configs(configs, named_args, **kwargs)


def _split_configs():
    # Decode is memory-bound on the KV load, so favour small key tiles and few
    # warps; the miscompile blacklist still applies to the WMMA in the partial.
    return [
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps, num_stages=1)
        for block_n, num_warps in (
            (32, 1), (32, 2), (32, 4), (64, 2), (64, 4), (128, 4)
        )
    ]

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
    SAFE_SOFTMAX: tl.constexpr = False,
    softmax_scale=1.0, POST_SCALE_Q: tl.constexpr = False,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    bias_base=0, stride_bm=0, stride_bn=0, HAS_BIAS: tl.constexpr = False,
    alibi_slope=0.0, HAS_ALIBI: tl.constexpr = False,
    dropout_p=0.0, dropout_seed=0, dropout_base=0, DROPOUT: tl.constexpr = False,
):
    """Accumulate one contiguous band of key blocks into the online softmax.

    ``q`` is normally pre-scaled by ``softmax_scale * log2(e)`` so scores land
    directly in the log2 domain and ``exp2`` replaces ``exp``. The bf16 path
    scales the fp32 dot accumulator instead to preserve backward consistency.
    When ``MASKED`` is false the band is fully inside the valid, causal-kept
    region and no masking is emitted. ``PRE_LOAD_V`` loads V before the QK dot
    to overlap its latency, at the cost of holding it live across the softmax.
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
        if POST_SCALE_Q:
            qk *= softmax_scale * LOG2E
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
            # FlashAttention ALiBi: -slope * |query_pos - key_pos|, with query
            # positions bottom-right aligned when the sequence lengths differ.
            alibi = -tl.abs(offs_n[None, :] - offs_m[:, None] - (seqlen_k - seqlen_q)).to(tl.float32)
            qk += alibi_slope * alibi * LOG2E

        if MASKED:
            query_pos = offs_m[:, None] + (seqlen_k - seqlen_q)
            if IS_CAUSAL:
                # Bottom-right alignment: query i sits at absolute position
                # i + (seqlen_k - seqlen_q), so it attends keys j <= that.
                keep = (query_pos >= offs_n[None, :]) & (offs_n[None, :] < seqlen_k)
            else:
                keep = offs_n[None, :] < seqlen_k
            if WINDOW_LEFT >= 0:
                keep = keep & (offs_n[None, :] >= query_pos - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                keep = keep & (offs_n[None, :] <= query_pos + WINDOW_RIGHT)
            qk = tl.where(keep, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        if SAFE_SOFTMAX:
            # A tile fully outside a row's kept region leaves m_new at -inf; reset
            # it so the row contributes nothing instead of exp2(-inf - -inf) = NaN.
            # An additive bias can also fully mask an otherwise-unmasked tile. The
            # temporary safe max is not carried forward, so a later tile with very
            # negative finite logits still establishes its own numerically stable max.
            empty = m_new == float("-inf")
            m_safe = tl.where(empty, 0.0, m_new)
            m_next = tl.where(empty, m_i, m_new)
        else:
            m_safe = m_new
            m_next = m_new
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            if MASKED:
                v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs)
        if DROPOUT:
            # Drop after l_i (the normalizer keeps the undropped mass); kept
            # entries are scaled by 1/(1-p). Offset is a global (head, i, j) index
            # so the backward regenerates the identical mask.
            offs = (dropout_base + offs_m[:, None]) * seqlen_k + offs_n[None, :]
            keep_d = tl.rand(dropout_seed, offs) > dropout_p
            p = tl.where(keep_d, p * (1.0 / (1.0 - dropout_p)), 0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_next

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
    SAFE_LSE: tl.constexpr = False,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    bias_base=0, stride_bm=0, stride_bn=0, HAS_BIAS: tl.constexpr = False,
    alibi_slope=0.0, HAS_ALIBI: tl.constexpr = False,
    dropout_p=0.0, dropout_seed=0, dropout_base=0, DROPOUT: tl.constexpr = False,
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
            alibiT = -tl.abs(offs_n[:, None] - offs_m[None, :] - (seqlen_k - seqlen_q)).to(tl.float32)
            qkT += alibi_slope * alibiT * LOG2E
        if SAFE_LSE:
            lse_valid = lse != float("-inf")
            lse_safe = tl.where(lse_valid, lse, 0.0)
            pT = tl.exp2(qkT - lse_safe[None, :] * LOG2E)
            pT = tl.where(lse_valid[None, :], pT, 0.0)
        else:
            pT = tl.exp2(qkT - lse[None, :] * LOG2E)
        if MASKED:
            query_posT = offs_m[None, :] + (seqlen_k - seqlen_q)
            keep = (offs_n[:, None] < seqlen_k) & (offs_m[None, :] < seqlen_q)
            if IS_CAUSAL:
                keep = keep & (query_posT >= offs_n[:, None])
            if WINDOW_LEFT >= 0:
                keep = keep & (offs_n[:, None] >= query_posT - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                keep = keep & (offs_n[:, None] <= query_posT + WINDOW_RIGHT)
            pT = tl.where(keep, pT, 0.0)

        if DROPOUT:
            offsT = (dropout_base + offs_m[None, :]) * seqlen_k + offs_n[:, None]
            keepT = tl.rand(dropout_seed, offsT) > dropout_p
            inv_keep = 1.0 / (1.0 - dropout_p)
            pT_d = tl.where(keepT, pT * inv_keep, 0.0)
        else:
            pT_d = pT
        dv += tl.dot(pT_d.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        if DROPOUT:
            dpT = tl.where(keepT, dpT * inv_keep, 0.0)
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
    SAFE_LSE: tl.constexpr = False,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    bias_base=0, stride_bm=0, stride_bn=0, HAS_BIAS: tl.constexpr = False,
    alibi_slope=0.0, HAS_ALIBI: tl.constexpr = False,
    dropout_p=0.0, dropout_seed=0, dropout_base=0, DROPOUT: tl.constexpr = False,
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
            alibi = -tl.abs(offs_n[None, :] - offs_m[:, None] - (seqlen_k - seqlen_q)).to(tl.float32)
            qk += alibi_slope * alibi * LOG2E
        if SAFE_LSE:
            lse_valid = lse != float("-inf")
            lse_safe = tl.where(lse_valid, lse, 0.0)
            p = tl.exp2(qk - lse_safe[:, None] * LOG2E)
            p = tl.where(lse_valid[:, None], p, 0.0)
        else:
            p = tl.exp2(qk - lse[:, None] * LOG2E)
        if MASKED:
            query_pos = offs_m[:, None] + (seqlen_k - seqlen_q)
            keep = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            if IS_CAUSAL:
                keep = keep & (query_pos >= offs_n[None, :])
            if WINDOW_LEFT >= 0:
                keep = keep & (offs_n[None, :] >= query_pos - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                keep = keep & (offs_n[None, :] <= query_pos + WINDOW_RIGHT)
            p = tl.where(keep, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        if DROPOUT:
            offs = (dropout_base + offs_m[:, None]) * seqlen_k + offs_n[None, :]
            keep_d = tl.rand(dropout_seed, offs) > dropout_p
            dp = tl.where(keep_d, dp * (1.0 / (1.0 - dropout_p)), 0.0)
        ds = p * (dp - delta[:, None])
        if HAS_SOFTCAP:
            ds = ds * (1.0 - softcap_t * softcap_t)
        dq += tl.dot(ds.to(k.dtype), k)

    return dq
