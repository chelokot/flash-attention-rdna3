"""Triton FlashAttention-2 kernels tuned for AMD RDNA3 (gfx1100).

The forward kernel implements the FlashAttention-2 tiling with an online
softmax so the full attention matrix is never materialised. Block sizes are
autotuned over a small grid chosen for RDNA3: 32-lane WMMA fragments, a 64 KB
LDS budget per workgroup, and no async-copy pipelining (unlike CUDA cp.async),
which favours a modest number of stages.

The inner key/value loop is split into an unmasked region (full tiles strictly
below the causal diagonal, or before the key boundary) and a masked region (the
diagonal band and the ragged tail). The unmasked region carries no ``tl.where``
and no boundary-guarded loads, which is where most iterations live.
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


@triton.jit
def _attention_inner(
    acc, l_i, m_i, q,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_d,
    start_n, end_n, seqlen_k,
    BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    MASKED: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    """Accumulate one contiguous band of key blocks into the online softmax.

    ``q`` is pre-scaled by ``softmax_scale * log2(e)`` so scores land directly in
    the log2 domain and ``exp2`` replaces ``exp``. When ``MASKED`` is false the
    band is fully inside the valid, causal-kept region and no masking is emitted.
    """
    for start in range(start_n, end_n, BLOCK_N):
        start = tl.multiple_of(start, BLOCK_N)
        offs_n = start + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        if MASKED:
            k = tl.load(k_ptrs, mask=offs_n[None, :] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs)

        qk = tl.dot(q, k)

        if MASKED:
            if IS_CAUSAL:
                keep = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seqlen_k)
            else:
                keep = offs_n[None, :] < seqlen_k
            qk = tl.where(keep, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if MASKED:
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            v = tl.load(v_ptrs)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    return acc, l_i, m_i


@triton.autotune(
    configs=_fwd_configs(),
    key=["seqlen_q_bucket", "seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL"],
    do_bench=_autotune_bench,
)
@triton.jit
def _attention_forward(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    num_heads, seqlen_q, seqlen_k,
    seqlen_q_bucket, seqlen_k_bucket,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    tl.static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0, "HEAD_DIM must be a power of two")

    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = (q * (softmax_scale * LOG2E)).to(q_ptr.dtype.element_ty)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M)
        # Key blocks strictly below the diagonal and inside the key bound need no
        # mask; the diagonal band and any ragged key tail inside it do.
        unmasked_n = tl.minimum(block_m_idx * BLOCK_M, seqlen_k) // BLOCK_N * BLOCK_N
    else:
        max_n = seqlen_k
        # Only the ragged tail past the last whole block needs a boundary mask.
        unmasked_n = seqlen_k // BLOCK_N * BLOCK_N

    acc, l_i, m_i = _attention_inner(
        acc, l_i, m_i, q, k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        offs_m, offs_d, 0, unmasked_n, seqlen_k,
        BLOCK_N, HEAD_DIM, False, IS_CAUSAL,
    )
    acc, l_i, m_i = _attention_inner(
        acc, l_i, m_i, q, k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        offs_m, offs_d, unmasked_n, max_n, seqlen_k,
        BLOCK_N, HEAD_DIM, True, IS_CAUSAL,
    )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    out_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)

    lse = m_i / LOG2E + tl.log(l_safe)
    lse_base = lse_ptr + batch_idx * stride_lb + head_idx * stride_lh
    lse_ptrs = lse_base + offs_m * stride_lm
    tl.store(lse_ptrs, lse, mask=offs_m < seqlen_q)


def _bwd_configs():
    # The dK/dV and dQ kernels carry more live fp32 accumulators than the
    # forward (two head-dim tiles plus the score tile), so the same blacklist
    # applies and small tiles dominate. Reuse the validated forward geometries.
    return _fwd_configs()


@triton.jit
def _attention_bwd_preprocess(
    out_ptr, dout_ptr, delta_ptr,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_db, stride_dh, stride_dm,
    num_heads, seqlen_q,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """delta_i = sum_d O_id * dO_id, the per-row correction used by dS."""
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_mask = offs_m[:, None] < seqlen_q

    o_ptrs = (out_ptr + batch_idx * stride_ob + head_idx * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    do_ptrs = (dout_ptr + batch_idx * stride_dob + head_idx * stride_doh
               + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    o = tl.load(o_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + batch_idx * stride_db + head_idx * stride_dh + offs_m * stride_dm
    tl.store(delta_ptrs, delta, mask=offs_m < seqlen_q)


@triton.autotune(configs=_bwd_configs(),
                 key=["seqlen_q_bucket", "seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL"],
                 do_bench=_autotune_bench)
@triton.jit
def _attention_bwd_dkdv(
    q_ptr, k_ptr, v_ptr, dout_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_deb, stride_deh, stride_dem,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    num_heads, seqlen_q, seqlen_k,
    seqlen_q_bucket, seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """One key block per program; accumulate dK, dV by looping over query blocks.

    Scores are held transposed as [BLOCK_N, BLOCK_M] so no probability tile is
    transposed between matmuls. P is recomputed from Q, K and the stored LSE:
    ``P = exp2(scale*log2(e) * QK^T - log2(e) * LSE)``.
    """
    block_n_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)

    qk_scale = softmax_scale * LOG2E
    offs_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k_ptrs = (k_ptr + batch_idx * stride_kb + head_idx * stride_kh
              + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = (v_ptr + batch_idx * stride_vb + head_idx * stride_vh
              + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    n_mask = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=n_mask, other=0.0)
    v = tl.load(v_ptrs, mask=n_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    do_base = dout_ptr + batch_idx * stride_dob + head_idx * stride_doh
    lse_base = lse_ptr + batch_idx * stride_lb + head_idx * stride_lh
    delta_base = delta_ptr + batch_idx * stride_deb + head_idx * stride_deh

    # Only queries at or below this key block contribute under a causal mask.
    start_m = (block_n_idx * BLOCK_N) // BLOCK_M * BLOCK_M if IS_CAUSAL else 0

    for m in range(start_m, seqlen_q, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        m_mask = offs_m[:, None] < seqlen_q
        q = tl.load(q_ptrs, mask=m_mask, other=0.0)
        do = tl.load(do_ptrs, mask=m_mask, other=0.0)
        lse = tl.load(lse_base + offs_m * stride_lm, mask=offs_m < seqlen_q, other=0.0)
        delta = tl.load(delta_base + offs_m * stride_dem, mask=offs_m < seqlen_q, other=0.0)

        qkT = tl.dot(k, tl.trans(q))
        pT = tl.exp2(qkT * qk_scale - lse[None, :] * LOG2E)
        keep = (offs_n[:, None] < seqlen_k) & (offs_m[None, :] < seqlen_q)
        if IS_CAUSAL:
            keep = keep & (offs_m[None, :] >= offs_n[:, None])
        pT = tl.where(keep, pT, 0.0)

        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)

    dk *= softmax_scale
    dk_ptrs = (dk_ptr + batch_idx * stride_dkb + head_idx * stride_dkh
               + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    dv_ptrs = (dv_ptr + batch_idx * stride_dvb + head_idx * stride_dvh
               + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)


@triton.autotune(configs=_bwd_configs(),
                 key=["seqlen_q_bucket", "seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL"],
                 do_bench=_autotune_bench)
@triton.jit
def _attention_bwd_dq(
    q_ptr, k_ptr, v_ptr, dout_ptr, lse_ptr, delta_ptr, dq_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_deb, stride_deh, stride_dem,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    num_heads, seqlen_q, seqlen_k,
    seqlen_q_bucket, seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """One query block per program; accumulate dQ by looping over key blocks."""
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)

    qk_scale = softmax_scale * LOG2E
    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m[:, None] < seqlen_q

    q_ptrs = (q_ptr + batch_idx * stride_qb + head_idx * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    do_ptrs = (dout_ptr + batch_idx * stride_dob + head_idx * stride_doh
               + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    q = tl.load(q_ptrs, mask=m_mask, other=0.0)
    do = tl.load(do_ptrs, mask=m_mask, other=0.0)
    lse = tl.load(lse_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm,
                  mask=offs_m < seqlen_q, other=0.0)
    delta = tl.load(delta_ptr + batch_idx * stride_deb + head_idx * stride_deh + offs_m * stride_dem,
                    mask=offs_m < seqlen_q, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M) if IS_CAUSAL else seqlen_k

    for n in range(0, max_n, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        n_mask = offs_n[:, None] < seqlen_k
        k = tl.load(k_ptrs, mask=n_mask, other=0.0)
        v = tl.load(v_ptrs, mask=n_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k))
        p = tl.exp2(qk * qk_scale - lse[:, None] * LOG2E)
        keep = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
        if IS_CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(keep, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    dq *= softmax_scale
    dq_ptrs = (dq_ptr + batch_idx * stride_dqb + head_idx * stride_dqh
               + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
    tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)
