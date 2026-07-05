"""Variable-length (cu_seqlens) FlashAttention kernels for RDNA3."""

import triton
import triton.language as tl

from ._common import LOG2E, _autotune_bench, _fwd_configs, _bwd_configs, _attention_inner, _bwd_dkdv_inner, _bwd_dq_inner


@triton.autotune(
    configs=_fwd_configs(),
    key=["max_seqlen_q_bucket", "max_seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL"],
    do_bench=_autotune_bench,
)
@triton.jit
def _attention_forward_varlen(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    softmax_scale,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lh, stride_lt,
    num_heads,
    max_seqlen_q_bucket, max_seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr, PRE_LOAD_V: tl.constexpr = True,
):
    tl.static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0, "HEAD_DIM must be a power of two")

    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = (batch_head % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE

    q_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    seqlen_q = tl.load(cu_seqlens_q_ptr + batch_idx + 1) - q_start
    if block_m_idx * BLOCK_M >= seqlen_q:
        return
    k_start = tl.load(cu_seqlens_k_ptr + batch_idx)
    seqlen_k = tl.load(cu_seqlens_k_ptr + batch_idx + 1) - k_start

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + q_start.to(tl.int64) * stride_qt + head_idx * stride_qh
    k_base = k_ptr + k_start.to(tl.int64) * stride_kt + kv_head_idx * stride_kh
    v_base = v_ptr + k_start.to(tl.int64) * stride_vt + kv_head_idx * stride_vh

    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = (q * (softmax_scale * LOG2E)).to(q_ptr.dtype.element_ty)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M)
        unmasked_n = tl.minimum(block_m_idx * BLOCK_M, seqlen_k) // BLOCK_N * BLOCK_N
    else:
        max_n = seqlen_k
        unmasked_n = seqlen_k // BLOCK_N * BLOCK_N

    acc, l_i, m_i = _attention_inner(
        acc, l_i, m_i, q, k_base, v_base,
        stride_kt, stride_kd, stride_vt, stride_vd,
        offs_m, offs_d, 0, unmasked_n, seqlen_k, seqlen_q,
        BLOCK_N, HEAD_DIM, False, IS_CAUSAL, PRE_LOAD_V,
    )
    acc, l_i, m_i = _attention_inner(
        acc, l_i, m_i, q, k_base, v_base,
        stride_kt, stride_kd, stride_vt, stride_vd,
        offs_m, offs_d, unmasked_n, max_n, seqlen_k, seqlen_q,
        BLOCK_N, HEAD_DIM, True, IS_CAUSAL, PRE_LOAD_V,
    )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    out_base = out_ptr + q_start.to(tl.int64) * stride_ot + head_idx * stride_oh
    out_ptrs = out_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)

    lse = m_i / LOG2E + tl.log(l_safe)
    lse_ptrs = lse_ptr + head_idx * stride_lh + (q_start.to(tl.int64) + offs_m) * stride_lt
    tl.store(lse_ptrs, lse, mask=offs_m < seqlen_q)


@triton.jit
def _attention_bwd_preprocess_varlen(
    out_ptr, dout_ptr, delta_ptr, cu_seqlens_q_ptr,
    stride_ot, stride_oh, stride_od,
    stride_dot, stride_doh, stride_dod,
    stride_dh, stride_dt,
    num_heads,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = (batch_head % num_heads).to(tl.int64)

    q_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    seqlen_q = tl.load(cu_seqlens_q_ptr + batch_idx + 1) - q_start
    if block_m_idx * BLOCK_M >= seqlen_q:
        return

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_mask = offs_m[:, None] < seqlen_q
    token = (q_start.to(tl.int64) + offs_m)

    o_ptrs = (out_ptr + token[:, None] * stride_ot + head_idx * stride_oh
              + offs_d[None, :] * stride_od)
    do_ptrs = (dout_ptr + token[:, None] * stride_dot + head_idx * stride_doh
               + offs_d[None, :] * stride_dod)
    o = tl.load(o_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    tl.store(delta_ptr + head_idx * stride_dh + token * stride_dt, delta, mask=offs_m < seqlen_q)


@triton.autotune(configs=_bwd_configs(),
                 key=["max_seqlen_q_bucket", "max_seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL"],
                 do_bench=_autotune_bench)
@triton.jit
def _attention_bwd_dkdv_varlen(
    q_ptr, k_ptr, v_ptr, dout_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    softmax_scale,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_dot, stride_doh, stride_dod,
    stride_lh, stride_lt,
    stride_deh, stride_det,
    stride_dkt, stride_dkh, stride_dkd,
    stride_dvt, stride_dvh, stride_dvd,
    num_heads,
    max_seqlen_q_bucket, max_seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr,
):
    block_n_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    kv_head_idx = (batch_head % num_heads).to(tl.int64)

    k_start = tl.load(cu_seqlens_k_ptr + batch_idx)
    seqlen_k = tl.load(cu_seqlens_k_ptr + batch_idx + 1) - k_start
    if block_n_idx * BLOCK_N >= seqlen_k:
        return
    q_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    seqlen_q = tl.load(cu_seqlens_q_ptr + batch_idx + 1) - q_start

    qk_scale = softmax_scale * LOG2E
    offs_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k_ptrs = (k_ptr + (k_start.to(tl.int64) + offs_n)[:, None] * stride_kt
              + kv_head_idx * stride_kh + offs_d[None, :] * stride_kd)
    v_ptrs = (v_ptr + (k_start.to(tl.int64) + offs_n)[:, None] * stride_vt
              + kv_head_idx * stride_vh + offs_d[None, :] * stride_vd)
    n_mask = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=n_mask, other=0.0)
    v = tl.load(v_ptrs, mask=n_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    start_m = (block_n_idx * BLOCK_N) // BLOCK_M * BLOCK_M if IS_CAUSAL else 0
    n_block_full = (block_n_idx + 1) * BLOCK_N <= seqlen_k
    unmasked_end = seqlen_q // BLOCK_M * BLOCK_M
    if IS_CAUSAL:
        diag_end = ((block_n_idx + 1) * BLOCK_N + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    else:
        diag_end = 0
    unmasked_start = tl.where(n_block_full, tl.minimum(diag_end, unmasked_end), unmasked_end)

    q_seq = q_ptr + q_start.to(tl.int64) * stride_qt
    do_seq = dout_ptr + q_start.to(tl.int64) * stride_dot
    lse_seq = lse_ptr + q_start.to(tl.int64) * stride_lt
    delta_seq = delta_ptr + q_start.to(tl.int64) * stride_det

    for g in range(0, GROUP_SIZE, 1):
        q_head_idx = kv_head_idx * GROUP_SIZE + g
        q_base = q_seq + q_head_idx * stride_qh
        do_base = do_seq + q_head_idx * stride_doh
        lse_base = lse_seq + q_head_idx * stride_lh
        delta_base = delta_seq + q_head_idx * stride_deh

        dk, dv = _bwd_dkdv_inner(
            dk, dv, k, v, q_base, do_base, lse_base, delta_base,
            stride_qt, stride_qd, stride_dot, stride_dod, stride_lt, stride_det,
            offs_n, offs_d, start_m, unmasked_start, seqlen_q, seqlen_k, qk_scale,
            BLOCK_M, HEAD_DIM, True, IS_CAUSAL)
        dk, dv = _bwd_dkdv_inner(
            dk, dv, k, v, q_base, do_base, lse_base, delta_base,
            stride_qt, stride_qd, stride_dot, stride_dod, stride_lt, stride_det,
            offs_n, offs_d, unmasked_start, unmasked_end, seqlen_q, seqlen_k, qk_scale,
            BLOCK_M, HEAD_DIM, False, IS_CAUSAL)
        dk, dv = _bwd_dkdv_inner(
            dk, dv, k, v, q_base, do_base, lse_base, delta_base,
            stride_qt, stride_qd, stride_dot, stride_dod, stride_lt, stride_det,
            offs_n, offs_d, unmasked_end, seqlen_q, seqlen_q, seqlen_k, qk_scale,
            BLOCK_M, HEAD_DIM, True, IS_CAUSAL)

    dk *= softmax_scale
    token_n = k_start.to(tl.int64) + offs_n
    dk_ptrs = (dk_ptr + token_n[:, None] * stride_dkt + kv_head_idx * stride_dkh
               + offs_d[None, :] * stride_dkd)
    dv_ptrs = (dv_ptr + token_n[:, None] * stride_dvt + kv_head_idx * stride_dvh
               + offs_d[None, :] * stride_dvd)
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)


@triton.autotune(configs=_bwd_configs(),
                 key=["max_seqlen_q_bucket", "max_seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL"],
                 do_bench=_autotune_bench)
@triton.jit
def _attention_bwd_dq_varlen(
    q_ptr, k_ptr, v_ptr, dout_ptr, lse_ptr, delta_ptr, dq_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    softmax_scale,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_dot, stride_doh, stride_dod,
    stride_lh, stride_lt,
    stride_deh, stride_det,
    stride_dqt, stride_dqh, stride_dqd,
    num_heads,
    max_seqlen_q_bucket, max_seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = (batch_head % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE

    q_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    seqlen_q = tl.load(cu_seqlens_q_ptr + batch_idx + 1) - q_start
    if block_m_idx * BLOCK_M >= seqlen_q:
        return
    k_start = tl.load(cu_seqlens_k_ptr + batch_idx)
    seqlen_k = tl.load(cu_seqlens_k_ptr + batch_idx + 1) - k_start

    qk_scale = softmax_scale * LOG2E
    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    token = q_start.to(tl.int64) + offs_m
    m_mask = offs_m[:, None] < seqlen_q

    q_ptrs = q_ptr + token[:, None] * stride_qt + head_idx * stride_qh + offs_d[None, :] * stride_qd
    do_ptrs = dout_ptr + token[:, None] * stride_dot + head_idx * stride_doh + offs_d[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=m_mask, other=0.0)
    do = tl.load(do_ptrs, mask=m_mask, other=0.0)
    lse = tl.load(lse_ptr + head_idx * stride_lh + token * stride_lt, mask=offs_m < seqlen_q, other=0.0)
    delta = tl.load(delta_ptr + head_idx * stride_deh + token * stride_det, mask=offs_m < seqlen_q, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    k_base = k_ptr + k_start.to(tl.int64) * stride_kt + kv_head_idx * stride_kh
    v_base = v_ptr + k_start.to(tl.int64) * stride_vt + kv_head_idx * stride_vh

    if IS_CAUSAL:
        max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M)
        unmasked_n = tl.minimum(block_m_idx * BLOCK_M, seqlen_k) // BLOCK_N * BLOCK_N
    else:
        max_n = seqlen_k
        unmasked_n = seqlen_k // BLOCK_N * BLOCK_N

    dq = _bwd_dq_inner(
        dq, q, do, lse, delta, k_base, v_base,
        stride_kt, stride_kd, stride_vt, stride_vd,
        offs_m, offs_d, 0, unmasked_n, seqlen_q, seqlen_k, qk_scale,
        BLOCK_N, HEAD_DIM, False, IS_CAUSAL)
    dq = _bwd_dq_inner(
        dq, q, do, lse, delta, k_base, v_base,
        stride_kt, stride_kd, stride_vt, stride_vd,
        offs_m, offs_d, unmasked_n, max_n, seqlen_q, seqlen_k, qk_scale,
        BLOCK_N, HEAD_DIM, True, IS_CAUSAL)

    dq *= softmax_scale
    dq_ptrs = dq_ptr + token[:, None] * stride_dqt + head_idx * stride_dqh + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)
