"""FlashAttention-2 backward kernels for RDNA3 (preprocess, dK/dV, dQ)."""

import triton
import triton.language as tl

from ._common import LOG2E, _autotune_bench, _bwd_configs, _bwd_dkdv_inner, _bwd_dq_inner


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
    bias_ptr, stride_bb, stride_bh, stride_bm, stride_bn,
    num_heads, seqlen_q, seqlen_k,
    seqlen_q_bucket, seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    HAS_BIAS: tl.constexpr = False,
):
    """One K/V-head block per program; accumulate dK, dV by looping over query blocks.

    Scores are held transposed as [BLOCK_N, BLOCK_M] so no probability tile is
    transposed between matmuls. P is recomputed from Q, K and the stored LSE:
    ``P = exp2(scale*log2(e) * QK^T - log2(e) * LSE)``. Under grouped-query
    attention this K/V head is shared by ``GROUP_SIZE`` query heads, so their
    contributions are summed here (which keeps dK/dV race-free without atomics).
    """
    block_n_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    kv_head_idx = (batch_head % num_heads).to(tl.int64)

    qk_scale = softmax_scale * LOG2E
    causal_offset = seqlen_k - seqlen_q  # bottom-right causal alignment
    offs_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k_ptrs = (k_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
              + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = (v_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh
              + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    n_mask = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=n_mask, other=0.0)
    v = tl.load(v_ptrs, mask=n_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Bands depend only on the key block and shape, not on which query head.
    if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
        # Only query rows whose window covers this key block contribute.
        n_lo = block_n_idx * BLOCK_N
        if IS_CAUSAL:
            win_m_lo = tl.maximum(n_lo - causal_offset, 0) // BLOCK_M * BLOCK_M
        elif WINDOW_RIGHT >= 0:
            win_m_lo = tl.maximum(n_lo - WINDOW_RIGHT, 0) // BLOCK_M * BLOCK_M
        else:
            win_m_lo = 0
        if WINDOW_LEFT >= 0:
            win_m_hi = tl.minimum(seqlen_q, (block_n_idx + 1) * BLOCK_N + WINDOW_LEFT)
        else:
            win_m_hi = seqlen_q
    else:
        start_m = tl.maximum(block_n_idx * BLOCK_N - causal_offset, 0) // BLOCK_M * BLOCK_M if IS_CAUSAL else 0
        # Query blocks strictly above the diagonal need no mask, but only if this
        # key block is fully in bounds; the ragged query tail always needs a
        # boundary mask. Clamp to unmasked_end so the diagonal band and the
        # ragged-tail band never overlap.
        n_block_full = (block_n_idx + 1) * BLOCK_N <= seqlen_k
        unmasked_end = seqlen_q // BLOCK_M * BLOCK_M
        if IS_CAUSAL:
            diag_end = (tl.maximum((block_n_idx + 1) * BLOCK_N - causal_offset, 0) + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        else:
            diag_end = 0
        unmasked_start = tl.where(n_block_full, tl.minimum(diag_end, unmasked_end), unmasked_end)

    for g in range(0, GROUP_SIZE, 1):
        q_head_idx = kv_head_idx * GROUP_SIZE + g
        q_base = q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
        do_base = dout_ptr + batch_idx * stride_dob + q_head_idx * stride_doh
        lse_base = lse_ptr + batch_idx * stride_lb + q_head_idx * stride_lh
        delta_base = delta_ptr + batch_idx * stride_deb + q_head_idx * stride_deh
        bias_base = bias_ptr + batch_idx * stride_bb + q_head_idx * stride_bh

        if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qm, stride_qd, stride_dom, stride_dod, stride_lm, stride_dem,
                offs_n, offs_d, win_m_lo, win_m_hi, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, True, IS_CAUSAL, WINDOW_LEFT, WINDOW_RIGHT,
                softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS)
        else:
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qm, stride_qd, stride_dom, stride_dod, stride_lm, stride_dem,
                offs_n, offs_d, start_m, unmasked_start, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, True, IS_CAUSAL,
                softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS)
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qm, stride_qd, stride_dom, stride_dod, stride_lm, stride_dem,
                offs_n, offs_d, unmasked_start, unmasked_end, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, False, IS_CAUSAL,
                softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS)
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qm, stride_qd, stride_dom, stride_dod, stride_lm, stride_dem,
                offs_n, offs_d, unmasked_end, seqlen_q, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, True, IS_CAUSAL,
                softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS)

    dk *= softmax_scale
    dk_ptrs = (dk_ptr + batch_idx * stride_dkb + kv_head_idx * stride_dkh
               + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    dv_ptrs = (dv_ptr + batch_idx * stride_dvb + kv_head_idx * stride_dvh
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
    bias_ptr, stride_bb, stride_bh, stride_bm, stride_bn,
    num_heads, seqlen_q, seqlen_k,
    seqlen_q_bucket, seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False,
    HAS_BIAS: tl.constexpr = False,
):
    """One query block per program; accumulate dQ by looping over key blocks."""
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE

    qk_scale = softmax_scale * LOG2E
    causal_offset = seqlen_k - seqlen_q  # bottom-right causal alignment
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

    k_base = k_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh
    bias_base = bias_ptr + batch_idx * stride_bb + head_idx * stride_bh

    if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
        if WINDOW_LEFT >= 0:
            win_lo = tl.maximum(block_m_idx * BLOCK_M - WINDOW_LEFT, 0) // BLOCK_N * BLOCK_N
        else:
            win_lo = 0
        if IS_CAUSAL:
            win_hi = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset)
        elif WINDOW_RIGHT >= 0:
            win_hi = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + WINDOW_RIGHT)
        else:
            win_hi = seqlen_k
        dq = _bwd_dq_inner(
            dq, q, do, lse, delta, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, win_lo, win_hi, seqlen_q, seqlen_k, qk_scale,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL, WINDOW_LEFT, WINDOW_RIGHT,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
            bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS)
    else:
        if IS_CAUSAL:
            max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset)
            unmasked_n = tl.minimum(tl.maximum(block_m_idx * BLOCK_M + causal_offset, 0), seqlen_k) // BLOCK_N * BLOCK_N
        else:
            max_n = seqlen_k
            unmasked_n = seqlen_k // BLOCK_N * BLOCK_N

        dq = _bwd_dq_inner(
            dq, q, do, lse, delta, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, 0, unmasked_n, seqlen_q, seqlen_k, qk_scale,
            BLOCK_N, HEAD_DIM, False, IS_CAUSAL,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
            bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS)
        dq = _bwd_dq_inner(
            dq, q, do, lse, delta, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, unmasked_n, max_n, seqlen_q, seqlen_k, qk_scale,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
            bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS)

    dq *= softmax_scale
    dq_ptrs = (dq_ptr + batch_idx * stride_dqb + head_idx * stride_dqh
               + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
    tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)
