"""Variable-length (cu_seqlens) FlashAttention kernels for RDNA3."""

import triton
import triton.language as tl

from ._common import (
    LOG2E,
    _attention_inner,
    _autotune_bench,
    _bwd_dkdv_configs,
    _bwd_dkdv_inner,
    _bwd_dq_configs,
    _bwd_dq_inner,
    _fwd_configs,
    _prune_bwd_dkdv_configs,
    _prune_bwd_dq_configs,
    _prune_configs_by_head_dim,
)


@triton.jit
def _bounded_sequence(cu_seqlens_ptr, batch_idx, total_tokens):
    start = tl.load(cu_seqlens_ptr + batch_idx)
    end = tl.load(cu_seqlens_ptr + batch_idx + 1)
    start = tl.minimum(tl.maximum(start, 0), total_tokens)
    end = tl.minimum(tl.maximum(end, 0), total_tokens)
    return start, tl.maximum(end - start, 0)


@triton.autotune(
    configs=_fwd_configs(include_d64_specializations=False),
    key=[
        "max_seqlen_q_bucket", "max_seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL", "GROUP_SIZE",
        "WINDOW_LEFT", "WINDOW_RIGHT", "HAS_SOFTCAP", "HAS_ALIBI",
        "POST_SCALE_Q",
    ],
    do_bench=_autotune_bench,
    prune_configs_by={"early_config_prune": _prune_configs_by_head_dim},
    cache_results=True,
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
    alibi_ptr,
    num_heads, total_q, total_k,
    max_seqlen_q_bucket, max_seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False, HAS_ALIBI: tl.constexpr = False,
    POST_SCALE_Q: tl.constexpr = False,
    PRE_LOAD_V: tl.constexpr = True,
):
    tl.static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0, "HEAD_DIM must be a power of two")

    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = (batch_head % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE

    q_start, seqlen_q = _bounded_sequence(cu_seqlens_q_ptr, batch_idx, total_q)
    if block_m_idx * BLOCK_M >= seqlen_q:
        return
    k_start, seqlen_k = _bounded_sequence(cu_seqlens_k_ptr, batch_idx, total_k)
    causal_offset = seqlen_k - seqlen_q
    safe_softmax = IS_CAUSAL or WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + q_start.to(tl.int64) * stride_qt + head_idx * stride_qh
    k_base = k_ptr + k_start.to(tl.int64) * stride_kt + kv_head_idx * stride_kh
    v_base = v_ptr + k_start.to(tl.int64) * stride_vt + kv_head_idx * stride_vh

    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    if not POST_SCALE_Q:
        q = (q * (softmax_scale * LOG2E)).to(q_ptr.dtype.element_ty)
    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_ptr + head_idx)
    else:
        alibi_slope = 0.0

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
        if WINDOW_LEFT >= 0:
            win_lo = tl.maximum(block_m_idx * BLOCK_M + causal_offset - WINDOW_LEFT, 0) // BLOCK_N * BLOCK_N
        else:
            win_lo = 0
        if IS_CAUSAL:
            win_hi = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset)
        elif WINDOW_RIGHT >= 0:
            win_hi = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset + WINDOW_RIGHT)
        else:
            win_hi = seqlen_k
        acc, l_i, m_i = _attention_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kt, stride_kd, stride_vt, stride_vd,
            offs_m, offs_d, win_lo, win_hi, seqlen_k, seqlen_q,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL, PRE_LOAD_V, WINDOW_LEFT, WINDOW_RIGHT,
            SAFE_SOFTMAX=safe_softmax,
            softmax_scale=softmax_scale, POST_SCALE_Q=POST_SCALE_Q,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP, alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)
    else:
        if IS_CAUSAL:
            max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset)
            unmasked_n = tl.minimum(tl.maximum(block_m_idx * BLOCK_M + causal_offset, 0), seqlen_k) // BLOCK_N * BLOCK_N
        else:
            max_n = seqlen_k
            unmasked_n = seqlen_k // BLOCK_N * BLOCK_N

        acc, l_i, m_i = _attention_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kt, stride_kd, stride_vt, stride_vd,
            offs_m, offs_d, 0, unmasked_n, seqlen_k, seqlen_q,
            BLOCK_N, HEAD_DIM, False, IS_CAUSAL, PRE_LOAD_V,
            SAFE_SOFTMAX=safe_softmax,
            softmax_scale=softmax_scale, POST_SCALE_Q=POST_SCALE_Q,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP, alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)
        acc, l_i, m_i = _attention_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kt, stride_kd, stride_vt, stride_vd,
            offs_m, offs_d, unmasked_n, max_n, seqlen_k, seqlen_q,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL, PRE_LOAD_V,
            SAFE_SOFTMAX=safe_softmax,
            softmax_scale=softmax_scale, POST_SCALE_Q=POST_SCALE_Q,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP, alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)

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
    num_heads, total_q,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = (batch_head % num_heads).to(tl.int64)

    q_start, seqlen_q = _bounded_sequence(cu_seqlens_q_ptr, batch_idx, total_q)
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


@triton.autotune(configs=_bwd_dkdv_configs(include_d64_specialization=False),
                 key=["max_seqlen_q_bucket", "max_seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL", "GROUP_SIZE",
                      "WINDOW_LEFT", "WINDOW_RIGHT", "HAS_SOFTCAP", "HAS_ALIBI"],
                 do_bench=_autotune_bench,
                 prune_configs_by={"early_config_prune": _prune_bwd_dkdv_configs},
                 cache_results=True)
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
    alibi_ptr,
    num_heads, total_q, total_k,
    max_seqlen_q_bucket, max_seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False, HAS_ALIBI: tl.constexpr = False,
):
    block_n_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    kv_head_idx = (batch_head % num_heads).to(tl.int64)

    k_start, seqlen_k = _bounded_sequence(cu_seqlens_k_ptr, batch_idx, total_k)
    if block_n_idx * BLOCK_N >= seqlen_k:
        return
    q_start, seqlen_q = _bounded_sequence(cu_seqlens_q_ptr, batch_idx, total_q)
    causal_offset = seqlen_k - seqlen_q
    safe_lse = IS_CAUSAL or WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0

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

    if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
        n_lo = block_n_idx * BLOCK_N
        if IS_CAUSAL:
            win_m_lo = tl.maximum(n_lo - causal_offset, 0) // BLOCK_M * BLOCK_M
        elif WINDOW_RIGHT >= 0:
            win_m_lo = tl.maximum(n_lo - causal_offset - WINDOW_RIGHT, 0) // BLOCK_M * BLOCK_M
        else:
            win_m_lo = 0
        if WINDOW_LEFT >= 0:
            win_m_hi = tl.minimum(seqlen_q, (block_n_idx + 1) * BLOCK_N - causal_offset + WINDOW_LEFT)
        else:
            win_m_hi = seqlen_q
    else:
        start_m = tl.maximum(block_n_idx * BLOCK_N - causal_offset, 0) // BLOCK_M * BLOCK_M if IS_CAUSAL else 0
        n_block_full = (block_n_idx + 1) * BLOCK_N <= seqlen_k
        unmasked_end = seqlen_q // BLOCK_M * BLOCK_M
        if IS_CAUSAL:
            diag_end = (tl.maximum((block_n_idx + 1) * BLOCK_N - causal_offset, 0) + BLOCK_M - 1) // BLOCK_M * BLOCK_M
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
        if HAS_ALIBI:
            alibi_slope = tl.load(alibi_ptr + q_head_idx)
        else:
            alibi_slope = 0.0
        if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qt, stride_qd, stride_dot, stride_dod, stride_lt, stride_det,
                offs_n, offs_d, win_m_lo, win_m_hi, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, True, IS_CAUSAL, WINDOW_LEFT, WINDOW_RIGHT,
                SAFE_LSE=safe_lse, softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)
        else:
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qt, stride_qd, stride_dot, stride_dod, stride_lt, stride_det,
                offs_n, offs_d, start_m, unmasked_start, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, True, IS_CAUSAL,
                SAFE_LSE=safe_lse, softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qt, stride_qd, stride_dot, stride_dod, stride_lt, stride_det,
                offs_n, offs_d, unmasked_start, unmasked_end, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, False, IS_CAUSAL,
                SAFE_LSE=safe_lse, softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v, q_base, do_base, lse_base, delta_base,
                stride_qt, stride_qd, stride_dot, stride_dod, stride_lt, stride_det,
                offs_n, offs_d, unmasked_end, seqlen_q, seqlen_q, seqlen_k, qk_scale,
                BLOCK_M, HEAD_DIM, True, IS_CAUSAL,
                SAFE_LSE=safe_lse, softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
                alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)

    dk *= softmax_scale
    token_n = k_start.to(tl.int64) + offs_n
    dk_ptrs = (dk_ptr + token_n[:, None] * stride_dkt + kv_head_idx * stride_dkh
               + offs_d[None, :] * stride_dkd)
    dv_ptrs = (dv_ptr + token_n[:, None] * stride_dvt + kv_head_idx * stride_dvh
               + offs_d[None, :] * stride_dvd)
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)


@triton.autotune(configs=_bwd_dq_configs(include_d64_specialization=False),
                 key=["max_seqlen_q_bucket", "max_seqlen_k_bucket", "HEAD_DIM", "IS_CAUSAL", "GROUP_SIZE",
                      "WINDOW_LEFT", "WINDOW_RIGHT", "HAS_SOFTCAP", "HAS_ALIBI"],
                 do_bench=_autotune_bench,
                 prune_configs_by={"early_config_prune": _prune_bwd_dq_configs},
                 cache_results=True)
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
    alibi_ptr,
    num_heads, total_q, total_k,
    max_seqlen_q_bucket, max_seqlen_k_bucket,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, GROUP_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1, WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0, HAS_SOFTCAP: tl.constexpr = False, HAS_ALIBI: tl.constexpr = False,
):
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = (batch_head % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE

    q_start, seqlen_q = _bounded_sequence(cu_seqlens_q_ptr, batch_idx, total_q)
    if block_m_idx * BLOCK_M >= seqlen_q:
        return
    k_start, seqlen_k = _bounded_sequence(cu_seqlens_k_ptr, batch_idx, total_k)
    causal_offset = seqlen_k - seqlen_q
    safe_lse = IS_CAUSAL or WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0

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
    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_ptr + head_idx)
    else:
        alibi_slope = 0.0

    if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
        if WINDOW_LEFT >= 0:
            win_lo = tl.maximum(block_m_idx * BLOCK_M + causal_offset - WINDOW_LEFT, 0) // BLOCK_N * BLOCK_N
        else:
            win_lo = 0
        if IS_CAUSAL:
            win_hi = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset)
        elif WINDOW_RIGHT >= 0:
            win_hi = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset + WINDOW_RIGHT)
        else:
            win_hi = seqlen_k
        dq = _bwd_dq_inner(
            dq, q, do, lse, delta, k_base, v_base,
            stride_kt, stride_kd, stride_vt, stride_vd,
            offs_m, offs_d, win_lo, win_hi, seqlen_q, seqlen_k, qk_scale,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL, WINDOW_LEFT, WINDOW_RIGHT,
            SAFE_LSE=safe_lse,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP, alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)
    else:
        if IS_CAUSAL:
            max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset)
            unmasked_n = tl.minimum(tl.maximum(block_m_idx * BLOCK_M + causal_offset, 0), seqlen_k) // BLOCK_N * BLOCK_N
        else:
            max_n = seqlen_k
            unmasked_n = seqlen_k // BLOCK_N * BLOCK_N

        dq = _bwd_dq_inner(
            dq, q, do, lse, delta, k_base, v_base,
            stride_kt, stride_kd, stride_vt, stride_vd,
            offs_m, offs_d, 0, unmasked_n, seqlen_q, seqlen_k, qk_scale,
            BLOCK_N, HEAD_DIM, False, IS_CAUSAL,
            SAFE_LSE=safe_lse,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP, alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)
        dq = _bwd_dq_inner(
            dq, q, do, lse, delta, k_base, v_base,
            stride_kt, stride_kd, stride_vt, stride_vd,
            offs_m, offs_d, unmasked_n, max_n, seqlen_q, seqlen_k, qk_scale,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL,
            SAFE_LSE=safe_lse,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP, alibi_slope=alibi_slope, HAS_ALIBI=HAS_ALIBI)

    dq *= softmax_scale
    dq_ptrs = dq_ptr + token[:, None] * stride_dqt + head_idx * stride_dqh + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)
