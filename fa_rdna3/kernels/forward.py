"""FlashAttention-2 forward kernel for RDNA3."""

import triton
import triton.language as tl

from ._common import LOG2E, _autotune_bench, _fwd_configs, _attention_inner


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
    bias_ptr, stride_bb, stride_bh, stride_bm, stride_bn,
    num_heads, seqlen_q, seqlen_k,
    seqlen_q_bucket, seqlen_k_bucket,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr = -1,
    WINDOW_RIGHT: tl.constexpr = -1,
    softcap=0.0,
    HAS_SOFTCAP: tl.constexpr = False,
    HAS_BIAS: tl.constexpr = False,
    PRE_LOAD_V: tl.constexpr = True,
):
    tl.static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0, "HEAD_DIM must be a power of two")

    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE  # grouped-query: queries in a group share a K/V head

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh
    bias_base = bias_ptr + batch_idx * stride_bb + head_idx * stride_bh

    causal_offset = seqlen_k - seqlen_q  # bottom-right causal alignment
    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = (q * (softmax_scale * LOG2E)).to(q_ptr.dtype.element_ty)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
        # Sliding window: keys outside [i-left, i+right] are dropped, so the loop
        # is bounded to the window band and every tile is masked (the window can
        # cut through what would otherwise be a full unmasked tile).
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
        acc, l_i, m_i = _attention_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, win_lo, win_hi, seqlen_k, seqlen_q,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL, PRE_LOAD_V, WINDOW_LEFT, WINDOW_RIGHT,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
            bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS,
        )
    else:
        if IS_CAUSAL:
            max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M + causal_offset)
            # Key blocks strictly below the diagonal and inside the key bound need
            # no mask; the diagonal band and any ragged key tail inside it do.
            unmasked_n = tl.minimum(tl.maximum(block_m_idx * BLOCK_M + causal_offset, 0), seqlen_k) // BLOCK_N * BLOCK_N
        else:
            max_n = seqlen_k
            # Only the ragged tail past the last whole block needs a boundary mask.
            unmasked_n = seqlen_k // BLOCK_N * BLOCK_N

        acc, l_i, m_i = _attention_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, 0, unmasked_n, seqlen_k, seqlen_q,
            BLOCK_N, HEAD_DIM, False, IS_CAUSAL, PRE_LOAD_V,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
            bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS,
        )
        acc, l_i, m_i = _attention_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, unmasked_n, max_n, seqlen_k, seqlen_q,
            BLOCK_N, HEAD_DIM, True, IS_CAUSAL, PRE_LOAD_V,
            softcap=softcap, HAS_SOFTCAP=HAS_SOFTCAP,
            bias_base=bias_base, stride_bm=stride_bm, stride_bn=stride_bn, HAS_BIAS=HAS_BIAS,
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
