"""Split-K decode kernels for RDNA3 (partial attention + LSE combine)."""

import triton
import triton.language as tl

from ._common import LOG2E, _autotune_bench, _split_configs, _attention_inner


@triton.autotune(configs=_split_configs(),
                 key=["seqlen_k_bucket", "HEAD_DIM", "GROUP_SIZE"], do_bench=_autotune_bench,
                 cache_results=True)
@triton.jit
def _attention_split(
    q_ptr, k_ptr, v_ptr, o_partial_ptr, lse_partial_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ops, stride_opb, stride_oph, stride_opm, stride_opd,
    stride_lps, stride_lpb, stride_lph, stride_lpm,
    num_heads, seqlen_q, seqlen_k, seqlen_k_bucket, num_splits,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Attention of one query block against one contiguous slice of the keys.

    Grid is (split, m_block, batch*head). Each program owns ``ceil(seqlen_k /
    num_splits)`` keys, so a tiny query (decode) spreads across the whole GPU
    instead of one workgroup walking the entire cache. Emits a per-split
    normalised output and natural-log LSE for the combine pass to merge.
    """
    split_idx = tl.program_id(0)
    block_m_idx = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE

    blocks_per_split = tl.cdiv(tl.cdiv(seqlen_k, BLOCK_N), num_splits)
    n_start = split_idx * blocks_per_split * BLOCK_N
    n_end = tl.minimum(n_start + blocks_per_split * BLOCK_N, seqlen_k)

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = (q * (softmax_scale * LOG2E)).to(q_ptr.dtype.element_ty)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    acc, l_i, m_i = _attention_inner(
        acc, l_i, m_i, q, k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        offs_m, offs_d, n_start, n_end, seqlen_k, seqlen_q,
        BLOCK_N, HEAD_DIM, True, False, True,
    )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = m_i / LOG2E + tl.log(l_safe)  # -inf for an empty split; ignored on merge

    o_base = (o_partial_ptr + split_idx * stride_ops + batch_idx * stride_opb
              + head_idx * stride_oph)
    o_ptrs = o_base + offs_m[:, None] * stride_opm + offs_d[None, :] * stride_opd
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < seqlen_q)

    lse_base = (lse_partial_ptr + split_idx * stride_lps + batch_idx * stride_lpb
                + head_idx * stride_lph)
    tl.store(lse_base + offs_m * stride_lpm, lse, mask=offs_m < seqlen_q)


@triton.jit
def _attention_combine(
    o_partial_ptr, lse_partial_ptr, out_ptr,
    stride_ops, stride_opb, stride_oph, stride_opm, stride_opd,
    stride_lps, stride_lpb, stride_lph, stride_lpm,
    stride_ob, stride_oh, stride_om, stride_od,
    num_heads, seqlen_q, num_splits,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Merge the per-split partials into the final output via LSE reweighting."""
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = (batch_head // num_heads).to(tl.int64)
    head_idx = (batch_head % num_heads).to(tl.int64)

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_mask = offs_m < seqlen_q

    lse_head = lse_partial_ptr + batch_idx * stride_lpb + head_idx * stride_lph
    o_head = o_partial_ptr + batch_idx * stride_opb + head_idx * stride_oph

    m_global = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    for s in range(0, num_splits, 1):
        lse_s = tl.load(lse_head + s * stride_lps + offs_m * stride_lpm,
                        mask=row_mask, other=float("-inf"))
        m_global = tl.maximum(m_global, lse_s)
    has_any = m_global != float("-inf")
    m_safe = tl.where(has_any, m_global, 0.0)

    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for s in range(0, num_splits, 1):
        lse_s = tl.load(lse_head + s * stride_lps + offs_m * stride_lpm,
                        mask=row_mask, other=float("-inf"))
        weight = tl.where(lse_s != float("-inf"), tl.exp(lse_s - m_safe), 0.0)
        denom += weight
        o_ptrs = (o_head + s * stride_ops + offs_m[:, None] * stride_opm
                  + offs_d[None, :] * stride_opd)
        o_s = tl.load(o_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        acc += weight[:, None] * o_s

    denom_safe = tl.where(denom == 0.0, 1.0, denom)
    out = acc / denom_safe[:, None]
    out_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)
