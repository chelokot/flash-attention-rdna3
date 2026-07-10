"""Paged-KV decode kernel for RDNA3 (vLLM-style block-table cache).

The K/V cache is stored as fixed-size physical blocks; each sequence's logical
positions are mapped to physical blocks by a per-sequence block table. A decode
step (one query row) gathers its keys/values one page at a time. seqlen_q == 1,
so the attention is a vector-matrix reduction — no WMMA — which suits the tiny
query and keeps the gather simple (one key tile == one page).
"""

import triton
import triton.language as tl

from ._common import LOG2E


@triton.jit
def _attention_decode_paged(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr, lse_ptr, block_table_ptr, context_lens_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kblk, stride_kpos, stride_kh, stride_kd,
    stride_vblk, stride_vpos, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_btb, stride_btk,
    stride_clb,
    num_heads, num_cache_blocks, max_table_blocks, num_splits,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    split_idx = tl.program_id(0)
    pid = tl.program_id(1)
    batch_idx = (pid // num_heads).to(tl.int64)
    head_idx = (pid % num_heads).to(tl.int64)
    kv_head_idx = head_idx // GROUP_SIZE

    context_len = tl.load(context_lens_ptr + batch_idx * stride_clb)
    context_len = tl.minimum(tl.maximum(context_len, 0), max_table_blocks * BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_pos = tl.arange(0, BLOCK_SIZE)

    q = tl.load(q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd)
    q = (q.to(tl.float32) * (softmax_scale * LOG2E))

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    num_blocks = tl.cdiv(context_len, BLOCK_SIZE)
    blocks_per_split = num_blocks // num_splits
    extra_blocks = num_blocks % num_splits
    block_begin = split_idx * blocks_per_split + tl.minimum(split_idx, extra_blocks)
    block_count = blocks_per_split + (split_idx < extra_blocks).to(tl.int32)
    block_end = block_begin + block_count
    for blk in range(block_begin, block_end, 1):
        phys = tl.load(block_table_ptr + batch_idx * stride_btb + blk * stride_btk).to(tl.int64)
        pos = blk * BLOCK_SIZE + offs_pos
        valid_phys = (phys >= 0) & (phys < num_cache_blocks)
        pos_mask = (pos < context_len) & valid_phys

        k_ptrs = (k_cache_ptr + phys * stride_kblk + offs_pos[:, None] * stride_kpos
                  + kv_head_idx * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=pos_mask[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1)   # log2-domain scores, one page
        qk = tl.where(pos_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        empty = m_new == float("-inf")
        m_safe = tl.where(empty, 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        v_ptrs = (v_cache_ptr + phys * stride_vblk + offs_pos[:, None] * stride_vpos
                  + kv_head_idx * stride_vh + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=pos_mask[:, None], other=0.0)
        acc += tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = tl.where(empty, m_i, m_new)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe
    out_ptrs = (out_ptr + batch_idx * stride_ob + head_idx * stride_oh
                + split_idx * stride_os + offs_d * stride_od)
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty))
    if STORE_LSE:
        lse = m_i / LOG2E + tl.log(l_safe)
        lse_offset = batch_idx * stride_lb + head_idx * stride_lh + split_idx * stride_ls
        tl.store(lse_ptr + lse_offset, lse)
