"""Triton FlashAttention-2 kernels tuned for AMD RDNA3 (gfx1100).

The forward kernel implements the FlashAttention-2 tiling with an online
softmax so the full attention matrix is never materialised. Block sizes are
autotuned over a small grid chosen for RDNA3: 32-lane WMMA fragments, a 128 KB
LDS budget per workgroup, and no async-copy pipelining (unlike CUDA cp.async),
which favours a modest number of stages.
"""

import triton
import triton.language as tl


# (BLOCK_M, BLOCK_N, num_warps) geometries that miscompile in the ROCm Triton
# WMMA backend on gfx1100: they run fast enough to win autotuning but return
# large localized errors for some dtypes. Found by bench/config_sweep.py; kept
# out of the search space so autotuning can only pick a numerically correct
# config.
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


@triton.autotune(configs=_fwd_configs(), key=["seqlen_q", "seqlen_k", "HEAD_DIM", "IS_CAUSAL"])
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
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = batch_head % num_heads

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = softmax_scale * 1.44269504089  # fold in log2(e) so we can use exp2

    if IS_CAUSAL:
        max_n = tl.minimum(seqlen_k, (block_m_idx + 1) * BLOCK_M)
    else:
        max_n = seqlen_k

    for start_n in range(0, max_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k_mask = offs_n[None, :] < seqlen_k
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, k) * qk_scale

        if IS_CAUSAL:
            keep = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seqlen_k)
            qk = tl.where(keep, qk, float("-inf"))
        else:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    out_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)

    lse = m_i / 1.44269504089 + tl.log(l_safe)
    lse_base = lse_ptr + batch_idx * stride_lb + head_idx * stride_lh
    lse_ptrs = lse_base + offs_m * stride_lm
    tl.store(lse_ptrs, lse, mask=offs_m < seqlen_q)
