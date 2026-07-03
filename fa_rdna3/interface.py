"""User-facing FlashAttention entry point for RDNA3.

Accepts ``(batch, heads, seqlen, head_dim)`` tensors matching the layout of
``torch.nn.functional.scaled_dot_product_attention`` and dispatches to the
Triton forward/backward kernels through an ``autograd.Function`` so the kernel
is a drop-in differentiable operation.
"""

import math

import torch
import triton

from .kernels import (
    _attention_forward,
    _attention_bwd_preprocess,
    _attention_bwd_dkdv,
    _attention_bwd_dq,
)

_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128, 256)


def _forward(query, key, value, causal, softmax_scale):
    batch, heads, seqlen_q, head_dim = query.shape
    seqlen_k = key.shape[2]

    out = torch.empty_like(query)
    lse = torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device=query.device)

    grid = lambda meta: (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * heads)
    _attention_forward[grid](
        query, key, value, out, lse,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        heads, seqlen_q, seqlen_k,
        triton.next_power_of_2(seqlen_q), triton.next_power_of_2(seqlen_k),
        HEAD_DIM=head_dim,
        IS_CAUSAL=causal,
    )
    return out, lse


def _backward(dout, query, key, value, out, lse, causal, softmax_scale):
    batch, heads, seqlen_q, head_dim = query.shape
    seqlen_k = key.shape[2]
    dout = dout.contiguous()

    delta = torch.empty_like(lse)
    dquery = torch.empty_like(query)
    dkey = torch.empty_like(key)
    dvalue = torch.empty_like(value)

    q_bucket = triton.next_power_of_2(seqlen_q)
    k_bucket = triton.next_power_of_2(seqlen_k)

    pre_grid = lambda meta: (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * heads)
    _attention_bwd_preprocess[pre_grid](
        out, dout, delta,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        heads, seqlen_q,
        HEAD_DIM=head_dim, BLOCK_M=128,
    )

    dkdv_grid = lambda meta: (triton.cdiv(seqlen_k, meta["BLOCK_N"]), batch * heads)
    _attention_bwd_dkdv[dkdv_grid](
        query, key, value, dout, lse, delta, dkey, dvalue,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dkey.stride(0), dkey.stride(1), dkey.stride(2), dkey.stride(3),
        dvalue.stride(0), dvalue.stride(1), dvalue.stride(2), dvalue.stride(3),
        heads, seqlen_q, seqlen_k, q_bucket, k_bucket,
        HEAD_DIM=head_dim, IS_CAUSAL=causal,
    )

    dq_grid = lambda meta: (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * heads)
    _attention_bwd_dq[dq_grid](
        query, key, value, dout, lse, delta, dquery,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dquery.stride(0), dquery.stride(1), dquery.stride(2), dquery.stride(3),
        heads, seqlen_q, seqlen_k, q_bucket, k_bucket,
        HEAD_DIM=head_dim, IS_CAUSAL=causal,
    )
    return dquery, dkey, dvalue


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, causal, softmax_scale):
        out, lse = _forward(query, key, value, causal, softmax_scale)
        ctx.save_for_backward(query, key, value, out, lse)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, out, lse = ctx.saved_tensors
        dquery, dkey, dvalue = _backward(
            dout, query, key, value, out, lse, ctx.causal, ctx.softmax_scale)
        return dquery, dkey, dvalue, None, None


def flash_attention(query, key, value, causal=False, softmax_scale=None):
    """Compute scaled dot-product attention with the RDNA3 Triton kernel.

    Args:
        query, key, value: tensors shaped ``(batch, heads, seqlen, head_dim)``
            in float16 or bfloat16. Key and value share the key sequence length.
        causal: apply a causal mask over the query/key positions.
        softmax_scale: scale applied to the logits; defaults to ``1/sqrt(head_dim)``.

    Returns:
        The attention output shaped like ``query``. Differentiable in q, k, v.
    """
    head_dim = query.shape[-1]
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim {head_dim} not supported; expected one of {_SUPPORTED_HEAD_DIMS}")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {query.dtype}; expected float16 or bfloat16")
    for name, tensor in (("key", key), ("value", value)):
        if tensor.dtype != query.dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} does not match query dtype {query.dtype}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    return _FlashAttention.apply(query, key, value, causal, softmax_scale)
