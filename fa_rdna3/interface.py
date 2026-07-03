"""User-facing FlashAttention entry point for RDNA3.

Accepts ``(batch, heads, seqlen, head_dim)`` tensors matching the layout of
``torch.nn.functional.scaled_dot_product_attention`` and dispatches to the
Triton forward kernel.
"""

import math

import torch
import triton

from .kernels import _attention_forward

_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128, 256)


def flash_attention(query, key, value, causal=False, softmax_scale=None):
    """Compute scaled dot-product attention with the RDNA3 Triton kernel.

    Args:
        query, key, value: tensors shaped ``(batch, heads, seqlen, head_dim)``
            in float16 or bfloat16. Key and value share the key sequence length.
        causal: apply a causal mask over the query/key positions.
        softmax_scale: scale applied to the logits; defaults to ``1/sqrt(head_dim)``.

    Returns:
        The attention output shaped like ``query``.
    """
    batch, heads, seqlen_q, head_dim = query.shape
    seqlen_k = key.shape[2]

    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim {head_dim} not supported; expected one of {_SUPPORTED_HEAD_DIMS}")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {query.dtype}; expected float16 or bfloat16")
    for name, tensor in (("key", key), ("value", value)):
        if tensor.dtype != query.dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} does not match query dtype {query.dtype}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

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
    return out
