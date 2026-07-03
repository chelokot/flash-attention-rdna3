"""Drop-in ``scaled_dot_product_attention`` backend for RDNA3.

Calling :func:`enable_rdna3_flash_attention` installs an override of
``torch.nn.functional.scaled_dot_product_attention`` so existing code
(ComfyUI, diffusers, transformers) transparently uses the Triton kernel for the
cases it supports, and defers to the original implementation otherwise.
"""

import torch
import torch.nn.functional as F

from .interface import flash_attention, _SUPPORTED_HEAD_DIMS

_original_sdpa = None


def _is_supported(query, key, value, attn_mask, dropout_p):
    return (
        query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dim() == 4
        and query.shape[-1] in _SUPPORTED_HEAD_DIMS
        and key.shape[-1] == query.shape[-1]
        and value.shape[-1] == query.shape[-1]
        and key.shape[-2] == value.shape[-2]
        and attn_mask is None
        and dropout_p == 0.0
    )


def _dispatch_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
    if not enable_gqa and _is_supported(query, key, value, attn_mask, dropout_p):
        return flash_attention(query, key, value, causal=is_causal, softmax_scale=scale)
    return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def enable_rdna3_flash_attention():
    """Route supported ``scaled_dot_product_attention`` calls to the RDNA3 kernel."""
    global _original_sdpa
    if _original_sdpa is None:
        _original_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _dispatch_sdpa


def disable_rdna3_flash_attention():
    """Restore the original ``scaled_dot_product_attention``."""
    global _original_sdpa
    if _original_sdpa is not None:
        F.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
