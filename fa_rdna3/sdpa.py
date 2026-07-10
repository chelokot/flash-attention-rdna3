"""Drop-in ``scaled_dot_product_attention`` backend for RDNA3.

Calling :func:`enable_rdna3_flash_attention` installs an override of
``torch.nn.functional.scaled_dot_product_attention`` so existing code
(ComfyUI, diffusers, transformers) transparently uses the Triton kernel for the
cases it supports, and defers to the original implementation otherwise.
"""

from contextlib import contextmanager
import math
from numbers import Real
import threading

import torch
import torch.nn.functional as F

from ._validation import is_available
from .interface import flash_attention, flash_attention_decode, _SUPPORTED_HEAD_DIMS

_original_sdpa = None
_patch_lock = threading.RLock()
_context_depth = 0
_context_installed = False

# Below this many query rows against a long enough cache, split-K decode wins by
# fanning the cache across the GPU instead of one workgroup per (batch, head).
_DECODE_MAX_QUERY = 8
_DECODE_MIN_KEY = 1024

# For very short sequences our backward launches three separate kernels whose
# fixed overhead dominates the trivial compute, so torch's fused math backward
# wins (measured: 12-token GQA fwd+bwd 0.68ms vs 0.50ms). The forward alone still
# wins, so only step aside when a backward will follow (grad enabled). torch's
# math backward is cheap and stable at this size — unlike its large-S path, which
# is the erratic one we exist to replace, so the threshold stays deliberately low.
_MIN_TRAIN_SEQLEN = 16


def _is_supported_mask(attn_mask, query, key):
    if attn_mask is None:
        return True
    if not isinstance(attn_mask, torch.Tensor):
        return False
    if (attn_mask.device != query.device or attn_mask.requires_grad
            or attn_mask.dtype not in (torch.bool, query.dtype, torch.float32)):
        return False
    target = (query.shape[0], query.shape[1], query.shape[2], key.shape[2])
    try:
        return torch.broadcast_shapes(tuple(attn_mask.shape), target) == target
    except RuntimeError:
        return False


def _is_supported_scale(scale):
    return scale is None or (
        not isinstance(scale, bool) and isinstance(scale, Real) and math.isfinite(float(scale)))


def _is_supported(query, key, value, attn_mask, dropout_p, enable_gqa, scale):
    return (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and query.is_cuda
        and is_available(query.device)
        and key.device == query.device
        and value.device == query.device
        and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.dim() == 4
        and key.dim() == 4
        and value.dim() == 4
        and query.shape[0] == key.shape[0] == value.shape[0]
        and query.shape[-2] > 0
        and key.shape[-2] > 0
        and query.shape[-1] <= 512  # non-power-of-two dims are padded internally
        and query.shape[-1] > 0
        and key.shape[-1] == query.shape[-1]
        and value.shape[-1] == query.shape[-1]
        and key.shape[-2] == value.shape[-2]
        and key.shape[-3] == value.shape[-3]
        and query.shape[-3] > 0
        and key.shape[-3] > 0
        and (query.shape[-3] == key.shape[-3]
             or (enable_gqa and query.shape[-3] % key.shape[-3] == 0))
        and _is_supported_mask(attn_mask, query, key)
        and _is_supported_scale(scale)
        and not isinstance(dropout_p, bool)
        and isinstance(dropout_p, Real)
        and dropout_p == 0.0
    )


def _mask_to_bias(attn_mask):
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        return torch.zeros_like(attn_mask, dtype=torch.float32).masked_fill(~attn_mask, float("-inf"))
    return attn_mask


def _dispatch_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
    if (_is_supported(query, key, value, attn_mask, dropout_p, enable_gqa, scale)
            and (not is_causal
                 or (attn_mask is None and query.shape[-2] == key.shape[-2]))):
        requires_backward = torch.is_grad_enabled() and (
            query.requires_grad or key.requires_grad or value.requires_grad)
        if (requires_backward
                and query.shape[-2] <= _MIN_TRAIN_SEQLEN and key.shape[-2] <= _MIN_TRAIN_SEQLEN):
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                  is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        bias = _mask_to_bias(attn_mask)
        if (bias is None and not is_causal and not requires_backward
                and query.dtype in (torch.float16, torch.bfloat16)
                and query.shape[-1] in _SUPPORTED_HEAD_DIMS
                and query.shape[-2] <= _DECODE_MAX_QUERY
                and key.shape[-2] >= _DECODE_MIN_KEY):
            return flash_attention_decode(query, key, value, softmax_scale=scale)
        return flash_attention(query, key, value, causal=is_causal, softmax_scale=scale, bias=bias)
    return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def enable_rdna3_flash_attention():
    """Route supported ``scaled_dot_product_attention`` calls to the RDNA3 kernel."""
    global _original_sdpa
    with _patch_lock:
        if _original_sdpa is None:
            _original_sdpa = F.scaled_dot_product_attention
        if F.scaled_dot_product_attention is _original_sdpa:
            F.scaled_dot_product_attention = _dispatch_sdpa


def disable_rdna3_flash_attention():
    """Restore the original ``scaled_dot_product_attention``."""
    with _patch_lock:
        if _original_sdpa is not None and F.scaled_dot_product_attention is _dispatch_sdpa:
            F.scaled_dot_product_attention = _original_sdpa


@contextmanager
def use_rdna3_flash_attention():
    """Temporarily route supported SDPA calls through the RDNA3 kernel."""
    global _context_depth, _context_installed
    with _patch_lock:
        if _context_depth == 0:
            already_installed = F.scaled_dot_product_attention is _dispatch_sdpa
            enable_rdna3_flash_attention()
            _context_installed = (
                not already_installed and F.scaled_dot_product_attention is _dispatch_sdpa)
        _context_depth += 1
    try:
        yield
    finally:
        with _patch_lock:
            _context_depth -= 1
            if _context_depth == 0:
                if _context_installed:
                    disable_rdna3_flash_attention()
                _context_installed = False
