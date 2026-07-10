"""Input contracts and platform detection for the public attention APIs."""

from functools import lru_cache
import math
from numbers import Integral, Real

import torch

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SUPPORTED_HEAD_DIMS = (16, 32, 64, 128, 256, 512)
LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


@lru_cache(maxsize=None)
def _device_arch(device_index):
    properties = torch.cuda.get_device_properties(device_index)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0]


def unsupported_reason(device=None):
    if device is not None:
        device = torch.device(device)
    if torch.version.hip is None:
        return "fa-rdna3 requires a ROCm build of PyTorch"
    if not torch.cuda.is_available():
        return "fa-rdna3 requires an available ROCm GPU"
    if device is not None and device.type != "cuda":
        return f"fa-rdna3 requires a ROCm GPU tensor, got device {device}"
    device_index = torch.cuda.current_device() if device is None or device.index is None else device.index
    architecture = _device_arch(device_index)
    if architecture != "gfx1100":
        return f"fa-rdna3 supports gfx1100, got {architecture or 'an unknown GPU architecture'}"
    return None


def is_available(device=None):
    return unsupported_reason(device) is None


def _require_platform(device):
    reason = unsupported_reason(device)
    if reason is not None:
        raise RuntimeError(reason)


def _require_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def validate_dense_inputs(query, key, value, *, allow_gqa=True, dtypes=SUPPORTED_DTYPES):
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _require_tensor(name, tensor)
        if tensor.dim() != 4:
            raise ValueError(f"{name} must have shape (batch, heads, seqlen, head_dim)")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key and value must be on the same device")
    if query.dtype not in dtypes:
        expected = ", ".join(str(dtype).removeprefix("torch.") for dtype in dtypes)
        raise ValueError(f"unsupported dtype {query.dtype}; expected {expected}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key and value must have the same dtype")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("query, key and value must have the same batch size")
    if key.shape[1] != value.shape[1]:
        raise ValueError("key and value must have the same number of heads")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key and value must have the same sequence length")
    if query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
        raise ValueError("query, key and value must have the same head_dim")
    if min(query.shape[0], query.shape[1], key.shape[1], query.shape[2], key.shape[2], query.shape[3]) <= 0:
        raise ValueError("batch, heads, sequence lengths and head_dim must be positive")
    if query.shape[3] > SUPPORTED_HEAD_DIMS[-1]:
        raise ValueError(f"head_dim {query.shape[3]} exceeds the supported maximum {SUPPORTED_HEAD_DIMS[-1]}")
    if query.shape[1] != key.shape[1]:
        if not allow_gqa:
            raise ValueError("different query and key/value head counts require enable_gqa=True")
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("query heads must be a multiple of key/value heads")
    _require_platform(query.device)


def validate_softmax_scale(softmax_scale):
    if softmax_scale is None:
        return
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, Real):
        raise TypeError("softmax_scale must be a real number or None")
    if not math.isfinite(float(softmax_scale)):
        raise ValueError("softmax_scale must be finite")


def validate_attention_options(query, key, *, softmax_scale, window_size, softcap, bias,
                               alibi_slopes, dropout_p):
    validate_softmax_scale(softmax_scale)
    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        raise ValueError("window_size must contain exactly (left, right)")
    if any(isinstance(side, bool) or not isinstance(side, Integral) or side < -1 for side in window_size):
        raise ValueError("window_size values must be integers greater than or equal to -1")
    if isinstance(softcap, bool) or not isinstance(softcap, Real) or not math.isfinite(float(softcap)):
        raise ValueError("softcap must be a finite nonnegative number")
    if softcap < 0:
        raise ValueError("softcap must be nonnegative")
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, Real) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must satisfy 0 <= dropout_p < 1")
    batch, heads, seqlen_q = query.shape[:3]
    if bias is not None:
        _require_tensor("bias", bias)
        if bias.device != query.device:
            raise ValueError("bias must be on the same device as query")
        if bias.dtype not in (query.dtype, torch.float32):
            raise ValueError("bias dtype must match query dtype or be float32")
        if bias.requires_grad:
            raise ValueError("bias gradients are not supported")
        target = (batch, heads, seqlen_q, key.shape[2])
        try:
            broadcast = torch.broadcast_shapes(tuple(bias.shape), target)
        except RuntimeError:
            broadcast = None
        if broadcast != target:
            raise ValueError("bias must be broadcastable to (batch, heads, seqlen_q, seqlen_k)")
    if alibi_slopes is not None:
        _require_tensor("alibi_slopes", alibi_slopes)
        if alibi_slopes.device != query.device:
            raise ValueError("alibi_slopes must be on the same device as query")
        if not alibi_slopes.dtype.is_floating_point:
            raise ValueError("alibi_slopes must have a floating-point dtype")
        if alibi_slopes.requires_grad:
            raise ValueError("ALiBi slope gradients are not supported")
        if alibi_slopes.shape != (heads,):
            raise ValueError(f"alibi_slopes must have shape ({heads},)")


def validate_inference_only(*tensors):
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("this decode API is inference-only and does not support autograd")


def validate_varlen_inputs(query, key, value, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k):
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _require_tensor(name, tensor)
        if tensor.dim() != 3:
            raise ValueError(f"{name} must have shape (total_tokens, heads, head_dim)")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key and value must be on the same device")
    if query.dtype not in LOW_PRECISION_DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("varlen query, key and value must share a float16 or bfloat16 dtype")
    if key.shape != value.shape:
        raise ValueError("varlen key and value must have the same shape")
    if query.shape[2] != key.shape[2]:
        raise ValueError("varlen query, key and value must have the same head_dim")
    if min(query.shape[0], key.shape[0], query.shape[1], key.shape[1], query.shape[2]) <= 0:
        raise ValueError("total tokens, heads and head_dim must be positive")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be a multiple of key/value heads")
    if query.shape[2] not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim {query.shape[2]} is not supported by varlen attention")
    for name, cumulative in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        _require_tensor(name, cumulative)
        if cumulative.device != query.device or cumulative.dtype != torch.int32 or not cumulative.is_contiguous():
            raise ValueError(f"{name} must be a contiguous int32 tensor on {query.device}")
        if cumulative.dim() != 1 or cumulative.numel() < 2:
            raise ValueError(f"{name} must have shape (batch + 1,)")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")
    for name, maximum, total in (("max_seqlen_q", max_seqlen_q, query.shape[0]),
                                 ("max_seqlen_k", max_seqlen_k, key.shape[0])):
        if isinstance(maximum, bool) or not isinstance(maximum, Integral) or not 0 < maximum <= total:
            raise ValueError(f"{name} must be a positive integer no larger than total tokens")
    _require_platform(query.device)


def validate_paged_inputs(query, key, value, block_table, context_lens):
    _require_tensor("query", query)
    _require_tensor("k_cache", key)
    _require_tensor("v_cache", value)
    if query.dim() != 3 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("paged query must be (batch, heads, dim) and caches must be (blocks, block, heads, dim)")
    if key.shape != value.shape:
        raise ValueError("paged key and value caches must have the same shape")
    if query.device != key.device or query.device != value.device:
        raise ValueError("paged query, key and value must be on the same device")
    if query.dtype not in LOW_PRECISION_DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("paged query, key and value must share a float16 or bfloat16 dtype")
    if query.shape[2] != key.shape[3] or query.shape[2] not in SUPPORTED_HEAD_DIMS:
        raise ValueError("paged query and cache head_dim must match a supported head dimension")
    if min(query.shape[0], query.shape[1], query.shape[2], key.shape[0], key.shape[1], key.shape[2]) <= 0:
        raise ValueError("paged batch, heads, blocks, block size and head_dim must be positive")
    if query.shape[1] % key.shape[2] != 0:
        raise ValueError("query heads must be a multiple of cache heads")
    block_size = key.shape[1]
    if block_size & (block_size - 1):
        raise ValueError("paged cache block size must be a power of two")
    for name, metadata, expected_shape in (
        ("block_table", block_table, (query.shape[0], None)),
        ("context_lens", context_lens, (query.shape[0],)),
    ):
        _require_tensor(name, metadata)
        if metadata.device != query.device or metadata.dtype != torch.int32:
            raise ValueError(f"{name} must be an int32 tensor on {query.device}")
        if metadata.dim() != len(expected_shape) or metadata.shape[0] != expected_shape[0]:
            raise ValueError(f"{name} has an invalid shape for batch size {query.shape[0]}")
    if block_table.shape[1] <= 0:
        raise ValueError("block_table must contain at least one logical block")
    _require_platform(query.device)
