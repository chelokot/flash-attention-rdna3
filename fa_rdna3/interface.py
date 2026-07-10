"""User-facing FlashAttention entry points for RDNA3.

``flash_attention`` takes ``(batch, heads, seqlen, head_dim)`` tensors matching
``torch.nn.functional.scaled_dot_product_attention`` and is a drop-in
differentiable op (grouped-query attention supported). ``flash_attention_decode``
is the split-K path for small-query / long-KV decode. ``flash_attention_varlen``
handles packed variable-length sequences via ``cu_seqlens``. Dense attention is
registered as a ``torch.library`` custom op; the packed path uses an
``autograd.Function`` wrapper.
"""

import math
from numbers import Integral

import torch
import triton

from ._validation import (
    LOW_PRECISION_DTYPES,
    SUPPORTED_DTYPES,
    SUPPORTED_HEAD_DIMS,
    validate_attention_options,
    validate_dense_inputs,
    validate_inference_only,
    validate_paged_inputs,
    validate_softmax_scale,
    validate_varlen_inputs,
)
from .kernels import (
    _attention_forward,
    _attention_bwd_preprocess,
    _attention_bwd_dkdv,
    _attention_bwd_dq,
    _attention_split,
    _attention_combine,
    _attention_forward_varlen,
    _attention_bwd_preprocess_varlen,
    _attention_bwd_dkdv_varlen,
    _attention_bwd_dq_varlen,
    _attention_decode_paged,
)

_SUPPORTED_HEAD_DIMS = SUPPORTED_HEAD_DIMS

# The XTX reports 48 workgroup processors through PyTorch. Four programs per
# processor preserves the measured 192-program saturation target on this card.
_DECODE_TARGET_PROGRAMS = 192
_DECODE_BLOCK_M = 16


def _decode_num_splits(batch, heads, seqlen_q, seqlen_k, target_programs=_DECODE_TARGET_PROGRAMS):
    m_blocks = triton.cdiv(seqlen_q, _DECODE_BLOCK_M)
    base = batch * heads * m_blocks
    wanted = (target_programs + base - 1) // base
    key_blocks = triton.cdiv(seqlen_k, 64)
    return max(1, min(32, wanted, key_blocks))


def _paged_num_splits(batch, heads, block_size, max_blocks, target_programs,
                      max_context_len=None, requested_splits=None):
    if requested_splits is not None:
        return requested_splits
    if max_context_len is None:
        active_blocks = max_blocks
        active_tokens = active_blocks * block_size
    else:
        active_blocks = min(max_blocks, triton.cdiv(max_context_len, block_size))
        active_tokens = min(max_blocks * block_size, max_context_len)
    base_programs = batch * heads
    if active_tokens <= 64:
        return 1
    split_cap = min(16, active_blocks, target_programs)
    occupancy_splits = triton.cdiv(5 * target_programs, base_programs)
    work_splits = min(16, triton.cdiv(active_tokens, 256))
    return min(split_cap, max(occupancy_splits, work_splits))


_DTYPE_KEY = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_SUPPORTED_DTYPES = SUPPORTED_DTYPES


def _autotune_seqlen_key(seqlen, dtype):
    # RDNA3 WMMA miscompiles are dtype-dependent, and the autotuner keys on seqlen
    # buckets, not dtype. Fold the dtype into the bucket so a config tuned for one
    # dtype is never reused for another: cross-dtype reuse silently selects a config
    # that miscompiles for the other dtype, producing large localized errors.
    return triton.next_power_of_2(seqlen) * 4 + _DTYPE_KEY[dtype]


def _bias_args(bias, query, batch, heads, seqlen_q, seqlen_k):
    """Return (bias_or_dummy_tensor, sbb, sbh, sbm, sbn, has_bias) for a kernel call.

    A None bias uses ``query`` as an unused dummy pointer with zero strides. A
    present bias is broadcast to (batch, heads, seqlen_q, seqlen_k); broadcast
    dimensions carry stride 0 so the kernel reads the shared value.
    """
    if bias is None:
        return query, 0, 0, 0, 0, False
    bias = bias.expand(batch, heads, seqlen_q, seqlen_k)
    sbb, sbh, sbm, sbn = bias.stride()
    return bias, sbb, sbh, sbm, sbn, True


def _alibi_args(alibi_slopes, query):
    """Return (per-head-slopes-or-dummy tensor, has_alibi). A None uses query."""
    if alibi_slopes is None:
        return query, False
    return alibi_slopes.to(torch.float32).contiguous(), True


def alibi_slopes(n_heads, device="cuda"):
    """Standard ALiBi per-head slopes (Press et al., 2021) as a (n_heads,) tensor."""
    if isinstance(n_heads, bool) or not isinstance(n_heads, Integral) or n_heads <= 0:
        raise ValueError("n_heads must be a positive integer")

    def pow2_slopes(n):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = pow2_slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = pow2_slopes(closest)
        slopes += pow2_slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32, device=device)


def _forward(query, key, value, causal, softmax_scale, window, softcap, bias, alibi, dropout_p, dropout_seed):
    batch, heads, seqlen_q, head_dim = query.shape
    seqlen_k = key.shape[2]
    group_size = heads // key.shape[1]

    out = torch.empty_like(query)
    lse = torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device=query.device)
    grid = lambda meta: (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * heads)
    common_args = (
        query, key, value, out, lse,
        softmax_scale,
        *query.stride(), *key.stride(), *value.stride(), *out.stride(), *lse.stride(),
    )
    shape_args = (
        heads, seqlen_q, seqlen_k,
        _autotune_seqlen_key(seqlen_q, query.dtype), _autotune_seqlen_key(seqlen_k, query.dtype),
    )
    bias_t, sbb, sbh, sbm, sbn, has_bias = _bias_args(
        bias, query, batch, heads, seqlen_q, seqlen_k)
    alibi_t, has_alibi = _alibi_args(alibi, query)
    safe_softmax = (
        has_bias
        or window[0] >= 0
        or window[1] >= 0
        or (seqlen_q > seqlen_k and causal)
    )
    _attention_forward[grid](
        *common_args,
        bias_t, sbb, sbh, sbm, sbn,
        alibi_t, dropout_p, dropout_seed,
        *shape_args,
        HEAD_DIM=head_dim,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        PARALLELISM_BUCKET=int(batch * heads > 1),
        WINDOW_LEFT=window[0],
        WINDOW_RIGHT=window[1],
        softcap=softcap,
        HAS_SOFTCAP=softcap > 0.0,
        HAS_BIAS=has_bias,
        HAS_ALIBI=has_alibi,
        DROPOUT=dropout_p > 0.0,
        SAFE_SOFTMAX=safe_softmax,
        POST_SCALE_Q=query.dtype == torch.bfloat16,
    )
    return out, lse


def _backward(dout, query, key, value, out, lse, causal, softmax_scale, window, softcap, bias, alibi, dropout_p, dropout_seed):
    batch, heads, seqlen_q, head_dim = query.shape
    seqlen_k = key.shape[2]
    kv_heads = key.shape[1]
    group_size = heads // kv_heads
    dout = dout.contiguous()
    bias_t, sbb, sbh, sbm, sbn, has_bias = _bias_args(bias, query, batch, heads, seqlen_q, seqlen_k)
    alibi_t, has_alibi = _alibi_args(alibi, query)
    safe_softmax = (
        has_bias
        or window[0] >= 0
        or window[1] >= 0
        or (seqlen_q > seqlen_k and causal)
    )

    delta = torch.empty_like(lse)
    dquery = torch.empty_like(query)
    dkey = torch.empty_like(key)
    dvalue = torch.empty_like(value)

    q_bucket = _autotune_seqlen_key(seqlen_q, query.dtype)
    k_bucket = _autotune_seqlen_key(seqlen_k, query.dtype)

    pre_grid = lambda meta: (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * heads)
    _attention_bwd_preprocess[pre_grid](
        out, dout, delta,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        heads, seqlen_q,
        HEAD_DIM=head_dim, BLOCK_M=128,
    )

    dkdv_grid = lambda meta: (triton.cdiv(seqlen_k, meta["BLOCK_N"]), batch * kv_heads)
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
        bias_t, sbb, sbh, sbm, sbn,
        alibi_t, dropout_p, dropout_seed,
        kv_heads, seqlen_q, seqlen_k, q_bucket, k_bucket,
        HEAD_DIM=head_dim, IS_CAUSAL=causal, GROUP_SIZE=group_size,
        WINDOW_LEFT=window[0], WINDOW_RIGHT=window[1],
        softcap=softcap, HAS_SOFTCAP=softcap > 0.0, HAS_BIAS=has_bias, HAS_ALIBI=has_alibi,
        DROPOUT=dropout_p > 0.0,
        SAFE_SOFTMAX=safe_softmax,
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
        bias_t, sbb, sbh, sbm, sbn,
        alibi_t, dropout_p, dropout_seed,
        heads, seqlen_q, seqlen_k, q_bucket, k_bucket,
        HEAD_DIM=head_dim, IS_CAUSAL=causal, GROUP_SIZE=group_size,
        WINDOW_LEFT=window[0], WINDOW_RIGHT=window[1],
        softcap=softcap, HAS_SOFTCAP=softcap > 0.0, HAS_BIAS=has_bias, HAS_ALIBI=has_alibi,
        DROPOUT=dropout_p > 0.0,
        SAFE_SOFTMAX=safe_softmax,
    )
    return dquery, dkey, dvalue


# Registered as a torch.library custom op (not a bare autograd.Function) so it is
# opaque to torch.compile with a fake/meta rule, passes torch.library.opcheck, and
# stays functional — the dropout seed is an explicit input, not internal state.
@torch.library.custom_op("fa_rdna3::flash_fwd", mutates_args=())
def _flash_fwd(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               causal: bool, softmax_scale: float, window_left: int, window_right: int,
               softcap: float, bias: "torch.Tensor | None", alibi: "torch.Tensor | None",
               dropout_p: float, dropout_seed: int) -> "tuple[torch.Tensor, torch.Tensor]":
    return _forward(query, key, value, causal, softmax_scale, (window_left, window_right),
                    softcap, bias, alibi, dropout_p, dropout_seed)


@_flash_fwd.register_fake
def _(query, key, value, causal, softmax_scale, window_left, window_right,
      softcap, bias, alibi, dropout_p, dropout_seed):
    out = torch.empty_like(query)
    lse = query.new_empty((query.shape[0], query.shape[1], query.shape[2]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("fa_rdna3::flash_bwd", mutates_args=())
def _flash_bwd(dout: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               out: torch.Tensor, lse: torch.Tensor, causal: bool, softmax_scale: float,
               window_left: int, window_right: int, softcap: float,
               bias: "torch.Tensor | None", alibi: "torch.Tensor | None",
               dropout_p: float, dropout_seed: int) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    return _backward(dout, query, key, value, out, lse, causal, softmax_scale,
                     (window_left, window_right), softcap, bias, alibi, dropout_p, dropout_seed)


@_flash_bwd.register_fake
def _(dout, query, key, value, out, lse, causal, softmax_scale, window_left, window_right,
      softcap, bias, alibi, dropout_p, dropout_seed):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


def _flash_fwd_setup_context(ctx, inputs, output):
    (query, key, value, causal, softmax_scale, window_left, window_right,
     softcap, bias, alibi, dropout_p, dropout_seed) = inputs
    out, lse = output
    ctx.save_for_backward(query, key, value, out, lse, bias, alibi)
    ctx.args = (causal, softmax_scale, window_left, window_right, softcap, dropout_p, dropout_seed)


def _flash_fwd_backward(ctx, grad_out, grad_lse):
    query, key, value, out, lse, bias, alibi = ctx.saved_tensors
    causal, softmax_scale, window_left, window_right, softcap, dropout_p, dropout_seed = ctx.args
    dquery, dkey, dvalue = _flash_bwd(
        grad_out.contiguous(), query, key, value, out, lse, causal, softmax_scale,
        window_left, window_right, softcap, bias, alibi, dropout_p, dropout_seed)
    return dquery, dkey, dvalue, None, None, None, None, None, None, None, None, None


_flash_fwd.register_autograd(_flash_fwd_backward, setup_context=_flash_fwd_setup_context)


def flash_attention(query, key, value, causal=False, softmax_scale=None, window_size=(-1, -1),
                    softcap=0.0, bias=None, alibi_slopes=None, dropout_p=0.0):
    """Compute scaled dot-product attention with the RDNA3 Triton kernel.

    Args:
        query, key, value: tensors shaped ``(batch, heads, seqlen, head_dim)``
            in float16 or bfloat16. Key and value share the key sequence length.
        causal: apply a causal mask over the query/key positions.
        softmax_scale: scale applied to the logits; defaults to ``1/sqrt(head_dim)``.
        window_size: ``(left, right)`` sliding window — query ``i`` attends keys
            ``j`` with ``i - left <= j <= i + right``. ``-1`` on a side means no
            limit (the default ``(-1, -1)`` is full attention). Composes with
            ``causal`` (e.g. Mistral: ``causal=True, window_size=(w - 1, 0)``).
        softcap: if > 0, cap logits as ``softcap * tanh(logit / softcap)`` before
            softmax (Gemma2's attention logit soft-capping). ``0`` disables it.
        bias: optional additive logit bias / mask, broadcastable to
            ``(batch, heads, seqlen_q, seqlen_k)`` and added to the scores before
            softmax (use ``-inf`` entries for a hard mask). Treated as a constant
            (not differentiable); gradients still flow to q, k, v.
        alibi_slopes: optional ``(heads,)`` per-head ALiBi slopes; adds
            ``-slope * abs(query_pos - key_pos)`` to the logits in-kernel (no
            materialised bias). Use :func:`alibi_slopes` for the standard values.
        dropout_p: attention dropout probability applied to the softmax weights
            (kept entries scaled by ``1/(1-p)``). A fresh RNG seed is drawn per
            forward and reused in backward so the mask matches. ``0`` disables it.

    Returns:
        The attention output shaped like ``query``. Differentiable in q, k, v.
    """
    validate_dense_inputs(query, key, value)
    validate_attention_options(
        query, key, softmax_scale=softmax_scale, window_size=window_size, softcap=softcap,
        bias=bias, alibi_slopes=alibi_slopes, dropout_p=dropout_p)
    head_dim = query.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    else:
        softmax_scale = float(softmax_scale)
    window_size = tuple(window_size)
    softcap = float(softcap)
    dropout_p = float(dropout_p)

    # Non-power-of-two head_dim: pad q/k/v to the next supported size with zeros
    # (which contribute nothing to q.k^T or p.v) and slice the output back. The
    # pad/slice are autograd-tracked around the op, so gradients need no change.
    pad = 0
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        padded = max(16, triton.next_power_of_2(head_dim))
        if padded not in _SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim {head_dim} not supported (would pad to {padded})")
        pad = padded - head_dim
        query = torch.nn.functional.pad(query, (0, pad))
        key = torch.nn.functional.pad(key, (0, pad))
        value = torch.nn.functional.pad(value, (0, pad))

    dropout_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item()) if dropout_p > 0.0 else 0
    out, _ = _flash_fwd(query, key, value, causal, softmax_scale, window_size[0], window_size[1],
                        softcap, bias, alibi_slopes, dropout_p, dropout_seed)
    if pad:
        out = out[..., :head_dim]
    return out


def flash_attention_decode(query, key, value, softmax_scale=None):
    """Split-K attention for autoregressive decode (small query, long KV).

    Splits the key/value cache across workgroups so a tiny query saturates the
    GPU instead of leaving one workgroup to walk the whole cache, then merges the
    partial results with an LSE reduction. Non-causal (the cache holds exactly
    the visible keys) and inference-only (not differentiable).
    """
    validate_dense_inputs(query, key, value, dtypes=LOW_PRECISION_DTYPES)
    validate_inference_only(query, key, value)
    validate_softmax_scale(softmax_scale)
    batch, heads, seqlen_q, head_dim = query.shape
    seqlen_k = key.shape[2]
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim {head_dim} not supported; expected one of {_SUPPORTED_HEAD_DIMS}")
    group_size = heads // key.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    else:
        softmax_scale = float(softmax_scale)

    target_programs = 4 * torch.cuda.get_device_properties(query.device).multi_processor_count
    num_splits = _decode_num_splits(batch, heads, seqlen_q, seqlen_k, target_programs)
    o_partial = torch.empty((num_splits, batch, heads, seqlen_q, head_dim),
                            dtype=torch.float32, device=query.device)
    lse_partial = torch.empty((num_splits, batch, heads, seqlen_q),
                              dtype=torch.float32, device=query.device)
    out = torch.empty_like(query)

    split_grid = lambda meta: (num_splits, triton.cdiv(seqlen_q, _DECODE_BLOCK_M), batch * heads)
    _attention_split[split_grid](
        query, key, value, o_partial, lse_partial,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3), o_partial.stride(4),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2), lse_partial.stride(3),
        heads, seqlen_q, seqlen_k, _autotune_seqlen_key(seqlen_k, query.dtype), num_splits,
        HEAD_DIM=head_dim, BLOCK_M=_DECODE_BLOCK_M, GROUP_SIZE=group_size,
    )

    combine_grid = lambda meta: (triton.cdiv(seqlen_q, 64), batch * heads)
    _attention_combine[combine_grid](
        o_partial, lse_partial, out,
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3), o_partial.stride(4),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2), lse_partial.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        heads, seqlen_q, num_splits,
        HEAD_DIM=head_dim, BLOCK_M=64,
    )
    return out


def _forward_varlen(query, key, value, cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k, causal, softmax_scale, window, softcap, alibi):
    total_q, q_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    group_size = q_heads // kv_heads
    batch = cu_seqlens_q.numel() - 1

    out = torch.empty_like(query)
    lse = torch.empty((q_heads, total_q), dtype=torch.float32, device=query.device)
    alibi_t, has_alibi = _alibi_args(alibi, query)

    grid = lambda meta: (triton.cdiv(max_seqlen_q, meta["BLOCK_M"]), batch * q_heads)
    _attention_forward_varlen[grid](
        query, key, value, out, lse,
        cu_seqlens_q, cu_seqlens_k,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        alibi_t,
        q_heads, total_q, key.shape[0],
        _autotune_seqlen_key(max_seqlen_q, query.dtype), _autotune_seqlen_key(max_seqlen_k, query.dtype),
        HEAD_DIM=head_dim, IS_CAUSAL=causal, GROUP_SIZE=group_size,
        WINDOW_LEFT=window[0], WINDOW_RIGHT=window[1],
        softcap=softcap, HAS_SOFTCAP=softcap > 0.0, HAS_ALIBI=has_alibi,
        POST_SCALE_Q=query.dtype == torch.bfloat16,
    )
    return out, lse


def _backward_varlen(dout, query, key, value, out, lse, cu_seqlens_q, cu_seqlens_k,
                     max_seqlen_q, max_seqlen_k, causal, softmax_scale, window, softcap, alibi):
    total_q, q_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    group_size = q_heads // kv_heads
    batch = cu_seqlens_q.numel() - 1
    dout = dout.contiguous()

    delta = torch.empty_like(lse)
    dquery = torch.empty_like(query)
    dkey = torch.empty_like(key)
    dvalue = torch.empty_like(value)
    q_bucket = _autotune_seqlen_key(max_seqlen_q, query.dtype)
    k_bucket = _autotune_seqlen_key(max_seqlen_k, query.dtype)
    alibi_t, has_alibi = _alibi_args(alibi, query)

    pre_grid = lambda meta: (triton.cdiv(max_seqlen_q, meta["BLOCK_M"]), batch * q_heads)
    _attention_bwd_preprocess_varlen[pre_grid](
        out, dout, delta, cu_seqlens_q,
        out.stride(0), out.stride(1), out.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        delta.stride(0), delta.stride(1),
        q_heads, total_q,
        HEAD_DIM=head_dim, BLOCK_M=128,
    )

    dkdv_grid = lambda meta: (triton.cdiv(max_seqlen_k, meta["BLOCK_N"]), batch * kv_heads)
    _attention_bwd_dkdv_varlen[dkdv_grid](
        query, key, value, dout, lse, delta, dkey, dvalue,
        cu_seqlens_q, cu_seqlens_k,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        dkey.stride(0), dkey.stride(1), dkey.stride(2),
        dvalue.stride(0), dvalue.stride(1), dvalue.stride(2),
        alibi_t,
        kv_heads, total_q, key.shape[0],
        q_bucket, k_bucket,
        HEAD_DIM=head_dim, IS_CAUSAL=causal, GROUP_SIZE=group_size,
        WINDOW_LEFT=window[0], WINDOW_RIGHT=window[1],
        softcap=softcap, HAS_SOFTCAP=softcap > 0.0, HAS_ALIBI=has_alibi,
    )

    dq_grid = lambda meta: (triton.cdiv(max_seqlen_q, meta["BLOCK_M"]), batch * q_heads)
    _attention_bwd_dq_varlen[dq_grid](
        query, key, value, dout, lse, delta, dquery,
        cu_seqlens_q, cu_seqlens_k,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        dquery.stride(0), dquery.stride(1), dquery.stride(2),
        alibi_t,
        q_heads, total_q, key.shape[0],
        q_bucket, k_bucket,
        HEAD_DIM=head_dim, IS_CAUSAL=causal, GROUP_SIZE=group_size,
        WINDOW_LEFT=window[0], WINDOW_RIGHT=window[1],
        softcap=softcap, HAS_SOFTCAP=softcap > 0.0, HAS_ALIBI=has_alibi,
    )
    return dquery, dkey, dvalue


class _FlashAttentionVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, causal, softmax_scale, window, softcap, alibi):
        out, lse = _forward_varlen(query, key, value, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k, causal, softmax_scale, window, softcap, alibi)
        ctx.save_for_backward(query, key, value, out, lse, cu_seqlens_q, cu_seqlens_k, alibi)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.window = window
        ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, out, lse, cu_seqlens_q, cu_seqlens_k, alibi = ctx.saved_tensors
        dquery, dkey, dvalue = _backward_varlen(
            dout, query, key, value, out, lse, cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.causal, ctx.softmax_scale, ctx.window, ctx.softcap, alibi)
        return dquery, dkey, dvalue, None, None, None, None, None, None, None, None, None


def flash_attention_varlen(query, key, value, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, causal=False, softmax_scale=None,
                           window_size=(-1, -1), softcap=0.0, alibi_slopes=None):
    """Attention over variable-length sequences packed without padding.

    Args:
        query: ``(total_q, q_heads, head_dim)`` — all sequences concatenated.
        key, value: ``(total_k, kv_heads, head_dim)`` (kv_heads may be fewer for GQA).
        cu_seqlens_q, cu_seqlens_k: int32 ``(batch + 1,)`` cumulative sequence
            lengths, so sequence ``i`` is rows ``cu_seqlens[i]:cu_seqlens[i+1]``.
        max_seqlen_q, max_seqlen_k: longest query / key sequence (grid sizing).
        causal: causal mask within each sequence.
        softmax_scale: defaults to ``1/sqrt(head_dim)``.
        window_size, softcap, alibi_slopes: same as :func:`flash_attention`, applied
            per sequence. (Additive ``bias`` and ``dropout`` are batched-only for now.)

    Returns:
        ``(total_q, q_heads, head_dim)``. Differentiable in q, k, v.
    """
    validate_varlen_inputs(
        query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    validate_attention_options(
        query, key, softmax_scale=softmax_scale, window_size=window_size, softcap=softcap,
        bias=None, alibi_slopes=alibi_slopes, dropout_p=0.0)
    head_dim = query.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    else:
        softmax_scale = float(softmax_scale)

    return _FlashAttentionVarlen.apply(query, key, value, cu_seqlens_q, cu_seqlens_k,
                                       int(max_seqlen_q), int(max_seqlen_k), causal, softmax_scale,
                                       tuple(window_size), float(softcap), alibi_slopes)


def flash_attention_decode_paged(query, k_cache, v_cache, block_table, context_lens,
                                 softmax_scale=None, *, max_context_len=None, num_splits=None):
    """Decode attention over a paged (block-table) KV cache — vLLM style.

    Args:
        query: ``(batch, q_heads, head_dim)`` — one query row per sequence.
        k_cache, v_cache: ``(num_blocks, block_size, kv_heads, head_dim)`` physical
            block pool (kv_heads may be fewer than q_heads for GQA).
        block_table: int32 ``(batch, max_blocks_per_seq)`` — physical block id for
            each logical block of a sequence.
        context_lens: int32 ``(batch,)`` — number of valid keys per sequence.
        softmax_scale: defaults to ``1/sqrt(head_dim)``.
        max_context_len: optional host-known upper bound used only for automatic
            split selection. Defaults to the block-table capacity.
        num_splits: optional explicit split count from 1 to 32.

    Returns:
        ``(batch, q_heads, head_dim)``. Non-causal, inference-only.
    """
    validate_paged_inputs(query, k_cache, v_cache, block_table, context_lens)
    validate_inference_only(query, k_cache, v_cache)
    validate_softmax_scale(softmax_scale)
    if (max_context_len is not None
            and (isinstance(max_context_len, bool) or not isinstance(max_context_len, Integral)
                 or max_context_len < 0)):
        raise ValueError("max_context_len must be a nonnegative integer or None")
    if (num_splits is not None
            and (isinstance(num_splits, bool) or not isinstance(num_splits, Integral)
                 or not 1 <= num_splits <= 32)):
        raise ValueError("num_splits must be an integer between 1 and 32 or None")
    if max_context_len is not None:
        max_context_len = int(max_context_len)
    if num_splits is not None:
        num_splits = int(num_splits)
    batch, q_heads, head_dim = query.shape
    block_size, kv_heads = k_cache.shape[1], k_cache.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    else:
        softmax_scale = float(softmax_scale)

    target_programs = 2 * torch.cuda.get_device_properties(query.device).multi_processor_count
    selected_splits = int(_paged_num_splits(
        batch, q_heads, block_size, block_table.shape[1], target_programs,
        max_context_len=max_context_len, requested_splits=num_splits))
    out = torch.empty_like(query)
    if selected_splits == 1:
        partial = out
        lse_partial = query
        stride_os = 0
        stride_ob, stride_oh, stride_od = out.stride()
        stride_lb = stride_lh = stride_ls = 0
    else:
        partial = torch.empty(
            (batch, q_heads, selected_splits, head_dim), dtype=torch.float32, device=query.device)
        lse_partial = torch.empty(
            (batch, q_heads, selected_splits), dtype=torch.float32, device=query.device)
        stride_ob, stride_oh, stride_os, stride_od = partial.stride()
        stride_lb, stride_lh, stride_ls = lse_partial.stride()

    grid = (selected_splits, batch * q_heads)
    _attention_decode_paged[grid](
        query, k_cache, v_cache, partial, lse_partial, block_table, context_lens,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lb, stride_lh, stride_ls,
        block_table.stride(0), block_table.stride(1),
        context_lens.stride(0),
        q_heads, k_cache.shape[0], block_table.shape[1], selected_splits,
        HEAD_DIM=head_dim, BLOCK_SIZE=block_size, GROUP_SIZE=q_heads // kv_heads,
        STORE_LSE=selected_splits > 1,
    )
    if selected_splits > 1:
        combine_grid = (1, batch * q_heads)
        _attention_combine[combine_grid](
            partial, lse_partial, out,
            partial.stride(2), partial.stride(0), partial.stride(1), 0, partial.stride(3),
            lse_partial.stride(2), lse_partial.stride(0), lse_partial.stride(1), 0,
            out.stride(0), out.stride(1), 0, out.stride(2),
            q_heads, 1, selected_splits,
            HEAD_DIM=head_dim, BLOCK_M=1,
        )
    return out
