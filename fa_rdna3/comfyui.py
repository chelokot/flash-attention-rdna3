"""ComfyUI attention adapter and model patch node."""

from collections.abc import Callable
from functools import wraps
import logging
from typing import Protocol

import torch

from ._validation import unsupported_reason
from .interface import flash_attention
from .sdpa import (
    _bypass_rdna3_flash_attention,
    _is_supported,
    _mask_to_bias,
    enable_rdna3_flash_attention,
)

AttentionFunction = Callable[..., torch.Tensor]
PrecisionResolver = Callable[[torch.dtype | None, torch.dtype], torch.dtype | None]
logger = logging.getLogger(__name__)


class ComfyModel(Protocol):
    model_options: dict[str, dict[str, object]]

    def clone(self) -> "ComfyModel": ...


def _reshape_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    skip_reshape: bool,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0:
        return None
    expected_rank = 4 if skip_reshape else 3
    if any(tensor.dim() != expected_rank for tensor in (query, key, value)):
        return None
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        return None
    if skip_reshape:
        if query.shape[1] != heads:
            return None
        return query, key, value

    if query.shape[2] % heads != 0:
        return None
    head_dim = query.shape[2] // heads
    if head_dim == 0 or key.shape[2] % head_dim != 0 or value.shape[2] % head_dim != 0:
        return None
    key_heads = key.shape[2] // head_dim
    value_heads = value.shape[2] // head_dim
    if key_heads != value_heads:
        return None
    if enable_gqa:
        if key_heads == 0 or heads % key_heads != 0:
            return None
    elif key_heads != heads:
        return None

    batch = query.shape[0]
    query = query.reshape(batch, query.shape[1], heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, key.shape[1], key_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, value.shape[1], value_heads, head_dim).transpose(1, 2)
    return query, key, value


def _normalize_mask(mask: object) -> object:
    if not isinstance(mask, torch.Tensor):
        return mask
    if mask.dim() == 2:
        return mask.unsqueeze(0)
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    return mask


class ComfyUIAttentionOverride:
    def __init__(
        self,
        previous_override: AttentionFunction | None = None,
        precision_resolver: PrecisionResolver | None = None,
    ):
        self.previous_override = previous_override
        self.precision_resolver = precision_resolver or (lambda precision, dtype: precision)

    def _call_attention(
        self,
        attention: AttentionFunction,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        mask: object,
        attn_precision: torch.dtype | None,
        skip_reshape: bool,
        skip_output_reshape: bool,
        kwargs: dict[str, object],
    ) -> torch.Tensor:
        call_kwargs = {
            "mask": mask,
            "attn_precision": attn_precision,
            "skip_reshape": skip_reshape,
            "skip_output_reshape": skip_output_reshape,
            **kwargs,
        }
        with _bypass_rdna3_flash_attention():
            return attention(query, key, value, heads, **call_kwargs)

    def _run(
        self,
        original_attention: AttentionFunction,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        mask: object = None,
        attn_precision: torch.dtype | None = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        enable_gqa = kwargs.get("enable_gqa") is True
        effective_precision = self.precision_resolver(attn_precision, query.dtype)
        reshaped = _reshape_qkv(
            query, key, value, heads, skip_reshape, enable_gqa)
        normalized_mask = _normalize_mask(mask)
        if (
            reshaped is None
            or (effective_precision == torch.float32 and query.dtype != torch.float32)
            or kwargs.get("low_precision_attention", True) is False
        ):
            return self._call_attention(
                original_attention,
                query,
                key,
                value,
                heads,
                mask,
                attn_precision,
                skip_reshape,
                skip_output_reshape,
                kwargs,
            )
        query_heads, key_heads, value_heads = reshaped
        if not _is_supported(
            query_heads,
            key_heads,
            value_heads,
            normalized_mask,
            0.0,
            enable_gqa,
            kwargs.get("scale"),
        ):
            return self._call_attention(
                original_attention,
                query,
                key,
                value,
                heads,
                mask,
                attn_precision,
                skip_reshape,
                skip_output_reshape,
                kwargs,
            )

        output = flash_attention(
            query_heads,
            key_heads,
            value_heads,
            softmax_scale=kwargs.get("scale"),
            bias=_mask_to_bias(normalized_mask),
        )
        if skip_output_reshape:
            return output
        return output.transpose(1, 2).reshape(
            output.shape[0], output.shape[2], heads * output.shape[3])

    def __call__(
        self,
        original_attention: AttentionFunction,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        mask: object = None,
        attn_precision: torch.dtype | None = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.previous_override is None:
            return self._run(
                original_attention,
                query,
                key,
                value,
                heads,
                mask,
                attn_precision,
                skip_reshape,
                skip_output_reshape,
                **kwargs,
            )

        @wraps(original_attention)
        def rdna3_attention(
            nested_query: torch.Tensor,
            nested_key: torch.Tensor,
            nested_value: torch.Tensor,
            nested_heads: int,
            mask: object = None,
            attn_precision: torch.dtype | None = None,
            skip_reshape: bool = False,
            skip_output_reshape: bool = False,
            **nested_kwargs: object,
        ) -> torch.Tensor:
            return self._run(
                original_attention,
                nested_query,
                nested_key,
                nested_value,
                nested_heads,
                mask,
                attn_precision,
                skip_reshape,
                skip_output_reshape,
                **nested_kwargs,
            )

        with _bypass_rdna3_flash_attention():
            return self.previous_override(
                rdna3_attention,
                query,
                key,
                value,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )


def _get_comfyui_precision_resolver() -> PrecisionResolver:
    from comfy.ldm.modules.attention import get_attn_precision

    return get_attn_precision


class ApplyRDNA3FlashAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = (
        "Routes this model's supported attention calls through the RDNA3 "
        "FlashAttention kernel and safely falls back for unsupported calls."
    )

    def patch(self, model: ComfyModel) -> tuple[ComfyModel]:
        reason = unsupported_reason()
        if reason is not None:
            raise RuntimeError(f"Cannot enable RDNA3 Flash Attention: {reason}")

        model_clone = model.clone()
        transformer_options = model_clone.model_options["transformer_options"]
        previous = transformer_options.get("optimized_attention_override")
        if previous is not None and not callable(previous):
            raise TypeError("optimized_attention_override must be callable")
        if previous is not None and not isinstance(previous, ComfyUIAttentionOverride):
            logger.info("Composing RDNA3 Flash Attention under an existing attention override")
        if isinstance(previous, ComfyUIAttentionOverride):
            attention_override = previous
        else:
            attention_override = ComfyUIAttentionOverride(
                previous,
                _get_comfyui_precision_resolver(),
            )
        transformer_options["optimized_attention_override"] = attention_override
        enable_rdna3_flash_attention()
        logger.info("Enabled RDNA3 Flash Attention for the patched ComfyUI model")
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "ApplyRDNA3FlashAttention": ApplyRDNA3FlashAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyRDNA3FlashAttention": "RDNA3 Flash Attention",
}
