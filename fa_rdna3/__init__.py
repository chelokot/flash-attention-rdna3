"""FlashAttention-2 for AMD RDNA3 (gfx1100) via Triton."""

from .interface import (
    flash_attention,
    flash_attention_decode,
    flash_attention_varlen,
    alibi_slopes,
)

__all__ = ["flash_attention", "flash_attention_decode", "flash_attention_varlen", "alibi_slopes"]
