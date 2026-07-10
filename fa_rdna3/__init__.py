"""FlashAttention-2 for AMD RDNA3 (gfx1100) via Triton."""

from ._validation import is_available, unsupported_reason
from .interface import (
    flash_attention,
    flash_attention_decode,
    flash_attention_varlen,
    flash_attention_decode_paged,
    alibi_slopes,
)
from .sdpa import (
    disable_rdna3_flash_attention,
    enable_rdna3_flash_attention,
    use_rdna3_flash_attention,
)

__all__ = [
    "flash_attention",
    "flash_attention_decode",
    "flash_attention_varlen",
    "flash_attention_decode_paged",
    "alibi_slopes",
    "is_available",
    "unsupported_reason",
    "enable_rdna3_flash_attention",
    "disable_rdna3_flash_attention",
    "use_rdna3_flash_attention",
]
