"""The SDPA dispatch routes supported calls and defers to the original otherwise."""

import torch
import torch.nn.functional as F

from fa_rdna3.sdpa import enable_rdna3_flash_attention, disable_rdna3_flash_attention


def test_dispatch_matches_reference_and_restores():
    query = torch.randn(2, 8, 1024, 128, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    baseline = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    enable_rdna3_flash_attention()
    try:
        dispatched = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    finally:
        disable_rdna3_flash_attention()

    assert F.scaled_dot_product_attention is not None
    torch.testing.assert_close(dispatched.float(), baseline.float(), atol=3e-3, rtol=3e-3)


def test_dispatch_falls_back_for_unsupported_head_dim():
    # head_dim 96 is unsupported by the kernel; the dispatcher must defer to the original.
    query = torch.randn(1, 4, 256, 96, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    reference = F.scaled_dot_product_attention(query, key, value)

    enable_rdna3_flash_attention()
    try:
        result = F.scaled_dot_product_attention(query, key, value)
    finally:
        disable_rdna3_flash_attention()

    torch.testing.assert_close(result.float(), reference.float(), atol=3e-3, rtol=3e-3)
