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


def test_dispatch_routes_decode():
    # A one-row query against a long cache with grad disabled must route to the
    # split-K decode path and still match the reference.
    query = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)

    reference = F.scaled_dot_product_attention(query, key, value)

    enable_rdna3_flash_attention()
    try:
        with torch.no_grad():
            result = F.scaled_dot_product_attention(query, key, value)
    finally:
        disable_rdna3_flash_attention()

    torch.testing.assert_close(result.float(), reference.float(), atol=3e-3, rtol=3e-3)


def test_dispatch_routes_attn_mask():
    # A float attn_mask must route through the kernel as an additive bias.
    torch.manual_seed(2)
    query = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    mask = torch.randn(2, 4, 256, 256, device="cuda", dtype=torch.float16)

    reference = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    enable_rdna3_flash_attention()
    try:
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    finally:
        disable_rdna3_flash_attention()
    torch.testing.assert_close(result.float(), reference.float(), atol=3e-3, rtol=3e-3)


def test_dispatch_falls_back_for_unsupported_head_dim():
    # head_dim 640 is above the 512 cap; the dispatcher must defer to the original.
    query = torch.randn(1, 4, 256, 640, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    reference = F.scaled_dot_product_attention(query, key, value)

    enable_rdna3_flash_attention()
    try:
        result = F.scaled_dot_product_attention(query, key, value)
    finally:
        disable_rdna3_flash_attention()

    torch.testing.assert_close(result.float(), reference.float(), atol=3e-3, rtol=3e-3)


def test_dispatch_defers_tiny_sequence_in_training():
    # A very short sequence with grad enabled must defer to torch (its backward
    # wins there); the same shape under no_grad must stay on the kernel.
    import fa_rdna3.sdpa as sdpa_mod

    tiny = lambda: torch.randn(1, 8, 12, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    big = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.float16, requires_grad=True)

    enable_rdna3_flash_attention()
    try:
        real_orig = sdpa_mod._original_sdpa
        deferrals = {"n": 0}

        def spy(*args, **kwargs):
            deferrals["n"] += 1
            return real_orig(*args, **kwargs)

        sdpa_mod._original_sdpa = spy

        F.scaled_dot_product_attention(tiny(), tiny(), tiny(), is_causal=True)
        assert deferrals["n"] == 1  # tiny + grad -> deferred

        with torch.no_grad():
            F.scaled_dot_product_attention(tiny(), tiny(), tiny(), is_causal=True)
        assert deferrals["n"] == 1  # tiny + no_grad -> kept on kernel

        F.scaled_dot_product_attention(big, big, big, is_causal=True)
        assert deferrals["n"] == 1  # long + grad -> kept on kernel
    finally:
        sdpa_mod._original_sdpa = real_orig
        disable_rdna3_flash_attention()
