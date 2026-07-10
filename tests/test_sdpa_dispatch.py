"""The SDPA dispatch routes supported calls and defers to the original otherwise."""

import pytest
import torch
import torch.nn.functional as F

from fa_rdna3.sdpa import (
    disable_rdna3_flash_attention,
    enable_rdna3_flash_attention,
    use_rdna3_flash_attention,
)


def test_dispatch_matches_reference_and_restores():
    query = torch.randn(2, 8, 1024, 128, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    original = F.scaled_dot_product_attention
    baseline = original(query, key, value, is_causal=True)

    enable_rdna3_flash_attention()
    try:
        dispatched = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    finally:
        disable_rdna3_flash_attention()

    assert F.scaled_dot_product_attention is original
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


def test_dispatch_preserves_pytorch_causal_cross_attention_alignment():
    query = torch.zeros(1, 1, 1, 16, device="cuda", dtype=torch.float16)
    key = torch.zeros(1, 1, 3, 16, device="cuda", dtype=torch.float16)
    values = torch.tensor([1.0, 2.0, 4.0], device="cuda", dtype=torch.float16)
    value = values.view(1, 1, 3, 1).expand_as(key).contiguous()
    reference = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    with use_rdna3_flash_attention():
        result = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    torch.testing.assert_close(result, reference)


def test_dispatch_preserves_causal_mask_rejection():
    query = torch.randn(1, 1, 4, 16, device="cuda", dtype=torch.float16)
    mask = torch.ones(4, 4, device="cuda", dtype=torch.bool)

    with use_rdna3_flash_attention(), pytest.raises(RuntimeError):
        F.scaled_dot_product_attention(
            query, query, query, attn_mask=mask, is_causal=True)


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


def test_dispatch_respects_enable_gqa():
    query = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)

    with use_rdna3_flash_attention():
        try:
            F.scaled_dot_product_attention(query, key, value)
        except RuntimeError:
            pass
        else:
            raise AssertionError("different Q/KV head counts must require enable_gqa=True")

        result = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)

    reference = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    torch.testing.assert_close(result.float(), reference.float(), atol=3e-3, rtol=3e-3)


def test_dispatch_uses_decode_for_inference_tensors_with_grad_mode_enabled():
    import fa_rdna3.sdpa as sdpa_mod

    query = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    calls = {"decode": 0}
    real_decode = sdpa_mod.flash_attention_decode

    def spy(*args, **kwargs):
        calls["decode"] += 1
        return real_decode(*args, **kwargs)

    sdpa_mod.flash_attention_decode = spy
    try:
        with use_rdna3_flash_attention():
            F.scaled_dot_product_attention(query, key, value)
    finally:
        sdpa_mod.flash_attention_decode = real_decode

    assert calls["decode"] == 1


def test_disable_preserves_later_third_party_override():
    original = F.scaled_dot_product_attention
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    reference = original(query, key, value)

    enable_rdna3_flash_attention()
    installed = F.scaled_dot_product_attention

    def third_party(*args, **kwargs):
        return installed(*args, **kwargs)

    F.scaled_dot_product_attention = third_party
    disable_rdna3_flash_attention()
    try:
        assert F.scaled_dot_product_attention is third_party
        result = F.scaled_dot_product_attention(query, key, value)
        torch.testing.assert_close(result, reference)

        enable_rdna3_flash_attention()
        assert F.scaled_dot_product_attention is third_party
    finally:
        F.scaled_dot_product_attention = original


def test_context_manager_is_nestable():
    original = F.scaled_dot_product_attention
    with use_rdna3_flash_attention():
        installed = F.scaled_dot_product_attention
        with use_rdna3_flash_attention():
            assert F.scaled_dot_product_attention is installed
        assert F.scaled_dot_product_attention is installed
    assert F.scaled_dot_product_attention is original


def test_context_manager_overlaps_are_refcounted():
    original = F.scaled_dot_product_attention
    first = use_rdna3_flash_attention()
    second = use_rdna3_flash_attention()

    first.__enter__()
    installed = F.scaled_dot_product_attention
    second.__enter__()
    first.__exit__(None, None, None)
    assert F.scaled_dot_product_attention is installed
    second.__exit__(None, None, None)
    assert F.scaled_dot_product_attention is original
