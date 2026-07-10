"""ComfyUI model-patch adapter behavior without importing ComfyUI itself."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import fa_rdna3.comfyui as comfyui
import fa_rdna3.sdpa as sdpa


def test_attention_override_reshapes_and_restores_comfyui_layout(monkeypatch):
    query = torch.randn(2, 5, 12)
    key = torch.randn(2, 7, 12)
    value = torch.randn(2, 7, 12)
    mask = torch.ones(5, 7, dtype=torch.bool)
    mask[:, -1] = False
    captured = {}

    monkeypatch.setattr(comfyui, "_is_supported", lambda *args: True)

    def flash_attention(query_heads, key_heads, value_heads, **kwargs):
        captured.update(
            query=query_heads,
            key=key_heads,
            value=value_heads,
            bias=kwargs["bias"],
            scale=kwargs["softmax_scale"],
        )
        return query_heads

    monkeypatch.setattr(comfyui, "flash_attention", flash_attention)
    override = comfyui.ComfyUIAttentionOverride()
    result = override(
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
        query,
        key,
        value,
        3,
        mask=mask,
        scale=0.25,
    )

    assert captured["query"].shape == (2, 3, 5, 4)
    assert captured["key"].shape == (2, 3, 7, 4)
    assert captured["value"].shape == (2, 3, 7, 4)
    assert captured["scale"] == 0.25
    assert captured["bias"].shape == (1, 5, 7)
    assert torch.all(captured["bias"][..., :-1] == 0)
    assert torch.all(captured["bias"][..., -1] == float("-inf"))
    torch.testing.assert_close(
        result,
        captured["query"].transpose(1, 2).reshape(2, 5, 12),
    )


def test_attention_override_preserves_head_layout_and_gqa(monkeypatch):
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 2, 7, 8)
    value = torch.randn_like(key)
    captured = {}

    monkeypatch.setattr(comfyui, "_is_supported", lambda *args: True)

    def flash_attention(query_heads, key_heads, value_heads, **kwargs):
        captured["shapes"] = (
            query_heads.shape,
            key_heads.shape,
            value_heads.shape,
        )
        return query_heads

    monkeypatch.setattr(comfyui, "flash_attention", flash_attention)
    result = comfyui.ComfyUIAttentionOverride()(
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
        query,
        key,
        value,
        4,
        skip_reshape=True,
        skip_output_reshape=True,
        enable_gqa=True,
    )

    assert captured["shapes"] == (query.shape, key.shape, value.shape)
    assert result is query


def test_attention_override_composes_with_existing_fallback(monkeypatch):
    calls = []

    def existing_override(attention, *args, **kwargs):
        calls.append(attention)
        return attention(*args, **kwargs)

    monkeypatch.setattr(comfyui, "_is_supported", lambda *args: False)
    original_attention = lambda *args, **kwargs: "original"
    query = torch.randn(1, 4, 16)
    result = comfyui.ComfyUIAttentionOverride(existing_override)(
        original_attention,
        query,
        query,
        query,
        2,
    )

    assert result == "original"
    assert len(calls) == 1
    assert calls[0] is not original_attention


def test_existing_backend_override_keeps_precedence(monkeypatch):
    monkeypatch.setattr(
        comfyui,
        "flash_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("kernel must not run")),
    )
    existing_override = lambda attention, *args, **kwargs: "existing"
    query = torch.randn(1, 4, 16)
    result = comfyui.ComfyUIAttentionOverride(existing_override)(
        lambda *args, **kwargs: "original",
        query,
        query,
        query,
        2,
    )
    assert result == "existing"


def test_existing_semantic_override_can_delegate_to_rdna3(monkeypatch):
    monkeypatch.setattr(comfyui, "_is_supported", lambda *args: True)
    monkeypatch.setattr(
        comfyui,
        "flash_attention",
        lambda query, key, value, **kwargs: query,
    )

    def add_one(attention, query, key, value, heads, **kwargs):
        return attention(query + 1, key, value, heads, **kwargs)

    query = torch.zeros(1, 4, 16)
    result = comfyui.ComfyUIAttentionOverride(add_one)(
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
        query,
        query,
        query,
        2,
    )
    torch.testing.assert_close(result, torch.ones_like(query))


def test_attention_override_honors_requested_fp32_accumulation(monkeypatch):
    monkeypatch.setattr(
        comfyui,
        "flash_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("kernel must not run")),
    )
    query = torch.randn(1, 4, 16, dtype=torch.float16)
    result = comfyui.ComfyUIAttentionOverride()(
        lambda *args, **kwargs: "fallback",
        query,
        query,
        query,
        2,
        attn_precision=torch.float32,
    )
    assert result == "fallback"


def test_attention_override_honors_comfyui_global_upcast(monkeypatch):
    monkeypatch.setattr(
        comfyui,
        "flash_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("kernel must not run")),
    )
    query = torch.randn(1, 4, 16, dtype=torch.float16)
    override = comfyui.ComfyUIAttentionOverride(
        precision_resolver=lambda precision, dtype: torch.float32)
    result = override(
        lambda *args, **kwargs: "fallback",
        query,
        query,
        query,
        2,
    )
    assert result == "fallback"


def test_attention_fallback_bypasses_process_sdpa_dispatch(monkeypatch):
    query = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    original_sdpa = F.scaled_dot_product_attention
    reference = original_sdpa(query, key, value)
    monkeypatch.setattr(
        sdpa,
        "flash_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dispatcher leaked")),
    )

    def pytorch_attention(query, key, value, heads, **kwargs):
        return F.scaled_dot_product_attention(query, key, value)

    sdpa.enable_rdna3_flash_attention()
    try:
        result = comfyui.ComfyUIAttentionOverride()(
            pytorch_attention,
            query,
            key,
            value,
            2,
            skip_reshape=True,
            skip_output_reshape=True,
            low_precision_attention=False,
        )
    finally:
        sdpa.disable_rdna3_flash_attention()

    torch.testing.assert_close(result, reference)


class FakeModel:
    def __init__(self, transformer_options=None):
        self.model_options = {
            "transformer_options": dict(transformer_options or {}),
        }

    def clone(self):
        return FakeModel(self.model_options["transformer_options"])


def test_model_node_clones_composes_and_enables_global_sdpa(monkeypatch):
    global_calls = []
    previous = lambda *args, **kwargs: "previous"
    model = FakeModel({"optimized_attention_override": previous})
    monkeypatch.setattr(comfyui, "unsupported_reason", lambda: None)
    monkeypatch.setattr(
        comfyui,
        "enable_rdna3_flash_attention",
        lambda: global_calls.append(True),
    )
    monkeypatch.setattr(
        comfyui,
        "_get_comfyui_precision_resolver",
        lambda: (lambda precision, dtype: precision),
    )

    patched, = comfyui.ApplyRDNA3FlashAttention().patch(model)
    attention_override = patched.model_options["transformer_options"][
        "optimized_attention_override"]

    assert patched is not model
    assert model.model_options["transformer_options"]["optimized_attention_override"] is previous
    assert isinstance(attention_override, comfyui.ComfyUIAttentionOverride)
    assert attention_override.previous_override is previous
    assert global_calls == [True]


def test_model_node_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(comfyui, "unsupported_reason", lambda: "unsupported test GPU")
    with pytest.raises(RuntimeError, match="Cannot enable RDNA3 Flash Attention: unsupported test GPU"):
        comfyui.ApplyRDNA3FlashAttention().patch(FakeModel())


def test_model_node_rejects_invalid_existing_override(monkeypatch):
    monkeypatch.setattr(comfyui, "unsupported_reason", lambda: None)
    model = FakeModel({"optimized_attention_override": object()})
    with pytest.raises(TypeError, match="optimized_attention_override must be callable"):
        comfyui.ApplyRDNA3FlashAttention().patch(model)


def test_comfyui_node_mappings_are_exported():
    assert comfyui.NODE_CLASS_MAPPINGS == {
        "ApplyRDNA3FlashAttention": comfyui.ApplyRDNA3FlashAttention,
    }
    assert comfyui.NODE_DISPLAY_NAME_MAPPINGS == {
        "ApplyRDNA3FlashAttention": "RDNA3 Flash Attention",
    }


def test_custom_node_entrypoint_ignores_conflicting_top_level_package(tmp_path):
    conflict = tmp_path / "fa_rdna3"
    conflict.mkdir()
    (conflict / "__init__.py").write_text("SOURCE = 'stale install'\n")
    repository = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.util
import sys

module_name = "rdna3_custom_node_test"
spec = importlib.util.spec_from_file_location(
    module_name,
    {str(repository / "__init__.py")!r},
    submodule_search_locations=[{str(repository)!r}],
)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
node = module.NODE_CLASS_MAPPINGS["ApplyRDNA3FlashAttention"]
assert node.__module__.startswith(module_name + ".fa_rdna3")
"""
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
    )


def test_attention_override_matches_sdpa_for_comfyui_layout():
    torch.manual_seed(0)
    batch, query_tokens, key_tokens, heads, head_dim = 2, 96, 137, 4, 64
    query = torch.randn(
        batch, query_tokens, heads * head_dim,
        device="cuda", dtype=torch.float16,
    )
    key = torch.randn(
        batch, key_tokens, heads * head_dim,
        device="cuda", dtype=torch.float16,
    )
    value = torch.randn_like(key)
    mask = torch.zeros(query_tokens, key_tokens, device="cuda", dtype=torch.float32)
    mask[:, -11:] = float("-inf")
    query_heads = query.reshape(batch, query_tokens, heads, head_dim).transpose(1, 2)
    key_heads = key.reshape(batch, key_tokens, heads, head_dim).transpose(1, 2)
    value_heads = value.reshape(batch, key_tokens, heads, head_dim).transpose(1, 2)
    reference = F.scaled_dot_product_attention(
        query_heads, key_heads, value_heads, attn_mask=mask.unsqueeze(0))
    reference = reference.transpose(1, 2).reshape(
        batch, query_tokens, heads * head_dim)

    result = comfyui.ComfyUIAttentionOverride()(
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
        query,
        key,
        value,
        heads,
        mask=mask,
    )

    torch.testing.assert_close(result.float(), reference.float(), atol=3e-3, rtol=3e-3)


def test_attention_override_matches_sdpa_for_head_layout_gqa():
    torch.manual_seed(1)
    query = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 192, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    reference = F.scaled_dot_product_attention(
        query, key, value, enable_gqa=True)

    result = comfyui.ComfyUIAttentionOverride()(
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
        query,
        key,
        value,
        8,
        skip_reshape=True,
        skip_output_reshape=True,
        enable_gqa=True,
    )

    torch.testing.assert_close(result.float(), reference.float(), atol=2e-2, rtol=2e-2)
