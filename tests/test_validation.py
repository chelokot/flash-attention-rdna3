"""Public API contracts reject invalid inputs before launching Triton kernels."""

import pytest
import torch

import fa_rdna3._validation as validation
from fa_rdna3 import (
    flash_attention,
    flash_attention_decode,
    flash_attention_decode_paged,
    flash_attention_varlen,
)


def _dense_inputs():
    query = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    return query, torch.randn_like(query), torch.randn_like(query)


def test_platform_detection_distinguishes_rocm_architectures(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(validation, "_device_arch", lambda device_index: "gfx1100")
    assert validation.unsupported_reason(torch.device("cuda:0")) is None
    assert validation.unsupported_reason("cuda:0") is None

    monkeypatch.setattr(validation, "_device_arch", lambda device_index: "gfx1101")
    assert "gfx1100" in validation.unsupported_reason(torch.device("cuda:0"))

    monkeypatch.setattr(torch.version, "hip", None)
    assert "ROCm" in validation.unsupported_reason(torch.device("cuda:0"))


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda q, k, v: (q[0], k, v), "query must have shape"),
        (lambda q, k, v: (q, k.expand(2, -1, -1, -1), v), "same batch size"),
        (lambda q, k, v: (q, k, v.double()), "same dtype"),
        (lambda q, k, v: (q, k[:, :3], v[:, :3]), "multiple"),
        (lambda q, k, v: (q, k[:, :, :-1], v), "same sequence length"),
    ],
)
def test_dense_shape_contracts_precede_kernel_launch(monkeypatch, mutate, match):
    monkeypatch.setattr(validation, "_require_platform", lambda device: None)
    query, key, value = mutate(*_dense_inputs())
    with pytest.raises(ValueError, match=match):
        flash_attention(query, key, value)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"window_size": (1,)}, "window_size"),
        ({"window_size": (-2, 0)}, "window_size"),
        ({"softcap": -1.0}, "softcap"),
        ({"dropout_p": 1.0}, "dropout_p"),
        ({"softmax_scale": float("nan")}, "softmax_scale"),
        ({"bias": torch.zeros(3, 7)}, "broadcastable"),
        ({"alibi_slopes": torch.ones(3)}, "shape"),
    ],
)
def test_dense_modifier_contracts_precede_kernel_launch(monkeypatch, kwargs, match):
    monkeypatch.setattr(validation, "_require_platform", lambda device: None)
    with pytest.raises((TypeError, ValueError), match=match):
        flash_attention(*_dense_inputs(), **kwargs)


def test_decode_rejects_autograd(monkeypatch):
    monkeypatch.setattr(validation, "_require_platform", lambda device: None)
    query, key, value = _dense_inputs()
    query.requires_grad_(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        flash_attention_decode(query, key, value)


def test_varlen_metadata_contracts(monkeypatch):
    monkeypatch.setattr(validation, "_require_platform", lambda device: None)
    query = torch.randn(8, 4, 64, dtype=torch.float16)
    key = torch.randn(8, 2, 64, dtype=torch.float16)
    value = torch.randn_like(key)
    cu_int64 = torch.tensor([0, 8], dtype=torch.int64)
    cu_int32 = cu_int64.to(torch.int32)

    with pytest.raises(ValueError, match="contiguous int32"):
        flash_attention_varlen(query, key, value, cu_int64, cu_int64, 8, 8)
    with pytest.raises(ValueError, match="same batch size"):
        flash_attention_varlen(query, key, value, cu_int32, torch.tensor([0, 4, 8], dtype=torch.int32), 8, 8)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        flash_attention_varlen(query, key, value, cu_int32, cu_int32, 9, 8)


def test_paged_metadata_contracts(monkeypatch):
    monkeypatch.setattr(validation, "_require_platform", lambda device: None)
    query = torch.randn(2, 4, 64, dtype=torch.float16)
    key = torch.randn(4, 24, 2, 64, dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.zeros(2, 1, dtype=torch.int32)
    context_lens = torch.ones(2, dtype=torch.int32)

    with pytest.raises(ValueError, match="power of two"):
        flash_attention_decode_paged(query, key, value, block_table, context_lens)

    key = torch.randn(4, 16, 2, 64, dtype=torch.float16)
    value = torch.randn_like(key)
    with pytest.raises(ValueError, match="int32"):
        flash_attention_decode_paged(query, key, value, block_table.long(), context_lens)
