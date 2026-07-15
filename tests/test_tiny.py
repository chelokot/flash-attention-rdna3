import math
import sys
import threading

import pytest
import torch
import torch.nn.functional as F

from fa_rdna3 import (
    disable_rdna3_flash_attention,
    enable_rdna3_flash_attention,
    flash_attention,
)
from fa_rdna3.interface import _can_use_tiny_forward
from fa_rdna3 import sdpa as sdpa_module


DEVICE = "cuda"
HEADS = 8
HEAD_DIM = 128


def _packed_inputs(sequence):
    packed = torch.randn(2, sequence, 3, HEADS, HEAD_DIM, device=DEVICE)
    return tuple(tensor.transpose(1, 2) for tensor in packed.unbind(2))


def _reference_attention(query, key, value, bias=None):
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    if bias is not None:
        scores = scores + bias
    probabilities = torch.softmax(scores, dim=-1).nan_to_num()
    return torch.matmul(probabilities, value)


@pytest.mark.parametrize("sequence", [1, 6, 19, 32])
@pytest.mark.parametrize("bias_kind", ["none", "broadcast", "full", "masked_row"])
def test_tiny_matches_reference_for_packed_views(sequence, bias_kind):
    torch.manual_seed(sequence)
    query, key, value = _packed_inputs(sequence)
    if bias_kind == "none":
        bias = None
    elif bias_kind == "broadcast":
        bias = torch.randn(2, 1, 1, sequence, device=DEVICE)
    else:
        bias = torch.randn(2, HEADS, sequence, sequence, device=DEVICE)
        if bias_kind == "masked_row":
            bias[:, :, sequence // 2, :] = float("-inf")

    actual = flash_attention(query, key, value, bias=bias)
    expected = _reference_attention(query, key, value, bias)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert (actual - expected).abs().max().item() < 2e-6


@pytest.mark.parametrize("has_bias", [False, True])
def test_tiny_triton_op_passes_opcheck(has_bias):
    query, key, value = _packed_inputs(6)
    bias = torch.randn(2, 1, 1, 6, device=DEVICE) if has_bias else None
    torch.library.opcheck(
        torch.ops.fa_rdna3.flash_tiny_fwd.default,
        (query, key, value, 1.0 / math.sqrt(HEAD_DIM), bias, 8),
    )


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="torch.compile unsupported on Python 3.14+")
def test_sdpa_compile_matches_stock_for_ardy_shapes():
    original_sdpa = F.scaled_dot_product_attention

    def attention(packed, mask):
        query, key, value = (
            tensor.transpose(1, 2) for tensor in packed.unbind(2)
        )
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    enable_rdna3_flash_attention()
    try:
        compiled = torch.compile(attention, fullgraph=True, dynamic=False)
        for sequence in (5, 6, 19):
            packed = torch.randn(2, sequence, 3, HEADS, HEAD_DIM, device=DEVICE)
            mask = torch.randn(2, 1, 1, sequence, device=DEVICE)
            query, key, value = (
                tensor.transpose(1, 2) for tensor in packed.unbind(2)
            )
            expected = original_sdpa(query, key, value, attn_mask=mask)
            actual = compiled(packed, mask)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            assert (actual - expected).abs().max().item() < 2e-6
    finally:
        disable_rdna3_flash_attention()
        torch._dynamo.reset()


def test_tiny_route_is_inference_only():
    query, key, value = _packed_inputs(19)
    options = (False, (-1, -1), 0.0, None, 0.0, HEAD_DIM)

    assert _can_use_tiny_forward(query, key, value, *options)

    query.requires_grad_(True)
    assert not _can_use_tiny_forward(query, key, value, *options)
    query.requires_grad_(False)

    assert not _can_use_tiny_forward(
        query.half(), key.half(), value.half(), *options
    )
    assert not _can_use_tiny_forward(
        query, key, value, True, (-1, -1), 0.0, None, 0.0, HEAD_DIM
    )
    assert not _can_use_tiny_forward(
        query, key, value, False, (-1, -1), 0.0, None, 0.0, 64
    )

    long_query, long_key, long_value = _packed_inputs(33)
    assert not _can_use_tiny_forward(
        long_query, long_key, long_value, *options
    )
    assert not _can_use_tiny_forward(
        query, key[:, :, :-1], value[:, :, :-1], *options
    )
    assert not _can_use_tiny_forward(
        query, key[:, :4], value[:, :4], *options
    )
    assert not _can_use_tiny_forward(
        query, key, value, False, (4, 0), 0.0, None, 0.0, HEAD_DIM
    )
    assert not _can_use_tiny_forward(
        query, key, value, False, (-1, -1), 10.0, None, 0.0, HEAD_DIM
    )
    slopes = torch.zeros(HEADS, device=DEVICE)
    assert not _can_use_tiny_forward(
        query, key, value, False, (-1, -1), 0.0, slopes, 0.0, HEAD_DIM
    )
    assert not _can_use_tiny_forward(
        query, key, value, False, (-1, -1), 0.0, None, 0.1, HEAD_DIM
    )


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="torch.compile unsupported on Python 3.14+")
def test_compiled_dispatch_preserves_active_bypass():
    query = torch.randn(1, 1, 6, HEAD_DIM, device=DEVICE)
    saved_original = sdpa_module._original_sdpa

    def original_sdpa(query, key, value, **kwargs):
        return query + 1.0

    sdpa_module._original_sdpa = original_sdpa
    try:
        compiled = torch.compile(
            sdpa_module._dispatch_sdpa, fullgraph=True, dynamic=False
        )
        with sdpa_module._bypass_rdna3_flash_attention():
            actual = compiled(query, query, query)
        torch.testing.assert_close(actual, query + 1.0)
    finally:
        sdpa_module._original_sdpa = saved_original
        torch._dynamo.reset()


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="torch.compile unsupported on Python 3.14+")
def test_bypass_in_another_thread_does_not_break_compile():
    query = torch.randn(1, 1, 6, HEAD_DIM, device=DEVICE)
    bypass_entered = threading.Event()
    release_bypass = threading.Event()
    saved_original = sdpa_module._original_sdpa

    def hold_bypass():
        with sdpa_module._bypass_rdna3_flash_attention():
            bypass_entered.set()
            release_bypass.wait(timeout=30)

    bypass_thread = threading.Thread(target=hold_bypass)
    bypass_thread.start()
    assert bypass_entered.wait(timeout=5)
    sdpa_module._original_sdpa = lambda query, key, value, **kwargs: query + 1.0
    try:
        compiled = torch.compile(
            sdpa_module._dispatch_sdpa, fullgraph=True, dynamic=False
        )
        actual = compiled(query, query, query)
        expected = flash_attention(query, query, query)
        torch.testing.assert_close(actual, expected)
    finally:
        release_bypass.set()
        bypass_thread.join(timeout=5)
        sdpa_module._original_sdpa = saved_original
        torch._dynamo.reset()
