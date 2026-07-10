"""Correctness tests for the RDNA3 FlashAttention forward kernel.

The reference is an explicit fp32 attention so the tolerance reflects only the
fp16/bf16 rounding of the kernel's inputs and accumulation, independent of
whatever backend ``scaled_dot_product_attention`` happens to select on ROCm.
"""

import math

import pytest
import torch

from fa_rdna3 import flash_attention

DEVICE = "cuda"


def reference_attention(query, key, value, causal, softmax_scale):
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * softmax_scale
    if causal:
        seqlen_q, seqlen_k = logits.shape[-2], logits.shape[-1]
        row = torch.arange(seqlen_q, device=logits.device)[:, None]
        col = torch.arange(seqlen_k, device=logits.device)[None, :]
        logits = logits.masked_fill(row < col, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, value.float())


def tolerance(dtype):
    return {torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 200, 512, 1000, 2048])
def test_matches_reference(dtype, causal, head_dim, seqlen):
    torch.manual_seed(seqlen + head_dim)
    batch, heads = 2, 4
    shape = (batch, heads, seqlen, head_dim)
    query = torch.randn(shape, device=DEVICE, dtype=dtype)
    key = torch.randn(shape, device=DEVICE, dtype=dtype)
    value = torch.randn(shape, device=DEVICE, dtype=dtype)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out = flash_attention(query, key, value, causal=causal, softmax_scale=softmax_scale)
    ref = reference_attention(query, key, value, causal, softmax_scale)

    torch.testing.assert_close(out.float(), ref, atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("head_dim", [16, 32, 64, 128, 256])
def test_head_dims(head_dim):
    torch.manual_seed(head_dim)
    shape = (1, 2, 512, head_dim)
    query = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    key = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    value = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out = flash_attention(query, key, value, causal=False, softmax_scale=softmax_scale)
    ref = reference_attention(query, key, value, False, softmax_scale)

    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_batch_head_d64_specialization(dtype):
    torch.manual_seed(37)
    query = torch.randn(1, 1, 1024, 64, device=DEVICE, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scale = 1.0 / math.sqrt(64)

    out = flash_attention(query, key, value, softmax_scale=scale)
    ref = reference_attention(query, key, value, False, scale)
    torch.testing.assert_close(out.float(), ref, atol=tolerance(dtype), rtol=tolerance(dtype))


def test_cross_attention_shapes():
    torch.manual_seed(1)
    query = torch.randn(2, 8, 333, 64, device=DEVICE, dtype=torch.float16)
    key = torch.randn(2, 8, 777, 64, device=DEVICE, dtype=torch.float16)
    value = torch.randn(2, 8, 777, 64, device=DEVICE, dtype=torch.float16)
    softmax_scale = 1.0 / math.sqrt(64)

    out = flash_attention(query, key, value, causal=False, softmax_scale=softmax_scale)
    ref = reference_attention(query, key, value, False, softmax_scale)

    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-3)


def test_rejects_unsupported_dtype():
    query = torch.randn(1, 1, 64, 64, device=DEVICE, dtype=torch.float64)
    with pytest.raises(ValueError, match="unsupported dtype"):
        flash_attention(query, query, query)
