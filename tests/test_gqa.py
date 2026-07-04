"""Grouped-query attention: fewer K/V heads than query heads."""

import math

import pytest
import torch

from fa_rdna3 import flash_attention, flash_attention_decode

DEVICE = "cuda"


def reference_gqa(query, key, value, causal, softmax_scale):
    # Expand K/V heads to match query heads, then plain attention.
    q_heads, kv_heads = query.shape[1], key.shape[1]
    group = q_heads // kv_heads
    key = key.repeat_interleave(group, dim=1)
    value = value.repeat_interleave(group, dim=1)
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * softmax_scale
    if causal:
        sq, sk = logits.shape[-2], logits.shape[-1]
        row = torch.arange(sq, device=logits.device)[:, None]
        col = torch.arange(sk, device=logits.device)[None, :]
        logits = logits.masked_fill(row < col, float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=-1), value.float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads", [(32, 8), (16, 4), (8, 1)])
def test_gqa_forward(dtype, causal, q_heads, kv_heads):
    torch.manual_seed(q_heads + kv_heads)
    batch, seqlen, head_dim = 2, 512, 128
    scale = 1.0 / math.sqrt(head_dim)
    query = torch.randn(batch, q_heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    key = torch.randn(batch, kv_heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    value = torch.randn(batch, kv_heads, seqlen, head_dim, device=DEVICE, dtype=dtype)

    out = flash_attention(query, key, value, causal=causal, softmax_scale=scale)
    ref = reference_gqa(query, key, value, causal, scale)
    tol = {torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads", [(16, 4), (8, 1)])
def test_gqa_backward(causal, q_heads, kv_heads):
    torch.manual_seed(q_heads * 10 + kv_heads)
    batch, seqlen, head_dim, dtype = 2, 512, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)

    def grads(fn, q, k, v, dout):
        q = q.detach().clone().requires_grad_(True)
        k = k.detach().clone().requires_grad_(True)
        v = v.detach().clone().requires_grad_(True)
        fn(q, k, v).backward(dout)
        return q.grad, k.grad, v.grad

    query = torch.randn(batch, q_heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    key = torch.randn(batch, kv_heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    value = torch.randn(batch, kv_heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    dout = torch.randn(batch, q_heads, seqlen, head_dim, device=DEVICE, dtype=dtype)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale), query, key, value, dout)
    exact = grads(lambda q, k, v: reference_gqa(q, k, v, causal, scale),
                  query.float(), key.float(), value.float(), dout.float())
    naive = grads(lambda q, k, v: reference_gqa(q, k, v, causal, scale), query, key, value, dout)
    for kernel_g, naive_g, exact_g in zip(kernel, naive, exact):
        kernel_err = (kernel_g.float() - exact_g).abs().max().item()
        naive_err = (naive_g.float() - exact_g).abs().max().item()
        assert kernel_err <= 2.0 * naive_err + 1e-3


@pytest.mark.parametrize("q_heads,kv_heads", [(32, 8), (16, 2)])
def test_gqa_decode(q_heads, kv_heads):
    torch.manual_seed(q_heads + kv_heads)
    batch, seqlen_k, head_dim, dtype = 1, 8192, 128, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    query = torch.randn(batch, q_heads, 1, head_dim, device=DEVICE, dtype=dtype)
    key = torch.randn(batch, kv_heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)
    value = torch.randn(batch, kv_heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)

    out = flash_attention_decode(query, key, value, scale)
    ref = reference_gqa(query, key, value, False, scale)
    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-3)


def test_gqa_rejects_indivisible_heads():
    query = torch.randn(1, 6, 128, 64, device=DEVICE, dtype=torch.float16)
    key = torch.randn(1, 4, 128, 64, device=DEVICE, dtype=torch.float16)
    with pytest.raises(ValueError, match="multiple"):
        flash_attention(query, key, key)
