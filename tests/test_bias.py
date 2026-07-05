"""Additive attention bias / mask: an arbitrary tensor added to logits.

Covers float relative-position-style biases and hard (-inf) masks, matching
torch SDPA's float ``attn_mask`` semantics.
"""

import math

import pytest
import torch

from fa_rdna3 import flash_attention

DEVICE = "cuda"


def reference_bias(query, key, value, scale, bias):
    logits = (torch.matmul(query, key.transpose(-1, -2)) * scale).float()
    logits = logits + bias.float()
    return torch.matmul(torch.softmax(logits, dim=-1).to(value.dtype), value)


def grads(fn, q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_shape", ["full", "broadcast_batch", "broadcast_head"])
@pytest.mark.parametrize("seqlen", [200, 512])
def test_bias_forward(dtype, bias_shape, seqlen):
    torch.manual_seed(seqlen + len(bias_shape))
    batch, heads, head_dim = 2, 4, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    shape = {"full": (batch, heads, seqlen, seqlen),
             "broadcast_batch": (1, heads, seqlen, seqlen),
             "broadcast_head": (batch, 1, seqlen, seqlen)}[bias_shape]
    bias = torch.randn(shape, device=DEVICE, dtype=torch.float32)

    out = flash_attention(q, k, v, softmax_scale=scale, bias=bias)
    full_bias = bias.expand(batch, heads, seqlen, seqlen)
    exact = reference_bias(q.float(), k.float(), v.float(), scale, full_bias)
    naive = reference_bias(q, k, v, scale, full_bias)
    ke = (out.float() - exact).abs().max().item()
    ne = (naive.float() - exact).abs().max().item()
    assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"


def test_bias_hard_mask():
    # -inf mask entries: every query keeps a random subset (at least the diagonal).
    torch.manual_seed(3)
    batch, heads, seqlen, head_dim = 2, 4, 256, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    keep = torch.rand(batch, heads, seqlen, seqlen, device=DEVICE) < 0.5
    keep |= torch.eye(seqlen, device=DEVICE, dtype=torch.bool)  # ensure each row has a key
    bias = torch.where(keep, 0.0, float("-inf")).float()

    out = flash_attention(q, k, v, softmax_scale=scale, bias=bias)
    exact = reference_bias(q.float(), k.float(), v.float(), scale, bias)
    torch.testing.assert_close(out.float(), exact, atol=3e-3, rtol=3e-3)


def test_bias_backward():
    torch.manual_seed(1)
    batch, heads, seqlen, head_dim = 2, 4, 512, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    bias = torch.randn(batch, heads, seqlen, seqlen, device=DEVICE, dtype=torch.float32)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, softmax_scale=scale, bias=bias), q, k, v, dout)
    exact = grads(lambda q, k, v: reference_bias(q, k, v, scale, bias),
                  q.float(), k.float(), v.float(), dout.float())
    naive = grads(lambda q, k, v: reference_bias(q, k, v, scale, bias), q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive, exact):
        ke = (kg.float() - eg).abs().max().item()
        ne = (ng.float() - eg).abs().max().item()
        assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"
