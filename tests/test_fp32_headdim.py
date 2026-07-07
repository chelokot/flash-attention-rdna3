"""fp32 inputs and non-power-of-two head dims (handled by zero-padding)."""

import math

import pytest
import torch

from fa_rdna3 import flash_attention

DEVICE = "cuda"


def reference(query, key, value, causal, scale):
    # dtype-aware: matmuls in the input dtype, softmax in fp32. Passed fp32 inputs
    # it is the exact reference; passed bf16/fp16 it is the naive same-dtype
    # baseline (so the error-ratio bound has a non-zero denominator).
    logits = (torch.matmul(query, key.transpose(-1, -2)) * scale).float()
    if causal:
        sq, sk = logits.shape[-2], logits.shape[-1]
        row = torch.arange(sq, device=DEVICE)[:, None]
        col = torch.arange(sk, device=DEVICE)[None, :]
        logits = logits.masked_fill((row + (sk - sq)) < col, float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=-1).to(value.dtype), value)


def grads(fn, q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("causal", [False, True])
def test_fp32_forward_backward(causal):
    torch.manual_seed(int(causal))
    batch, heads, seqlen, head_dim = 2, 4, 512, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal, softmax_scale=scale)
    ref = reference(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref.float(), atol=2e-3, rtol=2e-3)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale), q, k, v, dout)
    exact = grads(lambda q, k, v: reference(q, k, v, causal, scale), q, k, v, dout)
    for kg, eg in zip(kernel, exact):
        torch.testing.assert_close(kg, eg.float(), atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("causal", [False, True])
def test_large_head_dim(head_dim, causal):
    # head_dim 512 covers VAE spatial self-attention (AutoencoderKL); it is
    # heavier on registers, so this checks both correctness and that the pruned
    # autotune still finds a config that fits. Kept small — the large-tile prune
    # leaves few configs at head_dim 512 and each is slow to autotune.
    torch.manual_seed(head_dim + int(causal))
    batch, heads, seqlen, dtype = 1, 1, 128, torch.bfloat16
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal, softmax_scale=scale)
    exact = reference(q.float(), k.float(), v.float(), causal, scale)
    naive = reference(q, k, v, causal, scale)
    assert (out.float() - exact).abs().max().item() <= 2.0 * (naive.float() - exact).abs().max().item() + 2e-3

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale), q, k, v, dout)
    exact_g = grads(lambda q, k, v: reference(q, k, v, causal, scale), q.float(), k.float(), v.float(), dout.float())
    naive_g = grads(lambda q, k, v: reference(q, k, v, causal, scale), q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive_g, exact_g):
        assert kg.shape[-1] == head_dim
        assert (kg.float() - eg).abs().max().item() <= 2.0 * (ng.float() - eg).abs().max().item() + 2e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [48, 80, 96, 192])
@pytest.mark.parametrize("causal", [False, True])
def test_nonpow2_head_dim(dtype, head_dim, causal):
    torch.manual_seed(head_dim + int(causal))
    batch, heads, seqlen = 2, 4, 256
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal, softmax_scale=scale)
    exact = reference(q.float(), k.float(), v.float(), causal, scale)
    naive = reference(q, k, v, causal, scale)
    ke = (out.float() - exact).abs().max().item()
    ne = (naive.float() - exact).abs().max().item()
    assert ke <= 2.0 * ne + 2e-3, f"fwd err {ke:.2e} vs naive {ne:.2e}"

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale), q, k, v, dout)
    exact_g = grads(lambda q, k, v: reference(q, k, v, causal, scale), q.float(), k.float(), v.float(), dout.float())
    naive_g = grads(lambda q, k, v: reference(q, k, v, causal, scale), q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive_g, exact_g):
        assert kg.shape[-1] == head_dim
        ke = (kg.float() - eg).abs().max().item()
        ne = (ng.float() - eg).abs().max().item()
        assert ke <= 2.0 * ne + 2e-3
