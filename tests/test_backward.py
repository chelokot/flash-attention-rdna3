"""Gradient correctness for the RDNA3 FlashAttention backward kernels.

Absolute tolerances on fp16/bf16 gradients are either flaky or blind, so each
kernel gradient is judged against an fp32 reference relative to the error that a
naive attention in the same low precision already incurs: the kernel must not be
more than a small factor worse than unavoidable rounding.
"""

import math

import pytest
import torch

from fa_rdna3 import flash_attention
from fa_rdna3.interface import _forward

DEVICE = "cuda"


def reference_attention(query, key, value, causal, softmax_scale):
    logits = torch.matmul(query, key.transpose(-1, -2)) * softmax_scale
    if causal:
        seqlen_q, seqlen_k = logits.shape[-2], logits.shape[-1]
        row = torch.arange(seqlen_q, device=logits.device)[:, None]
        col = torch.arange(seqlen_k, device=logits.device)[None, :]
        logits = logits.masked_fill(row < col, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, value)


def grads(fn, query, key, value, dout):
    q = query.detach().clone().requires_grad_(True)
    k = key.detach().clone().requires_grad_(True)
    v = value.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 200, 512, 1000])
def test_backward_matches_reference(dtype, causal, head_dim, seqlen):
    torch.manual_seed(seqlen + head_dim)
    batch, heads = 2, 4
    scale = 1.0 / math.sqrt(head_dim)
    shape = (batch, heads, seqlen, head_dim)
    query = torch.randn(shape, device=DEVICE, dtype=dtype)
    key = torch.randn(shape, device=DEVICE, dtype=dtype)
    value = torch.randn(shape, device=DEVICE, dtype=dtype)
    dout = torch.randn(shape, device=DEVICE, dtype=dtype)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale),
                   query, key, value, dout)
    naive = grads(lambda q, k, v: reference_attention(q, k, v, causal, scale),
                  query, key, value, dout)
    exact = grads(lambda q, k, v: reference_attention(q, k, v, causal, scale),
                  query.float(), key.float(), value.float(), dout.float())

    for kernel_g, naive_g, exact_g, name in zip(kernel, naive, exact, ("dq", "dk", "dv")):
        kernel_err = (kernel_g.float() - exact_g).abs().max().item()
        naive_err = (naive_g.float() - exact_g).abs().max().item()
        assert kernel_err <= 2.0 * naive_err + 1e-3, (
            f"{name}: kernel_err={kernel_err:.2e} naive_err={naive_err:.2e}")


def test_lse_matches_logsumexp():
    torch.manual_seed(0)
    head_dim, seqlen = 64, 512
    scale = 1.0 / math.sqrt(head_dim)
    shape = (2, 4, seqlen, head_dim)
    query = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    key = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    value = torch.randn(shape, device=DEVICE, dtype=torch.float16)

    _, lse = _forward(query, key, value, False, scale)
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    expected = torch.logsumexp(logits, dim=-1)
    torch.testing.assert_close(lse, expected, atol=2e-2, rtol=2e-2)


def test_backward_cross_shapes():
    torch.manual_seed(1)
    scale = 1.0 / math.sqrt(64)
    query = torch.randn(2, 8, 333, 64, device=DEVICE, dtype=torch.float16)
    key = torch.randn(2, 8, 777, 64, device=DEVICE, dtype=torch.float16)
    value = torch.randn(2, 8, 777, 64, device=DEVICE, dtype=torch.float16)
    dout = torch.randn(2, 8, 333, 64, device=DEVICE, dtype=torch.float16)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, False, scale),
                   query, key, value, dout)
    exact = grads(lambda q, k, v: reference_attention(q, k, v, False, scale),
                  query.float(), key.float(), value.float(), dout.float())
    naive = grads(lambda q, k, v: reference_attention(q, k, v, False, scale),
                  query, key, value, dout)
    for kernel_g, naive_g, exact_g in zip(kernel, naive, exact):
        kernel_err = (kernel_g.float() - exact_g).abs().max().item()
        naive_err = (naive_g.float() - exact_g).abs().max().item()
        assert kernel_err <= 2.0 * naive_err + 1e-3
