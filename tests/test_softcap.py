"""Attention logit soft-capping: softcap * tanh(logit / softcap) before softmax.

Matches Gemma2's ``attn_logit_softcapping`` (value 50.0).
"""

import math

import pytest
import torch

from fa_rdna3 import flash_attention

DEVICE = "cuda"


def reference_softcap(query, key, value, causal, scale, softcap):
    """Dense reference, input dtype preserved (softmax in fp32)."""
    sq, sk = query.shape[-2], key.shape[-2]
    logits = (torch.matmul(query, key.transpose(-1, -2)) * scale).float()
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    if causal:
        row = torch.arange(sq, device=DEVICE)[:, None]
        col = torch.arange(sk, device=DEVICE)[None, :]
        logits = logits.masked_fill(row < col, float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=-1).to(value.dtype), value)


def grads(fn, q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softcap", [30.0, 50.0])
@pytest.mark.parametrize("seqlen", [200, 512])
def test_softcap_forward(dtype, causal, softcap, seqlen):
    torch.manual_seed(seqlen + int(softcap) + int(causal))
    batch, heads, head_dim = 2, 4, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal, softmax_scale=scale, softcap=softcap)
    exact = reference_softcap(q.float(), k.float(), v.float(), causal, scale, softcap)
    naive = reference_softcap(q, k, v, causal, scale, softcap)
    ke = (out.float() - exact).abs().max().item()
    ne = (naive.float() - exact).abs().max().item()
    assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"


@pytest.mark.parametrize("causal", [False, True])
def test_softcap_backward(causal):
    torch.manual_seed(int(causal))
    batch, heads, seqlen, head_dim, dtype, softcap = 2, 4, 512, 64, torch.float16, 50.0
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale, softcap=softcap), q, k, v, dout)
    exact = grads(lambda q, k, v: reference_softcap(q, k, v, causal, scale, softcap),
                  q.float(), k.float(), v.float(), dout.float())
    naive = grads(lambda q, k, v: reference_softcap(q, k, v, causal, scale, softcap), q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive, exact):
        ke = (kg.float() - eg).abs().max().item()
        ne = (ng.float() - eg).abs().max().item()
        assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"
