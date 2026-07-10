"""Bottom-right causal alignment for seqlen_q != seqlen_k.

Query i sits at absolute position i + (seqlen_k - seqlen_q) and attends keys up
to there (Dao-AILab/flash-attention's default causal since 2.1). For square
attention this is identical to top-left, so it does not change self-attention.
"""

import math

import pytest
import torch

from fa_rdna3 import flash_attention

DEVICE = "cuda"


def reference_br_causal(query, key, value, scale):
    sq, sk = query.shape[-2], key.shape[-2]
    logits = (torch.matmul(query, key.transpose(-1, -2)) * scale).float()
    row = torch.arange(sq, device=DEVICE)[:, None]
    col = torch.arange(sk, device=DEVICE)[None, :]
    keep = (row + (sk - sq)) >= col
    logits = logits.masked_fill(~keep, float("-inf"))
    valid = keep.any(dim=-1, keepdim=True)
    logits = torch.where(valid, logits, torch.zeros_like(logits))
    probs = torch.where(valid, torch.softmax(logits, dim=-1), 0.0)
    return torch.matmul(probs.to(value.dtype), value)


def grads(fn, q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (128, 384), (64, 256), (1, 256), (200, 200), (333, 512),
    (384, 128), (256, 64),
])
def test_br_causal_forward(dtype, seqlen_q, seqlen_k):
    torch.manual_seed(seqlen_q + seqlen_k)
    batch, heads, head_dim = 2, 4, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen_q, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch, heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)

    out = flash_attention(q, k, v, causal=True, softmax_scale=scale)
    exact = reference_br_causal(q.float(), k.float(), v.float(), scale)
    naive = reference_br_causal(q, k, v, scale)
    ke = (out.float() - exact).abs().max().item()
    ne = (naive.float() - exact).abs().max().item()
    assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"


@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 384), (200, 200), (384, 128)])
def test_br_causal_backward(seqlen_q, seqlen_k):
    torch.manual_seed(seqlen_q + seqlen_k)
    batch, heads, head_dim, dtype = 2, 4, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen_q, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch, heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)
    dout = torch.randn(batch, heads, seqlen_q, head_dim, device=DEVICE, dtype=dtype)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal=True, softmax_scale=scale), q, k, v, dout)
    exact = grads(lambda q, k, v: reference_br_causal(q, k, v, scale), q.float(), k.float(), v.float(), dout.float())
    naive = grads(lambda q, k, v: reference_br_causal(q, k, v, scale), q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive, exact):
        ke = (kg.float() - eg).abs().max().item()
        ne = (ng.float() - eg).abs().max().item()
        assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"
