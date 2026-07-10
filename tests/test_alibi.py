"""ALiBi: per-head linear positional bias -slope * abs(query_pos - key_pos).

Convention follows Press et al. (2021) / Dao-AILab/flash-attention.
"""

import math

import pytest
import torch

from fa_rdna3 import flash_attention, alibi_slopes

DEVICE = "cuda"


@pytest.mark.parametrize("n_heads", [0, -1, 1.5, True])
def test_alibi_slopes_rejects_invalid_head_count(n_heads):
    with pytest.raises(ValueError, match="positive integer"):
        alibi_slopes(n_heads, device="cpu")


def reference_alibi(query, key, value, scale, slopes, causal):
    sq, sk = query.shape[-2], key.shape[-2]
    logits = (torch.matmul(query, key.transpose(-1, -2)) * scale).float()
    row = torch.arange(sq, device=DEVICE)[:, None]
    col = torch.arange(sk, device=DEVICE)[None, :]
    distance = (col - row - (sk - sq)).abs().float()
    logits = logits - slopes.view(1, -1, 1, 1) * distance
    if causal:
        logits = logits.masked_fill((row + (sk - sq)) < col, float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=-1).to(value.dtype), value)


def grads(fn, q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads", [8, 12])
@pytest.mark.parametrize("seqlen", [200, 512])
def test_alibi_forward(dtype, causal, heads, seqlen):
    torch.manual_seed(seqlen + heads + int(causal))
    batch, head_dim = 2, 64
    scale = 1.0 / math.sqrt(head_dim)
    slopes = alibi_slopes(heads, device=DEVICE)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal, softmax_scale=scale, alibi_slopes=slopes)
    exact = reference_alibi(q.float(), k.float(), v.float(), scale, slopes, causal)
    naive = reference_alibi(q, k, v, scale, slopes, causal)
    ke = (out.float() - exact).abs().max().item()
    ne = (naive.float() - exact).abs().max().item()
    assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"


@pytest.mark.parametrize("causal", [False, True])
def test_alibi_backward(causal):
    torch.manual_seed(int(causal))
    batch, heads, seqlen, head_dim, dtype = 2, 8, 512, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    slopes = alibi_slopes(heads, device=DEVICE)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale, alibi_slopes=slopes), q, k, v, dout)
    exact = grads(lambda q, k, v: reference_alibi(q, k, v, scale, slopes, causal),
                  q.float(), k.float(), v.float(), dout.float())
    naive = grads(lambda q, k, v: reference_alibi(q, k, v, scale, slopes, causal), q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive, exact):
        ke = (kg.float() - eg).abs().max().item()
        ne = (ng.float() - eg).abs().max().item()
        assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"
