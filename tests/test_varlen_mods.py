"""Score modifiers (sliding window, softcap, ALiBi) on the varlen path."""

import math

import pytest
import torch

from fa_rdna3 import flash_attention_varlen, alibi_slopes

DEVICE = "cuda"


def _cu(lengths):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=DEVICE, dtype=torch.int32)


def reference(query, key, value, lengths, causal, scale, window, softcap, slopes):
    left, right = window
    group = query.shape[1] // key.shape[1]
    outs, qs = [], 0
    for L in lengths:
        q = query[qs:qs + L].transpose(0, 1)                       # (heads, L, d)
        k = key[qs:qs + L].transpose(0, 1).repeat_interleave(group, 0)
        v = value[qs:qs + L].transpose(0, 1).repeat_interleave(group, 0)
        logits = (torch.matmul(q, k.transpose(-1, -2)) * scale).float()
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        row = torch.arange(L, device=DEVICE)[:, None]
        col = torch.arange(L, device=DEVICE)[None, :]
        if slopes is not None:
            logits = logits - slopes.view(-1, 1, 1) * (col - row).abs().float()
        keep = torch.ones(L, L, dtype=torch.bool, device=DEVICE)
        if causal:
            keep &= row >= col
        if left >= 0:
            keep &= col >= row - left
        if right >= 0:
            keep &= col <= row + right
        logits = logits.masked_fill(~keep, float("-inf"))
        o = torch.matmul(torch.softmax(logits, dim=-1).to(v.dtype), v)
        outs.append(o.transpose(0, 1))
        qs += L
    return torch.cat(outs, dim=0)


def grads(fn, q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


CASES = [
    dict(causal=True, window=(31, 0), softcap=0.0, use_alibi=False),   # Mistral-style window
    dict(causal=False, window=(-1, -1), softcap=30.0, use_alibi=False),  # softcap
    dict(causal=True, window=(-1, -1), softcap=0.0, use_alibi=True),    # ALiBi causal
]


@pytest.mark.parametrize("case", CASES)
def test_varlen_mod_forward(case):
    torch.manual_seed(hash(str(case)) % 10000)
    lengths, heads, head_dim, dtype = [128, 65, 200], 8, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    total = sum(lengths)
    cu = _cu(lengths)
    slopes = alibi_slopes(heads, device=DEVICE) if case["use_alibi"] else None
    q = torch.randn(total, heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = flash_attention_varlen(q, k, v, cu, cu, max(lengths), max(lengths),
                                 causal=case["causal"], softmax_scale=scale,
                                 window_size=case["window"], softcap=case["softcap"],
                                 alibi_slopes=slopes)
    exact = reference(q.float(), k.float(), v.float(), lengths, case["causal"], scale,
                      case["window"], case["softcap"], slopes)
    naive = reference(q, k, v, lengths, case["causal"], scale, case["window"], case["softcap"], slopes)
    ke = (out.float() - exact).abs().max().item()
    ne = (naive.float() - exact).abs().max().item()
    assert ke <= 2.0 * ne + 2e-3, f"err {ke:.2e} vs naive {ne:.2e}"


@pytest.mark.parametrize("case", CASES)
def test_varlen_mod_backward(case):
    torch.manual_seed(hash(str(case)) % 9999)
    lengths, heads, head_dim, dtype = [128, 200], 8, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    total = sum(lengths)
    cu = _cu(lengths)
    slopes = alibi_slopes(heads, device=DEVICE) if case["use_alibi"] else None
    q = torch.randn(total, heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    fn = lambda q, k, v: flash_attention_varlen(q, k, v, cu, cu, max(lengths), max(lengths),
                                                causal=case["causal"], softmax_scale=scale,
                                                window_size=case["window"], softcap=case["softcap"],
                                                alibi_slopes=slopes)
    ref = lambda q, k, v: reference(q, k, v, lengths, case["causal"], scale,
                                    case["window"], case["softcap"], slopes)
    kernel = grads(fn, q, k, v, dout)
    exact = grads(ref, q.float(), k.float(), v.float(), dout.float())
    naive = grads(ref, q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive, exact):
        ke = (kg.float() - eg).abs().max().item()
        ne = (ng.float() - eg).abs().max().item()
        assert ke <= 2.0 * ne + 2e-3, f"err {ke:.2e} vs naive {ne:.2e}"
