"""Attention dropout: philox mask applied to the softmax weights.

The forward draws a seed and the backward reuses it, so the two see the same
mask. Tested three ways: p=0 exactness, forward unbiasedness (E ≈ no-dropout),
and a fixed-seed directional finite difference (which fails if the forward and
backward masks disagree).
"""

import math

import torch

from fa_rdna3 import flash_attention
from fa_rdna3.interface import _forward, _backward

DEVICE = "cuda"


def test_dropout_p0_matches_no_dropout():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 512, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(64)
    a = flash_attention(q, k, v, causal=True, softmax_scale=scale)
    b = flash_attention(q, k, v, causal=True, softmax_scale=scale, dropout_p=0.0)
    torch.testing.assert_close(a, b)


def test_dropout_forward_unbiased():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 256, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(64)
    ref = flash_attention(q, k, v, softmax_scale=scale).float()
    acc = torch.zeros_like(ref)
    n = 400
    for _ in range(n):
        acc += flash_attention(q, k, v, softmax_scale=scale, dropout_p=0.25).float()
    mean = acc / n
    assert (mean - ref).abs().mean().item() < 5e-3


def test_dropout_backward_consistency():
    # Fixed seed => deterministic; directional FD of L = <out, dout> vs <dq, direction>.
    torch.manual_seed(2)
    seed, p, scale = 777, 0.2, 1.0 / math.sqrt(64)
    q = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    out, lse = _forward(q, k, v, False, scale, (-1, -1), 0.0, None, None, p, seed)
    dq, dk, dv = _backward(dout, q, k, v, out, lse, False, scale, (-1, -1), 0.0, None, None, p, seed)
    base = (out.float() * dout.float()).sum()

    eps = 0.1
    for grad, tensor, name in ((dq, q, "dq"), (dk, k, "dk"), (dv, v, "dv")):
        direction = torch.randn_like(tensor)
        pert = {"dq": q, "dk": k, "dv": v}
        perturbed = {k_: t.clone() for k_, t in (("dq", q), ("dk", k), ("dv", v))}
        perturbed[name] = tensor + eps * direction
        out2, _ = _forward(perturbed["dq"], perturbed["dk"], perturbed["dv"],
                           False, scale, (-1, -1), 0.0, None, None, p, seed)
        fd = ((out2.float() * dout.float()).sum() - base) / eps
        analytical = (grad.float() * direction.float()).sum()
        rel = (fd - analytical).abs() / (analytical.abs() + 1.0)
        assert rel < 0.1, f"{name}: fd={fd.item():.3f} analytical={analytical.item():.3f}"
