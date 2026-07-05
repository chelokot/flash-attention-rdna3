"""Sliding-window attention: query i attends keys in [i-left, i+right].

Convention matches Dao-AILab/flash-attention's ``window_size=(left, right)``
(https://github.com/Dao-AILab/flash-attention).
"""

import math

import pytest
import torch

from fa_rdna3 import flash_attention

DEVICE = "cuda"


def reference_window(query, key, value, causal, scale, window):
    """Dense reference; input dtype preserved (softmax in fp32) so fp32 inputs
    give the exact reference and low-precision inputs give the naive baseline."""
    left, right = window
    sq, sk = query.shape[-2], key.shape[-2]
    logits = (torch.matmul(query, key.transpose(-1, -2)) * scale).float()
    row = torch.arange(sq, device=DEVICE)[:, None]
    col = torch.arange(sk, device=DEVICE)[None, :]
    keep = torch.ones_like(logits, dtype=torch.bool)
    if causal:
        keep &= row >= col
    if left >= 0:
        keep &= col >= row - left
    if right >= 0:
        keep &= col <= row + right
    logits = logits.masked_fill(~keep, float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=-1).to(value.dtype), value)


def grads(fn, q, k, v, dout):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    fn(q, k, v).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal,window", [
    (True, (63, 0)),      # Mistral-style causal window of 64
    (True, (255, 0)),     # window wider than a block
    (False, (48, 48)),    # symmetric bidirectional window
    (False, (100, 0)),    # left-only, non-causal
])
@pytest.mark.parametrize("seqlen", [200, 512])
def test_window_forward(dtype, causal, window, seqlen):
    torch.manual_seed(seqlen + window[0] + int(causal))
    batch, heads, head_dim = 2, 4, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal, softmax_scale=scale, window_size=window)
    exact = reference_window(q.float(), k.float(), v.float(), causal, scale, window)
    naive = reference_window(q, k, v, causal, scale, window)
    ke = (out.float() - exact).abs().max().item()
    ne = (naive.float() - exact).abs().max().item()
    assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"


@pytest.mark.parametrize("causal,window", [(True, (63, 0)), (False, (48, 48))])
def test_window_backward(causal, window):
    torch.manual_seed(window[0] + int(causal))
    batch, heads, seqlen, head_dim, dtype = 2, 4, 512, 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    kernel = grads(lambda q, k, v: flash_attention(q, k, v, causal, scale, window), q, k, v, dout)
    exact = grads(lambda q, k, v: reference_window(q, k, v, causal, scale, window),
                  q.float(), k.float(), v.float(), dout.float())
    naive = grads(lambda q, k, v: reference_window(q, k, v, causal, scale, window), q, k, v, dout)
    for kg, ng, eg in zip(kernel, naive, exact):
        ke = (kg.float() - eg).abs().max().item()
        ne = (ng.float() - eg).abs().max().item()
        assert ke <= 2.0 * ne + 1e-3, f"err {ke:.2e} vs naive {ne:.2e}"
