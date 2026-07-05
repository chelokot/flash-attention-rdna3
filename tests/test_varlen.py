"""Variable-length (cu_seqlens) attention: packed sequences, no padding."""

import math

import pytest
import torch

from fa_rdna3 import flash_attention_varlen

DEVICE = "cuda"


def _cu_seqlens(lengths):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=DEVICE, dtype=torch.int32)


def reference_varlen(query, key, value, lengths_q, lengths_k, causal, scale):
    """Per-sequence dense attention over the packed tensors, concatenated back.

    Computes in the input dtype (softmax in fp32), so passing fp32 inputs gives
    the exact reference and low-precision inputs give the naive baseline.
    """
    outs = []
    qs = ks = 0
    group = query.shape[1] // key.shape[1]
    for lq, lk in zip(lengths_q, lengths_k):
        q = query[qs:qs + lq].transpose(0, 1)                  # (heads, lq, d)
        k = key[ks:ks + lk].transpose(0, 1).repeat_interleave(group, dim=0)
        v = value[ks:ks + lk].transpose(0, 1).repeat_interleave(group, dim=0)
        logits = (torch.matmul(q, k.transpose(-1, -2)) * scale).float()
        if causal:
            row = torch.arange(lq, device=DEVICE)[:, None]
            col = torch.arange(lk, device=DEVICE)[None, :]
            logits = logits.masked_fill(row < col, float("-inf"))
        o = torch.matmul(torch.softmax(logits, dim=-1).to(v.dtype), v)
        outs.append(o.transpose(0, 1))                         # (lq, heads, d)
        qs += lq
        ks += lk
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 2)])
def test_varlen_forward(dtype, causal, q_heads, kv_heads):
    torch.manual_seed(sum(len(str(x)) for x in (dtype, causal, q_heads)) + kv_heads)
    lengths = [128, 57, 300, 1]
    head_dim = 64
    scale = 1.0 / math.sqrt(head_dim)
    total = sum(lengths)
    query = torch.randn(total, q_heads, head_dim, device=DEVICE, dtype=dtype)
    key = torch.randn(total, kv_heads, head_dim, device=DEVICE, dtype=dtype)
    value = torch.randn(total, kv_heads, head_dim, device=DEVICE, dtype=dtype)
    cu = _cu_seqlens(lengths)

    out = flash_attention_varlen(query, key, value, cu, cu, max(lengths), max(lengths),
                                 causal=causal, softmax_scale=scale)
    exact = reference_varlen(query.float(), key.float(), value.float(), lengths, lengths, causal, scale)
    naive = reference_varlen(query, key, value, lengths, lengths, causal, scale)
    kernel_err = (out.float() - exact).abs().max().item()
    naive_err = (naive.float() - exact).abs().max().item()
    assert kernel_err <= 2.0 * naive_err + 1e-3, f"kernel_err={kernel_err:.2e} naive_err={naive_err:.2e}"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 2)])
def test_varlen_backward(causal, q_heads, kv_heads):
    torch.manual_seed(q_heads + kv_heads + int(causal))
    lengths = [200, 64, 129]
    head_dim, dtype = 64, torch.float16
    scale = 1.0 / math.sqrt(head_dim)
    total = sum(lengths)
    cu = _cu_seqlens(lengths)

    def grads(fn, q, k, v, dout):
        q = q.detach().clone().requires_grad_(True)
        k = k.detach().clone().requires_grad_(True)
        v = v.detach().clone().requires_grad_(True)
        fn(q, k, v).backward(dout)
        return q.grad, k.grad, v.grad

    query = torch.randn(total, q_heads, head_dim, device=DEVICE, dtype=dtype)
    key = torch.randn(total, kv_heads, head_dim, device=DEVICE, dtype=dtype)
    value = torch.randn(total, kv_heads, head_dim, device=DEVICE, dtype=dtype)
    dout = torch.randn(total, q_heads, head_dim, device=DEVICE, dtype=dtype)

    kernel = grads(lambda q, k, v: flash_attention_varlen(
        q, k, v, cu, cu, max(lengths), max(lengths), causal, scale), query, key, value, dout)
    exact = grads(lambda q, k, v: reference_varlen(q, k, v, lengths, lengths, causal, scale),
                  query.float(), key.float(), value.float(), dout.float())
    naive = grads(lambda q, k, v: reference_varlen(q, k, v, lengths, lengths, causal, scale),
                  query, key, value, dout)
    for kernel_g, naive_g, exact_g in zip(kernel, naive, exact):
        kernel_err = (kernel_g.float() - exact_g).abs().max().item()
        naive_err = (naive_g.float() - exact_g).abs().max().item()
        assert kernel_err <= 2.0 * naive_err + 1e-3


def test_varlen_matches_padded_batched():
    # Equal-length sequences: varlen must equal the standard batched kernel.
    from fa_rdna3 import flash_attention
    torch.manual_seed(7)
    batch, heads, seqlen, head_dim = 3, 8, 128, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seqlen, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    batched = flash_attention(q, k, v, causal=True, softmax_scale=scale)
    packed_q = q.transpose(1, 2).reshape(batch * seqlen, heads, head_dim)
    packed_k = k.transpose(1, 2).reshape(batch * seqlen, heads, head_dim)
    packed_v = v.transpose(1, 2).reshape(batch * seqlen, heads, head_dim)
    cu = _cu_seqlens([seqlen] * batch)
    packed_out = flash_attention_varlen(packed_q, packed_k, packed_v, cu, cu, seqlen, seqlen,
                                        causal=True, softmax_scale=scale)
    packed_out = packed_out.reshape(batch, seqlen, heads, head_dim).transpose(1, 2)
    torch.testing.assert_close(packed_out.float(), batched.float(), atol=2e-3, rtol=2e-3)
