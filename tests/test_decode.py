"""Correctness for the split-K decode kernel (small query, long KV cache)."""

import math

import pytest
import torch

from fa_rdna3 import flash_attention_decode
from fa_rdna3.interface import _decode_num_splits

DEVICE = "cuda"


def reference_attention(query, key, value, softmax_scale):
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * softmax_scale
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, value.float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seqlen_q", [1, 4])
@pytest.mark.parametrize("seqlen_k", [128, 1000, 4096, 16384])
def test_decode_matches_reference(dtype, head_dim, seqlen_q, seqlen_k):
    torch.manual_seed(seqlen_k + head_dim + seqlen_q)
    batch, heads = 2, 8
    scale = 1.0 / math.sqrt(head_dim)
    query = torch.randn(batch, heads, seqlen_q, head_dim, device=DEVICE, dtype=dtype)
    key = torch.randn(batch, heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)
    value = torch.randn(batch, heads, seqlen_k, head_dim, device=DEVICE, dtype=dtype)

    out = flash_attention_decode(query, key, value, scale)
    ref = reference_attention(query, key, value, scale)
    tol = {torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_decode_actually_splits():
    # A long cache with few heads must fan out across more than one split,
    # otherwise the kernel offers nothing over the plain forward.
    assert _decode_num_splits(batch=1, heads=4, seqlen_q=1, seqlen_k=16384) > 1
    # A short cache that already fills the machine should not over-split.
    assert _decode_num_splits(batch=8, heads=32, seqlen_q=512, seqlen_k=256) == 1
