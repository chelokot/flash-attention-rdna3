"""Paged-KV decode: gather keys/values through a block table (vLLM-style)."""

import math

import pytest
import torch

from fa_rdna3 import flash_attention_decode_paged

DEVICE = "cuda"


def reference_paged(query, k_cache, v_cache, block_table, context_lens, scale):
    batch, q_heads, head_dim = query.shape
    block_size, kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = q_heads // kv_heads
    outs = []
    for b in range(batch):
        ctx = int(context_lens[b])
        nblk = math.ceil(ctx / block_size)
        phys = block_table[b, :nblk]
        kb = k_cache[phys].reshape(nblk * block_size, kv_heads, head_dim)[:ctx].float()
        vb = v_cache[phys].reshape(nblk * block_size, kv_heads, head_dim)[:ctx].float()
        kb = kb.repeat_interleave(group, dim=1)                 # (ctx, q_heads, d)
        vb = vb.repeat_interleave(group, dim=1)
        qb = query[b].float()                                   # (q_heads, d)
        logits = torch.einsum("hd,chd->hc", qb, kb) * scale
        p = torch.softmax(logits, dim=-1)
        outs.append(torch.einsum("hc,chd->hd", p, vb))
    return torch.stack(outs, dim=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 2)])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_paged_decode(dtype, q_heads, kv_heads, head_dim):
    torch.manual_seed(q_heads + kv_heads + head_dim)
    batch, block_size = 3, 16
    context = torch.tensor([100, 63, 16], device=DEVICE, dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_dim)
    max_blocks = math.ceil(int(context.max()) / block_size)

    query = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=dtype)
    k_cache = torch.randn(batch * max_blocks, block_size, kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(batch * max_blocks, device=DEVICE, dtype=torch.int32).reshape(batch, max_blocks)

    out = flash_attention_decode_paged(query, k_cache, v_cache, block_table, context, scale)
    ref = reference_paged(query, k_cache, v_cache, block_table, context, scale)
    tol = {torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)
