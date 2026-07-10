"""Paged-KV decode: gather keys/values through a block table (vLLM-style)."""

import math

import pytest
import torch

from fa_rdna3 import flash_attention_decode_paged
from fa_rdna3.interface import _paged_num_splits

DEVICE = "cuda"


def reference_paged(query, k_cache, v_cache, block_table, context_lens, scale):
    batch, q_heads, head_dim = query.shape
    block_size, kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = q_heads // kv_heads
    outs = []
    for b in range(batch):
        ctx = int(context_lens[b])
        if ctx == 0:
            outs.append(torch.zeros(q_heads, head_dim, device=query.device))
            continue
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


def test_paged_decode_zero_context_returns_zeros():
    batch, heads, head_dim, block_size = 2, 4, 64, 16
    query = torch.randn(batch, heads, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(4, block_size, heads, head_dim, device=DEVICE, dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.zeros(batch, 1, device=DEVICE, dtype=torch.int32)
    context_lens = torch.zeros(batch, device=DEVICE, dtype=torch.int32)

    out = flash_attention_decode_paged(query, key, value, block_table, context_lens)

    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.parametrize("num_splits", [1, 3, 7])
def test_paged_decode_forced_splits(num_splits):
    torch.manual_seed(31 + num_splits)
    contexts = torch.tensor([0, 1, 15, 16, 17, 49], device=DEVICE, dtype=torch.int32)
    batch, q_heads, kv_heads, head_dim, block_size = 6, 4, 2, 64, 16
    max_blocks = math.ceil(int(contexts.max()) / block_size)
    query = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(batch * max_blocks, block_size, kv_heads, head_dim,
                      device=DEVICE, dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.randperm(batch * max_blocks, device=DEVICE, dtype=torch.int32).reshape(batch, max_blocks)

    out = flash_attention_decode_paged(
        query, key, value, block_table, contexts, num_splits=num_splits)
    ref = reference_paged(
        query, key, value, block_table, contexts, 1.0 / math.sqrt(head_dim))

    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("block_size", [8, 32, 64])
def test_paged_decode_block_sizes(block_size):
    contexts = torch.tensor(
        [0, 1, block_size - 1, block_size, block_size + 5],
        device=DEVICE, dtype=torch.int32)
    batch, q_heads, kv_heads, head_dim = contexts.numel(), 4, 2, 64
    max_blocks = math.ceil(int(contexts.max()) / block_size)
    query = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(
        batch * max_blocks, block_size, kv_heads, head_dim,
        device=DEVICE, dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.arange(
        batch * max_blocks, device=DEVICE, dtype=torch.int32,
    ).reshape(batch, max_blocks)

    out = flash_attention_decode_paged(
        query, key, value, block_table, contexts, num_splits=3)
    ref = reference_paged(
        query, key, value, block_table, contexts, 1.0 / math.sqrt(head_dim))

    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-3)


def test_paged_decode_strided_metadata():
    batch, heads, head_dim, block_size, context = 3, 4, 64, 16, 31
    max_blocks = math.ceil(context / block_size)
    query = torch.randn(batch, heads, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(
        batch * max_blocks, block_size, heads, head_dim,
        device=DEVICE, dtype=torch.float16)
    value = torch.randn_like(key)
    block_storage = torch.zeros(
        batch, 2 * max_blocks, device=DEVICE, dtype=torch.int32)
    block_table = block_storage[:, ::2]
    block_table.copy_(torch.arange(
        batch * max_blocks, device=DEVICE, dtype=torch.int32,
    ).reshape(batch, max_blocks))
    context_storage = torch.zeros(2 * batch, device=DEVICE, dtype=torch.int32)
    context_lens = context_storage[::2]
    context_lens.fill_(context)

    out = flash_attention_decode_paged(
        query, key, value, block_table, context_lens, num_splits=3)
    ref = reference_paged(
        query, key, value, block_table, context_lens, 1.0 / math.sqrt(head_dim))

    assert not block_table.is_contiguous()
    assert not context_lens.is_contiguous()
    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-3)


def test_paged_decode_invalid_blocks_are_safely_ignored():
    batch, heads, head_dim, block_size = 2, 4, 64, 16
    query = torch.randn(batch, heads, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(2, block_size, heads, head_dim, device=DEVICE, dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.tensor([[-1], [99]], device=DEVICE, dtype=torch.int32)
    context_lens = torch.full((batch,), block_size, device=DEVICE, dtype=torch.int32)

    out = flash_attention_decode_paged(
        query, key, value, block_table, context_lens, num_splits=3)

    torch.testing.assert_close(out, torch.zeros_like(out))


def test_paged_decode_clamps_context_to_table_capacity():
    heads, head_dim, block_size = 2, 64, 16
    query = torch.zeros(1, heads, head_dim, device=DEVICE, dtype=torch.float16)
    key = torch.randn(1, block_size, heads, head_dim, device=DEVICE, dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.zeros(1, 1, device=DEVICE, dtype=torch.int32)
    context_lens = torch.tensor([2 * block_size], device=DEVICE, dtype=torch.int32)

    out = flash_attention_decode_paged(
        query, key, value, block_table, context_lens, num_splits=3)
    expected = value.float().mean(dim=1)

    torch.testing.assert_close(out.float(), expected, atol=3e-3, rtol=3e-3)


def test_paged_split_heuristic():
    assert _paged_num_splits(1, 8, 16, 256, 96, requested_splits=7) == 7
    assert _paged_num_splits(1, 8, 16, 4, 96) == 1
    assert _paged_num_splits(1, 8, 16, 256, 96) == 16
    assert _paged_num_splits(4, 32, 16, 256, 96, max_context_len=256) == 4
    assert _paged_num_splits(8, 32, 16, 256, 96, max_context_len=256) == 2
    assert _paged_num_splits(16, 32, 16, 256, 96, max_context_len=256) == 1
    assert _paged_num_splits(16, 32, 16, 256, 96, max_context_len=4096) == 16
