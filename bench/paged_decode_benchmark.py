"""Benchmark automatic split-K paged decode against one serial KV walker."""

import argparse
import math

import torch
import triton

from fa_rdna3 import flash_attention_decode_paged
from fa_rdna3.interface import _paged_num_splits


def timed(function):
    function()
    torch.cuda.synchronize()
    return float(triton.testing.do_bench(function, warmup=25, rep=100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    target_programs = 2 * torch.cuda.get_device_properties(0).multi_processor_count

    print(f"{'shape (b x qh/kvh x kv x d)':>32} {'split':>5} {'serial us':>10} "
          f"{'auto us':>9} {'speedup':>8} {'eff. GB/s':>9}")
    configs = (
        (1, 8, 8, 256, 128),
        (1, 8, 8, 1024, 128),
        (1, 8, 8, 4096, 128),
        (1, 32, 8, 4096, 128),
        (4, 32, 8, 4096, 128),
        (4, 32, 8, 16384, 128),
    )
    for batch, query_heads, kv_heads, context_len, head_dim in configs:
        blocks_per_sequence = math.ceil(context_len / args.block_size)
        query = torch.randn(batch, query_heads, head_dim, device="cuda", dtype=dtype)
        key = torch.randn(
            batch * blocks_per_sequence, args.block_size, kv_heads, head_dim,
            device="cuda", dtype=dtype)
        value = torch.randn_like(key)
        block_table = torch.arange(
            batch * blocks_per_sequence, device="cuda", dtype=torch.int32,
        ).reshape(batch, blocks_per_sequence)
        context_lens = torch.full(
            (batch,), context_len, device="cuda", dtype=torch.int32)

        serial_ms = timed(lambda: flash_attention_decode_paged(
            query, key, value, block_table, context_lens,
            max_context_len=context_len, num_splits=1))
        automatic_ms = timed(lambda: flash_attention_decode_paged(
            query, key, value, block_table, context_lens,
            max_context_len=context_len))
        selected_splits = _paged_num_splits(
            batch, query_heads, args.block_size, blocks_per_sequence,
            target_programs, max_context_len=context_len)
        kv_bytes = 2 * batch * query_heads * context_len * head_dim * query.element_size()
        bandwidth = kv_bytes / (automatic_ms * 1e-3) / 1e9
        label = f"b{batch} h{query_heads}/{kv_heads} kv{context_len} d{head_dim}"
        print(f"{label:>32} {selected_splits:>5} {serial_ms * 1e3:>10.1f} "
              f"{automatic_ms * 1e3:>9.1f} {serial_ms / automatic_ms:>7.2f}x "
              f"{bandwidth:>8.0f}")


if __name__ == "__main__":
    main()
