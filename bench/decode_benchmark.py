"""Benchmark split-K decode against the plain forward and default SDPA.

Decode (one query row, long KV cache) is latency- and bandwidth-bound, not
compute-bound, so this reports microseconds and the KV read bandwidth reached
rather than TFLOP/s. The plain forward launches one workgroup per (batch, head)
and lets it walk the whole cache; split-K fans the cache across the GPU.
"""

import argparse

import torch
import torch.nn.functional as F

from fa_rdna3 import flash_attention, flash_attention_decode


def timed(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"{'shape (b x h x kv x d)':>26} {'sdpa us':>9} {'fwd us':>9} "
          f"{'decode us':>10} {'vs fwd':>7} {'KV GB/s':>8}")
    configs = [
        (1, 8, 1024, 128),
        (1, 8, 4096, 128),
        (1, 8, 16384, 128),
        (1, 32, 4096, 128),
        (4, 32, 8192, 128),
        (1, 8, 16384, 64),
    ]
    for batch, heads, seqlen_k, head_dim in configs:
        q = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, heads, seqlen_k, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, heads, seqlen_k, head_dim, device="cuda", dtype=dtype)

        sdpa_us = timed(lambda: F.scaled_dot_product_attention(q, k, v)) * 1e3
        fwd_us = timed(lambda: flash_attention(q, k, v, causal=False)) * 1e3
        dec_us = timed(lambda: flash_attention_decode(q, k, v)) * 1e3

        kv_bytes = 2 * batch * heads * seqlen_k * head_dim * q.element_size()
        gbps = kv_bytes / (dec_us * 1e-6) / 1e9
        label = f"b{batch} h{heads} kv{seqlen_k} d{head_dim}"
        print(f"{label:>26} {sdpa_us:>9.1f} {fwd_us:>9.1f} {dec_us:>10.1f} "
              f"{fwd_us / dec_us:>6.2f}x {gbps:>8.0f}")


if __name__ == "__main__":
    main()
