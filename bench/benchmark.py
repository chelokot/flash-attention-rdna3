"""Benchmark the RDNA3 Triton kernel against PyTorch's default SDPA path.

On gfx1100 the default ``scaled_dot_product_attention`` falls back to a
non-fused math implementation (the fused flash/mem-efficient backends are
gated behind an experimental flag and lack RDNA3 kernels), so this measures
the speedup a fused Triton kernel buys over the stock experience.
"""

import argparse

import torch
import torch.nn.functional as F

from fa_rdna3 import flash_attention


def timed(fn, iters=50, warmup=10):
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


def tflops(batch, heads, seqlen, head_dim, causal, ms):
    flops = 2 * 2 * batch * heads * seqlen * seqlen * head_dim
    if causal:
        flops //= 2
    return flops / (ms * 1e-3) / 1e12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"{'shape':>28} {'causal':>7} {'torch ms':>10} {'triton ms':>10} "
          f"{'speedup':>8} {'triton TFLOP/s':>15}")
    configs = [
        (1, 32, 1024, 128),
        (2, 16, 2048, 128),
        (1, 16, 4096, 64),
        (1, 24, 4096, 128),
        (1, 16, 8192, 64),
    ]
    for causal in (False, True):
        for batch, heads, seqlen, head_dim in configs:
            shape = (batch, heads, seqlen, head_dim)
            q = torch.randn(shape, device="cuda", dtype=dtype)
            k = torch.randn(shape, device="cuda", dtype=dtype)
            v = torch.randn(shape, device="cuda", dtype=dtype)

            torch_ms = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
            triton_ms = timed(lambda: flash_attention(q, k, v, causal=causal))
            label = f"b{batch} h{heads} s{seqlen} d{head_dim}"
            print(f"{label:>28} {str(causal):>7} {torch_ms:>10.3f} {triton_ms:>10.3f} "
                  f"{torch_ms / triton_ms:>7.2f}x {tflops(batch, heads, seqlen, head_dim, causal, triton_ms):>14.1f}")


if __name__ == "__main__":
    main()
