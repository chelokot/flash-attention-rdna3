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
from fa_rdna3.comfyui import ComfyUIAttentionOverride


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


def fwd_flops(batch, heads, seqlen, head_dim, causal):
    flops = 2 * 2 * batch * heads * seqlen * seqlen * head_dim
    if causal:
        flops //= 2
    return flops


def tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def benchmark_comfyui(dtype):
    batch, heads, seqlen_q, head_dim = 2, 8, 4096, 64
    override = ComfyUIAttentionOverride()

    def unsupported(*args, **kwargs):
        raise RuntimeError("RDNA3 benchmark unexpectedly fell back")

    print(f"{'shape':>30} {'node ms':>9} {'torch ms':>9} {'speedup':>9} {'max diff':>10}")
    for seqlen_k in (77, 4096):
        query = torch.randn(
            batch, seqlen_q, heads * head_dim,
            device="cuda", dtype=dtype,
        )
        key = torch.randn(
            batch, seqlen_k, heads * head_dim,
            device="cuda", dtype=dtype,
        )
        value = torch.randn_like(key)

        def torch_attention():
            query_heads = query.reshape(
                batch, seqlen_q, heads, head_dim).transpose(1, 2)
            key_heads = key.reshape(
                batch, seqlen_k, heads, head_dim).transpose(1, 2)
            value_heads = value.reshape(
                batch, seqlen_k, heads, head_dim).transpose(1, 2)
            output = F.scaled_dot_product_attention(
                query_heads, key_heads, value_heads)
            return output.transpose(1, 2).reshape(
                batch, seqlen_q, heads * head_dim)

        def rdna3_attention():
            return override(unsupported, query, key, value, heads)

        reference = torch_attention()
        actual = rdna3_attention()
        node_ms = timed(rdna3_attention)
        torch_ms = timed(torch_attention)
        max_diff = float((actual.float() - reference.float()).abs().max())
        label = f"b{batch} h{heads} q{seqlen_q} k{seqlen_k} d{head_dim}"
        print(
            f"{label:>30} {node_ms:>9.3f} {torch_ms:>9.3f} "
            f"{torch_ms / node_ms:>8.2f}x {max_diff:>10.4g}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--comfyui", action="store_true")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if args.comfyui:
        benchmark_comfyui(dtype)
        return

    print(f"{'shape':>28} {'causal':>7} {'fwd ms':>8} {'fwd TF/s':>9} "
          f"{'bwd ms':>8} {'bwd TF/s':>9} {'vs torch':>9}")
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
            q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            dout = torch.randn(shape, device="cuda", dtype=dtype)

            torch_ms = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
            fwd_ms = timed(lambda: flash_attention(q, k, v, causal=causal))

            out = flash_attention(q, k, v, causal=causal)
            bwd_ms = timed(lambda: out.backward(dout, retain_graph=True))

            fflops = fwd_flops(batch, heads, seqlen, head_dim, causal)
            bflops = int(2.5 * fflops)  # standard FA convention: backward ~2.5x forward
            label = f"b{batch} h{heads} s{seqlen} d{head_dim}"
            print(f"{label:>28} {str(causal):>7} {fwd_ms:>8.3f} {tflops(fflops, fwd_ms):>9.1f} "
                  f"{bwd_ms:>8.3f} {tflops(bflops, bwd_ms):>9.1f} {torch_ms / fwd_ms:>8.2f}x")


if __name__ == "__main__":
    main()
