"""Sweep forward tile geometries for the non-causal head-dim-64 weak spot."""

import argparse
import math

import torch
import torch.nn.functional as F
import triton

from fa_rdna3 import flash_attention
from fa_rdna3.interface import _autotune_seqlen_key
from fa_rdna3.kernels import _attention_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--reference", choices=("torch", "fa"), default="torch")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--num-warps", type=int)
    args = parser.parse_args()

    selected_geometry = (args.block_m, args.block_n, args.num_warps)
    if any(value is not None for value in selected_geometry) and any(
            value is None for value in selected_geometry):
        parser.error("--block-m, --block-n and --num-warps must be provided together")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    shape = (args.batch, args.heads, args.seqlen, 64)
    query = torch.randn(shape, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    output = torch.empty_like(query)
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(64)
    if args.reference == "torch":
        reference = F.scaled_dot_product_attention(query, key, value)
    else:
        reference = flash_attention(query, key, value)
    q_bucket = _autotune_seqlen_key(args.seqlen, dtype)
    if args.block_m is not None:
        block_ms = (args.block_m,)
        block_ns = (args.block_n,)
        warp_counts = (args.num_warps,)
    else:
        block_ms = (32, 64, 128, 256) if args.extended else (64, 128)
        block_ns = (16, 32, 64, 128, 256) if args.extended else (16, 32, 64, 128)
        warp_counts = (1, 2, 4, 8) if args.extended else (2, 4, 8)

    measurements = []
    for block_m in block_ms:
        for block_n in block_ns:
            for num_warps in warp_counts:
                grid = (triton.cdiv(args.seqlen, block_m), args.batch * args.heads)

                def run():
                    _attention_forward.fn[grid](
                        query, key, value, output, lse,
                        scale,
                        *query.stride(), *key.stride(), *value.stride(),
                        *output.stride(), *lse.stride(),
                        query, 0, 0, 0, 0,
                        query, 0.0, 0,
                        args.heads, args.seqlen, args.seqlen, q_bucket, q_bucket,
                        HEAD_DIM=64, BLOCK_M=block_m, BLOCK_N=block_n,
                        IS_CAUSAL=False, GROUP_SIZE=1,
                        PARALLELISM_BUCKET=int(args.batch * args.heads > 1),
                        POST_SCALE_Q=dtype == torch.bfloat16,
                        num_warps=num_warps, num_stages=1,
                    )

                try:
                    run()
                    torch.cuda.synchronize()
                    first = output.clone()
                    run()
                    torch.cuda.synchronize()
                    error = float((output.float() - reference.float()).abs().max())
                    drift = float((output.float() - first.float()).abs().max())
                    elapsed_ms = float(triton.testing.do_bench(run, warmup=25, rep=100))
                except Exception as error_message:
                    print(
                        f"BM={block_m:<3} BN={block_n:<3} nw={num_warps}: "
                        f"ERROR {error_message}",
                        flush=True,
                    )
                    continue
                measurements.append((elapsed_ms, error, drift, block_m, block_n, num_warps))
                print(
                    f"BM={block_m:<3} BN={block_n:<3} nw={num_warps}: "
                    f"{elapsed_ms:.4f} ms error={error:.4g} drift={drift:.4g}",
                    flush=True,
                )

    print("\nFastest correct configurations:")
    correct = [measurement for measurement in measurements if measurement[1] <= 2e-3 and measurement[2] == 0.0]
    for elapsed_ms, error, _, block_m, block_n, num_warps in sorted(correct)[:15]:
        print(
            f"{elapsed_ms:.4f} ms  BM={block_m:<3} BN={block_n:<3} "
            f"nw={num_warps} error={error:.4g}"
        )


if __name__ == "__main__":
    main()
