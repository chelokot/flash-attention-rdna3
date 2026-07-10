"""Sweep dK/dV or dQ tile geometries for head-dim-64 attention."""

import argparse
import math

import torch
import triton

import fa_rdna3.interface as interface


class CapturedKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        self.grid = None
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def capture(*args, **kwargs):
            self.grid = grid
            self.args = args
            self.kwargs = kwargs
            return launch(*args, **kwargs)

        return capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", choices=("dkdv", "dq"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
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
    dout = torch.randn_like(query)
    scale = 1.0 / math.sqrt(64)
    out, lse = interface._forward(
        query, key, value, args.causal, scale, (-1, -1), 0.0, None, None, 0.0, 0)

    attribute = f"_attention_bwd_{args.kernel}"
    original = getattr(interface, attribute)
    captured = CapturedKernel(original)
    setattr(interface, attribute, captured)
    try:
        dquery, dkey, dvalue = interface._backward(
            dout, query, key, value, out, lse, args.causal, scale,
            (-1, -1), 0.0, None, None, 0.0, 0)
    finally:
        setattr(interface, attribute, original)
    torch.cuda.synchronize()

    baseline = (dkey.clone(), dvalue.clone()) if args.kernel == "dkdv" else (dquery.clone(),)
    outputs = (dkey, dvalue) if args.kernel == "dkdv" else (dquery,)
    default_block_ms = (16, 32, 64) if args.kernel == "dkdv" else (32, 64)
    block_ms = (args.block_m,) if args.block_m is not None else default_block_ms
    block_ns = (args.block_n,) if args.block_n is not None else (16, 32, 64)
    warp_counts = (args.num_warps,) if args.num_warps is not None else (1, 2, 4)
    measurements = []
    for block_m in block_ms:
        for block_n in block_ns:
            for num_warps in warp_counts:
                grid = captured.grid({"BLOCK_M": block_m, "BLOCK_N": block_n})

                def run():
                    original.fn[grid](
                        *captured.args,
                        **captured.kwargs,
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        num_warps=num_warps,
                        num_stages=1,
                    )

                try:
                    run()
                    torch.cuda.synchronize()
                    first = tuple(output.clone() for output in outputs)
                    run()
                    torch.cuda.synchronize()
                    error = max(
                        float((output.float() - expected.float()).abs().max())
                        for output, expected in zip(outputs, baseline)
                    )
                    drift = max(
                        float((output.float() - previous.float()).abs().max())
                        for output, previous in zip(outputs, first)
                    )
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

    print("\nFastest stable configurations:")
    tolerance = 3e-3 if dtype == torch.float16 else 2e-2
    stable = [
        measurement for measurement in measurements
        if measurement[1] <= tolerance and measurement[2] == 0.0
    ]
    for elapsed_ms, error, _, block_m, block_n, num_warps in sorted(stable)[:10]:
        print(
            f"{elapsed_ms:.4f} ms  BM={block_m:<3} BN={block_n:<3} "
            f"nw={num_warps} error={error:.4g}"
        )


if __name__ == "__main__":
    main()
