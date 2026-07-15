"""Benchmark short FP32 inference attention against PyTorch SDPA."""

import argparse
import os
import statistics

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import torch.nn.functional as F

from fa_rdna3 import flash_attention


def _timed(functions, iterations):
    samples = {name: [] for name, _ in functions}
    for iteration in range(iterations):
        order = functions if iteration % 2 == 0 else reversed(functions)
        for name, function in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
    return {name: statistics.median(values) for name, values in samples.items()}


def _inputs(sequence):
    packed = torch.randn(2, sequence, 3, 8, 128, device="cuda")
    query, key, value = (
        tensor.transpose(1, 2) for tensor in packed.unbind(2)
    )
    bias = torch.zeros(2, 1, 1, sequence, device="cuda")
    bias[..., : sequence // 2] = float("-inf")
    return query, key, value, bias


def _stock_attention(query, key, value, bias):
    return F.scaled_dot_product_attention(query, key, value, attn_mask=bias)


def _custom_attention(query, key, value, bias):
    return flash_attention(query, key, value, bias=bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    print("sequence  PyTorch SDPA  fa-rdna3  speedup")
    with torch.inference_mode():
        stock_attention = torch.compile(
            _stock_attention, fullgraph=True, dynamic=False
        )
        custom_attention = torch.compile(
            _custom_attention, fullgraph=True, dynamic=False
        )
        for sequence in (6, 19):
            query, key, value, bias = _inputs(sequence)
            stock = lambda: stock_attention(
                query, key, value, bias
            )
            custom = lambda: custom_attention(query, key, value, bias)
            for _ in range(10):
                stock()
                custom()
            torch.cuda.synchronize()
            timings = _timed(
                (("stock", stock), ("custom", custom)), args.iterations
            )
            stock_us = timings["stock"]
            custom_us = timings["custom"]
            print(
                f"{sequence:>8}  {stock_us:>10.2f} us  {custom_us:>7.2f} us  "
                f"{stock_us / custom_us:>6.2f}x"
            )


if __name__ == "__main__":
    main()
