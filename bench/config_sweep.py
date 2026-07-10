"""Exhaustively check every forward autotune config for correctness on RDNA3.

Some (tile, num_warps) geometries miscompile in the ROCm Triton WMMA backend
for specific dtypes, producing large localized errors while running fast enough
to win autotuning. This sweep finds the configs that are correct across every
dtype / causal / head_dim / seqlen combination so the kernel can restrict its
search space to them.
"""

import math
import itertools
import argparse

import torch
import triton

import fa_rdna3.kernels as kernels


def reference(q, k, v, causal, scale):
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        row = torch.arange(q.shape[2], device=q.device)[:, None]
        col = torch.arange(k.shape[2], device=q.device)[None, :]
        logits = logits.masked_fill(row < col, float("-inf"))
    return torch.matmul(torch.softmax(logits, -1), v.float()), torch.logsumexp(logits, dim=-1)


def launch(q, k, v, causal, scale, block_m, block_n, num_warps):
    batch, heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
    common_args = (
        q, k, v, out, lse, scale,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
    )
    shape_args = (
        heads, seqlen_q, seqlen_k,
        triton.next_power_of_2(seqlen_q) * 4,
        triton.next_power_of_2(seqlen_k) * 4,
    )
    meta = dict(
        HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        IS_CAUSAL=causal, GROUP_SIZE=1, POST_SCALE_Q=q.dtype == torch.bfloat16,
        PARALLELISM_BUCKET=1,
        num_warps=num_warps, num_stages=1)
    kernels._attention_forward.fn[grid](
        *common_args, q, 0, 0, 0, 0, q, 0.0, 0, *shape_args, **meta)
    return out, lse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    dtypes = (torch.float16, torch.bfloat16)
    causals = (False, True)
    head_dims = (64, 128)
    seqlens = (512, 1000, 2048)
    tiles = [(bm, bn) for bm in (64, 128) for bn in (32, 64, 128)]
    warps = (2, 4, 8)

    tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2}
    bad = {}
    for (block_m, block_n) in tiles:
        for num_warps in warps:
            worst = 0.0
            worst_case = None
            for dtype, causal, head_dim, seqlen in itertools.product(
                    dtypes, causals, head_dims, seqlens):
                torch.manual_seed(seqlen + head_dim)
                shape = (2, 4, seqlen, head_dim)
                q = torch.randn(shape, device="cuda", dtype=dtype)
                k = torch.randn(shape, device="cuda", dtype=dtype)
                v = torch.randn(shape, device="cuda", dtype=dtype)
                scale = 1.0 / math.sqrt(head_dim)
                ref_out, ref_lse = reference(q, k, v, causal, scale)
                for _ in range(args.repeats):
                    out, lse = launch(q, k, v, causal, scale, block_m, block_n, num_warps)
                    output_error = (out.float() - ref_out).abs().max().item()
                    lse_error = (lse - ref_lse).abs().max().item()
                    normalized = max(output_error / tolerance[dtype], lse_error / 3e-2)
                    if normalized > worst:
                        worst = normalized
                        worst_case = (
                            str(dtype).split(".")[-1], causal, head_dim, seqlen,
                            output_error, lse_error)
            status = "OK " if worst <= 1.0 else "BAD"
            if worst > 1.0:
                bad[(block_m, block_n, num_warps)] = worst_case
            print(f"{status} BM={block_m:<3} BN={block_n:<3} nw={num_warps} "
                  f"worst={worst:.2f}x tol  @ {worst_case}")

    print("BAD configs:", list(bad.keys()) if bad else "none")


if __name__ == "__main__":
    main()
