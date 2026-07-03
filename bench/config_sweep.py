"""Exhaustively check every autotune config for correctness on RDNA3.

Some (tile, num_warps) geometries miscompile in the ROCm Triton WMMA backend
for specific dtypes, producing large localized errors while running fast enough
to win autotuning. This sweep finds the configs that are correct across every
dtype / causal / head_dim / seqlen combination so the kernel can restrict its
search space to them.
"""

import math
import itertools

import torch
import triton

import fa_rdna3.kernels as kernels


def reference(q, k, v, causal, scale):
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        row = torch.arange(q.shape[2], device=q.device)[:, None]
        col = torch.arange(k.shape[2], device=q.device)[None, :]
        logits = logits.masked_fill(row < col, float("-inf"))
    return torch.matmul(torch.softmax(logits, -1), v.float())


def launch(q, k, v, causal, scale, block_m, block_n, num_warps):
    batch, heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
    kernels._attention_forward.fn[grid](
        q, k, v, out, lse, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        heads, seqlen_q, seqlen_k,
        triton.next_power_of_2(seqlen_q), triton.next_power_of_2(seqlen_k),
        HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n, IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=1,
    )
    return out


def main():
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
                out = launch(q, k, v, causal, scale, block_m, block_n, num_warps).float()
                ref = reference(q, k, v, causal, scale)
                err = (out - ref).abs().max().item()
                normalized = err / tolerance[dtype]
                if normalized > worst:
                    worst = normalized
                    worst_case = (str(dtype).split(".")[-1], causal, head_dim, seqlen, err)
            status = "OK " if worst <= 1.0 else "BAD"
            if worst > 1.0:
                bad[(block_m, block_n, num_warps)] = worst_case
            print(f"{status} BM={block_m:<3} BN={block_n:<3} nw={num_warps} "
                  f"worst={worst:.2f}x tol  @ {worst_case}")

    print("\nBAD configs:", list(bad.keys()) if bad else "none")


if __name__ == "__main__":
    main()
