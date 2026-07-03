"""Measure the sustained WMMA GEMM throughput on gfx1100.

FlashAttention can never exceed the throughput of the two matmuls it is built
from, so this establishes the real achievable ceiling on this card (not the
123 TFLOP/s marketing peak, which assumes dual-issue that does not occur in
practice). Forward/backward efficiency should be judged against this number.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(a_ptr, b_ptr, c_ptr, M, N, K,
          sam, sak, sbk, sbn, scm, scn,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def bench_gemm(M, N, K, dtype, block_m=128, block_n=128, block_k=64, num_warps=8):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    c = torch.empty((M, N), device="cuda", dtype=dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    def run():
        _gemm[grid](a, b, c, M, N, K,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                    num_warps=num_warps, num_stages=1)

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(run)
    flops = 2 * M * N * K
    return flops / (ms * 1e-3) / 1e12


def main():
    print(f"{'dtype':>8} {'M':>6} {'N':>6} {'K':>6} {'cfg':>16} {'TFLOP/s':>9}")
    for dtype in (torch.float16, torch.bfloat16):
        for M, N, K in ((4096, 4096, 4096), (8192, 8192, 8192)):
            best = 0.0
            best_cfg = None
            for bm, bn, bk, nw in ((128, 128, 32, 8), (128, 128, 64, 8),
                                   (128, 256, 64, 8), (256, 128, 64, 8),
                                   (128, 128, 64, 4), (64, 64, 32, 4)):
                try:
                    t = bench_gemm(M, N, K, dtype, bm, bn, bk, nw)
                except Exception:
                    continue
                if t > best:
                    best, best_cfg = t, (bm, bn, bk, nw)
            print(f"{str(dtype).split('.')[-1]:>8} {M:>6} {N:>6} {K:>6} "
                  f"{str(best_cfg):>16} {best:>9.1f}")


if __name__ == "__main__":
    main()
