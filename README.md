# FlashAttention-2 for AMD RDNA3 (gfx1100)

A single-file Triton implementation of the FlashAttention-2 forward pass tuned
for the AMD Radeon RX 7900 XT / XTX / GRE (RDNA3, `gfx1100`).

RDNA3 has been [the platform without a production-quality fused attention
kernel](https://llm-tracker.info/howto/AMD-GPUs) for years: the official
`flash-attention` Composable-Kernel backend ships CDNA-only kernels, and the
AOTriton path wired into PyTorch is still gated behind
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. In practice a stock ComfyUI /
diffusers / transformers install on a 7900 XTX runs the **non-fused math**
attention path and materialises the full `seqlen × seqlen` score matrix.

This kernel replaces that path.

## Results (Radeon RX 7900 XTX, ROCm 6.4, Triton 3.5.1, fp16)

Speedup over the default `scaled_dot_product_attention` path (non-fused math,
what you actually get out of the box on RDNA3):

| shape (batch × heads × seqlen × dim) | causal | speedup |
|---|---|---|
| 1 × 32 × 1024 × 128  | no  | 4.2× |
| 2 × 16 × 2048 × 128  | no  | 5.4× |
| 1 × 16 × 4096 × 64   | no  | 6.8× |
| 1 × 16 × 8192 × 64   | no  | 6.7× |
| 2 × 16 × 2048 × 128  | yes | 10.9× |
| 1 × 24 × 4096 × 128  | yes | 10.5× |
| 1 × 16 × 8192 × 64   | yes | 16.1× |

The gap widens with sequence length — the win is largest exactly where the
quadratic math path hurts most (long context, high-resolution diffusion).

Against AMD's own experimental AOTriton flash backend (forced on via the
experimental flag), same card:

| shape | causal | vs AOTriton |
|---|---|---|
| 1 × 16 × 4096 × 64  | no  | 0.93× (≈7% behind) |
| 1 × 24 × 4096 × 128 | yes | 1.44× faster |
| 1 × 16 × 8192 × 64  | yes | 1.71× faster |

Outputs match AOTriton to within fp16 tolerance (max abs diff ≤ 5e-4), an
independent cross-check on top of the fp32 reference tests.

## Usage

Direct call — tensors are `(batch, heads, seqlen, head_dim)`, fp16 or bf16:

```python
import torch
from fa_rdna3 import flash_attention

q = torch.randn(2, 16, 4096, 128, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attention(q, k, v, causal=True)
```

Transparent drop-in for ComfyUI / diffusers / transformers — installs an
override of `torch.nn.functional.scaled_dot_product_attention` and defers to
the original for anything unsupported (attention masks, GQA, fp32, exotic head
dims):

```python
from fa_rdna3.sdpa import enable_rdna3_flash_attention

enable_rdna3_flash_attention()  # call once at startup
```

## Supported

- Head dims 16, 32, 64, 128, 256
- fp16 and bf16
- Causal and full (bidirectional) attention
- Distinct query / key sequence lengths (cross-attention)
- Forward pass only (inference); backward is not implemented yet

## Testing / benchmarking

```bash
pip install -e ".[test]"
python -m pytest tests/          # correctness vs fp32 reference
python bench/benchmark.py        # vs default SDPA
python bench/config_sweep.py     # correctness of every autotune config
```

`config_sweep.py` exists because two `(tile, num_warps)` geometries
(`64×64` at 4 warps, `128×128` at 8 warps) miscompile in the ROCm Triton WMMA
backend on gfx1100 — they run fast enough to win autotuning but return large
localized errors for some dtypes. The sweep found them; they are excluded from
the kernel's search space (`_MISCOMPILED_ON_GFX1100` in `fa_rdna3/kernels.py`)
so autotuning can only ever select a numerically correct config.

## How it works

Standard FlashAttention-2 tiling with an online softmax, so the score matrix is
never materialised. RDNA3-specific choices live in `fa_rdna3/kernels.py`:

- Autotuned block sizes over a grid sized for RDNA3's 32-lane WMMA fragments and
  128 KB LDS budget.
- `num_stages=1` — RDNA3 has no `cp.async`/TMA equivalent, so deep software
  pipelining costs LDS without hiding latency the way it does on CUDA.
- The exponentials use `exp2` with `log2(e)` folded into the softmax scale,
  which maps to a native RDNA3 instruction.
- Causal blocks past the diagonal are skipped entirely rather than masked,
  which is where the large causal speedups come from.

## License

MIT
