# FlashAttention-2 for AMD RDNA3 (gfx1100)

A Triton implementation of FlashAttention-2 — forward, backward, split-K
decode, and variable-length (packed) sequences — tuned for the AMD Radeon
RX 7900 XT / XTX / GRE (RDNA3, `gfx1100`).

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
| 1 × 32 × 1024 × 128  | no  | 4.5× |
| 2 × 16 × 2048 × 128  | no  | 5.2× |
| 1 × 16 × 4096 × 64   | no  | 6.3× |
| 1 × 16 × 8192 × 64   | no  | 6.3× |
| 2 × 16 × 2048 × 128  | yes | 10.6× |
| 1 × 24 × 4096 × 128  | yes | 9.7× |
| 1 × 16 × 8192 × 64   | yes | 14.9× |

The gap widens with sequence length — the win is largest exactly where the
quadratic math path hurts most (long context, high-resolution diffusion).

**Absolute forward throughput** is 33–42 TFLOP/s (head_dim 128) and 26–37
TFLOP/s causal, against a measured ceiling of ~70 TFLOP/s for an optimal pure
WMMA GEMM on this card (`bench/gemm_ceiling.py`; the 123 TFLOP/s marketing peak
assumes dual-issue that does not occur in real kernels). Backward runs at 15–22
TFLOP/s.

Against AMD's own experimental AOTriton flash backend (forced on via the
experimental flag), same card:

| shape | causal | vs AOTriton |
|---|---|---|
| 1 × 16 × 4096 × 64  | no  | 0.86× (≈14% behind) |
| 2 × 16 × 2048 × 128 | yes | 1.75× faster |
| 1 × 24 × 4096 × 128 | yes | 1.63× faster |
| 1 × 16 × 8192 × 64  | yes | 1.79× faster |

Faster than AOTriton on causal (the loop split skips the masked upper triangle);
behind it on non-causal head_dim 64, which is the current weak spot. Outputs
match AOTriton to within fp16 tolerance (max abs diff ≤ 2e-3), an independent
cross-check on top of the fp32 reference tests.

### Decode (single query row, long KV cache)

Split-K decode against the plain forward on the same card, fp16, head_dim 128:

| batch × heads × kv | plain fwd | split-K decode | speedup | KV read |
|---|---|---|---|---|
| 1 × 8 × 4096   | 368 µs  | 105 µs | 3.5× | 159 GB/s |
| 1 × 8 × 16384  | 1441 µs | 108 µs | 13.4× | 623 GB/s |
| 1 × 32 × 4096  | 390 µs  | 110 µs | 3.6× | 613 GB/s |
| 4 × 32 × 8192  | 1155 µs | 620 µs | 1.9× | 865 GB/s |

Decode is memory-bound; split-K reaches up to 865 GB/s of KV read bandwidth
(≈90% of the card's ~960 GB/s peak). It helps most when few `batch × heads`
would otherwise leave most of the GPU idle while one workgroup walks the cache.

## Usage

Direct call — tensors are `(batch, heads, seqlen, head_dim)`, fp16 or bf16.
Differentiable, so it works inside training:

```python
import torch
from fa_rdna3 import flash_attention

q = torch.randn(2, 16, 4096, 128, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attention(q, k, v, causal=True)
out.sum().backward()  # dq, dk, dv
```

Decode (inference-only, non-causal, small query against a long cache):

```python
from fa_rdna3 import flash_attention_decode

out = flash_attention_decode(q_step, k_cache, v_cache)  # q_step is (b, h, 1, d)
```

Variable-length packed sequences — concatenate without padding and pass
cumulative lengths (`(total_tokens, heads, head_dim)` layout):

```python
from fa_rdna3 import flash_attention_varlen

# cu_seqlens = [0, len0, len0+len1, ...]; differentiable in q, k, v
out = flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k, causal=True)
```

Transparent drop-in for ComfyUI / diffusers / transformers — installs an
override of `torch.nn.functional.scaled_dot_product_attention` (float or
boolean `attn_mask` is routed as an additive bias) and defers to the original
for anything unsupported (fp32, dropout, exotic head dims):

```python
from fa_rdna3.sdpa import enable_rdna3_flash_attention

enable_rdna3_flash_attention()  # call once at startup
```

## Supported

- Head dims 16, 32, 64, 128, 256
- fp16 and bf16
- Causal (bottom-right aligned for `seqlen_q != seqlen_k`) and full attention
- Sliding-window / local attention (`window_size=(left, right)`, e.g. Mistral)
- Logit soft-capping (`softcap`, e.g. Gemma2)
- Additive attention bias / mask (`bias`, broadcastable; float or `-inf` mask)
- ALiBi positional bias (`alibi_slopes`, computed in-kernel — no materialised tensor)
- Distinct query / key sequence lengths (cross-attention)
- Grouped-query and multi-query attention (fewer K/V heads than query heads)
- Forward **and backward** (autograd `Function`; deterministic gradients)
- Split-K decode for the small-query / long-KV regime
- Variable-length packed sequences via `cu_seqlens` (no padding), differentiable

## Testing / benchmarking

```bash
pip install -e ".[test]"
python -m pytest tests/            # forward, backward, and decode vs fp32 reference
python bench/benchmark.py          # forward + backward, vs default SDPA
python bench/decode_benchmark.py   # split-K decode, vs forward and SDPA
python bench/gemm_ceiling.py       # sustained WMMA GEMM ceiling on this card
python bench/config_sweep.py       # correctness of every autotune config
```

`config_sweep.py` exists because two `(tile, num_warps)` geometries
(`64×64` at 4 warps, `128×128` at 8 warps) miscompile in the ROCm Triton WMMA
backend on gfx1100 — they run fast enough to win autotuning but return large
localized errors for some dtypes. The sweep found them (still reproducible on
Triton 3.5.1 / ROCm 6.4); they are excluded from the kernel's search space
(`_MISCOMPILED_ON_GFX1100` in `fa_rdna3/kernels/`) so autotuning can only ever
select a numerically correct config.

## How it works

Standard FlashAttention-2 tiling with an online softmax, so the score matrix is
never materialised. RDNA3-specific choices live in `fa_rdna3/kernels/`:

- Autotuned block sizes over a grid sized for RDNA3's 32-lane WMMA fragments and
  the 64 KB per-workgroup LDS limit.
- `num_stages=1` — RDNA3 has no `cp.async`/TMA equivalent, so deep software
  pipelining costs LDS without hiding latency the way it does on CUDA. Measured:
  `num_stages=2` is uniformly slower here.
- The exponentials use `exp2` with `log2(e)` folded into the softmax scale (and
  applied to `q` once before the loop), which maps to a native RDNA3
  instruction.
- V is loaded at the top of the key loop, before the QK dot, so its global-load
  latency overlaps the QK matmul and the softmax (+3–7% at head_dim 128).
- The inner key loop is split into an unmasked region (full tiles below the
  causal diagonal / before the key boundary) and a masked region (diagonal band
  and ragged tail), so the common path carries no `tl.where` and no
  boundary-guarded loads. This split is shared by the forward and both backward
  kernels.
- Backward is three kernels — a `delta = rowsum(dO ∘ O)` preprocess, a dK/dV
  kernel parallel over keys, and a dQ kernel parallel over queries — which keeps
  dQ accumulation race-free without atomics (deterministic gradients). P is
  recomputed from Q, K and the stored LSE; the `log2(e)` factor on the LSE ties
  the base-2 forward softmax to the natural-log LSE.
- Decode splits the KV cache across workgroups (each computes a partial output
  and LSE) and merges the partials with an LSE reduction, so a one-row query
  saturates the GPU instead of leaving one workgroup to walk the whole cache.

## Acknowledgements

Built on ideas and conventions from prior FlashAttention work:

- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the FlashAttention-2 algorithm and the `window_size=(left, right)` sliding-window convention.
- [ROCm/aotriton](https://github.com/ROCm/aotriton) — AMD's ahead-of-time Triton attention kernels; source of the RDNA `PRE_LOAD_V` idea and a benchmarking reference.
- [OpenAI Triton FlashAttention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) — the tiling and online-softmax kernel structure.

## License

MIT
