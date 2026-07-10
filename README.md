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
assumes dual-issue that does not occur in real kernels). Backward runs at 15–44
TFLOP/s, with the upper end reached by the D64 specialization.

Against AMD's own experimental AOTriton flash backend (forced on via the
experimental flag), same card:

| shape | causal | forward vs AOTriton |
|---|---|---|
| 1 × 16 × 4096 × 64  | no  | 1.03× faster |
| 2 × 16 × 2048 × 128 | yes | 1.75× faster |
| 1 × 24 × 4096 × 128 | yes | 1.63× faster |
| 1 × 16 × 8192 × 64  | yes | 1.79× faster |

Faster than AOTriton on the tested causal shapes and on non-causal head_dim 64.
The D64 occupancy specializations measure 1.03–1.04× faster at 16 heads for
sequence lengths 2048–8192; the single-head path ranges from 1.10× to 1.71×
faster over the same lengths. Outputs match AOTriton to within fp16 tolerance
(max abs diff ≤ 2.44e-4), an independent cross-check on top of the fp32
reference tests.

The same comparison for the complete backward pass (dQ + dK + dV):

| shape | causal | this kernel | AOTriton | speedup |
|---|---|---:|---:|---:|
| 1 × 16 × 4096 × 64 | no  | 3.922 ms | 5.291 ms | 1.35× |
| 1 × 16 × 4096 × 64 | yes | 2.270 ms | 3.705 ms | 1.63× |
| 1 × 16 × 2048 × 128 | no  | 4.568 ms | 4.945 ms | 1.08× |
| 1 × 16 × 2048 × 128 | yes | 2.524 ms | 3.121 ms | 1.24× |

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

Paged-KV decode now splits the logical block table with the same LSE-correct
reduction instead of assigning one serial cache walk to each query head:

| batch × q heads / KV heads × context | serial paged | automatic split-K | speedup |
|---|---:|---:|---:|
| 1 × 8 / 8 × 256   | 35.2 µs  | 20.0 µs  | 1.76× |
| 1 × 8 / 8 × 1024  | 155.3 µs | 28.0 µs  | 5.54× |
| 1 × 8 / 8 × 4096  | 422.7 µs | 58.8 µs  | 7.19× |
| 1 × 32 / 8 × 4096 | 419.8 µs | 102.0 µs | 4.12× |
| 4 × 32 / 8 × 4096 | 526.6 µs | 327.6 µs | 1.61× |
| 4 × 32 / 8 × 16384 | 2047.3 µs | 1208.1 µs | 1.69× |

The automatic policy was checked against a forced-split oracle across context
lengths 64–16384 and stays within 6% of its best measured split on the tested
matrix. Short contexts at or below 64 tokens remain unsplit when the supplied
`max_context_len` (or the block-table capacity used in its absence) reflects
that bound.

## Installation

Install a matching ROCm build of PyTorch 2.8 or newer first. Its
`pytorch-triton-rocm` package provides the matching Triton compiler. `fa-rdna3`
deliberately declares neither package as an automatic dependency, preventing
pip or ComfyUI Manager from replacing a working ROCm stack with generic PyPI
wheels.

The currently verified stack is Python 3.14, PyTorch 2.9.1 + ROCm 6.4, and
`pytorch-triton-rocm` 3.5.1 on `gfx1100`. Verify the environment, then install:

```bash
python -c 'import torch, triton; print(torch.__version__, torch.version.hip, triton.__version__)'
pip install -e .
```

### ComfyUI custom node

The repository is directly installable as a ComfyUI custom node; it does not
need a wrapper repository or a second Triton installation:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/chelokot/flash-attention-rdna3.git RDNA3-Flash-Attention
```

Restart ComfyUI, then insert **RDNA3 Flash Attention** between the checkpoint /
diffusion-model loader and the sampler. It appears under
`model_patches/attention` and returns a patched `MODEL`.

The node uses ComfyUI's per-model attention override, so no
`--use-pytorch-cross-attention` launch flag is required. It also installs the
safe process-wide SDPA dispatcher for model components that call PyTorch
directly. Unsupported calls keep using ComfyUI/PyTorch, existing attention
overrides remain outermost and can delegate to the RDNA3 backend, and
non-`gfx1100` systems are rejected with a clear error. ComfyUI 0.4.0 or newer
is required. The first execution of a new shape compiles and caches its Triton
specialization, so it is slower than subsequent generations.

Semantic overrides are preserved. A backend-selection override that deliberately
does not delegate (for example, a SageAttention selector) keeps precedence;
remove that conflicting backend node when you want RDNA3 to handle the call.

Backend-level timings through the actual ComfyUI tensor-layout adapter on the
verified RX 7900 XTX stack (fp16, `B=2`, `H=8`, `Q=4096`, `D=64`):

| attention | RDNA3 node path | stock PyTorch SDPA | speedup |
|---|---:|---:|---:|
| cross, `K=77` | 0.185 ms | 0.880 ms | 4.75× |
| self, `K=4096` | 1.838 ms | 14.378 ms | 7.82× |

These are attention-call timings, not an end-to-end generation claim; total
speedup depends on the model, resolution, sampler, offloading, and time spent
outside attention.

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

Paged-KV decode uses a vLLM-style physical block pool. `max_context_len` is an
optional host-known upper bound used to choose the split count without reading
the GPU `context_lens` tensor back to the CPU. `num_splits` is available as a
manual 1–32 override for benchmarking or unusual workloads:

```python
from fa_rdna3 import flash_attention_decode_paged

out = flash_attention_decode_paged(
    q_step, k_blocks, v_blocks, block_table, context_lens,
    max_context_len=4096,
)
```

Variable-length packed sequences — concatenate without padding and pass
cumulative lengths (`(total_tokens, heads, head_dim)` layout):

```python
from fa_rdna3 import flash_attention_varlen

# cu_seqlens = [0, len0, len0+len1, ...]; differentiable in q, k, v
out = flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k, causal=True)
```

The cumulative arrays must start at zero, be monotonically nondecreasing, and
end at the packed token totals; each `max_seqlen` must cover its longest
segment. Kernels clamp intervals to the packed storage bounds for memory safety,
but invalid metadata does not produce meaningful attention results.

Transparent Python drop-in for diffusers / transformers — installs an
override of `torch.nn.functional.scaled_dot_product_attention` (float or
boolean `attn_mask` is routed as an additive bias) and defers to the original
for anything unsupported (dropout, head dims above 512):

```python
from fa_rdna3 import enable_rdna3_flash_attention

enable_rdna3_flash_attention()  # call once at startup
```

## Supported

- Dense head dims up to 512 (incl. non-powers-of-two, zero-padded internally);
  packed and decode APIs use exact power-of-two dims from 16 through 512
- Dense fp16, bf16, and fp32; packed/decode fp16 and bf16
- Causal (bottom-right aligned for `seqlen_q != seqlen_k`) and full attention
- Sliding-window / local attention (`window_size=(left, right)`, e.g. Mistral)
- Logit soft-capping (`softcap`, e.g. Gemma2)
- Additive attention bias / mask (`bias`, broadcastable; float or `-inf` mask)
- ALiBi positional bias (`alibi_slopes`, computed in-kernel — no materialised tensor)
- Attention dropout (`dropout_p`, philox mask shared between forward and backward)
- Registered as a `torch.library` custom op — `torch.compile`-compatible and `opcheck`-clean
- Distinct query / key sequence lengths (cross-attention)
- Grouped-query and multi-query attention (fewer K/V heads than query heads)
- Forward **and backward** (registered autograd; deterministic gradients)
- Split-K decode for the small-query / long-KV regime
- Paged-KV decode over a block-table cache (`flash_attention_decode_paged`, vLLM-style)
- Variable-length packed sequences via `cu_seqlens` (no padding), differentiable

## Testing / benchmarking

```bash
pip install -e ".[test]"
python -m pytest tests/            # forward, backward, and decode vs fp32 reference
python bench/benchmark.py          # forward + backward, vs default SDPA
python bench/benchmark.py --comfyui  # ComfyUI-layout self/cross attention
python bench/decode_benchmark.py   # split-K decode, vs forward and SDPA
python bench/paged_decode_benchmark.py  # paged split-K vs a serial cache walk
python bench/gemm_ceiling.py       # sustained WMMA GEMM ceiling on this card
python bench/config_sweep.py       # correctness of every forward tile config
python bench/noncausal_d64_sweep.py  # D64 forward geometry timings + correctness
python -m bench.backward_sweep dkdv  # backward geometry timings + stability
python -m bench.backward_sweep dq --head-dim 128
```

`config_sweep.py` exists because three `(tile, num_warps)` geometries
(`64×64` at 4 warps, `128×64` at 8 warps, `128×128` at 8 warps) miscompile in
the ROCm Triton WMMA backend on gfx1100 — they run fast enough to win autotuning
but return large localized errors for some dtypes. A fourth geometry (`64×32`
at 4 warps) is unstable specifically when bf16 Q scaling happens after the dot.
The sweep found them (still reproducible on Triton 3.5.1 / ROCm 6.4); the first
set is excluded globally and the conditional case only from bf16 forward, so
autotuning can only select a numerically correct config. The production
shortlist keeps every geometry observed as a winner or top-three runner-up; the
benchmark retains the exhaustive grid for validating future compiler releases.
Four fp32 backward geometries also do not return on gfx1100. fp32 backward uses
one verified conservative geometry instead of cold-autotuning through fragile
codegen; the primary fp16/bf16 paths retain their measured search spaces.

## How it works

Standard FlashAttention-2 tiling with an online softmax, so the score matrix is
never materialised. RDNA3-specific choices live in `fa_rdna3/kernels/`:

- Autotuned block sizes over a grid sized for RDNA3's 32-lane WMMA fragments and
  the 64 KB per-workgroup LDS limit.
- The general production grids retain 22 measured candidates (forward 7,
  dK/dV 8, dQ 7), down from 45. The common D64 forward and D64/D128 backward
  paths bypass cold autotuning and compile one measured geometry per kernel.
- Plain non-causal D64 uses measured occupancy specializations: `32×64 / 2
  warps` for one batch-head at every length, and `64×16 / 4 warps` from two
  batch-heads up at sequence lengths of 512 or more.
- Plain dense backward uses `16×32 / 2 warps` for D64 dK/dV and `64×16 / 4
  warps` for D64 dQ; D128 uses `16×64 / 4 warps` and `32×16 / 2 warps`.
  These stable winners cover both causal modes and fp16/bf16; fp32, GQA,
  cross-attention, varlen, and score modifiers retain their existing
  specialized or autotuned paths.
- Autotune selections are keyed by dtype, shape, GQA grouping, and enabled score
  modifiers, then cached on disk so a new Python process does not benchmark the
  same specialization again.
- `num_stages=1` — RDNA3 has no `cp.async`/TMA equivalent, so deep software
  pipelining costs LDS without hiding latency the way it does on CUDA. Measured:
  `num_stages=2` is uniformly slower here.
- The exponentials use `exp2` with `log2(e)` folded into the softmax scale,
  which maps to a native RDNA3 instruction. fp16/fp32 apply it to `q` once;
  bf16 applies it to the fp32 score so backward recomputation follows the same
  arithmetic order and avoids a reproducible long-sequence gradient error.
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
- dQ and dK/dV use separate measured autotune spaces. This keeps every observed
  winning geometry while cutting cold backward candidates from 30 to 15 per
  specialization.
- Decode splits the KV cache across workgroups (each computes a partial output
  and LSE) and merges the partials with an LSE reduction, so a one-row query
  saturates the GPU instead of leaving one workgroup to walk the whole cache.
- Paged decode applies the same LSE-correct split/merge directly over a physical
  block table, clamps malformed lengths and block ids safely, and selects its
  split count from both GPU occupancy and serial work per program.
- Low-precision additive bias is consumed directly by the kernel; it is never
  expanded or copied into a materialised fp32 attention matrix.

## Acknowledgements

Built on ideas and conventions from prior FlashAttention work:

- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the FlashAttention-2 algorithm and the `window_size=(left, right)` sliding-window convention.
- [ROCm/aotriton](https://github.com/ROCm/aotriton) — AMD's ahead-of-time Triton attention kernels; source of the RDNA `PRE_LOAD_V` idea and a benchmarking reference.
- [OpenAI Triton FlashAttention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) — the tiling and online-softmax kernel structure.

## License

MIT
