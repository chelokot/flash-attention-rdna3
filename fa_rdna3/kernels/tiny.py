"""Short-sequence FP32 attention kernel for inference workloads."""

import triton
import triton.language as tl

from ._common import LOG2E


@triton.jit
def _attention_tiny_forward(
    q_ptr, k_ptr, v_ptr, out_ptr,
    bias_ptr, softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_bb, stride_bh, stride_bm, stride_bn,
    SEQLEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    offsets_d = tl.arange(0, HEAD_DIM)
    offsets_n = tl.arange(0, BLOCK_N)

    query_offsets = (
        batch_idx * stride_qb
        + head_idx * stride_qh
        + query_idx * stride_qm
        + offsets_d * stride_qd
    )
    query = tl.load(q_ptr + query_offsets) * (softmax_scale * LOG2E)
    scores = tl.full([BLOCK_N], float("-inf"), tl.float32)

    for key_idx in tl.static_range(0, SEQLEN):
        key_offsets = (
            batch_idx * stride_kb
            + head_idx * stride_kh
            + key_idx * stride_kn
            + offsets_d * stride_kd
        )
        key = tl.load(k_ptr + key_offsets)
        score = tl.sum(query * key, axis=0)
        if HAS_BIAS:
            bias_offset = (
                batch_idx * stride_bb
                + head_idx * stride_bh
                + query_idx * stride_bm
                + key_idx * stride_bn
            )
            score += tl.load(bias_ptr + bias_offset) * LOG2E
        scores = tl.where(offsets_n == key_idx, score, scores)

    maximum = tl.max(scores, axis=0)
    maximum_safe = tl.where(maximum == float("-inf"), 0.0, maximum)
    probabilities = tl.where(
        (offsets_n < SEQLEN) & (maximum != float("-inf")),
        tl.exp2(scores - maximum_safe),
        0.0,
    )
    denominator = tl.sum(probabilities, axis=0)
    denominator_safe = tl.where(denominator == 0.0, 1.0, denominator)
    accumulator = tl.zeros([HEAD_DIM], tl.float32)

    for key_idx in tl.static_range(0, SEQLEN):
        value_offsets = (
            batch_idx * stride_vb
            + head_idx * stride_vh
            + key_idx * stride_vn
            + offsets_d * stride_vd
        )
        value = tl.load(v_ptr + value_offsets)
        probability = tl.sum(tl.where(offsets_n == key_idx, probabilities, 0.0), axis=0)
        accumulator += probability * value

    output_offsets = (
        batch_idx * stride_ob
        + head_idx * stride_oh
        + query_idx * stride_om
        + offsets_d * stride_od
    )
    tl.store(out_ptr + output_offsets, accumulator / denominator_safe)
