"""Autotune policy shared by the production kernel entry points."""

import torch

from fa_rdna3.kernels import (
    _attention_bwd_dkdv,
    _attention_bwd_dkdv_varlen,
    _attention_bwd_dq,
    _attention_bwd_dq_varlen,
    _attention_forward,
    _attention_forward_varlen,
    _attention_split,
)
from fa_rdna3.kernels._common import (
    _D64_BACKWARD_DKDV_GEOMETRY,
    _D64_BACKWARD_DQ_GEOMETRY,
    _D64_NONCAUSAL_LOW_PARALLELISM_GEOMETRY,
    _D64_NONCAUSAL_NARROW_GEOMETRY,
    _FP32_BACKWARD_GEOMETRY,
    _MISCOMPILED_BF16_POST_SCALE,
    _MISCOMPILED_ON_GFX1100,
    _NONRETURNING_FP32_DKDV,
    _NONRETURNING_FP32_DQ,
    _prune_bwd_dkdv_configs,
    _prune_bwd_dq_configs,
    _prune_configs_by_head_dim,
)


AUTOTUNED_KERNELS = (
    _attention_forward,
    _attention_bwd_dkdv,
    _attention_bwd_dq,
    _attention_forward_varlen,
    _attention_bwd_dkdv_varlen,
    _attention_bwd_dq_varlen,
    _attention_split,
)


def test_autotune_results_are_cached_across_processes():
    assert all(kernel.cache_results for kernel in AUTOTUNED_KERNELS)


def test_batched_autotune_keys_cover_codegen_features():
    required = {
        "GROUP_SIZE",
        "WINDOW_LEFT",
        "WINDOW_RIGHT",
        "HAS_SOFTCAP",
        "HAS_BIAS",
        "HAS_ALIBI",
        "DROPOUT",
        "SAFE_SOFTMAX",
    }
    for kernel in (_attention_forward, _attention_bwd_dkdv, _attention_bwd_dq):
        assert required <= set(kernel.keys)
    assert "POST_SCALE_Q" in _attention_forward.keys
    assert "PARALLELISM_BUCKET" in _attention_forward.keys
    assert "POST_SCALE_Q" in _attention_forward_varlen.keys


def test_production_config_spaces_stay_bounded_and_blacklist_free():
    expected_counts = {
        _attention_forward: 9,
        _attention_bwd_dkdv: 9,
        _attention_bwd_dq: 8,
        _attention_forward_varlen: 7,
        _attention_bwd_dkdv_varlen: 8,
        _attention_bwd_dq_varlen: 7,
        _attention_split: 6,
    }
    for kernel, expected_count in expected_counts.items():
        assert len(kernel.configs) == expected_count
        for config in kernel.configs:
            if "BLOCK_M" in config.kwargs:
                geometry = (
                    config.kwargs["BLOCK_M"],
                    config.kwargs["BLOCK_N"],
                    config.num_warps,
                )
                assert geometry not in _MISCOMPILED_ON_GFX1100


def test_bfloat16_post_scale_prunes_unstable_geometry():
    kept = _prune_configs_by_head_dim(
        _attention_forward.configs,
        {"HEAD_DIM": 64, "POST_SCALE_Q": True},
    )
    geometries = {
        (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
        for config in kept
    }
    assert geometries.isdisjoint(_MISCOMPILED_BF16_POST_SCALE)


def test_float32_backward_prunes_nonreturning_geometries():
    query = torch.empty(1, dtype=torch.float32)
    cases = (
        (_attention_bwd_dkdv, _prune_bwd_dkdv_configs, _NONRETURNING_FP32_DKDV),
        (_attention_bwd_dq, _prune_bwd_dq_configs, _NONRETURNING_FP32_DQ),
    )
    for kernel, prune, blocked in cases:
        kept = prune(kernel.configs, {"HEAD_DIM": 64, "q_ptr": query})
        geometries = {
            (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
            for config in kept
        }
        assert geometries == {_FP32_BACKWARD_GEOMETRY}
        assert geometries.isdisjoint(blocked)


def test_plain_dense_d64_backward_uses_measured_specializations():
    common = {
        "HEAD_DIM": 64,
        "GROUP_SIZE": 1,
        "seqlen_q": 4096,
        "seqlen_k": 4096,
    }
    for dtype in (torch.float16, torch.bfloat16):
        for kernel, prune, expected in (
            (_attention_bwd_dkdv, _prune_bwd_dkdv_configs, _D64_BACKWARD_DKDV_GEOMETRY),
            (_attention_bwd_dq, _prune_bwd_dq_configs, _D64_BACKWARD_DQ_GEOMETRY),
        ):
            kept = prune(kernel.configs, {**common, "q_ptr": torch.empty(1, dtype=dtype)})
            geometries = {
                (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
                for config in kept
            }
            assert geometries == {expected}


def test_d64_backward_specializations_stay_dense_and_plain():
    common = {
        "HEAD_DIM": 64,
        "GROUP_SIZE": 1,
        "seqlen_q": 4096,
        "seqlen_k": 4096,
        "q_ptr": torch.empty(1, dtype=torch.float16),
    }
    exclusions = (
        {"seqlen_q": None, "seqlen_k": None},
        {"seqlen_q": 64, "seqlen_k": 64},
        {"seqlen_k": 2048},
        {"GROUP_SIZE": 2},
        {"HAS_BIAS": True},
        {"q_ptr": torch.empty(1, dtype=torch.float32)},
    )
    for kernel, prune, specialization in (
        (_attention_bwd_dkdv, _prune_bwd_dkdv_configs, _D64_BACKWARD_DKDV_GEOMETRY),
        (_attention_bwd_dq, _prune_bwd_dq_configs, _D64_BACKWARD_DQ_GEOMETRY),
    ):
        for exclusion in exclusions:
            kept = prune(kernel.configs, {**common, **exclusion})
            geometries = {
                (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
                for config in kept
            }
            assert specialization not in geometries


def test_plain_d64_noncausal_uses_measured_parallelism_specialization():
    common = {
        "HEAD_DIM": 64,
        "IS_CAUSAL": False,
        "seqlen_q": 4096,
        "seqlen_k": 4096,
        "q_ptr": torch.empty(1, dtype=torch.float16),
    }
    for parallelism, expected in (
        (0, _D64_NONCAUSAL_LOW_PARALLELISM_GEOMETRY),
        (1, _D64_NONCAUSAL_NARROW_GEOMETRY),
    ):
        kept = _prune_configs_by_head_dim(
            _attention_forward.configs,
            {**common, "PARALLELISM_BUCKET": parallelism},
        )
        geometries = {
            (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
            for config in kept
        }
        assert geometries == {expected}

    short = _prune_configs_by_head_dim(
        _attention_forward.configs,
        {**common, "seqlen_q": 256, "seqlen_k": 256, "PARALLELISM_BUCKET": 1},
    )
    short_geometries = {
        (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
        for config in short
    }
    assert _D64_NONCAUSAL_NARROW_GEOMETRY not in short_geometries
    assert _D64_NONCAUSAL_LOW_PARALLELISM_GEOMETRY not in short_geometries

    float32 = _prune_configs_by_head_dim(
        _attention_forward.configs,
        {**common, "q_ptr": torch.empty(1, dtype=torch.float32), "PARALLELISM_BUCKET": 1},
    )
    float32_geometries = {
        (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
        for config in float32
    }
    assert _D64_NONCAUSAL_NARROW_GEOMETRY not in float32_geometries
    assert _D64_NONCAUSAL_LOW_PARALLELISM_GEOMETRY not in float32_geometries
