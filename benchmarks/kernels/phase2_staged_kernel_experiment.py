# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

try:
    from .phase2_hotshape_utils import Phase2HotShapeSpec, benchmark_cuda
    from .phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        StageBufferPlan,
        make_reference_attention_inputs,
        make_packed_reference_inputs,
        stage_buffer_plan_to_dict,
        validate_stage_buffer_plan,
    )
except ImportError:
    from phase2_hotshape_utils import Phase2HotShapeSpec, benchmark_cuda
    from phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        StageBufferPlan,
        make_reference_attention_inputs,
        make_packed_reference_inputs,
        stage_buffer_plan_to_dict,
        validate_stage_buffer_plan,
    )


@dataclass(frozen=True)
class Stage1Artifacts:
    plan: StageBufferPlan
    mid_o: torch.Tensor  # [T, H, S, D+1], float32
    full_out_ref: torch.Tensor  # [T, H, D], float32
    full_lse_ref: torch.Tensor  # [H, T], float32
    impl: str
    replay: Callable[[], None] | None = None


@dataclass(frozen=True)
class Stage2Artifacts:
    output: torch.Tensor  # [T, H, D]
    lse: torch.Tensor  # [H, T]
    impl: str
    replay: Callable[[], None] | None = None


@dataclass(frozen=True)
class PipelineArtifacts:
    stage1: Stage1Artifacts
    stage2: Stage2Artifacts
    out_max_abs: float
    lse_max_abs: float


@dataclass(frozen=True)
class PairCheckResult:
    stage1_impl: str
    stage2_impl: str
    out_max_abs: float
    lse_max_abs: float
    ok: bool


@dataclass(frozen=True)
class RegistrySelfCheckResult:
    stage1_impls: tuple[str, ...]
    stage2_impls: tuple[str, ...]
    pair_results: tuple[PairCheckResult, ...]
    passed_pairs: int
    total_pairs: int


@dataclass(frozen=True)
class Stage1CompareResult:
    impl_a: str
    impl_b: str
    ms_a: float
    ms_b: float
    ms_ratio_b_over_a: float
    mid_o_max_abs: float
    full_out_ref_max_abs: float
    full_lse_ref_max_abs: float


@dataclass(frozen=True)
class Stage2CompareResult:
    stage1_impl: str
    impl_a: str
    impl_b: str
    ms_a: float
    ms_b: float
    ms_ratio_b_over_a: float
    output_max_abs: float
    lse_max_abs: float


@dataclass(frozen=True)
class PipelineCompareResult:
    stage1_impl_a: str
    stage2_impl_a: str
    stage1_impl_b: str
    stage2_impl_b: str
    stage1_ms_a: float
    stage1_ms_b: float
    stage1_ms_ratio_b_over_a: float
    stage2_ms_a: float
    stage2_ms_b: float
    stage2_ms_ratio_b_over_a: float
    ms_a: float
    ms_b: float
    ms_ratio_b_over_a: float
    output_max_abs: float
    lse_max_abs: float


Stage1ImplFn = Callable[
    [
        Phase2HotShapeSpec,
        torch.dtype,
        str,
        int,
        int,
        ReferenceAttentionInputs | PackedReferenceInputs | None,
    ],
    Stage1Artifacts,
]
Stage2ImplFn = Callable[[Stage1Artifacts], Stage2Artifacts]
Stage1MetadataGetter = Callable[[], dict[str, object] | None]

_STAGE1_IMPLS: dict[str, Stage1ImplFn] = {}
_STAGE2_IMPLS: dict[str, Stage2ImplFn] = {}
_STAGE1_METADATA_GETTERS: dict[str, Stage1MetadataGetter] = {}
_DISCOVERED_IMPL_MODULES: tuple[str, ...] = ()


def register_stage1_impl(name: str, fn: Stage1ImplFn) -> None:
    if name in _STAGE1_IMPLS:
        raise ValueError(f"stage1 impl '{name}' already registered")
    _STAGE1_IMPLS[name] = fn


def register_stage2_impl(name: str, fn: Stage2ImplFn) -> None:
    if name in _STAGE2_IMPLS:
        raise ValueError(f"stage2 impl '{name}' already registered")
    _STAGE2_IMPLS[name] = fn


def register_stage1_metadata_getter(name: str, fn: Stage1MetadataGetter) -> None:
    if name in _STAGE1_METADATA_GETTERS:
        raise ValueError(f"stage1 metadata getter '{name}' already registered")
    _STAGE1_METADATA_GETTERS[name] = fn


def list_stage1_impls() -> tuple[str, ...]:
    return tuple(sorted(_STAGE1_IMPLS))


def list_stage2_impls() -> tuple[str, ...]:
    return tuple(sorted(_STAGE2_IMPLS))


def list_discovered_impl_modules() -> tuple[str, ...]:
    return _DISCOVERED_IMPL_MODULES


def get_stage1_impl_metadata(impl: str) -> dict[str, object] | None:
    getter = _STAGE1_METADATA_GETTERS.get(impl)
    if getter is None:
        return None
    return getter()


def validate_stage1_artifacts(
    *,
    spec: Phase2HotShapeSpec,
    output_dtype: torch.dtype,
    num_kv_splits: int,
    query_tile_size_hint: int,
    artifacts: Stage1Artifacts,
) -> None:
    validate_stage_buffer_plan(
        spec=spec,
        num_kv_splits=num_kv_splits,
        plan=artifacts.plan,
        output_dtype=output_dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    expected_mid_o_shape = artifacts.plan.mid_o_shape
    expected_output_shape = artifacts.plan.output_shape
    expected_lse_shape = (artifacts.plan.output_shape[1], artifacts.plan.output_shape[0])
    if artifacts.plan.mid_o_shape != expected_mid_o_shape:
        raise ValueError(
            f"stage1 mid_o_shape mismatch: "
            f"{artifacts.plan.mid_o_shape} != {expected_mid_o_shape}"
        )
    if artifacts.plan.output_shape != expected_output_shape:
        raise ValueError(
            f"stage1 output_shape mismatch: "
            f"{artifacts.plan.output_shape} != {expected_output_shape}"
        )
    if tuple(artifacts.mid_o.shape) != expected_mid_o_shape:
        raise ValueError(
            f"stage1 mid_o tensor shape mismatch: "
            f"{tuple(artifacts.mid_o.shape)} != {expected_mid_o_shape}"
        )
    if tuple(artifacts.full_out_ref.shape) != expected_output_shape:
        raise ValueError(
            f"stage1 full_out_ref shape mismatch: "
            f"{tuple(artifacts.full_out_ref.shape)} != {expected_output_shape}"
        )
    if tuple(artifacts.full_lse_ref.shape) != expected_lse_shape:
        raise ValueError(
            f"stage1 full_lse_ref shape mismatch: "
            f"{tuple(artifacts.full_lse_ref.shape)} != {expected_lse_shape}"
        )
    if artifacts.mid_o.dtype != torch.float32:
        raise ValueError(f"stage1 mid_o dtype must be float32, got {artifacts.mid_o.dtype}")
    if artifacts.full_out_ref.dtype != torch.float32:
        raise ValueError(
            f"stage1 full_out_ref dtype must be float32, got {artifacts.full_out_ref.dtype}"
        )
    if artifacts.full_lse_ref.dtype != torch.float32:
        raise ValueError(
            f"stage1 full_lse_ref dtype must be float32, got {artifacts.full_lse_ref.dtype}"
        )


def validate_stage2_artifacts(
    *,
    stage1: Stage1Artifacts,
    artifacts: Stage2Artifacts,
) -> None:
    expected_output_shape = stage1.plan.output_shape
    expected_lse_shape = (
        stage1.plan.output_shape[1],
        stage1.plan.output_shape[0],
    )
    if tuple(artifacts.output.shape) != expected_output_shape:
        raise ValueError(
            f"stage2 output shape mismatch: "
            f"{tuple(artifacts.output.shape)} != {expected_output_shape}"
        )
    if tuple(artifacts.lse.shape) != expected_lse_shape:
        raise ValueError(
            f"stage2 lse shape mismatch: "
            f"{tuple(artifacts.lse.shape)} != {expected_lse_shape}"
        )
    if artifacts.output.dtype != torch.float32:
        raise ValueError(
            f"stage2 output dtype must be float32, got {artifacts.output.dtype}"
        )
    if artifacts.lse.dtype != torch.float32:
        raise ValueError(f"stage2 lse dtype must be float32, got {artifacts.lse.dtype}")
    if artifacts.output.device != stage1.mid_o.device:
        raise ValueError("stage2 output device must match stage1 mid_o device")
    if artifacts.lse.device != stage1.mid_o.device:
        raise ValueError("stage2 lse device must match stage1 mid_o device")


def run_stage1(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
    impl: str = "reference",
) -> Stage1Artifacts:
    if impl not in _STAGE1_IMPLS:
        raise NotImplementedError(
            f"Unsupported stage1 impl '{impl}'. Supported: {list_stage1_impls()}"
        )
    artifacts = _STAGE1_IMPLS[impl](
        spec,
        dtype,
        device,
        num_kv_splits,
        query_tile_size_hint,
        reference_inputs,
    )
    validate_stage1_artifacts(
        spec=spec,
        output_dtype=dtype,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        artifacts=artifacts,
    )
    return artifacts


def run_stage2(
    stage1: Stage1Artifacts,
    *,
    impl: str = "reference",
) -> Stage2Artifacts:
    if impl not in _STAGE2_IMPLS:
        raise NotImplementedError(
            f"Unsupported stage2 impl '{impl}'. Supported: {list_stage2_impls()}"
        )
    artifacts = _STAGE2_IMPLS[impl](stage1)
    validate_stage2_artifacts(stage1=stage1, artifacts=artifacts)
    return artifacts


def run_pipeline(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
    stage1_impl: str = "reference",
    stage2_impl: str = "reference",
) -> PipelineArtifacts:
    stage1 = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=reference_inputs,
        impl=stage1_impl,
    )
    stage2 = run_stage2(stage1, impl=stage2_impl)
    out_max_abs = (stage2.output - stage1.full_out_ref).abs().max().item()
    lse_max_abs = (stage2.lse - stage1.full_lse_ref).abs().max().item()
    return PipelineArtifacts(
        stage1=stage1,
        stage2=stage2,
        out_max_abs=out_max_abs,
        lse_max_abs=lse_max_abs,
    )


def run_registry_selfcheck(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
    out_tol: float = 1e-5,
    lse_tol: float = 1e-5,
) -> RegistrySelfCheckResult:
    stage1_impls = list_stage1_impls()
    stage2_impls = list_stage2_impls()
    pair_results: list[PairCheckResult] = []
    passed = 0
    for stage1_impl in stage1_impls:
        for stage2_impl in stage2_impls:
            try:
                pipeline = run_pipeline(
                    spec=spec,
                    dtype=dtype,
                    device=device,
                    num_kv_splits=num_kv_splits,
                    query_tile_size_hint=query_tile_size_hint,
                    reference_inputs=None,
                    stage1_impl=stage1_impl,
                    stage2_impl=stage2_impl,
                )
            except NotImplementedError:
                continue
            ok = pipeline.out_max_abs < out_tol and pipeline.lse_max_abs < lse_tol
            pair_results.append(
                PairCheckResult(
                    stage1_impl=stage1_impl,
                    stage2_impl=stage2_impl,
                    out_max_abs=pipeline.out_max_abs,
                    lse_max_abs=pipeline.lse_max_abs,
                    ok=ok,
                )
            )
            passed += int(ok)
    return RegistrySelfCheckResult(
        stage1_impls=stage1_impls,
        stage2_impls=stage2_impls,
        pair_results=tuple(pair_results),
        passed_pairs=passed,
        total_pairs=len(pair_results),
    )


def compare_stage1_impls(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    impl_a: str,
    impl_b: str,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> dict[str, float]:
    inputs = (
        make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
        if reference_inputs is None
        else reference_inputs
    )
    a = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=impl_a,
    )
    b = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=impl_b,
    )
    return {
        "mid_o_max_abs": (a.mid_o - b.mid_o).abs().max().item(),
        "full_out_ref_max_abs": (a.full_out_ref - b.full_out_ref).abs().max().item(),
        "full_lse_ref_max_abs": (a.full_lse_ref - b.full_lse_ref).abs().max().item(),
    }


def _benchmark_callable(
    fn,
    *,
    device: str,
    num_warmup_iters: int,
    num_iters: int,
) -> float:
    if device.startswith("cuda"):
        return benchmark_cuda(
            fn,
            num_warmup_iters=num_warmup_iters,
            num_iters=num_iters,
        )
    for _ in range(num_warmup_iters):
        fn()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / num_iters


def _prime_callable(fn, *, device: str) -> None:
    fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_compare_stage1_impls(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    impl_a: str,
    impl_b: str,
    num_warmup_iters: int,
    num_iters: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> Stage1CompareResult:
    inputs = (
        make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
        if reference_inputs is None
        else reference_inputs
    )
    fn_a = lambda: run_stage1(
            spec=spec,
            dtype=dtype,
            device=device,
            num_kv_splits=num_kv_splits,
            query_tile_size_hint=query_tile_size_hint,
            reference_inputs=inputs,
            impl=impl_a,
        )
    fn_b = lambda: run_stage1(
            spec=spec,
            dtype=dtype,
            device=device,
            num_kv_splits=num_kv_splits,
            query_tile_size_hint=query_tile_size_hint,
            reference_inputs=inputs,
            impl=impl_b,
        )
    _prime_callable(fn_a, device=device)
    _prime_callable(fn_b, device=device)
    ms_a = _benchmark_callable(
        fn_a,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    ms_b = _benchmark_callable(
        fn_b,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    comparison = compare_stage1_impls(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        impl_a=impl_a,
        impl_b=impl_b,
        reference_inputs=inputs,
    )
    return Stage1CompareResult(
        impl_a=impl_a,
        impl_b=impl_b,
        ms_a=ms_a,
        ms_b=ms_b,
        ms_ratio_b_over_a=ms_b / ms_a if ms_a != 0 else float("inf"),
        mid_o_max_abs=comparison["mid_o_max_abs"],
        full_out_ref_max_abs=comparison["full_out_ref_max_abs"],
        full_lse_ref_max_abs=comparison["full_lse_ref_max_abs"],
    )


def benchmark_compare_stage1_replays(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    impl_a: str,
    impl_b: str,
    num_warmup_iters: int,
    num_iters: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> Stage1CompareResult:
    inputs = (
        make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
        if reference_inputs is None
        else reference_inputs
    )
    a = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=impl_a,
    )
    b = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=impl_b,
    )
    if a.replay is None or b.replay is None:
        raise NotImplementedError(
            "compare_stage1_replay requires both stage1 impls to expose replay()"
        )
    _prime_callable(a.replay, device=device)
    _prime_callable(b.replay, device=device)
    ms_a = _benchmark_callable(
        a.replay,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    ms_b = _benchmark_callable(
        b.replay,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    a.replay()
    b.replay()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return Stage1CompareResult(
        impl_a=impl_a,
        impl_b=impl_b,
        ms_a=ms_a,
        ms_b=ms_b,
        ms_ratio_b_over_a=ms_b / ms_a if ms_a != 0 else float("inf"),
        mid_o_max_abs=(a.mid_o - b.mid_o).abs().max().item(),
        full_out_ref_max_abs=(a.full_out_ref - b.full_out_ref).abs().max().item(),
        full_lse_ref_max_abs=(a.full_lse_ref - b.full_lse_ref).abs().max().item(),
    )


def compare_stage2_impls(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    stage1_impl: str,
    impl_a: str,
    impl_b: str,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> dict[str, float]:
    inputs = (
        make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
        if reference_inputs is None
        else reference_inputs
    )
    stage1 = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=stage1_impl,
    )
    a = run_stage2(stage1, impl=impl_a)
    b = run_stage2(stage1, impl=impl_b)
    return {
        "output_max_abs": (a.output - b.output).abs().max().item(),
        "lse_max_abs": (a.lse - b.lse).abs().max().item(),
    }


def benchmark_compare_stage2_impls(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    stage1_impl: str,
    impl_a: str,
    impl_b: str,
    num_warmup_iters: int,
    num_iters: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> Stage2CompareResult:
    inputs = (
        make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
        if reference_inputs is None
        else reference_inputs
    )
    stage1 = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=stage1_impl,
    )
    fn_a = lambda: run_stage2(stage1, impl=impl_a)
    fn_b = lambda: run_stage2(stage1, impl=impl_b)
    _prime_callable(fn_a, device=device)
    _prime_callable(fn_b, device=device)
    ms_a = _benchmark_callable(
        fn_a,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    ms_b = _benchmark_callable(
        fn_b,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    comparison = compare_stage2_impls(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        stage1_impl=stage1_impl,
        impl_a=impl_a,
        impl_b=impl_b,
        reference_inputs=inputs,
    )
    return Stage2CompareResult(
        stage1_impl=stage1_impl,
        impl_a=impl_a,
        impl_b=impl_b,
        ms_a=ms_a,
        ms_b=ms_b,
        ms_ratio_b_over_a=ms_b / ms_a if ms_a != 0 else float("inf"),
        output_max_abs=comparison["output_max_abs"],
        lse_max_abs=comparison["lse_max_abs"],
    )


def benchmark_compare_pipelines(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    stage1_impl_a: str,
    stage2_impl_a: str,
    stage1_impl_b: str,
    stage2_impl_b: str,
    num_warmup_iters: int,
    num_iters: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> PipelineCompareResult:
    inputs = (
        make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
        if reference_inputs is None
        else reference_inputs
    )

    stage1_a = lambda: run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=stage1_impl_a,
    )
    stage1_b = lambda: run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=stage1_impl_b,
    )
    _prime_callable(stage1_a, device=device)
    _prime_callable(stage1_b, device=device)
    stage1_ms_a = _benchmark_callable(
        stage1_a,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    stage1_ms_b = _benchmark_callable(
        stage1_b,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )

    stage1_artifacts_a = stage1_a()
    stage1_artifacts_b = stage1_b()

    stage2_a = lambda: run_stage2(stage1_artifacts_a, impl=stage2_impl_a)
    stage2_b = lambda: run_stage2(stage1_artifacts_b, impl=stage2_impl_b)
    _prime_callable(stage2_a, device=device)
    _prime_callable(stage2_b, device=device)
    stage2_ms_a = _benchmark_callable(
        stage2_a,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    stage2_ms_b = _benchmark_callable(
        stage2_b,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )

    pipe_a = PipelineArtifacts(
        stage1=stage1_artifacts_a,
        stage2=stage2_a(),
        out_max_abs=0.0,
        lse_max_abs=0.0,
    )
    pipe_b = PipelineArtifacts(
        stage1=stage1_artifacts_b,
        stage2=stage2_b(),
        out_max_abs=0.0,
        lse_max_abs=0.0,
    )
    ms_a = stage1_ms_a + stage2_ms_a
    ms_b = stage1_ms_b + stage2_ms_b

    return PipelineCompareResult(
        stage1_impl_a=stage1_impl_a,
        stage2_impl_a=stage2_impl_a,
        stage1_impl_b=stage1_impl_b,
        stage2_impl_b=stage2_impl_b,
        stage1_ms_a=stage1_ms_a,
        stage1_ms_b=stage1_ms_b,
        stage1_ms_ratio_b_over_a=stage1_ms_b / stage1_ms_a if stage1_ms_a != 0 else float("inf"),
        stage2_ms_a=stage2_ms_a,
        stage2_ms_b=stage2_ms_b,
        stage2_ms_ratio_b_over_a=stage2_ms_b / stage2_ms_a if stage2_ms_a != 0 else float("inf"),
        ms_a=ms_a,
        ms_b=ms_b,
        ms_ratio_b_over_a=ms_b / ms_a if ms_a != 0 else float("inf"),
        output_max_abs=(pipe_a.stage2.output - pipe_b.stage2.output).abs().max().item(),
        lse_max_abs=(pipe_a.stage2.lse - pipe_b.stage2.lse).abs().max().item(),
    )


def benchmark_compare_pipeline_replays(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    stage1_impl_a: str,
    stage2_impl_a: str,
    stage1_impl_b: str,
    stage2_impl_b: str,
    num_warmup_iters: int,
    num_iters: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> PipelineCompareResult:
    inputs = (
        make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
        if reference_inputs is None
        else reference_inputs
    )
    stage1_a = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=stage1_impl_a,
    )
    stage1_b = run_stage1(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
        reference_inputs=inputs,
        impl=stage1_impl_b,
    )
    stage2_a = run_stage2(stage1_a, impl=stage2_impl_a)
    stage2_b = run_stage2(stage1_b, impl=stage2_impl_b)
    if (
        stage1_a.replay is None
        or stage1_b.replay is None
        or stage2_a.replay is None
        or stage2_b.replay is None
    ):
        raise NotImplementedError(
            "compare_pipeline_replay requires replay() on both stage1 and stage2 impls"
        )

    replay_pipe_a = lambda: (stage1_a.replay(), stage2_a.replay())
    replay_pipe_b = lambda: (stage1_b.replay(), stage2_b.replay())
    _prime_callable(stage1_a.replay, device=device)
    _prime_callable(stage1_b.replay, device=device)
    _prime_callable(stage2_a.replay, device=device)
    _prime_callable(stage2_b.replay, device=device)
    stage1_ms_a = _benchmark_callable(
        stage1_a.replay,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    stage1_ms_b = _benchmark_callable(
        stage1_b.replay,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    stage2_ms_a = _benchmark_callable(
        stage2_a.replay,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    stage2_ms_b = _benchmark_callable(
        stage2_b.replay,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    ms_a = _benchmark_callable(
        replay_pipe_a,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    ms_b = _benchmark_callable(
        replay_pipe_b,
        device=device,
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    replay_pipe_a()
    replay_pipe_b()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return PipelineCompareResult(
        stage1_impl_a=stage1_impl_a,
        stage2_impl_a=stage2_impl_a,
        stage1_impl_b=stage1_impl_b,
        stage2_impl_b=stage2_impl_b,
        stage1_ms_a=stage1_ms_a,
        stage1_ms_b=stage1_ms_b,
        stage1_ms_ratio_b_over_a=stage1_ms_b / stage1_ms_a if stage1_ms_a != 0 else float("inf"),
        stage2_ms_a=stage2_ms_a,
        stage2_ms_b=stage2_ms_b,
        stage2_ms_ratio_b_over_a=stage2_ms_b / stage2_ms_a if stage2_ms_a != 0 else float("inf"),
        ms_a=ms_a,
        ms_b=ms_b,
        ms_ratio_b_over_a=ms_b / ms_a if ms_a != 0 else float("inf"),
        output_max_abs=(stage2_a.output - stage2_b.output).abs().max().item(),
        lse_max_abs=(stage2_a.lse - stage2_b.lse).abs().max().item(),
    )


def compare_stage1_result_to_dict(
    *,
    impl_a: str,
    impl_b: str,
    comparison: dict[str, float],
) -> dict[str, object]:
    return {
        "compare_stage1_impl_a": impl_a,
        "compare_stage1_impl_b": impl_b,
        **comparison,
    }


def pair_check_result_to_dict(pair: PairCheckResult) -> dict[str, object]:
    return {
        "stage1_impl": pair.stage1_impl,
        "stage2_impl": pair.stage2_impl,
        "out_max_abs": pair.out_max_abs,
        "lse_max_abs": pair.lse_max_abs,
        "ok": pair.ok,
    }


def stage1_compare_result_to_dict(result: Stage1CompareResult) -> dict[str, object]:
    return {
        "compare_stage1_impl_a": result.impl_a,
        "compare_stage1_impl_b": result.impl_b,
        "compare_stage1_ms_a": result.ms_a,
        "compare_stage1_ms_b": result.ms_b,
        "compare_stage1_ms_ratio_b_over_a": result.ms_ratio_b_over_a,
        "mid_o_max_abs": result.mid_o_max_abs,
        "full_out_ref_max_abs": result.full_out_ref_max_abs,
        "full_lse_ref_max_abs": result.full_lse_ref_max_abs,
    }


def stage2_compare_result_to_dict(result: Stage2CompareResult) -> dict[str, object]:
    return {
        "compare_stage1_impl": result.stage1_impl,
        "compare_stage2_impl_a": result.impl_a,
        "compare_stage2_impl_b": result.impl_b,
        "compare_stage2_ms_a": result.ms_a,
        "compare_stage2_ms_b": result.ms_b,
        "compare_stage2_ms_ratio_b_over_a": result.ms_ratio_b_over_a,
        "output_max_abs": result.output_max_abs,
        "lse_max_abs": result.lse_max_abs,
    }


def pipeline_compare_result_to_dict(result: PipelineCompareResult) -> dict[str, object]:
    return {
        "compare_pipeline_stage1_impl_a": result.stage1_impl_a,
        "compare_pipeline_stage2_impl_a": result.stage2_impl_a,
        "compare_pipeline_stage1_impl_b": result.stage1_impl_b,
        "compare_pipeline_stage2_impl_b": result.stage2_impl_b,
        "compare_pipeline_stage1_ms_a": result.stage1_ms_a,
        "compare_pipeline_stage1_ms_b": result.stage1_ms_b,
        "compare_pipeline_stage1_ms_ratio_b_over_a": result.stage1_ms_ratio_b_over_a,
        "compare_pipeline_stage2_ms_a": result.stage2_ms_a,
        "compare_pipeline_stage2_ms_b": result.stage2_ms_b,
        "compare_pipeline_stage2_ms_ratio_b_over_a": result.stage2_ms_ratio_b_over_a,
        "compare_pipeline_ms_a": result.ms_a,
        "compare_pipeline_ms_b": result.ms_b,
        "compare_pipeline_ms_ratio_b_over_a": result.ms_ratio_b_over_a,
        "output_max_abs": result.output_max_abs,
        "lse_max_abs": result.lse_max_abs,
    }


def stage1_artifacts_to_dict(stage1: Stage1Artifacts) -> dict[str, object]:
    return {
        **stage_buffer_plan_to_dict(stage1.plan),
        "stage1_impl": stage1.impl,
        "stage1_supports_replay": stage1.replay is not None,
        "stage1_mid_o_shape": tuple(stage1.mid_o.shape),
        "stage1_mid_o_dtype": str(stage1.mid_o.dtype),
        "stage1_full_out_ref_shape": tuple(stage1.full_out_ref.shape),
        "stage1_full_out_ref_dtype": str(stage1.full_out_ref.dtype),
        "stage1_full_lse_ref_shape": tuple(stage1.full_lse_ref.shape),
        "stage1_full_lse_ref_dtype": str(stage1.full_lse_ref.dtype),
    }


def stage2_artifacts_to_dict(stage2: Stage2Artifacts) -> dict[str, object]:
    return {
        "stage2_impl": stage2.impl,
        "stage2_supports_replay": stage2.replay is not None,
        "reduced_output_shape": tuple(stage2.output.shape),
        "reduced_output_dtype": str(stage2.output.dtype),
        "reduced_lse_shape": tuple(stage2.lse.shape),
        "reduced_lse_dtype": str(stage2.lse.dtype),
    }


def pipeline_artifacts_to_dict(pipeline: PipelineArtifacts) -> dict[str, object]:
    return {
        **stage1_artifacts_to_dict(pipeline.stage1),
        **stage2_artifacts_to_dict(pipeline.stage2),
        "pipeline_out_max_abs": pipeline.out_max_abs,
        "pipeline_lse_max_abs": pipeline.lse_max_abs,
    }


def registry_selfcheck_to_dict(
    result: RegistrySelfCheckResult,
) -> dict[str, object]:
    return {
        "selfcheck_stage1_impls": result.stage1_impls,
        "selfcheck_stage2_impls": result.stage2_impls,
        "selfcheck_pairs": [pair_check_result_to_dict(pair) for pair in result.pair_results],
        "selfcheck_passed_pairs": result.passed_pairs,
        "selfcheck_total_pairs": result.total_pairs,
    }


def _discover_impl_modules() -> tuple[str, ...]:
    modules = []
    this_dir = Path(__file__).resolve().parent
    package_prefix = __package__
    for path in sorted(this_dir.glob("phase2_staged_impls_*.py")):
        if path.name == Path(__file__).name:
            continue
        module_name = path.stem
        qualified_name = (
            f"{package_prefix}.{module_name}" if package_prefix else module_name
        )
        importlib.import_module(qualified_name)
        modules.append(module_name)
    return tuple(modules)


_DISCOVERED_IMPL_MODULES = _discover_impl_modules()
