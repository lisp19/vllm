# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from contextlib import contextmanager

try:
    import pytest
except ImportError:
    @contextmanager
    def _raises(exc_type):
        try:
            yield
        except exc_type:
            return
        raise AssertionError(f"Expected {exc_type.__name__} to be raised")

    class _PytestFallback:
        raises = staticmethod(_raises)

    pytest = _PytestFallback()

from benchmarks.kernels.phase2_hotshape_utils import (
    CompiledKernelMetadata,
    Phase2HotShapeSpec,
    compiled_metadata_to_dict,
    hotshape_spec_to_dict,
    make_phase2_packed_inputs,
)
from benchmarks.kernels.phase2_staged_kernel_experiment import (
    Stage1Artifacts,
    Stage2Artifacts,
    benchmark_compare_pipelines,
    benchmark_compare_stage1_impls,
    benchmark_compare_stage2_impls,
    compare_stage1_impls,
    compare_stage2_impls,
    list_discovered_impl_modules,
    list_stage1_impls,
    list_stage2_impls,
    pipeline_artifacts_to_dict,
    registry_selfcheck_to_dict,
    run_pipeline,
    run_registry_selfcheck,
    stage1_artifacts_to_dict,
    stage2_artifacts_to_dict,
)
from benchmarks.kernels.phase2_staged_reference import (
    allocate_partial_buffers,
    allocate_stage2_outputs,
    allocate_stage_buffer,
    build_stage_buffer_plan,
    make_reference_attention_inputs_from_tensors,
    make_packed_reference_inputs,
    materialize_packed_factorized_partials,
    materialize_packed_factorized_partials_into,
    pack_stage_buffer,
    run_packed_factorized_pipeline,
    stage_buffer_plan_to_dict,
    unpack_stage_buffer,
    validate_stage_buffer_plan,
)


def _small_spec() -> Phase2HotShapeSpec:
    return Phase2HotShapeSpec(
        seq_len=128,
        query_len=16,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        block_size=16,
        sliding_window=-1,
        k_bits=5,
        v_bits=4,
    )


def test_phase2_staged_registry_discovers_expected_impls():
    assert "phase2_staged_impls_reference" in list_discovered_impl_modules()
    assert "phase2_staged_impls_qtile_reference" in list_discovered_impl_modules()
    assert "phase2_staged_impls_kside_splitkv_stub" in list_discovered_impl_modules()
    assert "phase2_staged_impls_merge_stub" in list_discovered_impl_modules()
    assert "phase2_staged_impls_merge_tree_reference" in list_discovered_impl_modules()
    assert set(list_stage1_impls()) >= {
        "kside_splitkv_chunkedkv_proto",
        "kside_splitkv_chunkedk_proto",
        "reference",
        "packed_factorized_reference",
        "passthrough",
        "qtile_reference",
        "qtile_packed_factorized_reference",
        "kside_splitkv_stub",
    }
    assert set(list_stage2_impls()) >= {
        "reference",
        "passthrough",
        "merge_stub",
        "merge_tree_reference",
    }


def test_stage_buffer_plan_roundtrip_metadata():
    spec = _small_spec()
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=4,
        output_dtype=torch.float32,
        query_tile_size_hint=8,
    )
    validate_stage_buffer_plan(
        spec=spec,
        num_kv_splits=4,
        plan=plan,
        output_dtype=torch.float32,
        query_tile_size_hint=8,
    )
    payload = stage_buffer_plan_to_dict(plan)
    assert payload["stage_split_ranges"] == ((0, 32), (32, 64), (64, 96), (96, 128))
    assert payload["stage_split_lengths"] == (32, 32, 32, 32)
    assert payload["stage_query_tile_size_hint"] == 8
    assert payload["stage_num_query_tiles"] == 2
    assert payload["stage_max_query_tile_tokens"] == 8


def test_hotshape_and_metadata_serialization_helpers():
    spec = _small_spec()
    spec_payload = hotshape_spec_to_dict(spec, dtype=torch.float16)
    assert spec_payload["seq_len"] == 128
    assert spec_payload["query_len"] == 16
    assert spec_payload["head_size"] == 64
    assert spec_payload["dtype"] == "torch.float16"

    metadata = CompiledKernelMetadata(
        n_regs=208,
        n_spills=0,
        num_warps=4,
        num_stages=3,
        shared=32768,
    )
    metadata_payload = compiled_metadata_to_dict(metadata)
    assert metadata_payload == {
        "compiled_n_regs": 208,
        "compiled_n_spills": 0,
        "compiled_num_warps": 4,
        "compiled_num_stages": 3,
        "compiled_shared": 32768,
    }


def test_stage_buffer_plan_validation_rejects_wrong_query_tile_hint():
    spec = _small_spec()
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=4,
        output_dtype=torch.float32,
        query_tile_size_hint=8,
    )
    with pytest.raises(ValueError):
        validate_stage_buffer_plan(
            spec=spec,
            num_kv_splits=4,
            plan=plan,
            output_dtype=torch.float32,
            query_tile_size_hint=16,
        )


def test_stage_buffer_pack_unpack_and_allocators():
    spec = _small_spec()
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=4,
        output_dtype=torch.float32,
        query_tile_size_hint=8,
    )
    partial_out, partial_lse = allocate_partial_buffers(plan, device="cpu")
    partial_out.normal_()
    partial_lse.normal_()
    mid_o = pack_stage_buffer(partial_out, partial_lse)
    roundtrip_out, roundtrip_lse = unpack_stage_buffer(mid_o)
    assert tuple(mid_o.shape) == plan.mid_o_shape
    assert mid_o.dtype == torch.float32
    assert torch.equal(roundtrip_out, partial_out)
    assert torch.equal(roundtrip_lse, partial_lse)
    stage_buf = allocate_stage_buffer(plan, device="cpu")
    stage2_out, stage2_lse = allocate_stage2_outputs(plan, device="cpu")
    assert tuple(stage_buf.shape) == plan.mid_o_shape
    assert tuple(stage2_out.shape) == plan.output_shape
    assert tuple(stage2_lse.shape) == (plan.output_shape[1], plan.output_shape[0])
    assert stage2_out.dtype == torch.float32
    assert stage2_lse.dtype == torch.float32


def test_reference_inputs_from_explicit_tensors_cpu():
    spec = _small_spec()
    torch.manual_seed(0)
    query = torch.randn(
        spec.query_len,
        spec.num_query_heads,
        spec.head_size,
        dtype=torch.float32,
    )
    key = torch.randn(
        spec.seq_len,
        spec.num_query_heads,
        spec.head_size,
        dtype=torch.float32,
    )
    value = torch.randn(
        spec.seq_len,
        spec.num_query_heads,
        spec.head_size_v,
        dtype=torch.float32,
    )
    inputs = make_reference_attention_inputs_from_tensors(
        spec=spec,
        query=query,
        key=key,
        value=value,
    )
    assert torch.equal(inputs.query, query)
    assert torch.equal(inputs.key, key)
    assert torch.equal(inputs.value, value)
    assert tuple(inputs.full_scores.shape) == (
        spec.query_len,
        spec.num_query_heads,
        spec.seq_len,
    )
    assert tuple(inputs.full_out_ref.shape) == (
        spec.query_len,
        spec.num_query_heads,
        spec.head_size_v,
    )
    assert tuple(inputs.full_lse_ref.shape) == (
        spec.num_query_heads,
        spec.query_len,
    )


def test_reference_pipeline_and_registry_selfcheck_cpu():
    spec = _small_spec()
    dtype = torch.float32
    pipeline = run_pipeline(
        spec=spec,
        dtype=dtype,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        stage1_impl="qtile_reference",
        stage2_impl="merge_tree_reference",
    )
    assert pipeline.out_max_abs < 1e-5
    assert pipeline.lse_max_abs < 1e-5
    stage1_payload = stage1_artifacts_to_dict(pipeline.stage1)
    stage2_payload = stage2_artifacts_to_dict(pipeline.stage2)
    pipeline_payload = pipeline_artifacts_to_dict(pipeline)
    assert stage1_payload["stage1_impl"] == "qtile_reference"
    assert stage1_payload["stage1_mid_o_shape"] == pipeline.stage1.plan.mid_o_shape
    assert stage2_payload["stage2_impl"] == "merge_tree_reference"
    assert stage2_payload["reduced_output_shape"] == pipeline.stage1.plan.output_shape
    assert pipeline_payload["stage1_impl"] == "qtile_reference"
    assert pipeline_payload["stage2_impl"] == "merge_tree_reference"
    assert pipeline_payload["pipeline_out_max_abs"] == pipeline.out_max_abs
    assert pipeline_payload["pipeline_lse_max_abs"] == pipeline.lse_max_abs
    result = run_registry_selfcheck(
        spec=spec,
        dtype=dtype,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        out_tol=1e-5,
        lse_tol=1e-5,
    )
    result_payload = registry_selfcheck_to_dict(result)
    assert result_payload["selfcheck_stage1_impls"] == result.stage1_impls
    assert result_payload["selfcheck_stage2_impls"] == result.stage2_impls
    assert result_payload["selfcheck_passed_pairs"] == result.passed_pairs
    assert result_payload["selfcheck_total_pairs"] == result.total_pairs
    assert len(result_payload["selfcheck_pairs"]) == result.total_pairs
    assert result.total_pairs >= 16
    assert result.passed_pairs == result.total_pairs


def test_stage1_artifacts_to_dict_reports_replay_support():
    spec = _small_spec()
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=4,
        output_dtype=torch.float32,
        query_tile_size_hint=8,
    )
    mid_o = allocate_stage_buffer(plan, device="cpu")
    full_out = torch.zeros(plan.output_shape, dtype=torch.float32)
    full_lse = torch.zeros(
        (plan.output_shape[1], plan.output_shape[0]), dtype=torch.float32
    )
    without_replay = Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=full_out,
        full_lse_ref=full_lse,
        impl="test_impl",
    )
    with_replay = Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=full_out,
        full_lse_ref=full_lse,
        impl="test_impl",
        replay=lambda: None,
    )
    assert stage1_artifacts_to_dict(without_replay)["stage1_supports_replay"] is False
    assert stage1_artifacts_to_dict(with_replay)["stage1_supports_replay"] is True


def test_materialize_packed_factorized_partials_into_matches_allocating_path():
    spec = _small_spec()
    packed_inputs = make_packed_reference_inputs(spec=spec, dtype=torch.float32, device="cpu")
    plan, partial_out, partial_lse, full_out, full_lse = (
        materialize_packed_factorized_partials(
            spec=spec,
            hotshape=packed_inputs.hotshape,
            num_kv_splits=4,
            query_tile_size_hint=8,
        )
    )
    replay_out, replay_lse = allocate_partial_buffers(plan, device="cpu")
    full_out_2, full_lse_2 = materialize_packed_factorized_partials_into(
        spec=spec,
        hotshape=packed_inputs.hotshape,
        plan=plan,
        partial_out=replay_out,
        partial_lse=replay_lse,
    )
    assert torch.allclose(replay_out, partial_out)
    assert torch.allclose(replay_lse, partial_lse)
    assert torch.allclose(full_out_2, full_out)
    assert torch.allclose(full_lse_2, full_lse)


def test_stage2_artifacts_to_dict_reports_replay_support():
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    lse = torch.zeros((3, 2), dtype=torch.float32)
    without_replay = Stage2Artifacts(output=output, lse=lse, impl="test_impl")
    with_replay = Stage2Artifacts(
        output=output,
        lse=lse,
        impl="test_impl",
        replay=lambda: None,
    )
    assert stage2_artifacts_to_dict(without_replay)["stage2_supports_replay"] is False
    assert stage2_artifacts_to_dict(with_replay)["stage2_supports_replay"] is True


def test_packed_factorized_pipeline_cpu():
    spec = _small_spec()
    hotshape = make_phase2_packed_inputs(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
    )
    plan, merged, merged_lse, out_max_abs, lse_max_abs = run_packed_factorized_pipeline(
        spec=spec,
        hotshape=hotshape,
        num_kv_splits=4,
        query_tile_size_hint=8,
    )
    assert tuple(merged.shape) == plan.output_shape
    assert tuple(merged_lse.shape) == (plan.output_shape[1], plan.output_shape[0])
    assert merged.dtype == torch.float32
    assert merged_lse.dtype == torch.float32
    assert out_max_abs < 1e-5
    assert lse_max_abs < 1e-5


def test_packed_factorized_reference_stage1_via_registry_cpu():
    spec = _small_spec()
    packed_inputs = make_packed_reference_inputs(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
    )
    pipeline = run_pipeline(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        reference_inputs=packed_inputs,
        stage1_impl="packed_factorized_reference",
        stage2_impl="reference",
    )
    assert pipeline.out_max_abs < 1e-5
    assert pipeline.lse_max_abs < 1e-5


def test_qtile_packed_factorized_reference_matches_packed_reference_cpu():
    spec = _small_spec()
    packed_inputs = make_packed_reference_inputs(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
    )
    comparison = compare_stage1_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        impl_a="packed_factorized_reference",
        impl_b="qtile_packed_factorized_reference",
        reference_inputs=packed_inputs,
    )
    assert comparison["mid_o_max_abs"] < 1e-5
    assert comparison["full_out_ref_max_abs"] == 0.0
    assert comparison["full_lse_ref_max_abs"] == 0.0


def test_kside_splitkv_proto_matches_packed_factorized_reference_cpu():
    spec = _small_spec()
    packed_inputs = make_packed_reference_inputs(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
    )
    comparison = compare_stage1_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        impl_a="packed_factorized_reference",
        impl_b="kside_splitkv_proto",
        reference_inputs=packed_inputs,
    )
    assert comparison["mid_o_max_abs"] < 1e-5
    assert comparison["full_out_ref_max_abs"] == 0.0
    assert comparison["full_lse_ref_max_abs"] == 0.0


def test_kside_splitkv_chunkedk_proto_matches_packed_factorized_reference_cpu():
    spec = _small_spec()
    packed_inputs = make_packed_reference_inputs(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
    )
    comparison = compare_stage1_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        impl_a="packed_factorized_reference",
        impl_b="kside_splitkv_chunkedk_proto",
        reference_inputs=packed_inputs,
    )
    assert comparison["mid_o_max_abs"] < 1e-5
    assert comparison["full_out_ref_max_abs"] == 0.0
    assert comparison["full_lse_ref_max_abs"] == 0.0


def test_kside_splitkv_chunkedkv_proto_matches_packed_factorized_reference_cpu():
    spec = _small_spec()
    packed_inputs = make_packed_reference_inputs(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
    )
    comparison = compare_stage1_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        impl_a="packed_factorized_reference",
        impl_b="kside_splitkv_chunkedkv_proto",
        reference_inputs=packed_inputs,
    )
    assert comparison["mid_o_max_abs"] < 1e-5
    assert comparison["full_out_ref_max_abs"] == 0.0
    assert comparison["full_lse_ref_max_abs"] == 0.0


def test_packed_reference_inputs_keep_high_precision_dense_view_cpu():
    spec = _small_spec()
    packed_inputs = make_packed_reference_inputs(
        spec=spec,
        dtype=torch.float16,
        device="cpu",
    )
    assert packed_inputs.dense_inputs.query.dtype == torch.float16
    assert packed_inputs.dense_inputs.key.dtype == torch.float16
    assert packed_inputs.dense_inputs.value.dtype == torch.float16
    assert packed_inputs.dense_inputs_high_precision.query.dtype == torch.float16
    assert packed_inputs.dense_inputs_high_precision.key.dtype == torch.float32
    assert packed_inputs.dense_inputs_high_precision.value.dtype == torch.float32


def test_stage1_impls_can_share_reference_inputs_cpu():
    spec = _small_spec()
    comparison = compare_stage1_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        impl_a="reference",
        impl_b="qtile_reference",
    )
    assert comparison["mid_o_max_abs"] < 1e-5
    assert comparison["full_out_ref_max_abs"] == 0.0
    assert comparison["full_lse_ref_max_abs"] == 0.0

    timed = benchmark_compare_stage1_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        impl_a="reference",
        impl_b="qtile_reference",
        num_warmup_iters=1,
        num_iters=2,
    )
    assert timed.ms_a > 0.0
    assert timed.ms_b > 0.0
    assert timed.mid_o_max_abs < 1e-5
    assert timed.full_out_ref_max_abs == 0.0
    assert timed.full_lse_ref_max_abs == 0.0


def test_stage2_impls_can_share_stage1_inputs_cpu():
    spec = _small_spec()
    comparison = compare_stage2_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        stage1_impl="qtile_reference",
        impl_a="reference",
        impl_b="merge_tree_reference",
    )
    assert comparison["output_max_abs"] < 1e-5
    assert comparison["lse_max_abs"] < 1e-5

    timed = benchmark_compare_stage2_impls(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        stage1_impl="qtile_reference",
        impl_a="reference",
        impl_b="merge_tree_reference",
        num_warmup_iters=1,
        num_iters=2,
    )
    assert timed.ms_a > 0.0
    assert timed.ms_b > 0.0
    assert timed.output_max_abs < 1e-5
    assert timed.lse_max_abs < 1e-5


def test_pipeline_compare_uses_composed_stage_timings_cpu():
    spec = _small_spec()
    torch.manual_seed(1)
    query = torch.randn(
        spec.query_len,
        spec.num_query_heads,
        spec.head_size,
        dtype=torch.float32,
    )
    key = torch.randn(
        spec.seq_len,
        spec.num_query_heads,
        spec.head_size,
        dtype=torch.float32,
    )
    value = torch.randn(
        spec.seq_len,
        spec.num_query_heads,
        spec.head_size_v,
        dtype=torch.float32,
    )
    inputs = make_reference_attention_inputs_from_tensors(
        spec=spec,
        query=query,
        key=key,
        value=value,
    )
    timed = benchmark_compare_pipelines(
        spec=spec,
        dtype=torch.float32,
        device="cpu",
        num_kv_splits=4,
        query_tile_size_hint=8,
        stage1_impl_a="reference",
        stage2_impl_a="reference",
        stage1_impl_b="qtile_reference",
        stage2_impl_b="merge_tree_reference",
        num_warmup_iters=1,
        num_iters=2,
        reference_inputs=inputs,
    )
    assert timed.ms_a == timed.stage1_ms_a + timed.stage2_ms_a
    assert timed.ms_b == timed.stage1_ms_b + timed.stage2_ms_b
    assert timed.output_max_abs < 1e-5
    assert timed.lse_max_abs < 1e-5
