# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental phase-2 scaffold for staged/split-KV kernel-family work.

This file is intentionally not a final optimization. It gives future phase-2
kernel-family work a stable entry point that:

- reuses the canonical Gemma4 hot-shape inputs
- reports the native packed-int baseline with compiled metadata
- defines the intermediate buffer shapes for a future stage1/stage2 design

The actual staged kernel implementation is still pending. The point here is to
stop rebuilding the same setup/shape/metadata logic from scratch every time a
new low-level idea is tested.
"""

from __future__ import annotations

import json
import random
import torch

from phase2_hotshape_utils import (
    compute_packed_factorized_reference,
    Phase2HotShapeSpec,
    benchmark_cuda,
    build_phase2_launch_config,
    extract_compiled_metadata,
    hotshape_spec_to_dict,
    make_phase2_packed_inputs,
)
from phase2_staged_kernel_experiment import (
    benchmark_compare_stage1_impls,
    benchmark_compare_stage1_replays,
    benchmark_compare_stage2_impls,
    benchmark_compare_pipelines,
    benchmark_compare_pipeline_replays,
    get_stage1_impl_metadata,
    list_discovered_impl_modules,
    list_stage1_impls,
    list_stage2_impls,
    pipeline_compare_result_to_dict,
    pipeline_artifacts_to_dict,
    registry_selfcheck_to_dict,
    run_registry_selfcheck,
    run_pipeline,
    run_stage1,
    run_stage2,
    stage1_artifacts_to_dict,
    stage1_compare_result_to_dict,
    stage2_artifacts_to_dict,
    stage2_compare_result_to_dict,
)
from phase2_staged_reference import (
    PackedReferenceInputs,
    StageBufferPlan,
    build_stage_buffer_plan,
    make_packed_reference_inputs,
    run_packed_factorized_pipeline,
    stage_buffer_plan_to_dict,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.ops.triton_packed_int_kv import (
    kernel_packed_int_attention,
    paged_attention_packed_int,
)


def _emit_json(payload: dict[str, object], *, enabled: bool) -> bool:
    if not enabled:
        return False
    print(json.dumps(payload, sort_keys=True))
    return True


def _seed_all(seed: int) -> None:
    try:
        set_random_seed(seed)
    except NotImplementedError:
        random.seed(seed)
        torch.manual_seed(seed)


def _print_stage_plan(plan: StageBufferPlan) -> None:
    for key, value in stage_buffer_plan_to_dict(plan).items():
        print(f"{key}={value}")


def _run_baseline(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_warmup_iters: int,
    num_iters: int,
) -> tuple[float, object]:
    hotshape = make_phase2_packed_inputs(spec=spec, dtype=dtype, device=device)
    query = hotshape["query"]
    packed_out = hotshape["out"]
    packed_layout = hotshape["layout"]
    packed_launch = build_phase2_launch_config(
        q_element_size=query.element_size(),
        spec=spec,
    )

    packed_kwargs = dict(
        q=query,
        key_cache=hotshape["key_cache"],
        value_cache=hotshape["value_cache"],
        k_scale_cache=hotshape["k_scale"],
        v_scale_cache=hotshape["v_scale"],
        out=packed_out,
        query_start_loc=hotshape["query_start_loc"],
        seq_lens=hotshape["seq_lens"],
        block_table=hotshape["block_table"],
        layout=packed_layout,
        softmax_scale=spec.softmax_scale,
        softcap=0.0,
        num_queries_per_kv=spec.num_queries_per_kv,
        sliding_window=(spec.sliding_window, -1),
        launch_config=packed_launch,
        max_query_len=spec.query_len,
        allow_single_query_override=True,
    )

    kernel_packed_int_attention.device_caches.clear()
    packed_ms = benchmark_cuda(
        lambda: paged_attention_packed_int(**packed_kwargs),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    metadata = extract_compiled_metadata()
    return packed_ms, metadata


def _make_packed_hotshape_reference_inputs(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
) -> tuple[PackedReferenceInputs, dict[str, object]]:
    packed_inputs = make_packed_reference_inputs(spec=spec, dtype=dtype, device=device)
    hotshape = packed_inputs.hotshape
    query = hotshape["query"]
    layout = hotshape["layout"]
    key = hotshape["key_cache"]
    value = hotshape["value_cache"]
    k_scale = hotshape["k_scale"]
    v_scale = hotshape["v_scale"]
    block_table = hotshape["block_table"]

    packed_kwargs = dict(
        q=query,
        key_cache=key,
        value_cache=value,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
        out=hotshape["out"],
        query_start_loc=hotshape["query_start_loc"],
        seq_lens=hotshape["seq_lens"],
        block_table=block_table,
        layout=layout,
        softmax_scale=spec.softmax_scale,
        softcap=0.0,
        num_queries_per_kv=spec.num_queries_per_kv,
        sliding_window=(spec.sliding_window, -1),
        launch_config=build_phase2_launch_config(
            q_element_size=query.element_size(),
            spec=spec,
        ),
        max_query_len=spec.query_len,
        allow_single_query_override=True,
    )
    return packed_inputs, packed_kwargs


def main() -> None:
    parser = FlexibleArgumentParser(
        description=(
            "Experimental phase-2 scaffold for staged/split-KV kernel-family work."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "compare_stage1",
            "compare_stage1_replay",
            "compare_stage2",
            "compare_pipeline",
            "compare_pipeline_replay",
            "compare_native_packed_pipeline",
            "compare_native_packed_pipeline_replay",
            "dryrun_splitkv",
            "inspect_stage1_impl",
            "packed_factorized_pipeline",
            "reference_reduce",
            "reference_parity",
            "reference_pipeline",
            "selfcheck_registry",
        ],
        default="baseline",
    )
    parser.add_argument("--seq-len", type=int, default=17037)
    parser.add_argument("--query-len", type=int, default=1024)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=512)
    parser.add_argument("--head-size-v", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sliding-window", type=int, default=-1)
    parser.add_argument("--k-bits", type=int, default=5)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--dtype", choices=sorted(STR_DTYPE_TO_TORCH_DTYPE), default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--num-kv-splits", type=int, default=4)
    parser.add_argument("--query-tile-size-hint", type=int, default=16)
    parser.add_argument("--stage1-impl", type=str, default="reference")
    parser.add_argument("--stage2-impl", type=str, default="reference")
    parser.add_argument("--compare-stage1-a", type=str, default="reference")
    parser.add_argument("--compare-stage1-b", type=str, default="qtile_reference")
    parser.add_argument("--compare-stage2-a", type=str, default="reference")
    parser.add_argument("--compare-stage2-b", type=str, default="merge_tree_reference")
    parser.add_argument("--compare-pipeline-a-stage1", type=str, default="reference")
    parser.add_argument("--compare-pipeline-a-stage2", type=str, default="reference")
    parser.add_argument("--compare-pipeline-b-stage1", type=str, default="qtile_triton_dense")
    parser.add_argument("--compare-pipeline-b-stage2", type=str, default="merge_triton_dense")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--packed-reference-inputs",
        action="store_true",
        help=(
            "Build one packed hot-shape reference bundle and feed its dense or "
            "packed views into the selected staged implementations."
        ),
    )
    parser.add_argument("--selfcheck-out-tol", type=float, default=1e-5)
    parser.add_argument("--selfcheck-lse-tol", type=float, default=1e-5)
    parser.add_argument(
        "--list-impls",
        action="store_true",
        help="List supported stage1/stage2 implementations and exit.",
    )
    args = parser.parse_args()

    if args.list_impls:
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "list_impls",
                "impl_modules": list_discovered_impl_modules(),
                "stage1_impls": list_stage1_impls(),
                "stage2_impls": list_stage2_impls(),
            },
            enabled=args.json,
        ):
            return
        print(f"impl_modules={list_discovered_impl_modules()}")
        print(f"stage1_impls={list_stage1_impls()}")
        print(f"stage2_impls={list_stage2_impls()}")
        return

    _seed_all(args.seed)
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    spec = Phase2HotShapeSpec(
        seq_len=args.seq_len,
        query_len=args.query_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        head_size_v=args.head_size_v,
        block_size=args.block_size,
        sliding_window=args.sliding_window,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
    )
    packed_reference_inputs: PackedReferenceInputs | None = None
    if args.packed_reference_inputs:
        packed_reference_inputs = make_packed_reference_inputs(
            spec=spec,
            dtype=dtype,
            device=args.device,
        )

    if not args.json:
        print(
            "shape:"
            f" seq_len={spec.seq_len}"
            f" query_len={spec.query_len}"
            f" q_heads={spec.num_query_heads}"
            f" kv_heads={spec.num_kv_heads}"
            f" head_size={spec.head_size}"
            f" head_size_v={spec.head_size_v}"
            f" k_bits={spec.k_bits}"
            f" v_bits={spec.v_bits}"
            f" dtype={dtype}"
        )
        print(f"selected_stage1_impl={args.stage1_impl}")
        print(f"selected_stage2_impl={args.stage2_impl}")

    if args.mode == "baseline":
        packed_ms, metadata = _run_baseline(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "baseline",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                "selected_stage1_impl": args.stage1_impl,
                "selected_stage2_impl": args.stage2_impl,
                "baseline_packed_ms": packed_ms,
                "baseline_n_regs": metadata.n_regs,
                "baseline_n_spills": metadata.n_spills,
                "baseline_num_warps": metadata.num_warps,
                "baseline_num_stages": metadata.num_stages,
                "baseline_shared": metadata.shared,
            },
            enabled=args.json,
        ):
            return
        print(f"baseline_packed_ms={packed_ms:.6f}")
        print(f"baseline_n_regs={metadata.n_regs}")
        print(f"baseline_n_spills={metadata.n_spills}")
        print(f"baseline_num_warps={metadata.num_warps}")
        print(f"baseline_num_stages={metadata.num_stages}")
        print(f"baseline_shared={metadata.shared}")
        return

    if args.mode == "compare_stage1":
        comparison = benchmark_compare_stage1_impls(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            impl_a=args.compare_stage1_a,
            impl_b=args.compare_stage1_b,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            reference_inputs=packed_reference_inputs,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "compare_stage1",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                **stage1_compare_result_to_dict(comparison),
            },
            enabled=args.json,
        ):
            return
        print(f"compare_stage1_impl_a={comparison.impl_a}")
        print(f"compare_stage1_impl_b={comparison.impl_b}")
        print(f"compare_stage1_ms_a={comparison.ms_a:.6f}")
        print(f"compare_stage1_ms_b={comparison.ms_b:.6f}")
        print(f"compare_stage1_ms_ratio_b_over_a={comparison.ms_ratio_b_over_a:.6f}")
        print(f"mid_o_max_abs={comparison.mid_o_max_abs:.6e}")
        print(f"full_out_ref_max_abs={comparison.full_out_ref_max_abs:.6e}")
        print(f"full_lse_ref_max_abs={comparison.full_lse_ref_max_abs:.6e}")
        return

    if args.mode == "compare_stage1_replay":
        comparison = benchmark_compare_stage1_replays(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            impl_a=args.compare_stage1_a,
            impl_b=args.compare_stage1_b,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            reference_inputs=packed_reference_inputs,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "compare_stage1_replay",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                **stage1_compare_result_to_dict(comparison),
            },
            enabled=args.json,
        ):
            return
        print(f"compare_stage1_impl_a={comparison.impl_a}")
        print(f"compare_stage1_impl_b={comparison.impl_b}")
        print(f"compare_stage1_ms_a={comparison.ms_a:.6f}")
        print(f"compare_stage1_ms_b={comparison.ms_b:.6f}")
        print(f"compare_stage1_ms_ratio_b_over_a={comparison.ms_ratio_b_over_a:.6f}")
        print(f"mid_o_max_abs={comparison.mid_o_max_abs:.6e}")
        print(f"full_out_ref_max_abs={comparison.full_out_ref_max_abs:.6e}")
        print(f"full_lse_ref_max_abs={comparison.full_lse_ref_max_abs:.6e}")
        return

    if args.mode == "compare_stage2":
        comparison = benchmark_compare_stage2_impls(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            stage1_impl=args.stage1_impl,
            impl_a=args.compare_stage2_a,
            impl_b=args.compare_stage2_b,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            reference_inputs=packed_reference_inputs,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "compare_stage2",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                **stage2_compare_result_to_dict(comparison),
            },
            enabled=args.json,
        ):
            return
        print(f"compare_stage1_impl={comparison.stage1_impl}")
        print(f"compare_stage2_impl_a={comparison.impl_a}")
        print(f"compare_stage2_impl_b={comparison.impl_b}")
        print(f"compare_stage2_ms_a={comparison.ms_a:.6f}")
        print(f"compare_stage2_ms_b={comparison.ms_b:.6f}")
        print(f"compare_stage2_ms_ratio_b_over_a={comparison.ms_ratio_b_over_a:.6f}")
        print(f"output_max_abs={comparison.output_max_abs:.6e}")
        print(f"lse_max_abs={comparison.lse_max_abs:.6e}")
        return

    if args.mode == "compare_pipeline":
        comparison = benchmark_compare_pipelines(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            stage1_impl_a=args.compare_pipeline_a_stage1,
            stage2_impl_a=args.compare_pipeline_a_stage2,
            stage1_impl_b=args.compare_pipeline_b_stage1,
            stage2_impl_b=args.compare_pipeline_b_stage2,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            reference_inputs=packed_reference_inputs,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "compare_pipeline",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                **pipeline_compare_result_to_dict(comparison),
            },
            enabled=args.json,
        ):
            return
        print(f"compare_pipeline_stage1_impl_a={comparison.stage1_impl_a}")
        print(f"compare_pipeline_stage2_impl_a={comparison.stage2_impl_a}")
        print(f"compare_pipeline_stage1_impl_b={comparison.stage1_impl_b}")
        print(f"compare_pipeline_stage2_impl_b={comparison.stage2_impl_b}")
        print(f"compare_pipeline_stage1_ms_a={comparison.stage1_ms_a:.6f}")
        print(f"compare_pipeline_stage1_ms_b={comparison.stage1_ms_b:.6f}")
        print(
            f"compare_pipeline_stage1_ms_ratio_b_over_a="
            f"{comparison.stage1_ms_ratio_b_over_a:.6f}"
        )
        print(f"compare_pipeline_stage2_ms_a={comparison.stage2_ms_a:.6f}")
        print(f"compare_pipeline_stage2_ms_b={comparison.stage2_ms_b:.6f}")
        print(
            f"compare_pipeline_stage2_ms_ratio_b_over_a="
            f"{comparison.stage2_ms_ratio_b_over_a:.6f}"
        )
        print(f"compare_pipeline_ms_a={comparison.ms_a:.6f}")
        print(f"compare_pipeline_ms_b={comparison.ms_b:.6f}")
        print(
            f"compare_pipeline_ms_ratio_b_over_a="
            f"{comparison.ms_ratio_b_over_a:.6f}"
        )
        print(f"output_max_abs={comparison.output_max_abs:.6e}")
        print(f"lse_max_abs={comparison.lse_max_abs:.6e}")
        return

    if args.mode == "compare_pipeline_replay":
        comparison = benchmark_compare_pipeline_replays(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            stage1_impl_a=args.compare_pipeline_a_stage1,
            stage2_impl_a=args.compare_pipeline_a_stage2,
            stage1_impl_b=args.compare_pipeline_b_stage1,
            stage2_impl_b=args.compare_pipeline_b_stage2,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            reference_inputs=packed_reference_inputs,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "compare_pipeline_replay",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                **pipeline_compare_result_to_dict(comparison),
            },
            enabled=args.json,
        ):
            return
        print(f"compare_pipeline_stage1_impl_a={comparison.stage1_impl_a}")
        print(f"compare_pipeline_stage2_impl_a={comparison.stage2_impl_a}")
        print(f"compare_pipeline_stage1_impl_b={comparison.stage1_impl_b}")
        print(f"compare_pipeline_stage2_impl_b={comparison.stage2_impl_b}")
        print(f"compare_pipeline_stage1_ms_a={comparison.stage1_ms_a:.6f}")
        print(f"compare_pipeline_stage1_ms_b={comparison.stage1_ms_b:.6f}")
        print(
            f"compare_pipeline_stage1_ms_ratio_b_over_a="
            f"{comparison.stage1_ms_ratio_b_over_a:.6f}"
        )
        print(f"compare_pipeline_stage2_ms_a={comparison.stage2_ms_a:.6f}")
        print(f"compare_pipeline_stage2_ms_b={comparison.stage2_ms_b:.6f}")
        print(
            f"compare_pipeline_stage2_ms_ratio_b_over_a="
            f"{comparison.stage2_ms_ratio_b_over_a:.6f}"
        )
        print(f"compare_pipeline_ms_a={comparison.ms_a:.6f}")
        print(f"compare_pipeline_ms_b={comparison.ms_b:.6f}")
        print(
            f"compare_pipeline_ms_ratio_b_over_a="
            f"{comparison.ms_ratio_b_over_a:.6f}"
        )
        print(f"output_max_abs={comparison.output_max_abs:.6e}")
        print(f"lse_max_abs={comparison.lse_max_abs:.6e}")
        return

    if args.mode == "compare_native_packed_pipeline":
        packed_reference_inputs, packed_kwargs = _make_packed_hotshape_reference_inputs(
            spec=spec,
            dtype=dtype,
            device=args.device,
        )
        hotshape = packed_reference_inputs.hotshape
        factored_packed_reference = compute_packed_factorized_reference(
            spec=spec,
            hotshape=hotshape,
        )
        kernel_packed_int_attention.device_caches.clear()
        native_packed_ms = benchmark_cuda(
            lambda: paged_attention_packed_int(**packed_kwargs),
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
        )
        comparison = benchmark_compare_pipelines(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            stage1_impl_a=args.compare_pipeline_a_stage1,
            stage2_impl_a=args.compare_pipeline_a_stage2,
            stage1_impl_b=args.compare_pipeline_b_stage1,
            stage2_impl_b=args.compare_pipeline_b_stage2,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            reference_inputs=packed_reference_inputs,
        )
        native_output = hotshape["out"].float()
        native_vs_factored_output_abs = (
            native_output - factored_packed_reference.output
        ).abs()
        reference_pipeline = run_pipeline(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            reference_inputs=packed_reference_inputs,
            stage1_impl=args.compare_pipeline_a_stage1,
            stage2_impl=args.compare_pipeline_a_stage2,
        )
        candidate_pipeline = run_pipeline(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            reference_inputs=packed_reference_inputs,
            stage1_impl=args.compare_pipeline_b_stage1,
            stage2_impl=args.compare_pipeline_b_stage2,
        )
        native_vs_reference_output_max_abs = (
            native_output - reference_pipeline.stage2.output
        ).abs().max().item()
        native_vs_candidate_output_max_abs = (
            native_output - candidate_pipeline.stage2.output
        ).abs().max().item()
        factored_vs_reference_output_abs = (
            factored_packed_reference.output - reference_pipeline.stage2.output
        ).abs()
        factored_vs_candidate_output_abs = (
            factored_packed_reference.output - candidate_pipeline.stage2.output
        ).abs()
        factored_vs_reference_lse_abs = (
            factored_packed_reference.lse - reference_pipeline.stage2.lse
        ).abs()
        factored_vs_candidate_lse_abs = (
            factored_packed_reference.lse - candidate_pipeline.stage2.lse
        ).abs()
        reference_vs_candidate_lse_max_abs = (
            reference_pipeline.stage2.lse - candidate_pipeline.stage2.lse
        ).abs().max().item()
        candidate_factorized_lse_max_abs = (
            packed_reference_inputs.factorized_lse_ref - candidate_pipeline.stage2.lse
        ).abs().max().item()
        payload = {
            "schema_version": 1,
            "tool": "benchmark_phase2_staged_prototype",
            "mode": "compare_native_packed_pipeline",
            "shape": hotshape_spec_to_dict(spec, dtype=dtype),
            "native_packed_ms": native_packed_ms,
            "selected_stage1_impl": args.compare_pipeline_b_stage1,
            "selected_stage2_impl": args.compare_pipeline_b_stage2,
            "native_vs_stage1_impl_a": args.compare_pipeline_a_stage1,
            "native_vs_stage2_impl_a": args.compare_pipeline_a_stage2,
            "native_vs_stage1_impl_b": args.compare_pipeline_b_stage1,
            "native_vs_stage2_impl_b": args.compare_pipeline_b_stage2,
            "native_vs_factored_output_max_abs": native_vs_factored_output_abs.max().item(),
            "native_vs_factored_output_mean_abs": native_vs_factored_output_abs.mean().item(),
            "native_vs_reference_output_max_abs": native_vs_reference_output_max_abs,
            "native_vs_candidate_output_max_abs": native_vs_candidate_output_max_abs,
            "factored_vs_reference_output_max_abs": factored_vs_reference_output_abs.max().item(),
            "factored_vs_reference_output_mean_abs": factored_vs_reference_output_abs.mean().item(),
            "factored_vs_candidate_output_max_abs": factored_vs_candidate_output_abs.max().item(),
            "factored_vs_candidate_output_mean_abs": factored_vs_candidate_output_abs.mean().item(),
            "factored_vs_reference_lse_max_abs": factored_vs_reference_lse_abs.max().item(),
            "factored_vs_candidate_lse_max_abs": factored_vs_candidate_lse_abs.max().item(),
            "reference_vs_candidate_lse_max_abs": reference_vs_candidate_lse_max_abs,
            "candidate_factorized_lse_max_abs": candidate_factorized_lse_max_abs,
            "candidate_pipeline_vs_native_ratio": (
                comparison.ms_b / native_packed_ms
                if native_packed_ms != 0
                else float("inf")
            ),
            **pipeline_compare_result_to_dict(comparison),
        }
        if _emit_json(payload, enabled=args.json):
            return
        print(f"native_packed_ms={native_packed_ms:.6f}")
        print(f"selected_stage1_impl={args.compare_pipeline_b_stage1}")
        print(f"selected_stage2_impl={args.compare_pipeline_b_stage2}")
        print(
            f"candidate_pipeline_vs_native_ratio="
            f"{payload['candidate_pipeline_vs_native_ratio']:.6f}"
        )
        print(
            "native_vs_factored_output_max_abs="
            f"{native_vs_factored_output_abs.max().item():.6e}"
        )
        print(
            "native_vs_factored_output_mean_abs="
            f"{native_vs_factored_output_abs.mean().item():.6e}"
        )
        print(
            "native_vs_reference_output_max_abs="
            f"{native_vs_reference_output_max_abs:.6e}"
        )
        print(
            "native_vs_candidate_output_max_abs="
            f"{native_vs_candidate_output_max_abs:.6e}"
        )
        print(
            "factored_vs_reference_output_max_abs="
            f"{factored_vs_reference_output_abs.max().item():.6e}"
        )
        print(
            "factored_vs_candidate_output_max_abs="
            f"{factored_vs_candidate_output_abs.max().item():.6e}"
        )
        print(
            "factored_vs_reference_lse_max_abs="
            f"{factored_vs_reference_lse_abs.max().item():.6e}"
        )
        print(
            "factored_vs_candidate_lse_max_abs="
            f"{factored_vs_candidate_lse_abs.max().item():.6e}"
        )
        print(
            "reference_vs_candidate_lse_max_abs="
            f"{reference_vs_candidate_lse_max_abs:.6e}"
        )
        print(
            "candidate_factorized_lse_max_abs="
            f"{candidate_factorized_lse_max_abs:.6e}"
        )
        print(f"compare_pipeline_stage1_ms_a={comparison.stage1_ms_a:.6f}")
        print(f"compare_pipeline_stage1_ms_b={comparison.stage1_ms_b:.6f}")
        print(f"compare_pipeline_stage2_ms_a={comparison.stage2_ms_a:.6f}")
        print(f"compare_pipeline_stage2_ms_b={comparison.stage2_ms_b:.6f}")
        print(f"compare_pipeline_ms_a={comparison.ms_a:.6f}")
        print(f"compare_pipeline_ms_b={comparison.ms_b:.6f}")
        return

    if args.mode == "compare_native_packed_pipeline_replay":
        packed_reference_inputs, packed_kwargs = _make_packed_hotshape_reference_inputs(
            spec=spec,
            dtype=dtype,
            device=args.device,
        )
        hotshape = packed_reference_inputs.hotshape
        factored_packed_reference = compute_packed_factorized_reference(
            spec=spec,
            hotshape=hotshape,
        )
        kernel_packed_int_attention.device_caches.clear()
        native_packed_ms = benchmark_cuda(
            lambda: paged_attention_packed_int(**packed_kwargs),
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
        )
        comparison = benchmark_compare_pipeline_replays(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            stage1_impl_a=args.compare_pipeline_a_stage1,
            stage2_impl_a=args.compare_pipeline_a_stage2,
            stage1_impl_b=args.compare_pipeline_b_stage1,
            stage2_impl_b=args.compare_pipeline_b_stage2,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            reference_inputs=packed_reference_inputs,
        )
        native_output = hotshape["out"].float()
        native_vs_factored_output_abs = (
            native_output - factored_packed_reference.output
        ).abs()
        reference_pipeline = run_pipeline(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            reference_inputs=packed_reference_inputs,
            stage1_impl=args.compare_pipeline_a_stage1,
            stage2_impl=args.compare_pipeline_a_stage2,
        )
        candidate_pipeline = run_pipeline(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            reference_inputs=packed_reference_inputs,
            stage1_impl=args.compare_pipeline_b_stage1,
            stage2_impl=args.compare_pipeline_b_stage2,
        )
        native_vs_reference_output_max_abs = (
            native_output - reference_pipeline.stage2.output
        ).abs().max().item()
        native_vs_candidate_output_max_abs = (
            native_output - candidate_pipeline.stage2.output
        ).abs().max().item()
        factored_vs_reference_output_abs = (
            factored_packed_reference.output - reference_pipeline.stage2.output
        ).abs()
        factored_vs_candidate_output_abs = (
            factored_packed_reference.output - candidate_pipeline.stage2.output
        ).abs()
        factored_vs_reference_lse_abs = (
            factored_packed_reference.lse - reference_pipeline.stage2.lse
        ).abs()
        factored_vs_candidate_lse_abs = (
            factored_packed_reference.lse - candidate_pipeline.stage2.lse
        ).abs()
        reference_vs_candidate_lse_max_abs = (
            reference_pipeline.stage2.lse - candidate_pipeline.stage2.lse
        ).abs().max().item()
        candidate_factorized_lse_max_abs = (
            packed_reference_inputs.factorized_lse_ref - candidate_pipeline.stage2.lse
        ).abs().max().item()
        payload = {
            "schema_version": 1,
            "tool": "benchmark_phase2_staged_prototype",
            "mode": "compare_native_packed_pipeline_replay",
            "shape": hotshape_spec_to_dict(spec, dtype=dtype),
            "native_packed_ms": native_packed_ms,
            "selected_stage1_impl": args.compare_pipeline_b_stage1,
            "selected_stage2_impl": args.compare_pipeline_b_stage2,
            "native_vs_stage1_impl_a": args.compare_pipeline_a_stage1,
            "native_vs_stage2_impl_a": args.compare_pipeline_a_stage2,
            "native_vs_stage1_impl_b": args.compare_pipeline_b_stage1,
            "native_vs_stage2_impl_b": args.compare_pipeline_b_stage2,
            "native_vs_factored_output_max_abs": native_vs_factored_output_abs.max().item(),
            "native_vs_factored_output_mean_abs": native_vs_factored_output_abs.mean().item(),
            "native_vs_reference_output_max_abs": native_vs_reference_output_max_abs,
            "native_vs_candidate_output_max_abs": native_vs_candidate_output_max_abs,
            "factored_vs_reference_output_max_abs": factored_vs_reference_output_abs.max().item(),
            "factored_vs_reference_output_mean_abs": factored_vs_reference_output_abs.mean().item(),
            "factored_vs_candidate_output_max_abs": factored_vs_candidate_output_abs.max().item(),
            "factored_vs_candidate_output_mean_abs": factored_vs_candidate_output_abs.mean().item(),
            "factored_vs_reference_lse_max_abs": factored_vs_reference_lse_abs.max().item(),
            "factored_vs_candidate_lse_max_abs": factored_vs_candidate_lse_abs.max().item(),
            "reference_vs_candidate_lse_max_abs": reference_vs_candidate_lse_max_abs,
            "candidate_factorized_lse_max_abs": candidate_factorized_lse_max_abs,
            "candidate_pipeline_vs_native_ratio": (
                comparison.ms_b / native_packed_ms
                if native_packed_ms != 0
                else float("inf")
            ),
            **pipeline_compare_result_to_dict(comparison),
        }
        if _emit_json(payload, enabled=args.json):
            return
        print(f"native_packed_ms={native_packed_ms:.6f}")
        print(f"selected_stage1_impl={args.compare_pipeline_b_stage1}")
        print(f"selected_stage2_impl={args.compare_pipeline_b_stage2}")
        print(
            f"candidate_pipeline_vs_native_ratio="
            f"{payload['candidate_pipeline_vs_native_ratio']:.6f}"
        )
        print(
            "native_vs_factored_output_max_abs="
            f"{native_vs_factored_output_abs.max().item():.6e}"
        )
        print(
            "native_vs_factored_output_mean_abs="
            f"{native_vs_factored_output_abs.mean().item():.6e}"
        )
        print(
            "native_vs_reference_output_max_abs="
            f"{native_vs_reference_output_max_abs:.6e}"
        )
        print(
            "native_vs_candidate_output_max_abs="
            f"{native_vs_candidate_output_max_abs:.6e}"
        )
        print(
            "factored_vs_reference_output_max_abs="
            f"{factored_vs_reference_output_abs.max().item():.6e}"
        )
        print(
            "factored_vs_candidate_output_max_abs="
            f"{factored_vs_candidate_output_abs.max().item():.6e}"
        )
        print(
            "factored_vs_reference_lse_max_abs="
            f"{factored_vs_reference_lse_abs.max().item():.6e}"
        )
        print(
            "factored_vs_candidate_lse_max_abs="
            f"{factored_vs_candidate_lse_abs.max().item():.6e}"
        )
        print(
            "reference_vs_candidate_lse_max_abs="
            f"{reference_vs_candidate_lse_max_abs:.6e}"
        )
        print(
            "candidate_factorized_lse_max_abs="
            f"{candidate_factorized_lse_max_abs:.6e}"
        )
        print(f"compare_pipeline_stage1_ms_a={comparison.stage1_ms_a:.6f}")
        print(f"compare_pipeline_stage1_ms_b={comparison.stage1_ms_b:.6f}")
        print(f"compare_pipeline_stage2_ms_a={comparison.stage2_ms_a:.6f}")
        print(f"compare_pipeline_stage2_ms_b={comparison.stage2_ms_b:.6f}")
        print(f"compare_pipeline_ms_a={comparison.ms_a:.6f}")
        print(f"compare_pipeline_ms_b={comparison.ms_b:.6f}")
        return

    if args.mode == "dryrun_splitkv":
        plan = build_stage_buffer_plan(
            spec,
            num_kv_splits=args.num_kv_splits,
            output_dtype=dtype,
            query_tile_size_hint=args.query_tile_size_hint,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "dryrun_splitkv",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                "selected_stage1_impl": args.stage1_impl,
                "selected_stage2_impl": args.stage2_impl,
                **stage_buffer_plan_to_dict(plan),
            },
            enabled=args.json,
        ):
            return
        _print_stage_plan(plan)
        return

    if args.mode == "inspect_stage1_impl":
        stage1 = run_stage1(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            reference_inputs=packed_reference_inputs,
            impl=args.stage1_impl,
        )
        metadata = get_stage1_impl_metadata(args.stage1_impl)
        payload = {
            "schema_version": 1,
            "tool": "benchmark_phase2_staged_prototype",
            "mode": "inspect_stage1_impl",
            "shape": hotshape_spec_to_dict(spec, dtype=dtype),
            "selected_stage1_impl": args.stage1_impl,
            **stage1_artifacts_to_dict(stage1),
        }
        if metadata is not None:
            payload.update(metadata)
        if _emit_json(payload, enabled=args.json):
            return
        _print_stage_plan(stage1.plan)
        print(f"stage1_impl={stage1.impl}")
        print(f"stage1_mid_o_shape={tuple(stage1.mid_o.shape)}")
        if metadata is not None:
            for key, value in metadata.items():
                print(f"{key}={value}")
        return

    if args.mode == "packed_factorized_pipeline":
        hotshape = make_phase2_packed_inputs(spec=spec, dtype=dtype, device=args.device)
        plan, merged, merged_lse, out_max_abs, lse_max_abs = (
            run_packed_factorized_pipeline(
                spec=spec,
                hotshape=hotshape,
                num_kv_splits=args.num_kv_splits,
                query_tile_size_hint=args.query_tile_size_hint,
            )
        )
        payload = {
            "schema_version": 1,
            "tool": "benchmark_phase2_staged_prototype",
            "mode": "packed_factorized_pipeline",
            "shape": hotshape_spec_to_dict(spec, dtype=dtype),
            **stage_buffer_plan_to_dict(plan),
            "packed_factorized_output_shape": tuple(merged.shape),
            "packed_factorized_output_dtype": str(merged.dtype),
            "packed_factorized_lse_shape": tuple(merged_lse.shape),
            "packed_factorized_lse_dtype": str(merged_lse.dtype),
            "packed_factorized_out_max_abs": out_max_abs,
            "packed_factorized_lse_max_abs": lse_max_abs,
        }
        if _emit_json(payload, enabled=args.json):
            return
        _print_stage_plan(plan)
        print(f"packed_factorized_output_shape={tuple(merged.shape)}")
        print(f"packed_factorized_output_dtype={merged.dtype}")
        print(f"packed_factorized_lse_shape={tuple(merged_lse.shape)}")
        print(f"packed_factorized_lse_dtype={merged_lse.dtype}")
        print(f"packed_factorized_out_max_abs={out_max_abs:.6e}")
        print(f"packed_factorized_lse_max_abs={lse_max_abs:.6e}")
        return

    stage1 = run_stage1(
        spec=spec,
        dtype=dtype,
        device=args.device,
        num_kv_splits=args.num_kv_splits,
        query_tile_size_hint=args.query_tile_size_hint,
        reference_inputs=packed_reference_inputs,
        impl=args.stage1_impl,
    ) if args.mode in {"reference_reduce", "reference_pipeline"} else None

    if args.mode == "reference_reduce":
        assert stage1 is not None
        stage2 = run_stage2(stage1, impl=args.stage2_impl)
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "reference_reduce",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                "selected_stage1_impl": args.stage1_impl,
                "selected_stage2_impl": args.stage2_impl,
                **stage1_artifacts_to_dict(stage1),
                **stage2_artifacts_to_dict(stage2),
            },
            enabled=args.json,
        ):
            return
        _print_stage_plan(stage1.plan)
        print(f"stage1_impl={stage1.impl}")
        print(f"stage1_mid_o_shape={tuple(stage1.mid_o.shape)}")
        print(f"stage1_mid_o_dtype={stage1.mid_o.dtype}")
        print(f"stage1_full_out_ref_shape={tuple(stage1.full_out_ref.shape)}")
        print(f"stage1_full_out_ref_dtype={stage1.full_out_ref.dtype}")
        print(f"stage1_full_lse_ref_shape={tuple(stage1.full_lse_ref.shape)}")
        print(f"stage1_full_lse_ref_dtype={stage1.full_lse_ref.dtype}")
        print(f"stage2_impl={stage2.impl}")
        print(f"reduced_output_shape={tuple(stage2.output.shape)}")
        print(f"reduced_output_dtype={stage2.output.dtype}")
        print(f"reduced_lse_shape={tuple(stage2.lse.shape)}")
        print(f"reduced_lse_dtype={stage2.lse.dtype}")
        return

    if args.mode == "reference_pipeline":
        pipeline = run_pipeline(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            reference_inputs=packed_reference_inputs,
            stage1_impl=args.stage1_impl,
            stage2_impl=args.stage2_impl,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "reference_pipeline",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                "selected_stage1_impl": args.stage1_impl,
                "selected_stage2_impl": args.stage2_impl,
                **pipeline_artifacts_to_dict(pipeline),
            },
            enabled=args.json,
        ):
            return
        _print_stage_plan(pipeline.stage1.plan)
        print(f"reduced_output_shape={tuple(pipeline.stage2.output.shape)}")
        print(f"reduced_output_dtype={pipeline.stage2.output.dtype}")
        print(f"reduced_lse_shape={tuple(pipeline.stage2.lse.shape)}")
        print(f"reduced_lse_dtype={pipeline.stage2.lse.dtype}")
        print(f"pipeline_out_max_abs={pipeline.out_max_abs:.6e}")
        print(f"pipeline_lse_max_abs={pipeline.lse_max_abs:.6e}")
        return

    if args.mode == "selfcheck_registry":
        result = run_registry_selfcheck(
            spec=spec,
            dtype=dtype,
            device=args.device,
            num_kv_splits=args.num_kv_splits,
            query_tile_size_hint=args.query_tile_size_hint,
            out_tol=args.selfcheck_out_tol,
            lse_tol=args.selfcheck_lse_tol,
        )
        if _emit_json(
            {
                "schema_version": 1,
                "tool": "benchmark_phase2_staged_prototype",
                "mode": "selfcheck_registry",
                "shape": hotshape_spec_to_dict(spec, dtype=dtype),
                "selected_stage1_impl": args.stage1_impl,
                "selected_stage2_impl": args.stage2_impl,
                **registry_selfcheck_to_dict(result),
            },
            enabled=args.json,
        ):
            if result.passed_pairs != result.total_pairs:
                raise RuntimeError("staged registry self-check failed")
            return
        print(f"selfcheck_stage1_impls={result.stage1_impls}")
        print(f"selfcheck_stage2_impls={result.stage2_impls}")
        for pair in result.pair_results:
            print(
                "selfcheck_pair="
                f"{pair.stage1_impl}/{pair.stage2_impl}"
                f" out_max_abs={pair.out_max_abs:.6e}"
                f" lse_max_abs={pair.lse_max_abs:.6e}"
                f" ok={int(pair.ok)}"
            )
        print(f"selfcheck_passed_pairs={result.passed_pairs}")
        print(f"selfcheck_total_pairs={result.total_pairs}")
        if result.passed_pairs != result.total_pairs:
            raise RuntimeError("staged registry self-check failed")
        return

    pipeline = run_pipeline(
        spec=spec,
        dtype=dtype,
        device=args.device,
        num_kv_splits=args.num_kv_splits,
        query_tile_size_hint=args.query_tile_size_hint,
        reference_inputs=packed_reference_inputs,
    )
    if _emit_json(
        {
            "schema_version": 1,
            "tool": "benchmark_phase2_staged_prototype",
            "mode": "reference_parity",
            "shape": hotshape_spec_to_dict(spec, dtype=dtype),
            "selected_stage1_impl": args.stage1_impl,
            "selected_stage2_impl": args.stage2_impl,
            "reference_num_kv_splits": args.num_kv_splits,
            "reference_out_max_abs": pipeline.out_max_abs,
            "reference_lse_max_abs": pipeline.lse_max_abs,
        },
        enabled=args.json,
    ):
        return
    print(f"reference_num_kv_splits={args.num_kv_splits}")
    print(f"reference_out_max_abs={pipeline.out_max_abs:.6e}")
    print(f"reference_lse_max_abs={pipeline.lse_max_abs:.6e}")


if __name__ == "__main__":
    main()
