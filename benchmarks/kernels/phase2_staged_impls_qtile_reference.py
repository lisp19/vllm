# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import torch

try:
    from .phase2_hotshape_utils import Phase2HotShapeSpec
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_impl,
    )
    from .phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        allocate_partial_buffers,
        build_stage_buffer_plan,
        pack_stage_buffer,
        resolve_dense_reference_inputs,
    )
except ImportError:
    from phase2_hotshape_utils import Phase2HotShapeSpec
    from phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_impl,
    )
    from phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        allocate_partial_buffers,
        build_stage_buffer_plan,
        pack_stage_buffer,
        resolve_dense_reference_inputs,
    )


def materialize_qtile_reference_partials(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # A more kernel-shaped reference than the fully vectorized implementation:
    # iterate explicitly over query tiles and KV splits using the same planning
    # metadata future Triton kernels are expected to consume.
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    partial_out, partial_lse = allocate_partial_buffers(plan, device=device)
    inputs = resolve_dense_reference_inputs(
        spec=spec,
        dtype=dtype,
        device=device,
        reference_inputs=reference_inputs,
        prefer_high_precision=True,
    )

    for split_idx, (k_start, k_stop) in enumerate(plan.split_ranges):
        split_scores = inputs.full_scores[..., k_start:k_stop]
        split_v = inputs.value[k_start:k_stop].float()
        for q_start, q_stop in plan.query_tile_ranges:
            tile_scores = split_scores[q_start:q_stop]
            valid = torch.isfinite(tile_scores).any(dim=-1)  # [T_tile, H]
            tile_lse = torch.where(
                valid,
                torch.logsumexp(tile_scores, dim=-1),
                torch.full(
                    (q_stop - q_start, spec.num_query_heads),
                    float("-inf"),
                    device=inputs.query.device,
                    dtype=torch.float32,
                ),
            )
            safe_scores = torch.where(
                valid[..., None],
                tile_scores,
                torch.zeros_like(tile_scores),
            )
            tile_prob = torch.softmax(safe_scores, dim=-1)
            tile_prob = torch.where(valid[..., None], tile_prob, torch.zeros_like(tile_prob))
            tile_out = torch.einsum("ths,shd->thd", tile_prob, split_v)
            partial_out[q_start:q_stop, :, split_idx, :].copy_(tile_out)
            partial_lse[split_idx, :, q_start:q_stop].copy_(tile_lse.transpose(0, 1))

    return (
        plan,
        partial_out,
        partial_lse,
        inputs.full_out_ref,
        inputs.full_lse_ref,
    )


def build_qtile_reference_artifacts(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
    *,
    impl_name: str,
) -> Stage1Artifacts:
    plan, partial_out, partial_lse, full_out_ref, full_lse_ref = (
        materialize_qtile_reference_partials(
            spec,
            dtype,
            device,
            num_kv_splits,
            query_tile_size_hint,
            reference_inputs,
        )
    )
    mid_o = pack_stage_buffer(partial_out, partial_lse)
    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=full_out_ref,
        full_lse_ref=full_lse_ref,
        impl=impl_name,
    )


def _stage1_qtile_reference(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    return build_qtile_reference_artifacts(
        spec,
        dtype,
        device,
        num_kv_splits,
        query_tile_size_hint,
        reference_inputs,
        impl_name="qtile_reference",
    )


def register_qtile_reference_impl() -> None:
    try:
        register_stage1_impl("qtile_reference", _stage1_qtile_reference)
    except ValueError:
        pass


register_qtile_reference_impl()
