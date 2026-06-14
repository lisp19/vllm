# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

try:
    from .phase2_hotshape_utils import Phase2HotShapeSpec
    from .phase2_staged_kernel_experiment import Stage1Artifacts, register_stage1_impl
    from .phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        allocate_partial_buffers,
        build_stage_buffer_plan,
        expand_packed_factorized_terms,
        pack_stage_buffer,
        resolve_packed_reference_inputs,
    )
except ImportError:
    from phase2_hotshape_utils import Phase2HotShapeSpec
    from phase2_staged_kernel_experiment import Stage1Artifacts, register_stage1_impl
    from phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        allocate_partial_buffers,
        build_stage_buffer_plan,
        expand_packed_factorized_terms,
        pack_stage_buffer,
        resolve_packed_reference_inputs,
    )


def materialize_qtile_packed_factorized_reference_partials(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None = None,
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_inputs = resolve_packed_reference_inputs(
        spec=spec,
        dtype=dtype,
        device=device,
        reference_inputs=reference_inputs,
    )
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    partial_out, partial_lse = allocate_partial_buffers(
        plan,
        device=packed_inputs.dense_inputs.query.device,
    )
    query, k_int, v_int, k_scales, v_scales = expand_packed_factorized_terms(
        spec=spec,
        hotshape=packed_inputs.hotshape,
    )
    context_len = spec.seq_len - spec.query_len
    q_abs = context_len + torch.arange(spec.query_len, device=query.device)
    k_abs = torch.arange(spec.seq_len, device=query.device)

    for split_idx, (k_start, k_stop) in enumerate(plan.split_ranges):
        split_k = k_int[k_start:k_stop]
        split_v = v_int[k_start:k_stop] * v_scales[k_start:k_stop].unsqueeze(-1)
        split_scale = k_scales[k_start:k_stop].transpose(0, 1).unsqueeze(0)
        split_mask = k_abs[k_start:k_stop].view(1, -1) <= q_abs.view(spec.query_len, 1)
        for q_start, q_stop in plan.query_tile_ranges:
            query_tile = query[q_start:q_stop]
            tile_mask = split_mask[q_start:q_stop]
            tile_scores = torch.einsum("thd,shd->ths", query_tile, split_k)
            tile_scores.mul_(spec.softmax_scale)
            tile_scores.mul_(split_scale)
            tile_scores = torch.where(
                tile_mask[:, None, :],
                tile_scores,
                torch.full_like(tile_scores, float("-inf")),
            )
            valid = torch.isfinite(tile_scores).any(dim=-1)
            tile_lse = torch.where(
                valid,
                torch.logsumexp(tile_scores, dim=-1),
                torch.full(
                    (q_stop - q_start, spec.num_query_heads),
                    float("-inf"),
                    device=query.device,
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
        packed_inputs.factorized_out_ref,
        packed_inputs.factorized_lse_ref,
    )


def build_qtile_packed_factorized_reference_artifacts(
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
        materialize_qtile_packed_factorized_reference_partials(
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


def _stage1_qtile_packed_factorized_reference(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    return build_qtile_packed_factorized_reference_artifacts(
        spec,
        dtype,
        device,
        num_kv_splits,
        query_tile_size_hint,
        reference_inputs,
        impl_name="qtile_packed_factorized_reference",
    )


def register_qtile_packed_factorized_reference_impl() -> None:
    try:
        register_stage1_impl(
            "qtile_packed_factorized_reference",
            _stage1_qtile_packed_factorized_reference,
        )
    except ValueError:
        pass


register_qtile_packed_factorized_reference_impl()
