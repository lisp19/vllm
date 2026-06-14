# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

try:
    from .phase2_hotshape_utils import Phase2HotShapeSpec
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        Stage2Artifacts,
        register_stage1_impl,
        register_stage2_impl,
    )
    from .phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        materialize_reference_stage_buffers,
        materialize_reference_stage_buffers_from_inputs,
        materialize_packed_factorized_partials_only_into,
        materialize_packed_factorized_stage_buffers,
        reduce_reference_stage_buffer_into,
        reduce_reference_stage_buffer,
        resolve_dense_reference_inputs,
        resolve_packed_reference_inputs,
    )
except ImportError:
    from phase2_hotshape_utils import Phase2HotShapeSpec
    from phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        Stage2Artifacts,
        register_stage1_impl,
        register_stage2_impl,
    )
    from phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        materialize_reference_stage_buffers,
        materialize_reference_stage_buffers_from_inputs,
        materialize_packed_factorized_partials_only_into,
        materialize_packed_factorized_stage_buffers,
        reduce_reference_stage_buffer_into,
        reduce_reference_stage_buffer,
        resolve_dense_reference_inputs,
        resolve_packed_reference_inputs,
    )


def _stage1_reference(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    if reference_inputs is None:
        plan, mid_o, full_out_ref, full_lse_ref = materialize_reference_stage_buffers(
            spec=spec,
            dtype=dtype,
            device=device,
            num_kv_splits=num_kv_splits,
            query_tile_size_hint=query_tile_size_hint,
        )
    else:
        dense_inputs = resolve_dense_reference_inputs(
            spec=spec,
            dtype=dtype,
            device=device,
            reference_inputs=reference_inputs,
            prefer_high_precision=True,
        )
        plan, mid_o, full_out_ref, full_lse_ref = (
            materialize_reference_stage_buffers_from_inputs(
                spec=spec,
                reference_inputs=dense_inputs,
                num_kv_splits=num_kv_splits,
                query_tile_size_hint=query_tile_size_hint,
            )
        )
    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=full_out_ref,
        full_lse_ref=full_lse_ref,
        impl="reference",
    )


def _stage1_packed_factorized_reference(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    packed_inputs = resolve_packed_reference_inputs(
        spec=spec,
        dtype=dtype,
        device=device,
        reference_inputs=reference_inputs,
    )
    plan, mid_o, full_out_ref, full_lse_ref = materialize_packed_factorized_stage_buffers(
        spec=spec,
        hotshape=packed_inputs.hotshape,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
    )
    partial_out = mid_o[..., :-1]
    partial_lse = mid_o[..., -1].permute(2, 1, 0)

    def replay() -> None:
        materialize_packed_factorized_partials_only_into(
            spec=spec,
            hotshape=packed_inputs.hotshape,
            plan=plan,
            partial_out=partial_out,
            partial_lse=partial_lse,
        )

    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=full_out_ref,
        full_lse_ref=full_lse_ref,
        impl="packed_factorized_reference",
        replay=replay,
    )


def _stage2_reference(stage1: Stage1Artifacts) -> Stage2Artifacts:
    output, lse = reduce_reference_stage_buffer(stage1.mid_o)

    def replay() -> None:
        reduce_reference_stage_buffer_into(stage1.mid_o, output, lse)

    return Stage2Artifacts(output=output, lse=lse, impl="reference", replay=replay)


def register_reference_impls() -> None:
    # Idempotent registration when imported multiple times within one process.
    try:
        register_stage1_impl("reference", _stage1_reference)
    except ValueError:
        pass
    try:
        register_stage1_impl(
            "packed_factorized_reference",
            _stage1_packed_factorized_reference,
        )
    except ValueError:
        pass
    try:
        register_stage2_impl("reference", _stage2_reference)
    except ValueError:
        pass


register_reference_impls()
