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
        reduce_reference_stage_buffer_into,
        reduce_reference_stage_buffer,
        resolve_dense_reference_inputs,
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
        reduce_reference_stage_buffer_into,
        reduce_reference_stage_buffer,
        resolve_dense_reference_inputs,
    )


def _stage1_passthrough(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    # Experimental scaffold:
    # replace the body of this function with a future non-reference stage1
    # kernel-family implementation, while preserving the returned contract.
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
        impl="passthrough",
    )


def _stage2_passthrough(stage1: Stage1Artifacts) -> Stage2Artifacts:
    # Experimental scaffold:
    # replace this with a future non-reference stage2 merge/reduction if needed.
    output, lse = reduce_reference_stage_buffer(stage1.mid_o)

    def replay() -> None:
        reduce_reference_stage_buffer_into(stage1.mid_o, output, lse)

    return Stage2Artifacts(output=output, lse=lse, impl="passthrough", replay=replay)


def register_passthrough_impls() -> None:
    try:
        register_stage1_impl("passthrough", _stage1_passthrough)
    except ValueError:
        pass
    try:
        register_stage2_impl("passthrough", _stage2_passthrough)
    except ValueError:
        pass


register_passthrough_impls()
