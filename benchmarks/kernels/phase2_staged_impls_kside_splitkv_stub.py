# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

try:
    from .phase2_hotshape_utils import Phase2HotShapeSpec
    from .phase2_staged_impls_qtile_packed_factorized_reference import (
        materialize_qtile_packed_factorized_reference_partials,
    )
    from .phase2_staged_impls_qtile_reference import materialize_qtile_reference_partials
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_impl,
    )
    from .phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        pack_stage_buffer,
    )
except ImportError:
    from phase2_hotshape_utils import Phase2HotShapeSpec
    from phase2_staged_impls_qtile_packed_factorized_reference import (
        materialize_qtile_packed_factorized_reference_partials,
    )
    from phase2_staged_impls_qtile_reference import materialize_qtile_reference_partials
    from phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_impl,
    )
    from phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        pack_stage_buffer,
    )


def _stage1_kside_splitkv_stub(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    # Intentional scaffold for the most credible next phase-2 direction:
    # a new stage1 kernel family that changes K-side shared staging while
    # preserving more of the native single-pass arithmetic efficiency.
    #
    # Today this delegates to the more kernel-shaped qtile partial generator
    # and then packs the canonical stage buffer explicitly, preserving its own
    # target-named implementation identity. Replace this body when the first
    # real non-reference stage1 kernel is ready.
    partial_builder = (
        materialize_qtile_packed_factorized_reference_partials
        if isinstance(reference_inputs, PackedReferenceInputs)
        else materialize_qtile_reference_partials
    )
    plan, partial_out, partial_lse, full_out_ref, full_lse_ref = partial_builder(
        spec,
        dtype,
        device,
        num_kv_splits,
        query_tile_size_hint,
        reference_inputs,
    )
    mid_o = pack_stage_buffer(partial_out, partial_lse)
    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=full_out_ref,
        full_lse_ref=full_lse_ref,
        impl="kside_splitkv_stub",
    )


def register_kside_splitkv_stub_impl() -> None:
    try:
        register_stage1_impl("kside_splitkv_stub", _stage1_kside_splitkv_stub)
    except ValueError:
        pass


register_kside_splitkv_stub_impl()
