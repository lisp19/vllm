# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

try:
    from .phase2_staged_impls_merge_tree_reference import (
        build_merge_tree_reference_artifacts,
    )
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        Stage2Artifacts,
        register_stage2_impl,
    )
except ImportError:
    from phase2_staged_impls_merge_tree_reference import build_merge_tree_reference_artifacts
    from phase2_staged_kernel_experiment import Stage1Artifacts, Stage2Artifacts, register_stage2_impl


def _stage2_merge_stub(stage1: Stage1Artifacts) -> Stage2Artifacts:
    # Intentional scaffold for a future non-reference stage2 merge/reduction
    # implementation. Today it delegates to the more kernel-shaped tree-style
    # host reference while preserving its own target-named stage2 identity.
    return build_merge_tree_reference_artifacts(
        stage1,
        impl_name="merge_stub",
    )


def register_merge_stub_impl() -> None:
    try:
        register_stage2_impl("merge_stub", _stage2_merge_stub)
    except ValueError:
        pass


register_merge_stub_impl()
