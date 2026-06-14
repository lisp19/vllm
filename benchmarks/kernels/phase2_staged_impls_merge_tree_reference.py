# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

try:
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        Stage2Artifacts,
        register_stage2_impl,
    )
    from .phase2_staged_reference import (
        allocate_stage2_outputs,
        merge_two_partials_torch,
        unpack_stage_buffer,
    )
except ImportError:
    from phase2_staged_kernel_experiment import Stage1Artifacts, Stage2Artifacts, register_stage2_impl
    from phase2_staged_reference import (
        allocate_stage2_outputs,
        merge_two_partials_torch,
        unpack_stage_buffer,
    )


def build_merge_tree_reference_artifacts(
    stage1: Stage1Artifacts,
    *,
    impl_name: str,
) -> Stage2Artifacts:
    output, lse = allocate_stage2_outputs(stage1.plan, device=stage1.mid_o.device)

    def replay() -> None:
        # More kernel-shaped host reference: reduce split partials in a pairwise
        # tree rather than a strict left fold.
        partial_out, partial_lse = unpack_stage_buffer(stage1.mid_o)
        out_nodes = [
            partial_out[:, :, idx, :].contiguous() for idx in range(partial_out.shape[2])
        ]
        lse_nodes = [
            partial_lse[idx].contiguous() for idx in range(partial_lse.shape[0])
        ]

        while len(out_nodes) > 1:
            next_out_nodes = []
            next_lse_nodes = []
            it = iter(range(0, len(out_nodes), 2))
            for start_idx in it:
                if start_idx + 1 >= len(out_nodes):
                    next_out_nodes.append(out_nodes[start_idx])
                    next_lse_nodes.append(lse_nodes[start_idx])
                    continue
                merged_out, merged_lse = merge_two_partials_torch(
                    out_nodes[start_idx],
                    lse_nodes[start_idx],
                    out_nodes[start_idx + 1],
                    lse_nodes[start_idx + 1],
                )
                next_out_nodes.append(merged_out)
                next_lse_nodes.append(merged_lse)
            out_nodes = next_out_nodes
            lse_nodes = next_lse_nodes
        output.copy_(out_nodes[0])
        lse.copy_(lse_nodes[0])

    replay()
    return Stage2Artifacts(output=output, lse=lse, impl=impl_name, replay=replay)


def _stage2_merge_tree_reference(stage1: Stage1Artifacts) -> Stage2Artifacts:
    return build_merge_tree_reference_artifacts(
        stage1,
        impl_name="merge_tree_reference",
    )


def register_merge_tree_reference_impl() -> None:
    try:
        register_stage2_impl("merge_tree_reference", _stage2_merge_tree_reference)
    except ValueError:
        pass


register_merge_tree_reference_impl()
