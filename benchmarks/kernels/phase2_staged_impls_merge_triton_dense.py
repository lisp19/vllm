# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

try:
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        Stage2Artifacts,
        register_stage2_impl,
    )
    from .phase2_staged_reference import allocate_stage2_outputs
    from vllm.triton_utils import tl, triton
except ImportError:
    from phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        Stage2Artifacts,
        register_stage2_impl,
    )
    from phase2_staged_reference import allocate_stage2_outputs
    from vllm.triton_utils import tl, triton


@triton.jit
def _merge_triton_dense_kernel(
    mid_o_ptr,
    out_ptr,
    lse_ptr,
    stride_mid_t: tl.int64,
    stride_mid_h: tl.int64,
    stride_mid_s: tl.int64,
    stride_out_t: tl.int64,
    stride_out_h: tl.int64,
    stride_lse_h: tl.int64,
    num_splits: tl.int32,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    mask_d = offs_d < HEAD_SIZE

    running_out = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)
    running_lse = float("-inf")

    for split_idx in range(0, num_splits):
        cur_ptr = (
            mid_o_ptr
            + token_idx * stride_mid_t
            + head_idx * stride_mid_h
            + split_idx * stride_mid_s
        )
        cur_out = tl.load(cur_ptr + offs_d, mask=mask_d, other=0.0)
        cur_lse = tl.load(cur_ptr + HEAD_SIZE)

        both_invalid = (running_lse == float("-inf")) & (cur_lse == float("-inf"))
        max_lse = tl.maximum(running_lse, cur_lse)
        safe_max = tl.where(both_invalid, 0.0, max_lse)
        p_se = tl.where(
            running_lse == float("-inf"),
            0.0,
            tl.exp(running_lse - safe_max),
        )
        s_se = tl.where(
            cur_lse == float("-inf"),
            0.0,
            tl.exp(cur_lse - safe_max),
        )
        out_se = p_se + s_se
        p_scale = p_se / out_se
        s_scale = s_se / out_se
        merged = running_out * p_scale + cur_out * s_scale
        running_out = tl.where(mask_d, tl.where(both_invalid, running_out, merged), 0.0)
        running_lse = tl.where(both_invalid, running_lse, tl.log(out_se) + safe_max)

    out_base = out_ptr + token_idx * stride_out_t + head_idx * stride_out_h
    tl.store(out_base + offs_d, running_out, mask=mask_d)
    tl.store(lse_ptr + head_idx * stride_lse_h + token_idx, running_lse)


def _stage2_merge_triton_dense(stage1: Stage1Artifacts) -> Stage2Artifacts:
    if stage1.mid_o.device.type != "cuda":
        raise NotImplementedError("merge_triton_dense is only available on CUDA")

    head_size = stage1.plan.output_shape[2]
    head_size_padded = triton.next_power_of_2(head_size)
    output, lse = allocate_stage2_outputs(stage1.plan, device=stage1.mid_o.device)
    grid = (stage1.plan.output_shape[0], stage1.plan.output_shape[1])

    def replay() -> None:
        _merge_triton_dense_kernel[grid](
            stage1.mid_o,
            output,
            lse,
            stage1.mid_o.stride(0),
            stage1.mid_o.stride(1),
            stage1.mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            stage1.plan.num_kv_splits,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            num_warps=4,
            num_stages=1,
        )

    replay()
    return Stage2Artifacts(
        output=output,
        lse=lse,
        impl="merge_triton_dense",
        replay=replay,
    )


def register_merge_triton_dense_impl() -> None:
    if not torch.cuda.is_available():
        return
    try:
        register_stage2_impl("merge_triton_dense", _stage2_merge_triton_dense)
    except ValueError:
        pass


register_merge_triton_dense_impl()
