# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

try:
    from .phase2_hotshape_utils import (
        Phase2HotShapeSpec,
        compiled_metadata_to_dict,
        extract_triton_jit_metadata,
        extract_triton_jit_shared_diagnostics,
    )
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_metadata_getter,
        register_stage1_impl,
    )
    from .phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        StageBufferPlan,
        build_stage_buffer_plan,
        resolve_dense_reference_inputs,
    )
    from vllm.triton_utils import tl, triton
except ImportError:
    from phase2_hotshape_utils import (
        Phase2HotShapeSpec,
        compiled_metadata_to_dict,
        extract_triton_jit_metadata,
        extract_triton_jit_shared_diagnostics,
    )
    from phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_metadata_getter,
        register_stage1_impl,
    )
    from phase2_staged_reference import (
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        StageBufferPlan,
        build_stage_buffer_plan,
        resolve_dense_reference_inputs,
    )
    from vllm.triton_utils import tl, triton


_LAST_QTILE_TRITON_VARIANT: str | None = None


@triton.jit
def _stage1_dense_splitkv_blocked_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mid_o_ptr,
    q_stride_0: tl.int64,
    q_stride_1: tl.int64,
    q_stride_2: tl.int64,
    k_stride_0: tl.int64,
    k_stride_1: tl.int64,
    k_stride_2: tl.int64,
    v_stride_0: tl.int64,
    v_stride_1: tl.int64,
    v_stride_2: tl.int64,
    mid_o_stride_0: tl.int64,
    mid_o_stride_1: tl.int64,
    mid_o_stride_2: tl.int64,
    mid_o_stride_3: tl.int64,
    query_len: tl.int32,
    seq_len: tl.int32,
    split_size: tl.int32,
    softmax_scale: tl.float32,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    q_start = q_tile_idx * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_valid = q_offsets < query_len

    d_offsets = tl.arange(0, HEAD_SIZE_PADDED)
    dv_offsets = tl.arange(0, HEAD_SIZE_V_PADDED)
    d_mask = d_offsets < HEAD_SIZE
    dv_mask = dv_offsets < HEAD_SIZE_V

    q_ptrs = (
        q_ptr
        + q_offsets[:, None] * q_stride_0
        + head_idx * q_stride_1
        + d_offsets[None, :] * q_stride_2
    )
    Q = tl.load(
        q_ptrs,
        mask=q_valid[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    context_len = seq_len - query_len
    q_abs = context_len + q_offsets

    split_start = split_idx * split_size
    split_end = tl.minimum(split_start + split_size, seq_len)

    m_prev = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_SIZE_V], dtype=tl.float32)

    for kv_start in range(split_start, split_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_valid = kv_offsets < split_end
        score_valid = q_valid[:, None] & kv_valid[None, :] & (
            kv_offsets[None, :] <= q_abs[:, None]
        )

        k_ptrs = (
            k_ptr
            + kv_offsets[:, None] * k_stride_0
            + head_idx * k_stride_1
            + d_offsets[None, :] * k_stride_2
        )
        v_ptrs = (
            v_ptr
            + kv_offsets[:, None] * v_stride_0
            + head_idx * v_stride_1
            + dv_offsets[None, :] * v_stride_2
        )
        K = tl.load(
            k_ptrs,
            mask=kv_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        V = tl.load(
            v_ptrs,
            mask=kv_valid[:, None] & dv_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score = tl.dot(Q, tl.trans(K)) * softmax_scale
        score = tl.where(score_valid, score, float("-inf"))

        block_m = tl.max(score, axis=1)
        m_curr = tl.maximum(m_prev, block_m)
        p_prev = tl.exp(m_prev - m_curr) * l_prev
        p_curr = tl.exp(score - m_curr[:, None])
        l_curr = p_prev + tl.sum(p_curr, axis=1)
        safe_l = tl.where(l_curr > 0, l_curr, 1.0)
        block_out = tl.dot((p_curr / safe_l[:, None]).to(tl.float32), V)
        acc = acc * (p_prev / safe_l)[:, None] + block_out
        m_prev = m_curr
        l_prev = l_curr

    lse = m_prev + tl.log(tl.where(l_prev > 0, l_prev, 1.0))
    out_ptrs = (
        mid_o_ptr
        + q_offsets[:, None] * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + dv_offsets[None, :] * mid_o_stride_3
    )
    tl.store(out_ptrs, acc, mask=q_valid[:, None] & dv_mask[None, :])
    lse_ptrs = (
        mid_o_ptr
        + q_offsets * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + HEAD_SIZE_V * mid_o_stride_3
    )
    tl.store(lse_ptrs, lse, mask=q_valid)


@triton.jit
def _stage1_dense_splitkv_scalar_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mid_o_ptr,
    q_stride_0: tl.int64,
    q_stride_1: tl.int64,
    q_stride_2: tl.int64,
    k_stride_0: tl.int64,
    k_stride_1: tl.int64,
    k_stride_2: tl.int64,
    v_stride_0: tl.int64,
    v_stride_1: tl.int64,
    v_stride_2: tl.int64,
    mid_o_stride_0: tl.int64,
    mid_o_stride_1: tl.int64,
    mid_o_stride_2: tl.int64,
    mid_o_stride_3: tl.int64,
    query_len: tl.int32,
    seq_len: tl.int32,
    split_size: tl.int32,
    softmax_scale: tl.float32,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    q_start = q_tile_idx * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_valid = q_offsets < query_len

    d_offsets = tl.arange(0, HEAD_SIZE)
    dv_offsets = tl.arange(0, HEAD_SIZE_V)

    q_ptrs = (
        q_ptr
        + q_offsets[:, None] * q_stride_0
        + head_idx * q_stride_1
        + d_offsets[None, :] * q_stride_2
    )
    Q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32)

    context_len = seq_len - query_len
    q_abs = context_len + q_offsets

    split_start = split_idx * split_size
    split_end = tl.minimum(split_start + split_size, seq_len)

    m_prev = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_SIZE_V], dtype=tl.float32)

    for kv_idx in range(split_start, split_end):
        k_ptrs = (
            k_ptr
            + kv_idx * k_stride_0
            + head_idx * k_stride_1
            + d_offsets * k_stride_2
        )
        v_ptrs = (
            v_ptr
            + kv_idx * v_stride_0
            + head_idx * v_stride_1
            + dv_offsets * v_stride_2
        )
        K = tl.load(k_ptrs).to(tl.float32)
        V = tl.load(v_ptrs).to(tl.float32)
        score = tl.sum(Q * K[None, :], axis=1) * softmax_scale
        score = tl.where(q_valid & (kv_idx <= q_abs), score, float("-inf"))

        m_curr = tl.maximum(m_prev, score)
        p_prev = tl.exp(m_prev - m_curr) * l_prev
        p_curr = tl.exp(score - m_curr)
        l_curr = p_prev + p_curr
        safe_l = tl.where(l_curr > 0, l_curr, 1.0)
        acc = acc * (p_prev / safe_l)[:, None] + V[None, :] * (
            p_curr / safe_l
        )[:, None]
        m_prev = m_curr
        l_prev = l_curr

    lse = m_prev + tl.log(tl.where(l_prev > 0, l_prev, 1.0))
    out_ptrs = (
        mid_o_ptr
        + q_offsets[:, None] * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + dv_offsets[None, :] * mid_o_stride_3
    )
    tl.store(out_ptrs, acc, mask=q_valid[:, None])
    lse_ptrs = (
        mid_o_ptr
        + q_offsets * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + HEAD_SIZE_V * mid_o_stride_3
    )
    tl.store(lse_ptrs, lse, mask=q_valid)


def build_qtile_triton_dense_artifacts(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    if device != "cuda":
        raise NotImplementedError("qtile_triton_dense is only available on CUDA")
    global _LAST_QTILE_TRITON_VARIANT

    inputs = resolve_dense_reference_inputs(
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
    mid_o = torch.empty(plan.mid_o_shape, device=device, dtype=torch.float32)
    grid = (plan.num_query_tiles, spec.num_query_heads, num_kv_splits)
    common_args = (
        inputs.query,
        inputs.key,
        inputs.value,
        mid_o,
        inputs.query.stride(0),
        inputs.query.stride(1),
        inputs.query.stride(2),
        inputs.key.stride(0),
        inputs.key.stride(1),
        inputs.key.stride(2),
        inputs.value.stride(0),
        inputs.value.stride(1),
        inputs.value.stride(2),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        spec.query_len,
        spec.seq_len,
        plan.max_split_tokens,
        spec.softmax_scale,
    )

    use_blocked_kernel = spec.head_size <= 128 and spec.head_size_v <= 128
    if use_blocked_kernel:
        _LAST_QTILE_TRITON_VARIANT = "blocked"

        def replay() -> None:
            _stage1_dense_splitkv_blocked_kernel[grid](
                *common_args,
                HEAD_SIZE=spec.head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(spec.head_size),
                HEAD_SIZE_V=spec.head_size_v,
                HEAD_SIZE_V_PADDED=triton.next_power_of_2(spec.head_size_v),
                BLOCK_Q=plan.query_tile_size_hint,
                BLOCK_KV=16,
                num_warps=4,
                num_stages=1,
            )
    else:
        _LAST_QTILE_TRITON_VARIANT = "scalar"

        def replay() -> None:
            _stage1_dense_splitkv_scalar_kernel[grid](
                *common_args,
                HEAD_SIZE=spec.head_size,
                HEAD_SIZE_V=spec.head_size_v,
                BLOCK_Q=plan.query_tile_size_hint,
                num_warps=4,
                num_stages=1,
            )
    replay()
    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=inputs.full_out_ref,
        full_lse_ref=inputs.full_lse_ref,
        impl="qtile_triton_dense",
        replay=replay,
    )


def _stage1_qtile_triton_dense(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    return build_qtile_triton_dense_artifacts(
        spec,
        dtype,
        device,
        num_kv_splits,
        query_tile_size_hint,
        reference_inputs,
    )


def register_qtile_triton_dense_impl() -> None:
    if not torch.cuda.is_available():
        return
    try:
        register_stage1_impl("qtile_triton_dense", _stage1_qtile_triton_dense)
    except ValueError:
        pass
    try:
        register_stage1_metadata_getter(
            "qtile_triton_dense",
            get_qtile_triton_dense_metadata,
        )
    except ValueError:
        pass


def get_qtile_triton_dense_metadata() -> dict[str, object] | None:
    if _LAST_QTILE_TRITON_VARIANT == "blocked":
        metadata = extract_triton_jit_metadata(_stage1_dense_splitkv_blocked_kernel)
        diagnostics = extract_triton_jit_shared_diagnostics(
            _stage1_dense_splitkv_blocked_kernel
        )
    elif _LAST_QTILE_TRITON_VARIANT == "scalar":
        metadata = extract_triton_jit_metadata(_stage1_dense_splitkv_scalar_kernel)
        diagnostics = extract_triton_jit_shared_diagnostics(
            _stage1_dense_splitkv_scalar_kernel
        )
    else:
        return None
    return {
        "stage1_triton_variant": _LAST_QTILE_TRITON_VARIANT,
        **compiled_metadata_to_dict(metadata),
        **diagnostics,
    }


register_qtile_triton_dense_impl()
