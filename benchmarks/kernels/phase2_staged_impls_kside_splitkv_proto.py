# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import torch

try:
    from .phase2_hotshape_utils import (
        Phase2HotShapeSpec,
        compiled_metadata_to_dict,
        extract_triton_jit_metadata,
        extract_triton_jit_shared_diagnostics,
    )
    from .phase2_staged_impls_qtile_packed_factorized_reference import (
        build_qtile_packed_factorized_reference_artifacts,
    )
    from .phase2_staged_impls_qtile_triton_dense import (
        build_qtile_triton_dense_artifacts,
    )
    from .phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_metadata_getter,
        register_stage1_impl,
    )
    from .phase2_staged_reference import (
        allocate_stage_buffer,
        build_stage_buffer_plan,
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        StageBufferPlan,
        resolve_packed_reference_inputs,
    )
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.ops.triton_attention_helpers import softmax_step
    from vllm.v1.attention.ops.triton_packed_int_kv import (
        _decode_packed_signed,
        _load_packed_k_tile,
        _load_packed_v_tile,
    )
except ImportError:
    from phase2_hotshape_utils import (
        Phase2HotShapeSpec,
        compiled_metadata_to_dict,
        extract_triton_jit_metadata,
        extract_triton_jit_shared_diagnostics,
    )
    from phase2_staged_impls_qtile_packed_factorized_reference import (
        build_qtile_packed_factorized_reference_artifacts,
    )
    from phase2_staged_impls_qtile_triton_dense import (
        build_qtile_triton_dense_artifacts,
    )
    from phase2_staged_kernel_experiment import (
        Stage1Artifacts,
        register_stage1_metadata_getter,
        register_stage1_impl,
    )
    from phase2_staged_reference import (
        allocate_stage_buffer,
        build_stage_buffer_plan,
        PackedReferenceInputs,
        ReferenceAttentionInputs,
        StageBufferPlan,
        resolve_packed_reference_inputs,
    )
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.ops.triton_attention_helpers import softmax_step
    from vllm.v1.attention.ops.triton_packed_int_kv import (
        _decode_packed_signed,
        _load_packed_k_tile,
        _load_packed_v_tile,
    )


_LAST_KSIDE_SPLITKV_TRITON_VARIANT: str | None = None


def build_kside_splitkv_stage1_grid(plan: StageBufferPlan) -> tuple[int, int, int]:
    """Return the canonical stage1 launch-grid shape for the current plan.

    The intended future Triton kernel decomposition is:
    - x: query tiles
    - y: query heads
    - z: KV splits
    """

    return (
        plan.num_query_tiles,
        plan.output_shape[1],
        plan.num_kv_splits,
    )


def build_kside_splitkv_packed_runtime_grid(
    *,
    query_len: int,
    num_query_heads: int,
    num_kv_splits: int,
    block_q_internal: int,
) -> tuple[int, int, int]:
    return (
        math.ceil(query_len / block_q_internal),
        num_query_heads,
        num_kv_splits,
    )


@triton.jit
def _stage1_packed_splitkv_scalar_kernel(
    mid_o_ptr,
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    q_stride_0: tl.int64,
    q_stride_1: tl.int64,
    q_stride_2: tl.int64,
    mid_o_stride_0: tl.int64,
    mid_o_stride_1: tl.int64,
    mid_o_stride_2: tl.int64,
    mid_o_stride_3: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_len: tl.int32,
    seq_len: tl.int32,
    num_queries_per_kv: tl.int32,
    split_size: tl.int32,
    softmax_scale: tl.float32,
    BLOCK_Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    q_offsets = q_tile_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_valid = q_offsets < query_len
    d_offsets = tl.arange(0, HEAD_SIZE_PADDED)
    dv_offsets = tl.arange(0, HEAD_SIZE_V_PADDED)
    q_ptrs = (
        q_ptr
        + q_offsets[:, None] * q_stride_0
        + head_idx * q_stride_1
        + d_offsets[None, :] * q_stride_2
    )
    q_mask = q_valid[:, None] & (d_offsets[None, :] < HEAD_SIZE)
    Q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    kv_head_idx = head_idx // num_queries_per_kv
    context_len = seq_len - query_len
    q_abs = context_len + q_offsets
    split_start = split_idx * split_size
    split_end = tl.minimum(split_start + split_size, seq_len)
    num_tiles = (split_end - split_start + TILE_SIZE - 1) // TILE_SIZE

    m_prev = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_SIZE_V_PADDED], dtype=tl.float32)
    offs_t = tl.arange(0, TILE_SIZE)

    for tile_idx in range(0, num_tiles):
        seq_offset = split_start + tile_idx * TILE_SIZE + offs_t
        tile_mask = seq_offset < split_end
        slot_in_block = seq_offset % BLOCK_SIZE
        physical_block_idx = tl.load(
            block_table_ptr + seq_offset // BLOCK_SIZE,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)
        k_scale_idx = (
            physical_block_idx * stride_ks_blk
            + slot_in_block * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        v_scale_idx = (
            physical_block_idx * stride_vs_blk
            + slot_in_block * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        k_token_head_scales = tl.load(
            k_scale_cache_ptr + k_scale_idx,
            mask=tile_mask,
            other=1.0,
        ).to(tl.float32)
        v_token_head_scales = tl.load(
            v_scale_cache_ptr + v_scale_idx,
            mask=tile_mask,
            other=1.0,
        ).to(tl.float32)
        K = _load_packed_k_tile(
            key_cache_ptr,
            physical_block_idx,
            kv_head_idx,
            seq_offset,
            tile_mask,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            BLOCK_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            K_BITS,
        ).to(tl.float32)
        V = _load_packed_v_tile(
            value_cache_ptr,
            physical_block_idx,
            kv_head_idx,
            seq_offset,
            tile_mask,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            BLOCK_SIZE,
            HEAD_SIZE_V,
            HEAD_SIZE_V_PADDED,
            V_BITS,
        ).to(tl.float32)
        causal_mask = q_valid[:, None] & tile_mask[None, :] & (
            q_abs[:, None] >= seq_offset[None, :]
        )
        score = tl.dot(Q, K) * (softmax_scale * k_token_head_scales[None, :])
        score = tl.where(causal_mask, score, float("-inf"))
        m_curr, l_curr, p_curr, alpha = softmax_step(score, m_prev, l_prev)
        if TILE_SIZE >= 16:
            block_out = tl.dot((p_curr * v_token_head_scales[None, :]).to(V.dtype), V)
        else:
            block_out = tl.sum(
                (p_curr * v_token_head_scales[None, :])[:, :, None] * V[None, :, :],
                axis=1,
            )
        acc = acc * alpha[:, None] + block_out
        m_prev = m_curr
        l_prev = l_curr

    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    acc = acc / safe_l[:, None]
    lse = m_prev + tl.log(safe_l)
    out_ptrs = (
        mid_o_ptr
        + q_offsets[:, None] * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + dv_offsets[None, :] * mid_o_stride_3
    )
    out_mask = q_valid[:, None] & (dv_offsets[None, :] < HEAD_SIZE_V)
    tl.store(out_ptrs, acc, mask=out_mask)
    lse_ptrs = (
        mid_o_ptr
        + q_offsets * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + HEAD_SIZE_V * mid_o_stride_3
    )
    tl.store(lse_ptrs, lse, mask=q_valid)


@triton.jit
def _load_packed_k_chunk_tile(
    key_cache_ptr,
    physical_block_idx,
    kv_head_idx,
    seq_offset,
    tile_mask,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    d_offsets,
    BLOCK_SIZE: tl.constexpr,
    K_BITS: tl.constexpr,
):
    slot_in_block = seq_offset % BLOCK_SIZE
    byte_base = (
        physical_block_idx[None, :] * stride_k_cache_0
        + slot_in_block[None, :] * stride_k_cache_1
        + kv_head_idx * stride_k_cache_2
    )
    decode_mask = tile_mask[None, :]
    decoded = _decode_packed_signed(
        key_cache_ptr,
        byte_base,
        stride_k_cache_3,
        K_BITS,
        d_offsets[:, None],
        decode_mask,
    )
    return tl.where(decode_mask, decoded, 0.0)


@triton.jit
def _load_packed_v_chunk_tile(
    value_cache_ptr,
    physical_block_idx,
    kv_head_idx,
    seq_offset,
    tile_mask,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    dv_offsets,
    BLOCK_SIZE: tl.constexpr,
    V_BITS: tl.constexpr,
):
    slot_in_block = seq_offset % BLOCK_SIZE
    byte_base = (
        physical_block_idx[:, None] * stride_v_cache_0
        + slot_in_block[:, None] * stride_v_cache_1
        + kv_head_idx * stride_v_cache_2
    )
    decode_mask = tile_mask[:, None]
    decoded = _decode_packed_signed(
        value_cache_ptr,
        byte_base,
        stride_v_cache_3,
        V_BITS,
        dv_offsets[None, :],
        decode_mask,
    )
    return tl.where(decode_mask, decoded, 0.0)


@triton.jit
def _stage1_packed_splitkv_chunkedk_kernel(
    mid_o_ptr,
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    q_stride_0: tl.int64,
    q_stride_1: tl.int64,
    q_stride_2: tl.int64,
    mid_o_stride_0: tl.int64,
    mid_o_stride_1: tl.int64,
    mid_o_stride_2: tl.int64,
    mid_o_stride_3: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_len: tl.int32,
    seq_len: tl.int32,
    num_queries_per_kv: tl.int32,
    split_size: tl.int32,
    softmax_scale: tl.float32,
    BLOCK_Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    q_offsets = q_tile_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_valid = q_offsets < query_len
    dv_offsets = tl.arange(0, HEAD_SIZE_V_PADDED)
    kv_head_idx = head_idx // num_queries_per_kv
    context_len = seq_len - query_len
    q_abs = context_len + q_offsets
    split_start = split_idx * split_size
    split_end = tl.minimum(split_start + split_size, seq_len)
    num_tiles = (split_end - split_start + TILE_SIZE - 1) // TILE_SIZE

    m_prev = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_SIZE_V_PADDED], dtype=tl.float32)
    offs_t = tl.arange(0, TILE_SIZE)

    for tile_idx in range(0, num_tiles):
        seq_offset = split_start + tile_idx * TILE_SIZE + offs_t
        tile_mask = seq_offset < split_end
        slot_in_block = seq_offset % BLOCK_SIZE
        physical_block_idx = tl.load(
            block_table_ptr + seq_offset // BLOCK_SIZE,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)
        k_scale_idx = (
            physical_block_idx * stride_ks_blk
            + slot_in_block * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        v_scale_idx = (
            physical_block_idx * stride_vs_blk
            + slot_in_block * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        k_token_head_scales = tl.load(
            k_scale_cache_ptr + k_scale_idx,
            mask=tile_mask,
            other=1.0,
        ).to(tl.float32)
        v_token_head_scales = tl.load(
            v_scale_cache_ptr + v_scale_idx,
            mask=tile_mask,
            other=1.0,
        ).to(tl.float32)

        score = tl.zeros([BLOCK_Q, TILE_SIZE], dtype=tl.float32)
        for d_base in range(0, HEAD_SIZE, BLOCK_D):
            d_offsets = d_base + tl.arange(0, BLOCK_D)
            q_ptrs = (
                q_ptr
                + q_offsets[:, None] * q_stride_0
                + head_idx * q_stride_1
                + d_offsets[None, :] * q_stride_2
            )
            q_mask = q_valid[:, None] & (d_offsets[None, :] < HEAD_SIZE)
            Q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            K = _load_packed_k_chunk_tile(
                key_cache_ptr,
                physical_block_idx,
                kv_head_idx,
                seq_offset,
                tile_mask,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                d_offsets,
                BLOCK_SIZE,
                K_BITS,
            ).to(tl.float32)
            score += tl.dot(Q, K)

        V = _load_packed_v_tile(
            value_cache_ptr,
            physical_block_idx,
            kv_head_idx,
            seq_offset,
            tile_mask,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            BLOCK_SIZE,
            HEAD_SIZE_V,
            HEAD_SIZE_V_PADDED,
            V_BITS,
        ).to(tl.float32)
        causal_mask = q_valid[:, None] & tile_mask[None, :] & (
            q_abs[:, None] >= seq_offset[None, :]
        )
        score = score * (softmax_scale * k_token_head_scales[None, :])
        score = tl.where(causal_mask, score, float("-inf"))
        block_m = tl.max(score, axis=1)
        m_curr = tl.maximum(m_prev, block_m)
        p_prev = tl.exp(m_prev - m_curr) * l_prev
        p_curr = tl.exp(score - m_curr[:, None])
        l_curr = p_prev + tl.sum(p_curr, axis=1)
        safe_l = tl.where(l_curr > 0, l_curr, 1.0)
        weighted_p = (p_curr * v_token_head_scales[None, :]) / safe_l[:, None]
        if TILE_SIZE >= 16:
            block_out = tl.dot(weighted_p.to(tl.float32), V)
        else:
            block_out = tl.sum(weighted_p[:, :, None] * V[None, :, :], axis=1)
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
    out_mask = q_valid[:, None] & (dv_offsets[None, :] < HEAD_SIZE_V)
    tl.store(out_ptrs, acc, mask=out_mask)
    lse_ptrs = (
        mid_o_ptr
        + q_offsets * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + HEAD_SIZE_V * mid_o_stride_3
    )
    tl.store(lse_ptrs, lse, mask=q_valid)


@triton.jit
def _stage1_packed_splitkv_chunkedkv_kernel(
    mid_o_ptr,
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    q_stride_0: tl.int64,
    q_stride_1: tl.int64,
    q_stride_2: tl.int64,
    mid_o_stride_0: tl.int64,
    mid_o_stride_1: tl.int64,
    mid_o_stride_2: tl.int64,
    mid_o_stride_3: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_len: tl.int32,
    seq_len: tl.int32,
    num_queries_per_kv: tl.int32,
    split_size: tl.int32,
    softmax_scale: tl.float32,
    BLOCK_Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_DV_CHUNKS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_chunk_idx = tl.program_id(2)
    split_idx = split_chunk_idx // NUM_DV_CHUNKS
    dv_chunk_idx = split_chunk_idx % NUM_DV_CHUNKS

    q_offsets = q_tile_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_valid = q_offsets < query_len
    dv_offsets = dv_chunk_idx * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_valid = dv_offsets < HEAD_SIZE_V
    kv_head_idx = head_idx // num_queries_per_kv
    context_len = seq_len - query_len
    q_abs = context_len + q_offsets
    split_start = split_idx * split_size
    split_end = tl.minimum(split_start + split_size, seq_len)
    num_tiles = (split_end - split_start + TILE_SIZE - 1) // TILE_SIZE

    m_prev = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_DV], dtype=tl.float32)
    offs_t = tl.arange(0, TILE_SIZE)

    for tile_idx in range(0, num_tiles):
        seq_offset = split_start + tile_idx * TILE_SIZE + offs_t
        tile_mask = seq_offset < split_end
        slot_in_block = seq_offset % BLOCK_SIZE
        physical_block_idx = tl.load(
            block_table_ptr + seq_offset // BLOCK_SIZE,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)
        k_scale_idx = (
            physical_block_idx * stride_ks_blk
            + slot_in_block * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        v_scale_idx = (
            physical_block_idx * stride_vs_blk
            + slot_in_block * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        k_token_head_scales = tl.load(
            k_scale_cache_ptr + k_scale_idx,
            mask=tile_mask,
            other=1.0,
        ).to(tl.float32)
        v_token_head_scales = tl.load(
            v_scale_cache_ptr + v_scale_idx,
            mask=tile_mask,
            other=1.0,
        ).to(tl.float32)

        score = tl.zeros([BLOCK_Q, TILE_SIZE], dtype=tl.float32)
        for d_base in range(0, HEAD_SIZE, BLOCK_DK):
            d_offsets = d_base + tl.arange(0, BLOCK_DK)
            q_ptrs = (
                q_ptr
                + q_offsets[:, None] * q_stride_0
                + head_idx * q_stride_1
                + d_offsets[None, :] * q_stride_2
            )
            q_mask = q_valid[:, None] & (d_offsets[None, :] < HEAD_SIZE)
            Q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            K = _load_packed_k_chunk_tile(
                key_cache_ptr,
                physical_block_idx,
                kv_head_idx,
                seq_offset,
                tile_mask,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                d_offsets,
                BLOCK_SIZE,
                K_BITS,
            ).to(tl.float32)
            score += tl.dot(Q, K)

        V = _load_packed_v_chunk_tile(
            value_cache_ptr,
            physical_block_idx,
            kv_head_idx,
            seq_offset,
            tile_mask,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            dv_offsets,
            BLOCK_SIZE,
            V_BITS,
        ).to(tl.float32)
        scaled_v = V * v_token_head_scales[:, None]
        causal_mask = q_valid[:, None] & tile_mask[None, :] & (
            q_abs[:, None] >= seq_offset[None, :]
        )
        score = score * (softmax_scale * k_token_head_scales[None, :])
        score = tl.where(causal_mask, score, float("-inf"))
        block_m = tl.max(score, axis=1)
        m_curr = tl.maximum(m_prev, block_m)
        p_prev = tl.exp(m_prev - m_curr) * l_prev
        p_curr = tl.exp(score - m_curr[:, None])
        l_curr = p_prev + tl.sum(p_curr, axis=1)
        safe_l = tl.where(l_curr > 0, l_curr, 1.0)
        p_norm = p_curr / safe_l[:, None]
        if TILE_SIZE >= 16:
            block_out = tl.dot(p_norm.to(tl.float32), scaled_v)
        else:
            block_out = tl.sum(p_norm[:, :, None] * scaled_v[None, :, :], axis=1)
        acc = acc * (p_prev / safe_l)[:, None] + block_out
        m_prev = m_curr
        l_prev = l_curr

    out_ptrs = (
        mid_o_ptr
        + q_offsets[:, None] * mid_o_stride_0
        + head_idx * mid_o_stride_1
        + split_idx * mid_o_stride_2
        + dv_offsets[None, :] * mid_o_stride_3
    )
    out_mask = q_valid[:, None] & dv_valid[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)
    if dv_chunk_idx == 0:
        lse = m_prev + tl.log(tl.where(l_prev > 0, l_prev, 1.0))
        lse_ptrs = (
            mid_o_ptr
            + q_offsets * mid_o_stride_0
            + head_idx * mid_o_stride_1
            + split_idx * mid_o_stride_2
            + HEAD_SIZE_V * mid_o_stride_3
        )
        tl.store(lse_ptrs, lse, mask=q_valid)


def build_kside_splitkv_packed_triton_artifacts(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: PackedReferenceInputs,
) -> Stage1Artifacts:
    global _LAST_KSIDE_SPLITKV_TRITON_VARIANT
    hotshape = reference_inputs.hotshape
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    mid_o = allocate_stage_buffer(plan, device=hotshape["query"].device)
    tile_size = 8 if max(spec.head_size, spec.head_size_v) >= 512 else 16
    common_args = (
        mid_o,
        hotshape["query"],
        hotshape["key_cache"],
        hotshape["value_cache"],
        hotshape["block_table"][0],
        hotshape["k_scale"],
        hotshape["v_scale"],
        hotshape["query"].stride(0),
        hotshape["query"].stride(1),
        hotshape["query"].stride(2),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        hotshape["key_cache"].stride(0),
        hotshape["key_cache"].stride(1),
        hotshape["key_cache"].stride(2),
        hotshape["key_cache"].stride(3),
        hotshape["value_cache"].stride(0),
        hotshape["value_cache"].stride(1),
        hotshape["value_cache"].stride(2),
        hotshape["value_cache"].stride(3),
        hotshape["k_scale"].stride(0),
        hotshape["k_scale"].stride(1),
        hotshape["k_scale"].stride(2),
        hotshape["v_scale"].stride(0),
        hotshape["v_scale"].stride(1),
        hotshape["v_scale"].stride(2),
        spec.query_len,
        spec.seq_len,
        spec.num_queries_per_kv,
        plan.max_split_tokens,
        spec.softmax_scale,
    )
    block_q_internal = plan.query_tile_size_hint
    grid = build_kside_splitkv_packed_runtime_grid(
        query_len=spec.query_len,
        num_query_heads=spec.num_query_heads,
        num_kv_splits=plan.num_kv_splits,
        block_q_internal=block_q_internal,
    )
    _LAST_KSIDE_SPLITKV_TRITON_VARIANT = f"scalar_tile{tile_size}_online_warps4"

    def replay() -> None:
        _stage1_packed_splitkv_scalar_kernel[grid](
            *common_args,
            BLOCK_Q=block_q_internal,
            BLOCK_SIZE=spec.block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=spec.head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(spec.head_size),
            HEAD_SIZE_V=spec.head_size_v,
            HEAD_SIZE_V_PADDED=triton.next_power_of_2(spec.head_size_v),
            K_BITS=hotshape["layout"].k_bits,
            V_BITS=hotshape["layout"].v_bits,
            num_warps=4,
            num_stages=1,
        )

    replay()
    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=reference_inputs.factorized_out_ref,
        full_lse_ref=reference_inputs.factorized_lse_ref,
        impl="kside_splitkv_proto",
        replay=replay,
    )


def build_kside_splitkv_packed_chunkedk_artifacts(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: PackedReferenceInputs,
) -> Stage1Artifacts:
    hotshape = reference_inputs.hotshape
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    mid_o = allocate_stage_buffer(plan, device=hotshape["query"].device)
    tile_size = 8 if max(spec.head_size, spec.head_size_v) >= 512 else 16
    common_args = (
        mid_o,
        hotshape["query"],
        hotshape["key_cache"],
        hotshape["value_cache"],
        hotshape["block_table"][0],
        hotshape["k_scale"],
        hotshape["v_scale"],
        hotshape["query"].stride(0),
        hotshape["query"].stride(1),
        hotshape["query"].stride(2),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        hotshape["key_cache"].stride(0),
        hotshape["key_cache"].stride(1),
        hotshape["key_cache"].stride(2),
        hotshape["key_cache"].stride(3),
        hotshape["value_cache"].stride(0),
        hotshape["value_cache"].stride(1),
        hotshape["value_cache"].stride(2),
        hotshape["value_cache"].stride(3),
        hotshape["k_scale"].stride(0),
        hotshape["k_scale"].stride(1),
        hotshape["k_scale"].stride(2),
        hotshape["v_scale"].stride(0),
        hotshape["v_scale"].stride(1),
        hotshape["v_scale"].stride(2),
        spec.query_len,
        spec.seq_len,
        spec.num_queries_per_kv,
        plan.max_split_tokens,
        spec.softmax_scale,
    )
    block_q_internal = plan.query_tile_size_hint
    grid = build_kside_splitkv_packed_runtime_grid(
        query_len=spec.query_len,
        num_query_heads=spec.num_query_heads,
        num_kv_splits=plan.num_kv_splits,
        block_q_internal=block_q_internal,
    )
    global _LAST_KSIDE_SPLITKV_TRITON_VARIANT
    _LAST_KSIDE_SPLITKV_TRITON_VARIANT = f"chunkedk_tile{tile_size}_blockd64_warps4"

    def replay() -> None:
        _stage1_packed_splitkv_chunkedk_kernel[grid](
            *common_args,
            BLOCK_Q=block_q_internal,
            BLOCK_SIZE=spec.block_size,
            TILE_SIZE=tile_size,
            BLOCK_D=64,
            HEAD_SIZE=spec.head_size,
            HEAD_SIZE_V=spec.head_size_v,
            HEAD_SIZE_V_PADDED=triton.next_power_of_2(spec.head_size_v),
            K_BITS=hotshape["layout"].k_bits,
            V_BITS=hotshape["layout"].v_bits,
            num_warps=4,
            num_stages=1,
        )

    replay()
    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=reference_inputs.factorized_out_ref,
        full_lse_ref=reference_inputs.factorized_lse_ref,
        impl="kside_splitkv_chunkedk_proto",
        replay=replay,
    )


def build_kside_splitkv_packed_chunkedkv_artifacts(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: PackedReferenceInputs,
) -> Stage1Artifacts:
    hotshape = reference_inputs.hotshape
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    mid_o = allocate_stage_buffer(plan, device=hotshape["query"].device)
    tile_size = 8 if max(spec.head_size, spec.head_size_v) >= 512 else 16
    common_args = (
        mid_o,
        hotshape["query"],
        hotshape["key_cache"],
        hotshape["value_cache"],
        hotshape["block_table"][0],
        hotshape["k_scale"],
        hotshape["v_scale"],
        hotshape["query"].stride(0),
        hotshape["query"].stride(1),
        hotshape["query"].stride(2),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        hotshape["key_cache"].stride(0),
        hotshape["key_cache"].stride(1),
        hotshape["key_cache"].stride(2),
        hotshape["key_cache"].stride(3),
        hotshape["value_cache"].stride(0),
        hotshape["value_cache"].stride(1),
        hotshape["value_cache"].stride(2),
        hotshape["value_cache"].stride(3),
        hotshape["k_scale"].stride(0),
        hotshape["k_scale"].stride(1),
        hotshape["k_scale"].stride(2),
        hotshape["v_scale"].stride(0),
        hotshape["v_scale"].stride(1),
        hotshape["v_scale"].stride(2),
        spec.query_len,
        spec.seq_len,
        spec.num_queries_per_kv,
        plan.max_split_tokens,
        spec.softmax_scale,
    )
    block_q_internal = plan.query_tile_size_hint
    grid = (
        build_kside_splitkv_packed_runtime_grid(
            query_len=spec.query_len,
            num_query_heads=spec.num_query_heads,
            num_kv_splits=plan.num_kv_splits * (spec.head_size_v // 128),
            block_q_internal=block_q_internal,
        )
    )
    global _LAST_KSIDE_SPLITKV_TRITON_VARIANT
    _LAST_KSIDE_SPLITKV_TRITON_VARIANT = f"chunkedkv_tile{tile_size}_warps4"

    def replay() -> None:
        _stage1_packed_splitkv_chunkedkv_kernel[grid](
            *common_args,
            BLOCK_Q=block_q_internal,
            BLOCK_SIZE=spec.block_size,
            TILE_SIZE=tile_size,
            BLOCK_DK=64,
            BLOCK_DV=128,
            NUM_DV_CHUNKS=spec.head_size_v // 128,
            HEAD_SIZE=spec.head_size,
            HEAD_SIZE_V=spec.head_size_v,
            K_BITS=hotshape["layout"].k_bits,
            V_BITS=hotshape["layout"].v_bits,
            num_warps=4,
            num_stages=1,
        )

    replay()
    return Stage1Artifacts(
        plan=plan,
        mid_o=mid_o,
        full_out_ref=reference_inputs.factorized_out_ref,
        full_lse_ref=reference_inputs.factorized_lse_ref,
        impl="kside_splitkv_chunkedkv_proto",
        replay=replay,
    )


def _stage1_kside_splitkv_proto(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    # First concrete implementation slot for the future K-side staged kernel.
    # Dense inputs still use the first real Triton qtile stage1 prototype.
    # Packed-backed inputs now take the packed-factorized qtile host reference
    # path unless CUDA is available, in which case they take the first real
    # packed Triton split-KV stage1 prototype. This keeps the target-named proto
    # slot aligned to the newer packed semantic contract while preserving its
    # own identity.
    if isinstance(reference_inputs, PackedReferenceInputs):
        packed_inputs = resolve_packed_reference_inputs(
            spec=spec,
            dtype=dtype,
            device=device,
            reference_inputs=reference_inputs,
        )
        if device.startswith("cuda"):
            artifacts = build_kside_splitkv_packed_triton_artifacts(
                spec,
                dtype,
                device,
                num_kv_splits,
                query_tile_size_hint,
                packed_inputs,
            )
        else:
            artifacts = build_qtile_packed_factorized_reference_artifacts(
                spec,
                dtype,
                device,
                num_kv_splits,
                query_tile_size_hint,
                packed_inputs,
                impl_name="kside_splitkv_proto",
            )
    else:
        artifacts = build_qtile_triton_dense_artifacts(
            spec,
            dtype,
            device,
            num_kv_splits,
            query_tile_size_hint,
            reference_inputs,
        )
    _ = build_kside_splitkv_stage1_grid(artifacts.plan)
    return Stage1Artifacts(
        plan=artifacts.plan,
        mid_o=artifacts.mid_o,
        full_out_ref=artifacts.full_out_ref,
        full_lse_ref=artifacts.full_lse_ref,
        impl="kside_splitkv_proto",
        replay=artifacts.replay,
    )


def _stage1_kside_splitkv_chunkedk_proto(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    if isinstance(reference_inputs, PackedReferenceInputs):
        packed_inputs = resolve_packed_reference_inputs(
            spec=spec,
            dtype=dtype,
            device=device,
            reference_inputs=reference_inputs,
        )
        if device.startswith("cuda"):
            artifacts = build_kside_splitkv_packed_chunkedk_artifacts(
                spec,
                dtype,
                device,
                num_kv_splits,
                query_tile_size_hint,
                packed_inputs,
            )
        else:
            artifacts = build_qtile_packed_factorized_reference_artifacts(
                spec,
                dtype,
                device,
                num_kv_splits,
                query_tile_size_hint,
                packed_inputs,
                impl_name="kside_splitkv_chunkedk_proto",
            )
    else:
        raise NotImplementedError(
            "kside_splitkv_chunkedk_proto is only implemented for packed-backed inputs"
        )
    return Stage1Artifacts(
        plan=artifacts.plan,
        mid_o=artifacts.mid_o,
        full_out_ref=artifacts.full_out_ref,
        full_lse_ref=artifacts.full_lse_ref,
        impl="kside_splitkv_chunkedk_proto",
        replay=artifacts.replay,
    )


def _stage1_kside_splitkv_chunkedkv_proto(
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> Stage1Artifacts:
    if isinstance(reference_inputs, PackedReferenceInputs):
        packed_inputs = resolve_packed_reference_inputs(
            spec=spec,
            dtype=dtype,
            device=device,
            reference_inputs=reference_inputs,
        )
        if device.startswith("cuda") and spec.head_size_v >= 128:
            artifacts = build_kside_splitkv_packed_chunkedkv_artifacts(
                spec,
                dtype,
                device,
                num_kv_splits,
                query_tile_size_hint,
                packed_inputs,
            )
        else:
            artifacts = build_qtile_packed_factorized_reference_artifacts(
                spec,
                dtype,
                device,
                num_kv_splits,
                query_tile_size_hint,
                packed_inputs,
                impl_name="kside_splitkv_chunkedkv_proto",
            )
    else:
        raise NotImplementedError(
            "kside_splitkv_chunkedkv_proto is only implemented for packed-backed inputs"
        )
    return Stage1Artifacts(
        plan=artifacts.plan,
        mid_o=artifacts.mid_o,
        full_out_ref=artifacts.full_out_ref,
        full_lse_ref=artifacts.full_lse_ref,
        impl="kside_splitkv_chunkedkv_proto",
        replay=artifacts.replay,
    )


def register_kside_splitkv_proto_impl() -> None:
    try:
        register_stage1_impl("kside_splitkv_proto", _stage1_kside_splitkv_proto)
    except ValueError:
        pass
    try:
        register_stage1_impl(
            "kside_splitkv_chunkedk_proto",
            _stage1_kside_splitkv_chunkedk_proto,
        )
    except ValueError:
        pass
    try:
        register_stage1_impl(
            "kside_splitkv_chunkedkv_proto",
            _stage1_kside_splitkv_chunkedkv_proto,
        )
    except ValueError:
        pass
    try:
        register_stage1_metadata_getter(
            "kside_splitkv_proto",
            get_kside_splitkv_proto_metadata,
        )
    except ValueError:
        pass
    try:
        register_stage1_metadata_getter(
            "kside_splitkv_chunkedk_proto",
            get_kside_splitkv_chunkedk_proto_metadata,
        )
    except ValueError:
        pass
    try:
        register_stage1_metadata_getter(
            "kside_splitkv_chunkedkv_proto",
            get_kside_splitkv_chunkedkv_proto_metadata,
        )
    except ValueError:
        pass
def get_kside_splitkv_proto_metadata() -> dict[str, object] | None:
    if _LAST_KSIDE_SPLITKV_TRITON_VARIANT is None:
        return None
    jit_kernel = _stage1_packed_splitkv_scalar_kernel
    metadata = extract_triton_jit_metadata(jit_kernel)
    diagnostics = extract_triton_jit_shared_diagnostics(jit_kernel)
    return {
        "stage1_triton_variant": _LAST_KSIDE_SPLITKV_TRITON_VARIANT,
        **compiled_metadata_to_dict(metadata),
        **diagnostics,
    }


def get_kside_splitkv_chunkedk_proto_metadata() -> dict[str, object] | None:
    metadata = extract_triton_jit_metadata(_stage1_packed_splitkv_chunkedk_kernel)
    diagnostics = extract_triton_jit_shared_diagnostics(
        _stage1_packed_splitkv_chunkedk_kernel
    )
    return {
        "stage1_triton_variant": _LAST_KSIDE_SPLITKV_TRITON_VARIANT or "chunkedk",
        **compiled_metadata_to_dict(metadata),
        **diagnostics,
    }


def get_kside_splitkv_chunkedkv_proto_metadata() -> dict[str, object] | None:
    metadata = extract_triton_jit_metadata(_stage1_packed_splitkv_chunkedkv_kernel)
    diagnostics = extract_triton_jit_shared_diagnostics(
        _stage1_packed_splitkv_chunkedkv_kernel
    )
    return {
        "stage1_triton_variant": "chunkedkv_tile8_warps4",
        **compiled_metadata_to_dict(metadata),
        **diagnostics,
    }










register_kside_splitkv_proto_impl()
