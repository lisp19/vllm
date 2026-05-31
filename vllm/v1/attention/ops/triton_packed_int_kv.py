# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.mem_utils import get_max_shared_memory_bytes
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_softcap,
    resolve_seq_and_query_len,
    softmax_step,
)
from vllm.v1.kv_cache_interface import PackedIntPerTokenHeadLayout


def _quant_bounds(bits: int) -> tuple[int, int]:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    return qmax, qmin


def _per_token_head_scale(values: torch.Tensor, qmax: int, bits: int) -> torch.Tensor:
    abs_values = values.abs()
    if bits <= 3:
        numel = abs_values.shape[-1]
        kth = max(1, (numel * 99 + 99) // 100)
        scale_base = abs_values.kthvalue(kth, dim=-1).values
    else:
        scale_base = abs_values.amax(dim=-1)
    return torch.clamp(scale_base / float(qmax), min=1e-6).to(torch.float32)


def _pack_signed_values(values: torch.Tensor, bits: int) -> torch.Tensor:
    values_i32 = values.to(torch.int32)
    mask = (1 << bits) - 1
    packed_bytes = (values.shape[-1] * bits + 7) // 8
    flat = values_i32.reshape(-1, values.shape[-1])
    codes = flat & mask
    bit_pos = torch.arange(flat.shape[-1], device=flat.device, dtype=torch.int32) * bits
    byte_idx = bit_pos // 8
    bit_off = bit_pos % 8
    out = torch.zeros(
        flat.shape[0], packed_bytes, device=flat.device, dtype=torch.int32
    )
    low = (codes << bit_off.unsqueeze(0)) & 0xFF
    out.scatter_add_(1, byte_idx.unsqueeze(0).expand_as(low), low)
    cross = bit_off + bits > 8
    if torch.any(cross):
        high = torch.where(
            cross.unsqueeze(0),
            codes >> (8 - bit_off).unsqueeze(0),
            torch.zeros_like(codes),
        )
        high_idx = (byte_idx + 1).clamp_max(packed_bytes - 1)
        out.scatter_add_(1, high_idx.unsqueeze(0).expand_as(high), high)
    return out.to(torch.uint8).reshape(*values.shape[:-1], packed_bytes)


def _unpack_signed_values(
    packed: torch.Tensor,
    *,
    bits: int,
    num_elements: int,
) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    if bits == 8:
        return packed.view(torch.int8).to(torch.int32)[..., :num_elements]

    bit_pos = torch.arange(num_elements, device=packed.device, dtype=torch.int32) * bits
    byte_idx = bit_pos // 8
    bit_off = bit_pos % 8
    low = torch.index_select(packed_i32, -1, byte_idx)
    values = low >> bit_off.view(*([1] * (low.ndim - 1)), -1)
    cross = bit_off + bits > 8
    if torch.any(cross):
        high_idx = (byte_idx + 1).clamp_max(packed.shape[-1] - 1)
        high = torch.index_select(packed_i32, -1, high_idx)
        values = values | torch.where(
            cross.view(*([1] * (high.ndim - 1)), -1),
            high << (8 - bit_off).view(*([1] * (high.ndim - 1)), -1),
            torch.zeros_like(high),
        )
    mask = (1 << bits) - 1
    values = values & mask
    sign_bit = 1 << (bits - 1)
    signed = torch.where(values >= sign_bit, values - (1 << bits), values)
    return signed.to(torch.int32)


def get_packed_int_cache_views(
    kv_cache: torch.Tensor,
    layout: PackedIntPerTokenHeadLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_u8 = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
    num_blocks, block_size, nkv, slot_bytes = kv_u8.shape
    assert slot_bytes >= layout.raw_bytes_per_token_head
    key_cache = torch.as_strided(
        kv_u8,
        size=(num_blocks, block_size, nkv, layout.k_data_bytes),
        stride=kv_u8.stride(),
        storage_offset=0,
    )
    value_cache = torch.as_strided(
        kv_u8,
        size=(num_blocks, block_size, nkv, layout.v_data_bytes),
        stride=kv_u8.stride(),
        storage_offset=layout.v_data_offset_bytes,
    )

    if any(x % 4 != 0 for x in (kv_u8.stride(0), kv_u8.stride(1), kv_u8.stride(2))):
        raise ValueError("Packed-int scale views require byte strides divisible by 4")
    if layout.k_scale_offset_bytes % 4 != 0 or layout.v_scale_offset_bytes % 4 != 0:
        raise ValueError("Packed-int scale offsets must be divisible by 4")

    raw = kv_u8.untyped_storage()
    base_f32 = torch.tensor([], dtype=torch.float32, device=kv_u8.device).set_(raw)
    block_stride_f32 = kv_u8.stride(0) // 4
    slot_stride_f32 = kv_u8.stride(1) // 4
    head_stride_f32 = kv_u8.stride(2) // 4
    k_scale_cache = torch.as_strided(
        base_f32,
        size=(num_blocks, block_size, nkv),
        stride=(block_stride_f32, slot_stride_f32, head_stride_f32),
        storage_offset=layout.k_scale_offset_bytes // 4,
    )
    v_scale_cache = torch.as_strided(
        base_f32,
        size=(num_blocks, block_size, nkv),
        stride=(block_stride_f32, slot_stride_f32, head_stride_f32),
        storage_offset=layout.v_scale_offset_bytes // 4,
    )
    return key_cache, value_cache, k_scale_cache, v_scale_cache


def reshape_and_cache_packed_int_per_token_head(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layout: PackedIntPerTokenHeadLayout,
) -> None:
    num_tokens, num_kv_heads, _ = key.shape
    device = key.device
    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    if not torch.any(valid):
        return
    token_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    slots = slots[valid]
    blk = slots // key_cache.shape[1]
    slot_in_blk = slots % key_cache.shape[1]
    head_idx = torch.arange(num_kv_heads, device=device, dtype=torch.int64)

    key_valid = key[token_idx].to(torch.float32)
    value_valid = value[token_idx].to(torch.float32)

    k_qmax, k_qmin = _quant_bounds(layout.k_bits)
    v_qmax, v_qmin = _quant_bounds(layout.v_bits)
    k_scale = _per_token_head_scale(key_valid, k_qmax, layout.k_bits)
    v_scale = _per_token_head_scale(value_valid, v_qmax, layout.v_bits)

    key_q = torch.clamp(
        torch.round(key_valid / k_scale.unsqueeze(-1)), k_qmin, k_qmax
    ).to(torch.int32)
    value_q = torch.clamp(
        torch.round(value_valid / v_scale.unsqueeze(-1)), v_qmin, v_qmax
    ).to(torch.int32)

    packed_k = _pack_signed_values(key_q, layout.k_bits)
    packed_v = _pack_signed_values(value_q, layout.v_bits)

    key_cache[blk[:, None], slot_in_blk[:, None], head_idx[None, :]] = packed_k
    value_cache[blk[:, None], slot_in_blk[:, None], head_idx[None, :]] = packed_v
    k_scale_cache[blk[:, None], slot_in_blk[:, None], head_idx[None, :]] = k_scale
    v_scale_cache[blk[:, None], slot_in_blk[:, None], head_idx[None, :]] = v_scale


def _materialize_sequence_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
    layout: PackedIntPerTokenHeadLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, device=block_table_row.device, dtype=torch.int64)
    blocks = block_table_row[pos // block_size]
    slots = pos % block_size
    packed_k = key_cache[blocks, slots]
    packed_v = value_cache[blocks, slots]
    k_scales = k_scale_cache[blocks, slots].to(torch.float32)
    v_scales = v_scale_cache[blocks, slots].to(torch.float32)
    key = _unpack_signed_values(
        packed_k, bits=layout.k_bits, num_elements=layout.head_size
    ).to(torch.float32) * k_scales.unsqueeze(-1)
    value = _unpack_signed_values(
        packed_v, bits=layout.v_bits, num_elements=layout.head_size_v
    ).to(torch.float32) * v_scales.unsqueeze(-1)
    return key, value


@triton.jit
def _decode_packed_signed(
    cache_ptr,
    byte_base,
    stride_cache_3: tl.int64,
    bit_width: tl.constexpr,
    dim_idx,
):
    bit_pos = dim_idx * bit_width
    byte_idx = bit_pos // 8
    bit_off = bit_pos % 8
    low = tl.load(cache_ptr + byte_base + byte_idx * stride_cache_3, other=0).to(
        tl.int32
    )
    needs_high = bit_off + bit_width > 8
    high = tl.load(
        cache_ptr + byte_base + (byte_idx + 1) * stride_cache_3,
        mask=needs_high,
        other=0,
    ).to(tl.int32)
    code = low >> bit_off
    code = tl.where(needs_high, code | (high << (8 - bit_off)), code)
    code = code & ((1 << bit_width) - 1)
    sign_bit = 1 << (bit_width - 1)
    return tl.where(code >= sign_bit, code - (1 << bit_width), code).to(tl.float32)


@triton.jit
def _load_packed_k_tile(
    key_cache_ptr,
    physical_block_idx,
    kv_head_idx,
    seq_offset,
    tile_mask,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    k_scale_cache_ptr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    K_BITS: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE
    slot_in_block = seq_offset % BLOCK_SIZE
    scale_idx = (
        physical_block_idx * stride_ks_blk
        + slot_in_block * stride_ks_slot
        + kv_head_idx * stride_ks_head
    )
    k_scales = tl.load(k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0)
    byte_base = (
        physical_block_idx[None, :] * stride_k_cache_0
        + slot_in_block[None, :] * stride_k_cache_1
        + kv_head_idx * stride_k_cache_2
    )
    decoded = _decode_packed_signed(
        key_cache_ptr,
        byte_base,
        stride_k_cache_3,
        K_BITS,
        offs_d[:, None],
    )
    decoded = decoded * k_scales[None, :]
    return tl.where(dim_mask[:, None] & tile_mask[None, :], decoded, 0.0)


@triton.jit
def _load_packed_v_tile(
    value_cache_ptr,
    physical_block_idx,
    kv_head_idx,
    seq_offset,
    tile_mask,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    v_scale_cache_ptr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    V_BITS: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_SIZE_V_PADDED)
    dim_mask = offs_d < HEAD_SIZE_V
    slot_in_block = seq_offset % BLOCK_SIZE
    scale_idx = (
        physical_block_idx * stride_vs_blk
        + slot_in_block * stride_vs_slot
        + kv_head_idx * stride_vs_head
    )
    v_scales = tl.load(v_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0)
    byte_base = (
        physical_block_idx[:, None] * stride_v_cache_0
        + slot_in_block[:, None] * stride_v_cache_1
        + kv_head_idx * stride_v_cache_2
    )
    decoded = _decode_packed_signed(
        value_cache_ptr,
        byte_base,
        stride_v_cache_3,
        V_BITS,
        offs_d[None, :],
    )
    decoded = decoded * v_scales[:, None]
    return tl.where(tile_mask[:, None] & dim_mask[None, :], decoded, 0.0)


@triton.jit
def kernel_packed_int_attention(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    softcap,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
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
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d_q = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d_v = tl.arange(0, HEAD_SIZE_V_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d_q[None, :]
    )

    dim_mask_q = tl.where(offs_d_q < HEAD_SIZE, 1, 0).to(tl.int1)
    dim_mask_v = tl.where(offs_d_v < HEAD_SIZE_V, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask_q[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_V_PADDED], dtype=tl.float32)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = tl.minimum(
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1,
        seq_len,
    )
    num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) // TILE_SIZE

    tile_start = 0
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        q_abs = context_len + qpos_lo
        first_allowed_key = q_abs - SLIDING_WINDOW + 1
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)

    for j in range(tile_start, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)

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
            stride_ks_blk,
            stride_ks_slot,
            stride_ks_head,
            k_scale_cache_ptr,
            BLOCK_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            K_BITS,
        )
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
            stride_vs_blk,
            stride_vs_slot,
            stride_vs_head,
            v_scale_cache_ptr,
            BLOCK_SIZE,
            HEAD_SIZE_V,
            HEAD_SIZE_V_PADDED,
            V_BITS,
        )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = query_abs_pos >= seq_offset[None, :]
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (query_abs_pos - seq_offset[None, :] < SLIDING_WINDOW)

        S = tl.dot(Q, K) * scale
        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]
        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]
    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d_v[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def paged_attention_packed_int(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    out: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    layout: PackedIntPerTokenHeadLayout,
    *,
    softmax_scale: float,
    softcap: float,
    num_queries_per_kv: int,
    sliding_window: tuple[int, int],
) -> None:
    block_size = value_cache.shape[1]
    num_seqs = len(seq_lens)
    num_query_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_size = q.shape[2]
    head_size_v = out.shape[2]

    block_m = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // block_q + num_seqs
    sliding_window_val = 1 + sliding_window[0] if sliding_window[0] >= 0 else 0
    tile_size = 32 if q.element_size() == 1 else 16
    if (
        tile_size == 32
        and triton.next_power_of_2(head_size) >= 512
        and q.element_size() >= 2
    ):
        max_shared_memory = (
            get_max_shared_memory_bytes() if current_platform.is_cuda() else 65536
        )
        if max_shared_memory < 98304:
            tile_size = 16

    head_size_padded = triton.next_power_of_2(head_size)
    head_size_v_padded = triton.next_power_of_2(head_size_v)
    num_warps = 8 if head_size_padded >= 256 else 4
    grid = (total_num_q_blocks, num_kv_heads)

    kernel_packed_int_attention[grid](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        scale=softmax_scale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        query_start_len_ptr=query_start_loc,
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=head_size_v_padded,
        K_BITS=layout.k_bits,
        V_BITS=layout.v_bits,
        USE_SOFTCAP=(softcap > 0),
        SLIDING_WINDOW=sliding_window_val,
        num_warps=num_warps,
    )
