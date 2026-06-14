# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PackedIntAttentionKernelConfig:
    block_q: int
    block_m: int
    sliding_window_val: int
    tile_size: int
    head_size_padded: int
    head_size_v_padded: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class PackedIntWriterKernelConfig:
    key_block: int
    value_block: int
    key_data_block: int
    value_data_block: int
    value_group_block: int
    num_warps: int
    num_stages: int


def _estimate_packed_int_attention_shared_bytes(
    *,
    block_m: int,
    tile_size: int,
    head_size_padded: int,
    head_size_v_padded: int,
) -> int:
    # The decode kernel's largest static buffers are:
    # - acc: [BLOCK_M, HEAD_SIZE_V_PADDED]
    # - K tile: [TILE_SIZE, HEAD_SIZE_PADDED]
    # - V tile: [TILE_SIZE, HEAD_SIZE_V_PADDED]
    # They are fp32 in the kernel today, so 4 bytes per element.
    return 4 * (
        block_m * head_size_v_padded
        + tile_size * head_size_padded
        + tile_size * head_size_v_padded
    )


def build_packed_int_attention_kernel_config(
    *,
    q_element_size: int,
    head_size: int,
    head_size_v: int,
    num_queries_per_kv: int,
    sliding_window: tuple[int, int],
) -> PackedIntAttentionKernelConfig:
    block_m = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    block_q = block_m // num_queries_per_kv
    sliding_window_val = 1 + sliding_window[0] if sliding_window[0] >= 0 else 0
    tile_size = 32 if q_element_size == 1 else 16
    if sliding_window_val == 1024 and head_size in (128, 256):
        tile_size = 32
    if (
        tile_size == 32
        and triton.next_power_of_2(head_size) >= 512
        and q_element_size >= 2
    ):
        max_shared_memory = (
            get_max_shared_memory_bytes() if current_platform.is_cuda() else 65536
        )
        if max_shared_memory < 98304:
            tile_size = 16

    head_size_padded = triton.next_power_of_2(head_size)
    head_size_v_padded = triton.next_power_of_2(head_size_v)
    num_warps = 8 if head_size_padded >= 256 else 4
    if current_platform.is_cuda():
        max_shared_memory = get_max_shared_memory_bytes()
        shared_budget = max(0, max_shared_memory - 4096)
        estimated_shared = _estimate_packed_int_attention_shared_bytes(
            block_m=block_m,
            tile_size=tile_size,
            head_size_padded=head_size_padded,
            head_size_v_padded=head_size_v_padded,
        )
        # On SM75-class GPUs with 64 KiB shared memory, large-head packed-int
        # decode can exceed the per-block limit even before any capture logic.
        # Prefer a narrow fallback here over letting Triton fail later with
        # OutOfResources during Gemma-style 512-d full-attention capture.
        if estimated_shared > max_shared_memory and (
            head_size_padded >= 512 or head_size_v_padded >= 512
        ):
            num_warps = min(num_warps, 4)
            if max_shared_memory <= 65536:
                tile_size = 8
            elif tile_size > 16:
                tile_size = 16
            estimated_shared = _estimate_packed_int_attention_shared_bytes(
                block_m=block_m,
                tile_size=tile_size,
                head_size_padded=head_size_padded,
                head_size_v_padded=head_size_v_padded,
            )
            while estimated_shared > shared_budget and block_q > 1:
                block_q = max(1, block_q // 2)
                block_m = block_q * num_queries_per_kv
                estimated_shared = _estimate_packed_int_attention_shared_bytes(
                    block_m=block_m,
                    tile_size=tile_size,
                    head_size_padded=head_size_padded,
                    head_size_v_padded=head_size_v_padded,
                )
    # On the local SM75 Gemma shape (256-d heads, GQA=2, sliding window 1024),
    # static scans consistently favored the smaller 16x4 launch across both
    # short decode and longer tail-prefill points.
    if (
        current_platform.is_cuda()
        and q_element_size >= 2
        and sliding_window_val == 1024
        and head_size == 256
        and num_queries_per_kv == 2
    ):
        tile_size = 16
        num_warps = 4
    return PackedIntAttentionKernelConfig(
        block_q=block_q,
        block_m=block_m,
        sliding_window_val=sliding_window_val,
        tile_size=tile_size,
        head_size_padded=head_size_padded,
        head_size_v_padded=head_size_v_padded,
        num_warps=num_warps,
        num_stages=3,
    )


def build_packed_int_writer_kernel_config(
    layout: PackedIntPerTokenHeadLayout,
) -> PackedIntWriterKernelConfig:
    key_block = triton.next_power_of_2(layout.head_size)
    value_block = triton.next_power_of_2(layout.head_size_v)
    return PackedIntWriterKernelConfig(
        key_block=key_block,
        value_block=value_block,
        key_data_block=triton.next_power_of_2(layout.k_data_bytes),
        value_data_block=triton.next_power_of_2(layout.v_data_bytes),
        value_group_block=triton.next_power_of_2((layout.head_size_v + 7) // 8),
        num_warps=8 if max(key_block, value_block) >= 512 else 4,
        num_stages=1,
    )


def _quant_bounds(bits: int, *, symmetric_3bit: bool = False) -> tuple[int, int]:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    if symmetric_3bit and bits == 3:
        qmin = -qmax
    return qmax, qmin


def _supports_triton_packed_int_writer(layout: PackedIntPerTokenHeadLayout) -> bool:
    return (
        current_platform.is_cuda()
        and 2 <= layout.k_bits <= 8
        and 2 <= layout.v_bits <= 8
        and layout.head_size <= 512
        and layout.head_size_v <= 512
    )


def _per_token_head_scale(values: torch.Tensor, qmax: int, bits: int) -> torch.Tensor:
    abs_values = values.abs()
    if bits <= 3:
        numel = abs_values.shape[-1]
        kth = max(1, (numel * 99 + 99) // 100)
        scale_base = abs_values.kthvalue(kth, dim=-1).values
    else:
        scale_base = abs_values.amax(dim=-1)
    return torch.clamp(scale_base / float(qmax), min=1e-6).to(torch.float32)


def _value_scale_headroom(bits: int) -> float:
    if bits == 3:
        return 1.05
    return 1.0


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
    kernel_config: PackedIntWriterKernelConfig | None = None,
) -> None:
    if kernel_config is not None:
        _reshape_and_cache_packed_int_triton(
            key,
            value,
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
            layout,
            kernel_config,
        )
        return

    if _supports_triton_packed_int_writer(layout):
        _reshape_and_cache_packed_int_triton(
            key,
            value,
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
            layout,
            kernel_config,
        )
        return

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
    v_qmax, v_qmin = _quant_bounds(layout.v_bits, symmetric_3bit=True)
    k_scale = _per_token_head_scale(key_valid, k_qmax, layout.k_bits)
    v_scale = _per_token_head_scale(value_valid, v_qmax, layout.v_bits)
    v_scale = v_scale * _value_scale_headroom(layout.v_bits)

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


if current_platform.is_rocm():

    @triton.jit
    def _round_to_int32(x):
        return tl.extra.hip.libdevice.nearbyint(x).to(tl.int32)


elif current_platform.is_xpu():

    @triton.jit
    def _round_to_int32(x):
        return tl.extra.intel.libdevice.nearbyint(x).to(tl.int32)


else:

    @triton.jit
    def _round_to_int32(x):
        return tl.extra.cuda.libdevice.nearbyint(x).to(tl.int32)


@triton.jit
def _compute_packed_scale(
    values,
    valid_mask,
    qmax: tl.constexpr,
    num_values: tl.constexpr,
    block_size: tl.constexpr,
    bits: tl.constexpr,
):
    abs_values = tl.abs(values)
    if bits <= 3:
        kth = (num_values * 99 + 99) // 100
        sorted_abs = tl.sort(
            tl.where(valid_mask, abs_values, float("inf")),
            descending=False,
        )
        kth_mask = tl.arange(0, block_size) == (kth - 1)
        scale_base = tl.max(tl.where(kth_mask, sorted_abs, 0.0), axis=0)
    else:
        scale_base = tl.max(tl.where(valid_mask, abs_values, 0.0), axis=0)
    return tl.maximum(scale_base / qmax, 1e-6)


@triton.jit
def _quantize_to_signed_codes(
    values,
    valid_mask,
    scale,
    bits: tl.constexpr,
    symmetric_3bit: tl.constexpr = False,
):
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    if symmetric_3bit and bits == 3:
        qmin = -qmax
    q = _round_to_int32(tl.math.div_rn(values, scale))
    q = tl.maximum(tl.minimum(q, qmax), qmin)
    return tl.where(valid_mask, q, 0)


@triton.jit
def _store_packed_4bit_from_src(
    src_ptr,
    base_offset,
    stride_dim: tl.int64,
    scale,
    num_values: tl.constexpr,
    dst_ptr,
    packed_block: tl.constexpr,
):
    offs_b = tl.arange(0, packed_block)
    even_idx = offs_b * 2
    odd_idx = even_idx + 1
    even_mask = even_idx < num_values
    odd_mask = odd_idx < num_values
    even_values = tl.load(
        src_ptr + base_offset + even_idx * stride_dim,
        mask=even_mask,
        other=0.0,
    ).to(tl.float32)
    odd_values = tl.load(
        src_ptr + base_offset + odd_idx * stride_dim,
        mask=odd_mask,
        other=0.0,
    ).to(tl.float32)
    even_codes = _quantize_to_signed_codes(even_values, even_mask, scale, 4) & 0xF
    odd_codes = _quantize_to_signed_codes(odd_values, odd_mask, scale, 4) & 0xF
    packed = (even_codes | (odd_codes << 4)).to(tl.uint8)
    tl.store(dst_ptr + offs_b, packed, mask=offs_b < ((num_values + 1) // 2))


@triton.jit
def _store_packed_8bit_from_src(
    src_ptr,
    base_offset,
    stride_dim: tl.int64,
    scale,
    num_values: tl.constexpr,
    dst_ptr,
    packed_block: tl.constexpr,
):
    offs = tl.arange(0, packed_block)
    valid = offs < num_values
    values = tl.load(
        src_ptr + base_offset + offs * stride_dim,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    codes = _quantize_to_signed_codes(values, valid, scale, 8) & 0xFF
    tl.store(dst_ptr + offs, codes.to(tl.uint8), mask=valid)


@triton.jit
def _store_packed_3bit_from_src(
    src_ptr,
    base_offset,
    stride_dim: tl.int64,
    scale,
    num_values: tl.constexpr,
    dst_ptr,
    group_block: tl.constexpr,
    symmetric_3bit: tl.constexpr = False,
):
    grp_idx = tl.arange(0, group_block)
    group_count = (num_values + 7) // 8
    group_mask = grp_idx < group_count
    lane_idx = tl.arange(0, 8)
    elem_idx = grp_idx[:, None] * 8 + lane_idx[None, :]
    elem_mask = group_mask[:, None] & (elem_idx < num_values)
    values = tl.load(
        src_ptr + base_offset + elem_idx * stride_dim,
        mask=elem_mask,
        other=0.0,
    ).to(tl.float32)
    codes = _quantize_to_signed_codes(values, elem_mask, scale, 3, symmetric_3bit) & 0x7
    shifts = lane_idx[None, :] * 3
    packed24 = tl.sum(codes << shifts, axis=1)
    byte_base = grp_idx * 3
    tl.store(dst_ptr + byte_base, (packed24 & 0xFF).to(tl.uint8), mask=group_mask)
    tl.store(
        dst_ptr + byte_base + 1,
        ((packed24 >> 8) & 0xFF).to(tl.uint8),
        mask=group_mask,
    )
    tl.store(
        dst_ptr + byte_base + 2,
        ((packed24 >> 16) & 0xFF).to(tl.uint8),
        mask=group_mask,
    )


@triton.jit
def _pack_src_to_bytes(
    src_ptr,
    base_offset,
    stride_dim: tl.int64,
    scale,
    num_values: tl.constexpr,
    packed_bytes: tl.constexpr,
    packed_block: tl.constexpr,
    bits: tl.constexpr,
):
    offs_b = tl.arange(0, packed_block)
    byte_mask = offs_b < packed_bytes
    start_elem = (offs_b * 8) // bits
    packed = tl.zeros((packed_block,), dtype=tl.int32)
    mask_bits = (1 << bits) - 1
    max_pack_elems = (8 + bits - 1) // bits + 1

    for i in range(max_pack_elems):
        elem_idx = start_elem + i
        valid = elem_idx < num_values
        values = tl.load(
            src_ptr + base_offset + elem_idx * stride_dim,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        code = _quantize_to_signed_codes(values, valid, scale, bits) & mask_bits
        shift = elem_idx * bits - offs_b * 8
        overlap = byte_mask & valid & (shift < 8) & (shift > -bits)
        contrib = tl.where(
            shift >= 0,
            (code << shift) & 0xFF,
            (code >> (-shift)) & 0xFF,
        )
        packed += tl.where(overlap, contrib, 0)
    return packed.to(tl.uint8)


@triton.jit
def _reshape_and_cache_packed_int_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_key_dim: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_val_dim: tl.int64,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ksc_blk: tl.int64,
    stride_ksc_slot: tl.int64,
    stride_ksc_head: tl.int64,
    stride_vsc_blk: tl.int64,
    stride_vsc_slot: tl.int64,
    stride_vsc_head: tl.int64,
    block_size: tl.int64,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    key_data_bytes: tl.constexpr,
    value_data_bytes: tl.constexpr,
    key_data_block: tl.constexpr,
    value_data_block: tl.constexpr,
    value_group_block: tl.constexpr,
    k_bits: tl.constexpr,
    v_bits: tl.constexpr,
    key_block: tl.constexpr,
    value_block: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // num_kv_heads
    kv_head_idx = pid % num_kv_heads

    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    blk = (slot // block_size).to(tl.int64)
    slot_in_blk = (slot % block_size).to(tl.int64)

    key_base = token_idx * stride_key_tok + kv_head_idx * stride_key_head
    value_base = token_idx * stride_val_tok + kv_head_idx * stride_val_head

    key_offs = tl.arange(0, key_block)
    key_mask = key_offs < head_size
    key_values = tl.load(
        key_ptr + key_base + key_offs * stride_key_dim,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    k_scale = _compute_packed_scale(
        key_values,
        key_mask,
        (1 << (k_bits - 1)) - 1,
        head_size,
        key_block,
        k_bits,
    )

    value_offs = tl.arange(0, value_block)
    value_mask = value_offs < head_size_v
    value_values = tl.load(
        value_ptr + value_base + value_offs * stride_val_dim,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    v_scale = _compute_packed_scale(
        value_values,
        value_mask,
        (1 << (v_bits - 1)) - 1,
        head_size_v,
        value_block,
        v_bits,
    )
    if v_bits == 3:
        v_scale = v_scale * 1.05

    kc_base = (
        blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + kv_head_idx * stride_kc_head
    )
    vc_base = (
        blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + kv_head_idx * stride_vc_head
    )

    if k_bits == 4:
        _store_packed_4bit_from_src(
            key_ptr,
            key_base,
            stride_key_dim,
            k_scale,
            head_size,
            key_cache_ptr + kc_base,
            key_data_block,
        )
    elif k_bits == 8:
        _store_packed_8bit_from_src(
            key_ptr,
            key_base,
            stride_key_dim,
            k_scale,
            head_size,
            key_cache_ptr + kc_base,
            key_data_block,
        )
    else:
        packed_key = _pack_src_to_bytes(
            key_ptr,
            key_base,
            stride_key_dim,
            k_scale,
            head_size,
            key_data_bytes,
            key_data_block,
            k_bits,
        )
        tl.store(
            key_cache_ptr + kc_base + tl.arange(0, key_data_block),
            packed_key,
            mask=tl.arange(0, key_data_block) < key_data_bytes,
        )

    if v_bits == 3:
        _store_packed_3bit_from_src(
            value_ptr,
            value_base,
            stride_val_dim,
            v_scale,
            head_size_v,
            value_cache_ptr + vc_base,
            value_group_block,
            True,
        )
    elif v_bits == 4:
        _store_packed_4bit_from_src(
            value_ptr,
            value_base,
            stride_val_dim,
            v_scale,
            head_size_v,
            value_cache_ptr + vc_base,
            value_data_block,
        )
    elif v_bits == 8:
        _store_packed_8bit_from_src(
            value_ptr,
            value_base,
            stride_val_dim,
            v_scale,
            head_size_v,
            value_cache_ptr + vc_base,
            value_data_block,
        )
    else:
        packed_value = _pack_src_to_bytes(
            value_ptr,
            value_base,
            stride_val_dim,
            v_scale,
            head_size_v,
            value_data_bytes,
            value_data_block,
            v_bits,
        )
        tl.store(
            value_cache_ptr + vc_base + tl.arange(0, value_data_block),
            packed_value,
            mask=tl.arange(0, value_data_block) < value_data_bytes,
        )

    # Store float32 per-(token, head) scales.
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ksc_blk
        + slot_in_blk * stride_ksc_slot
        + kv_head_idx * stride_ksc_head,
        k_scale,
    )
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vsc_blk
        + slot_in_blk * stride_vsc_slot
        + kv_head_idx * stride_vsc_head,
        v_scale,
    )


def _reshape_and_cache_packed_int_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layout: PackedIntPerTokenHeadLayout,
    kernel_config: PackedIntWriterKernelConfig | None = None,
) -> None:
    num_tokens, num_kv_heads, _ = key.shape
    grid = (num_tokens * num_kv_heads,)
    if kernel_config is None:
        kernel_config = build_packed_int_writer_kernel_config(layout)
    _reshape_and_cache_packed_int_kernel[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_key_dim=key.stride(2),
        stride_val_tok=value.stride(0),
        stride_val_head=value.stride(1),
        stride_val_dim=value.stride(2),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ksc_blk=k_scale_cache.stride(0),
        stride_ksc_slot=k_scale_cache.stride(1),
        stride_ksc_head=k_scale_cache.stride(2),
        stride_vsc_blk=v_scale_cache.stride(0),
        stride_vsc_slot=v_scale_cache.stride(1),
        stride_vsc_head=v_scale_cache.stride(2),
        block_size=key_cache.shape[1],
        num_kv_heads=num_kv_heads,
        head_size=layout.head_size,
        head_size_v=layout.head_size_v,
        key_data_bytes=layout.k_data_bytes,
        value_data_bytes=layout.v_data_bytes,
        key_data_block=kernel_config.key_data_block,
        value_data_block=kernel_config.value_data_block,
        value_group_block=kernel_config.value_group_block,
        k_bits=layout.k_bits,
        v_bits=layout.v_bits,
        key_block=kernel_config.key_block,
        value_block=kernel_config.value_block,
        num_warps=kernel_config.num_warps,
        num_stages=kernel_config.num_stages,
    )


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
    decode_mask,
):
    bit_pos = dim_idx * bit_width
    byte_idx = bit_pos // 8
    bit_off = bit_pos % 8
    low = tl.load(
        cache_ptr + byte_base + byte_idx * stride_cache_3,
        mask=decode_mask,
        other=0,
    ).to(tl.int32)
    needs_high = bit_off + bit_width > 8
    high = tl.load(
        cache_ptr + byte_base + (byte_idx + 1) * stride_cache_3,
        mask=decode_mask & needs_high,
        other=0,
    ).to(tl.int32)
    code = low >> bit_off
    code = tl.where(needs_high, code | (high << (8 - bit_off)), code)
    code = code & ((1 << bit_width) - 1)
    sign_bit = 1 << (bit_width - 1)
    return tl.where(code >= sign_bit, code - (1 << bit_width), code).to(tl.float32)


@triton.jit
def _decode_packed_signed_4bit(
    cache_ptr,
    byte_base,
    stride_cache_3: tl.int64,
    tile_mask,
    packed_elems: tl.constexpr,
    packed_valid_elems: tl.constexpr,
):
    offs_packed = tl.arange(0, packed_elems)
    packed = tl.load(
        cache_ptr + byte_base + offs_packed[None, :] * stride_cache_3,
        mask=tile_mask[:, None] & (offs_packed[None, :] < packed_valid_elems),
        other=0,
    ).to(tl.int32)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low = tl.where(low >= 8, low - 16, low).to(tl.float32)
    high = tl.where(high >= 8, high - 16, high).to(tl.float32)
    return tl.interleave(low, high)


@triton.jit
def _decode_packed_signed_5bit(
    cache_ptr,
    byte_base,
    stride_cache_3: tl.int64,
    tile_mask,
    packed_groups: tl.constexpr,
    valid_groups: tl.constexpr,
):
    offs_group = tl.arange(0, packed_groups)
    group_mask = tile_mask[:, None] & (offs_group[None, :] < valid_groups)
    byte_offs = offs_group[None, :] * 5

    b0 = tl.load(
        cache_ptr + byte_base + (byte_offs + 0) * stride_cache_3,
        mask=group_mask,
        other=0,
    ).to(tl.int32)
    b1 = tl.load(
        cache_ptr + byte_base + (byte_offs + 1) * stride_cache_3,
        mask=group_mask,
        other=0,
    ).to(tl.int32)
    b2 = tl.load(
        cache_ptr + byte_base + (byte_offs + 2) * stride_cache_3,
        mask=group_mask,
        other=0,
    ).to(tl.int32)
    b3 = tl.load(
        cache_ptr + byte_base + (byte_offs + 3) * stride_cache_3,
        mask=group_mask,
        other=0,
    ).to(tl.int32)
    b4 = tl.load(
        cache_ptr + byte_base + (byte_offs + 4) * stride_cache_3,
        mask=group_mask,
        other=0,
    ).to(tl.int32)

    c0 = b0 & 0x1F
    c1 = ((b0 >> 5) | ((b1 & 0x03) << 3)) & 0x1F
    c2 = (b1 >> 2) & 0x1F
    c3 = ((b1 >> 7) | ((b2 & 0x0F) << 1)) & 0x1F
    c4 = ((b2 >> 4) | ((b3 & 0x01) << 4)) & 0x1F
    c5 = (b3 >> 1) & 0x1F
    c6 = ((b3 >> 6) | ((b4 & 0x07) << 2)) & 0x1F
    c7 = (b4 >> 3) & 0x1F

    c0 = tl.where(c0 >= 16, c0 - 32, c0).to(tl.float32)
    c1 = tl.where(c1 >= 16, c1 - 32, c1).to(tl.float32)
    c2 = tl.where(c2 >= 16, c2 - 32, c2).to(tl.float32)
    c3 = tl.where(c3 >= 16, c3 - 32, c3).to(tl.float32)
    c4 = tl.where(c4 >= 16, c4 - 32, c4).to(tl.float32)
    c5 = tl.where(c5 >= 16, c5 - 32, c5).to(tl.float32)
    c6 = tl.where(c6 >= 16, c6 - 32, c6).to(tl.float32)
    c7 = tl.where(c7 >= 16, c7 - 32, c7).to(tl.float32)

    even_04 = tl.interleave(c0, c4)
    even_26 = tl.interleave(c2, c6)
    even = tl.interleave(even_04, even_26)
    odd_15 = tl.interleave(c1, c5)
    odd_37 = tl.interleave(c3, c7)
    odd = tl.interleave(odd_15, odd_37)
    return tl.interleave(even, odd)


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
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    K_BITS: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE
    slot_in_block = seq_offset % BLOCK_SIZE
    decode_mask = dim_mask[:, None] & tile_mask[None, :]
    byte_base = (
        physical_block_idx[:, None] * stride_k_cache_0
        + slot_in_block[:, None] * stride_k_cache_1
        + kv_head_idx * stride_k_cache_2
    )
    if K_BITS == 4:
        decoded = tl.trans(
            _decode_packed_signed_4bit(
                key_cache_ptr,
                byte_base,
                stride_k_cache_3,
                tile_mask,
                HEAD_SIZE_PADDED // 2,
                (HEAD_SIZE + 1) // 2,
            )
        )
    elif K_BITS == 5:
        decoded = tl.trans(
            _decode_packed_signed_5bit(
                key_cache_ptr,
                byte_base,
                stride_k_cache_3,
                tile_mask,
                HEAD_SIZE_PADDED // 8,
                (HEAD_SIZE + 7) // 8,
            )
        )
    else:
        decoded = _decode_packed_signed(
            key_cache_ptr,
            tl.trans(byte_base),
            stride_k_cache_3,
            K_BITS,
            offs_d[:, None],
            decode_mask,
        )
    return tl.where(decode_mask, decoded, 0.0)


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
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    V_BITS: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_SIZE_V_PADDED)
    dim_mask = offs_d < HEAD_SIZE_V
    slot_in_block = seq_offset % BLOCK_SIZE
    decode_mask = tile_mask[:, None] & dim_mask[None, :]
    byte_base = (
        physical_block_idx[:, None] * stride_v_cache_0
        + slot_in_block[:, None] * stride_v_cache_1
        + kv_head_idx * stride_v_cache_2
    )
    if V_BITS == 4:
        decoded = _decode_packed_signed_4bit(
            value_cache_ptr,
            byte_base,
            stride_v_cache_3,
            tile_mask,
            HEAD_SIZE_V_PADDED // 2,
            (HEAD_SIZE_V + 1) // 2,
        )
    else:
        decoded = _decode_packed_signed(
            value_cache_ptr,
            byte_base,
            stride_v_cache_3,
            V_BITS,
            offs_d[None, :],
            decode_mask,
        )
    return tl.where(decode_mask, decoded, 0.0)


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
        slot_in_block = seq_offset % BLOCK_SIZE
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,
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
            k_scale_cache_ptr + k_scale_idx, mask=tile_mask, other=1.0
        )
        v_token_head_scales = tl.load(
            v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
        )

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
            BLOCK_SIZE,
            HEAD_SIZE_V,
            HEAD_SIZE_V_PADDED,
            V_BITS,
        )
        K = K.to(Q.dtype)
        V = V.to(Q.dtype)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = query_abs_pos >= seq_offset[None, :]
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (query_abs_pos - seq_offset[None, :] < SLIDING_WINDOW)

        S = tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]
        P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
        if TILE_SIZE >= 16:
            acc += tl.dot(P_v, V)
        else:
            acc += tl.sum(P_v[:, :, None] * V[None, :, :], axis=1)

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
    launch_config: PackedIntAttentionKernelConfig | None = None,
    max_query_len: int | None = None,
    allow_single_query_override: bool = True,
) -> None:
    block_size = value_cache.shape[1]
    num_seqs = len(seq_lens)
    num_query_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_size = q.shape[2]
    head_size_v = out.shape[2]

    if launch_config is None:
        launch_config = build_packed_int_attention_kernel_config(
            q_element_size=q.element_size(),
            head_size=head_size,
            head_size_v=head_size_v,
            num_queries_per_kv=num_queries_per_kv,
            sliding_window=sliding_window,
        )
    if (
        allow_single_query_override
        and max_query_len == 1
        and q.element_size() >= 2
        and num_queries_per_kv == 2
        and head_size == 256
        and head_size_v == 256
        and launch_config.sliding_window_val == 1024
        and launch_config.tile_size == 16
        and launch_config.num_warps == 4
        and launch_config.block_m == 16
    ):
        launch_config = PackedIntAttentionKernelConfig(
            block_q=4,
            block_m=8,
            sliding_window_val=launch_config.sliding_window_val,
            tile_size=launch_config.tile_size,
            head_size_padded=launch_config.head_size_padded,
            head_size_v_padded=launch_config.head_size_v_padded,
            num_warps=launch_config.num_warps,
            num_stages=launch_config.num_stages,
        )
    if (
        q.element_size() >= 2
        and max_query_len is not None
        and max_query_len >= 512
        and layout.k_bits == 5
        and layout.v_bits == 4
        and head_size == 512
        and head_size_v == 512
        and num_queries_per_kv == 8
        and launch_config.sliding_window_val == 0
        and launch_config.tile_size == 8
    ):
        launch_config = PackedIntAttentionKernelConfig(
            block_q=2,
            block_m=16,
            sliding_window_val=launch_config.sliding_window_val,
            tile_size=16,
            head_size_padded=launch_config.head_size_padded,
            head_size_v_padded=launch_config.head_size_v_padded,
            num_warps=8,
            num_stages=launch_config.num_stages,
        )

    block_m = launch_config.block_m
    block_q = launch_config.block_q
    total_num_q_blocks = q.shape[0] // block_q + num_seqs
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
        TILE_SIZE=launch_config.tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=launch_config.head_size_padded,
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=launch_config.head_size_v_padded,
        K_BITS=layout.k_bits,
        V_BITS=layout.v_bits,
        USE_SOFTCAP=(softcap > 0),
        SLIDING_WINDOW=launch_config.sliding_window_val,
        num_warps=launch_config.num_warps,
        num_stages=launch_config.num_stages,
    )


def resolve_packed_int_attention_launch_config(
    *,
    q_element_size: int,
    head_size: int,
    head_size_v: int,
    num_queries_per_kv: int,
    sliding_window: tuple[int, int],
    layout: PackedIntPerTokenHeadLayout,
    launch_config: PackedIntAttentionKernelConfig | None = None,
    max_query_len: int | None = None,
    allow_single_query_override: bool = True,
) -> PackedIntAttentionKernelConfig:
    if launch_config is None:
        launch_config = build_packed_int_attention_kernel_config(
            q_element_size=q_element_size,
            head_size=head_size,
            head_size_v=head_size_v,
            num_queries_per_kv=num_queries_per_kv,
            sliding_window=sliding_window,
        )
    if (
        allow_single_query_override
        and max_query_len == 1
        and q_element_size >= 2
        and num_queries_per_kv == 2
        and head_size == 256
        and head_size_v == 256
        and launch_config.sliding_window_val == 1024
        and launch_config.tile_size == 16
        and launch_config.num_warps == 4
        and launch_config.block_m == 16
    ):
        launch_config = PackedIntAttentionKernelConfig(
            block_q=4,
            block_m=8,
            sliding_window_val=launch_config.sliding_window_val,
            tile_size=launch_config.tile_size,
            head_size_padded=launch_config.head_size_padded,
            head_size_v_padded=launch_config.head_size_v_padded,
            num_warps=launch_config.num_warps,
            num_stages=launch_config.num_stages,
        )
    if (
        q_element_size >= 2
        and max_query_len is not None
        and max_query_len >= 512
        and layout.k_bits == 5
        and layout.v_bits == 4
        and head_size == 512
        and head_size_v == 512
        and num_queries_per_kv == 8
        and launch_config.sliding_window_val == 0
        and launch_config.tile_size == 8
    ):
        launch_config = PackedIntAttentionKernelConfig(
            block_q=2,
            block_m=16,
            sliding_window_val=launch_config.sliding_window_val,
            tile_size=16,
            head_size_padded=launch_config.head_size_padded,
            head_size_v_padded=launch_config.head_size_v_padded,
            num_warps=8,
            num_stages=launch_config.num_stages,
        )
    return launch_config
