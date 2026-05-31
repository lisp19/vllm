# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

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
    num_reqs = query_start_loc.shape[0] - 1
    block_size = key_cache.shape[1]
    window = 1 + sliding_window[0] if sliding_window[0] >= 0 else None

    for req_idx in range(num_reqs):
        q_start = int(query_start_loc[req_idx].item())
        q_end = int(query_start_loc[req_idx + 1].item())
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        seq_len = int(seq_lens[req_idx].item())
        context_len = seq_len - q_len
        key_seq, value_seq = _materialize_sequence_kv(
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            block_table[req_idx],
            seq_len,
            block_size,
            layout,
        )
        key_seq = key_seq.repeat_interleave(num_queries_per_kv, dim=1).permute(1, 0, 2)
        value_seq = value_seq.repeat_interleave(num_queries_per_kv, dim=1).permute(
            1, 0, 2
        )
        q_seq = q[q_start:q_end].permute(1, 0, 2).to(torch.float32)

        query_abs = context_len + torch.arange(
            q_len, device=q.device, dtype=torch.int64
        )
        key_pos = torch.arange(seq_len, device=q.device, dtype=torch.int64)
        keep = key_pos.unsqueeze(0) <= query_abs.unsqueeze(1)
        if window is not None:
            keep = keep & (
                key_pos.unsqueeze(0) >= (query_abs.unsqueeze(1) - window + 1)
            )
        attn_bias = torch.zeros((q_len, seq_len), device=q.device, dtype=torch.float32)
        attn_bias.masked_fill_(~keep, float("-inf"))

        scores = torch.matmul(q_seq, key_seq.transpose(-1, -2)) * softmax_scale
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        scores = scores + attn_bias.unsqueeze(0)
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, value_seq)
        out[q_start:q_end].copy_(attn_out.permute(1, 0, 2).to(out.dtype))
