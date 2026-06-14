# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    from .phase2_hotshape_utils import (
        Phase2HotShapeSpec,
        compute_packed_factorized_reference,
        make_phase2_packed_inputs,
    )
except ImportError:
    from phase2_hotshape_utils import (
        Phase2HotShapeSpec,
        compute_packed_factorized_reference,
        make_phase2_packed_inputs,
    )


@dataclass(frozen=True)
class StageBufferPlan:
    num_kv_splits: int
    split_ranges: tuple[tuple[int, int], ...]
    split_lengths: tuple[int, ...]
    max_split_tokens: int
    query_tokens_per_split: int
    query_tile_size_hint: int
    query_tile_ranges: tuple[tuple[int, int], ...]
    num_query_tiles: int
    max_query_tile_tokens: int
    mid_o_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int]
    mid_o_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class ReferenceAttentionInputs:
    query: torch.Tensor  # [T, H, D], original dtype
    key: torch.Tensor  # [S, H, D], expanded to query-head count
    value: torch.Tensor  # [S, H, Dv], expanded to query-head count
    full_scores: torch.Tensor  # [T, H, S], float32
    full_out_ref: torch.Tensor  # [T, H, Dv], float32
    full_lse_ref: torch.Tensor  # [H, T], float32


@dataclass(frozen=True)
class PackedReferenceInputs:
    hotshape: dict[str, object]
    dense_inputs: ReferenceAttentionInputs
    dense_inputs_high_precision: ReferenceAttentionInputs
    factorized_out_ref: torch.Tensor  # [T, H, Dv], float32
    factorized_lse_ref: torch.Tensor  # [H, T], float32


def stage_buffer_plan_to_dict(plan: StageBufferPlan) -> dict[str, object]:
    return {
        "stage_num_kv_splits": plan.num_kv_splits,
        "stage_split_ranges": plan.split_ranges,
        "stage_split_lengths": plan.split_lengths,
        "stage_max_split_tokens": plan.max_split_tokens,
        "stage_query_tokens_per_split": plan.query_tokens_per_split,
        "stage_query_tile_size_hint": plan.query_tile_size_hint,
        "stage_num_query_tiles": plan.num_query_tiles,
        "stage_max_query_tile_tokens": plan.max_query_tile_tokens,
        "stage_mid_o_shape": plan.mid_o_shape,
        "stage_mid_o_bytes": plan.mid_o_bytes,
        "stage_output_shape": plan.output_shape,
        "stage_output_bytes": plan.output_bytes,
    }


def make_reference_attention_inputs(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
) -> ReferenceAttentionInputs:
    if spec.num_query_heads % spec.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    q = torch.randn(
        spec.query_len,
        spec.num_query_heads,
        spec.head_size,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        spec.seq_len,
        spec.num_kv_heads,
        spec.head_size,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        spec.seq_len,
        spec.num_kv_heads,
        spec.head_size_v,
        device=device,
        dtype=dtype,
    )

    q_group = spec.num_query_heads // spec.num_kv_heads
    k = k.repeat_interleave(q_group, dim=1)
    v = v.repeat_interleave(q_group, dim=1)

    return make_reference_attention_inputs_from_tensors(
        spec=spec,
        query=q,
        key=k,
        value=v,
    )


def make_reference_attention_inputs_from_tensors(
    *,
    spec: Phase2HotShapeSpec,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> ReferenceAttentionInputs:
    expected_query_shape = (
        spec.query_len,
        spec.num_query_heads,
        spec.head_size,
    )
    expected_key_shape = (
        spec.seq_len,
        spec.num_query_heads,
        spec.head_size,
    )
    expected_value_shape = (
        spec.seq_len,
        spec.num_query_heads,
        spec.head_size_v,
    )
    if tuple(query.shape) != expected_query_shape:
        raise ValueError(
            f"query shape mismatch: {tuple(query.shape)} != {expected_query_shape}"
        )
    if tuple(key.shape) != expected_key_shape:
        raise ValueError(
            f"key shape mismatch: {tuple(key.shape)} != {expected_key_shape}"
        )
    if tuple(value.shape) != expected_value_shape:
        raise ValueError(
            f"value shape mismatch: {tuple(value.shape)} != {expected_value_shape}"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError("query/key/value must be on the same device")

    context_len = spec.seq_len - spec.query_len
    q_abs = context_len + torch.arange(spec.query_len, device=query.device)
    k_abs = torch.arange(spec.seq_len, device=query.device)

    full_scores = torch.einsum("thd,shd->ths", query.float(), key.float())
    full_scores.mul_(spec.softmax_scale)
    causal_mask = k_abs.view(1, 1, spec.seq_len) <= q_abs.view(spec.query_len, 1, 1)
    full_scores = full_scores.masked_fill(~causal_mask, float("-inf"))
    full_lse = torch.logsumexp(full_scores, dim=-1)  # [T, H]
    full_prob = torch.softmax(full_scores, dim=-1)
    full_out = torch.einsum("ths,shd->thd", full_prob, value.float()).contiguous()

    return ReferenceAttentionInputs(
        query=query,
        key=key,
        value=value,
        full_scores=full_scores,
        full_out_ref=full_out,
        full_lse_ref=full_lse.transpose(0, 1).contiguous(),
    )


def make_packed_reference_inputs(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
) -> PackedReferenceInputs:
    from vllm.v1.attention.ops.triton_packed_int_kv import _materialize_sequence_kv

    hotshape = make_phase2_packed_inputs(spec=spec, dtype=dtype, device=device)
    dense_key, dense_value = _materialize_sequence_kv(
        hotshape["key_cache"],
        hotshape["value_cache"],
        hotshape["k_scale"],
        hotshape["v_scale"],
        hotshape["block_table"][0],
        spec.seq_len,
        spec.block_size,
        hotshape["layout"],
    )
    q_group = spec.num_query_heads // spec.num_kv_heads
    dense_inputs = make_reference_attention_inputs_from_tensors(
        spec=spec,
        query=hotshape["query"],
        key=dense_key.repeat_interleave(q_group, dim=1).contiguous().to(dtype),
        value=dense_value.repeat_interleave(q_group, dim=1).contiguous().to(dtype),
    )
    dense_inputs_high_precision = make_reference_attention_inputs_from_tensors(
        spec=spec,
        query=hotshape["query"],
        key=dense_key.repeat_interleave(q_group, dim=1).contiguous(),
        value=dense_value.repeat_interleave(q_group, dim=1).contiguous(),
    )
    factorized = compute_packed_factorized_reference(spec=spec, hotshape=hotshape)
    return PackedReferenceInputs(
        hotshape=hotshape,
        dense_inputs=dense_inputs,
        dense_inputs_high_precision=dense_inputs_high_precision,
        factorized_out_ref=factorized.output,
        factorized_lse_ref=factorized.lse,
    )


def resolve_dense_reference_inputs(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
    prefer_high_precision: bool = False,
) -> ReferenceAttentionInputs:
    if reference_inputs is None:
        return make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
    if isinstance(reference_inputs, PackedReferenceInputs):
        if prefer_high_precision:
            return reference_inputs.dense_inputs_high_precision
        return reference_inputs.dense_inputs
    return reference_inputs


def resolve_packed_reference_inputs(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs | None,
) -> PackedReferenceInputs:
    if isinstance(reference_inputs, PackedReferenceInputs):
        return reference_inputs
    if reference_inputs is not None:
        raise TypeError("Packed reference inputs required for packed-backed stage1 path")
    return make_packed_reference_inputs(spec=spec, dtype=dtype, device=device)


def build_stage_buffer_plan(
    spec: Phase2HotShapeSpec,
    *,
    num_kv_splits: int,
    output_dtype: torch.dtype,
    query_tile_size_hint: int = 16,
) -> StageBufferPlan:
    split_size = math.ceil(spec.seq_len / num_kv_splits)
    split_ranges = tuple(
        (
            split_idx * split_size,
            min((split_idx + 1) * split_size, spec.seq_len),
        )
        for split_idx in range(num_kv_splits)
    )
    split_lengths = tuple(stop - start for start, stop in split_ranges)
    query_tile_ranges = tuple(
        (
            start,
            min(start + query_tile_size_hint, spec.query_len),
        )
        for start in range(0, spec.query_len, query_tile_size_hint)
    )
    query_tile_lengths = tuple(stop - start for start, stop in query_tile_ranges)
    mid_o_shape = (
        spec.query_len,
        spec.num_query_heads,
        num_kv_splits,
        spec.head_size_v + 1,
    )
    output_shape = (spec.query_len, spec.num_query_heads, spec.head_size_v)
    mid_o_bytes = (
        spec.query_len
        * spec.num_query_heads
        * num_kv_splits
        * (spec.head_size_v + 1)
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    output_bytes = (
        spec.query_len
        * spec.num_query_heads
        * spec.head_size_v
        * torch.tensor([], dtype=output_dtype).element_size()
    )
    return StageBufferPlan(
        num_kv_splits=num_kv_splits,
        split_ranges=split_ranges,
        split_lengths=split_lengths,
        max_split_tokens=max(split_lengths) if split_lengths else 0,
        query_tokens_per_split=spec.query_len,
        query_tile_size_hint=query_tile_size_hint,
        query_tile_ranges=query_tile_ranges,
        num_query_tiles=len(query_tile_ranges),
        max_query_tile_tokens=max(query_tile_lengths) if query_tile_lengths else 0,
        mid_o_shape=mid_o_shape,
        output_shape=output_shape,
        mid_o_bytes=mid_o_bytes,
        output_bytes=output_bytes,
    )


def validate_stage_buffer_plan(
    *,
    spec: Phase2HotShapeSpec,
    num_kv_splits: int,
    plan: StageBufferPlan,
    output_dtype: torch.dtype,
    query_tile_size_hint: int = 16,
) -> None:
    expected = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=output_dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    if plan != expected:
        raise ValueError(
            "stage buffer plan mismatch:\n"
            f"actual={plan}\n"
            f"expected={expected}"
        )


def merge_two_partials_torch(
    prefix_out: torch.Tensor,  # [T, H, D]
    prefix_lse: torch.Tensor,  # [H, T]
    suffix_out: torch.Tensor,  # [T, H, D]
    suffix_lse: torch.Tensor,  # [H, T]
) -> tuple[torch.Tensor, torch.Tensor]:
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p_se = torch.exp(prefix_lse - max_lse)
    s_se = torch.exp(suffix_lse - max_lse)
    out_se = p_se + s_se
    p_scale = (p_se / out_se).transpose(0, 1).unsqueeze(-1)  # [T, H, 1]
    s_scale = (s_se / out_se).transpose(0, 1).unsqueeze(-1)  # [T, H, 1]
    merged_out = prefix_out * p_scale + suffix_out * s_scale
    merged_lse = torch.log(out_se) + max_lse
    return merged_out, merged_lse


def pack_stage_buffer(
    partial_out: torch.Tensor,  # [T, H, S, D], float32
    partial_lse: torch.Tensor,  # [S, H, T], float32
) -> torch.Tensor:
    num_tokens, num_heads, num_splits, head_size_v = partial_out.shape
    expected_lse_shape = (num_splits, num_heads, num_tokens)
    if tuple(partial_lse.shape) != expected_lse_shape:
        raise ValueError(
            f"partial_lse shape mismatch: {tuple(partial_lse.shape)} != {expected_lse_shape}"
        )
    if partial_out.dtype != torch.float32:
        raise ValueError(f"partial_out dtype must be float32, got {partial_out.dtype}")
    if partial_lse.dtype != torch.float32:
        raise ValueError(f"partial_lse dtype must be float32, got {partial_lse.dtype}")

    mid_o = torch.empty(
        num_tokens,
        num_heads,
        num_splits,
        head_size_v + 1,
        device=partial_out.device,
        dtype=torch.float32,
    )
    mid_o[..., :-1].copy_(partial_out)
    mid_o[..., -1].copy_(partial_lse.permute(2, 1, 0).contiguous())
    return mid_o


def allocate_stage_buffer(
    plan: StageBufferPlan,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    return torch.empty(plan.mid_o_shape, device=device, dtype=torch.float32)


def allocate_partial_buffers(
    plan: StageBufferPlan,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    partial_out = torch.empty(
        plan.output_shape[0],
        plan.output_shape[1],
        plan.num_kv_splits,
        plan.output_shape[2],
        device=device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        plan.num_kv_splits,
        plan.output_shape[1],
        plan.output_shape[0],
        device=device,
        dtype=torch.float32,
    )
    return partial_out, partial_lse


def allocate_stage2_outputs(
    plan: StageBufferPlan,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(plan.output_shape, device=device, dtype=torch.float32)
    lse = torch.empty(
        (plan.output_shape[1], plan.output_shape[0]),
        device=device,
        dtype=torch.float32,
    )
    return output, lse


def unpack_stage_buffer(
    mid_o: torch.Tensor,  # [T, H, S, D+1], float32
) -> tuple[torch.Tensor, torch.Tensor]:
    if mid_o.dtype != torch.float32:
        raise ValueError(f"mid_o dtype must be float32, got {mid_o.dtype}")
    partial_out = mid_o[..., :-1].contiguous()  # [T, H, S, D]
    partial_lse = mid_o[..., -1].permute(2, 1, 0).contiguous()  # [S, H, T]
    return partial_out, partial_lse


def reduce_reference_stage_buffer(
    mid_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    partial_out, partial_lse = unpack_stage_buffer(mid_o)
    num_kv_splits = partial_out.shape[2]

    merged = partial_out[:, :, 0, :].clone()
    merged_lse = partial_lse[0].clone()
    for split_idx in range(1, num_kv_splits):
        merged, merged_lse = merge_two_partials_torch(
            merged,
            merged_lse,
            partial_out[:, :, split_idx, :].contiguous(),
            partial_lse[split_idx].contiguous(),
        )
    return merged, merged_lse


def reduce_reference_stage_buffer_into(
    mid_o: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    merged, merged_lse = reduce_reference_stage_buffer(mid_o)
    output.copy_(merged)
    lse.copy_(merged_lse)


def materialize_reference_stage_buffers(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, torch.Tensor]:
    plan, partial_out, partial_lse, full_out, full_lse = materialize_reference_partials(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
    )
    mid_o = pack_stage_buffer(partial_out, partial_lse)
    return plan, mid_o, full_out, full_lse


def materialize_reference_stage_buffers_from_inputs(
    *,
    spec: Phase2HotShapeSpec,
    reference_inputs: ReferenceAttentionInputs | PackedReferenceInputs,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, torch.Tensor]:
    plan, partial_out, partial_lse, full_out, full_lse = (
        materialize_reference_partials_from_inputs(
            spec=spec,
            inputs=reference_inputs,
            num_kv_splits=num_kv_splits,
            query_tile_size_hint=query_tile_size_hint,
        )
    )
    mid_o = pack_stage_buffer(partial_out, partial_lse)
    return plan, mid_o, full_out, full_lse


def materialize_reference_partials(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = make_reference_attention_inputs(spec=spec, dtype=dtype, device=device)
    return materialize_reference_partials_from_inputs(
        spec=spec,
        inputs=inputs,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
    )


def materialize_reference_partials_from_inputs(
    *,
    spec: Phase2HotShapeSpec,
    inputs: ReferenceAttentionInputs | PackedReferenceInputs,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = resolve_dense_reference_inputs(
        spec=spec,
        dtype=inputs.dense_inputs.query.dtype if isinstance(inputs, PackedReferenceInputs) else inputs.query.dtype,
        device=(
            str(inputs.dense_inputs.query.device)
            if isinstance(inputs, PackedReferenceInputs)
            else str(inputs.query.device)
        ),
        reference_inputs=inputs,
    )
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=inputs.query.dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    partial_out, partial_lse = allocate_partial_buffers(plan, device=inputs.query.device)
    for split_idx, (start, stop) in enumerate(plan.split_ranges):
        if start >= stop:
            partial_out[:, :, split_idx, :].zero_()
            partial_lse[split_idx].fill_(float("-inf"))
            continue
        split_scores = inputs.full_scores[..., start:stop]
        valid = torch.isfinite(split_scores).any(dim=-1)  # [T, H]
        split_lse = torch.where(
            valid,
            torch.logsumexp(split_scores, dim=-1),
            torch.full_like(inputs.full_lse_ref.transpose(0, 1), float("-inf")),
        )
        safe_scores = torch.where(
            valid[..., None],
            split_scores,
            torch.zeros_like(split_scores),
        )
        split_prob = torch.softmax(safe_scores, dim=-1)
        split_prob = torch.where(valid[..., None], split_prob, torch.zeros_like(split_prob))
        split_out = torch.einsum(
            "ths,shd->thd", split_prob, inputs.value[start:stop].float()
        )
        partial_out[:, :, split_idx, :].copy_(split_out)
        partial_lse[split_idx].copy_(split_lse.transpose(0, 1))
    return (
        plan,
        partial_out,
        partial_lse,
        inputs.full_out_ref,
        inputs.full_lse_ref,
    )


def expand_packed_factorized_terms(
    *,
    spec: Phase2HotShapeSpec,
    hotshape: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = hotshape["query"].float()
    block_table = hotshape["block_table"]
    key_cache = hotshape["key_cache"]
    value_cache = hotshape["value_cache"]
    k_scale = hotshape["k_scale"]
    v_scale = hotshape["v_scale"]
    layout = hotshape["layout"]
    pos = torch.arange(spec.seq_len, device=query.device, dtype=torch.int64)
    blocks = block_table[0][pos // spec.block_size]
    slots = pos % spec.block_size
    from vllm.v1.attention.ops.triton_packed_int_kv import _unpack_signed_values

    packed_k = key_cache[blocks, slots]
    packed_v = value_cache[blocks, slots]
    k_scales = k_scale[blocks, slots].to(torch.float32)
    v_scales = v_scale[blocks, slots].to(torch.float32)
    k_int = _unpack_signed_values(
        packed_k,
        bits=layout.k_bits,
        num_elements=layout.head_size,
    ).to(torch.float32)
    v_int = _unpack_signed_values(
        packed_v,
        bits=layout.v_bits,
        num_elements=layout.head_size_v,
    ).to(torch.float32)
    q_group = spec.num_query_heads // spec.num_kv_heads
    return (
        query,
        k_int.repeat_interleave(q_group, dim=1).contiguous(),
        v_int.repeat_interleave(q_group, dim=1).contiguous(),
        k_scales.repeat_interleave(q_group, dim=1).contiguous(),
        v_scales.repeat_interleave(q_group, dim=1).contiguous(),
    )


def materialize_packed_factorized_partials(
    *,
    spec: Phase2HotShapeSpec,
    hotshape: dict[str, object],
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    plan = build_stage_buffer_plan(
        spec,
        num_kv_splits=num_kv_splits,
        output_dtype=hotshape["query"].dtype,
        query_tile_size_hint=query_tile_size_hint,
    )
    partial_out, partial_lse = allocate_partial_buffers(
        plan,
        device=hotshape["query"].device,
    )
    full_out, full_lse = materialize_packed_factorized_partials_into(
        spec=spec,
        hotshape=hotshape,
        plan=plan,
        partial_out=partial_out,
        partial_lse=partial_lse,
    )
    return (
        plan,
        partial_out,
        partial_lse,
        full_out,
        full_lse,
    )


def materialize_packed_factorized_partials_into(
    *,
    spec: Phase2HotShapeSpec,
    hotshape: dict[str, object],
    plan: StageBufferPlan,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    materialize_packed_factorized_partials_only_into(
        spec=spec,
        hotshape=hotshape,
        plan=plan,
        partial_out=partial_out,
        partial_lse=partial_lse,
    )
    factorized_full = compute_packed_factorized_reference(spec=spec, hotshape=hotshape)
    return factorized_full.output, factorized_full.lse


def materialize_packed_factorized_partials_only_into(
    *,
    spec: Phase2HotShapeSpec,
    hotshape: dict[str, object],
    plan: StageBufferPlan,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> None:
    expected_out_shape = (
        plan.output_shape[0],
        plan.output_shape[1],
        plan.num_kv_splits,
        plan.output_shape[2],
    )
    expected_lse_shape = (
        plan.num_kv_splits,
        plan.output_shape[1],
        plan.output_shape[0],
    )
    if tuple(partial_out.shape) != expected_out_shape:
        raise ValueError(
            f"partial_out shape mismatch: {tuple(partial_out.shape)} != {expected_out_shape}"
        )
    if tuple(partial_lse.shape) != expected_lse_shape:
        raise ValueError(
            f"partial_lse shape mismatch: {tuple(partial_lse.shape)} != {expected_lse_shape}"
    )
    query, k_int, v_int, k_scales, v_scales = expand_packed_factorized_terms(
        spec=spec,
        hotshape=hotshape,
    )
    context_len = spec.seq_len - spec.query_len
    q_abs = context_len + torch.arange(spec.query_len, device=query.device)
    k_abs = torch.arange(spec.seq_len, device=query.device)
    causal_mask = k_abs.view(1, spec.seq_len) <= q_abs.view(spec.query_len, 1)
    for split_idx, (start, stop) in enumerate(plan.split_ranges):
        if start >= stop:
            partial_out[:, :, split_idx, :].zero_()
            partial_lse[split_idx].fill_(float("-inf"))
            continue
        split_scores = torch.einsum("thd,shd->ths", query, k_int[start:stop])
        split_scores.mul_(spec.softmax_scale)
        split_scores.mul_(k_scales[start:stop].transpose(0, 1).unsqueeze(0))
        split_mask = causal_mask[:, start:stop]
        split_scores = torch.where(
            split_mask[:, None, :],
            split_scores,
            torch.full_like(split_scores, float("-inf")),
        )
        valid = torch.isfinite(split_scores).any(dim=-1)
        split_lse = torch.where(
            valid,
            torch.logsumexp(split_scores, dim=-1),
            torch.full(
                (spec.query_len, spec.num_query_heads),
                float("-inf"),
                device=query.device,
                dtype=torch.float32,
            ),
        )
        safe_scores = torch.where(valid[..., None], split_scores, torch.zeros_like(split_scores))
        split_prob = torch.softmax(safe_scores, dim=-1)
        split_prob = torch.where(valid[..., None], split_prob, torch.zeros_like(split_prob))
        scaled_v = v_int[start:stop] * v_scales[start:stop].unsqueeze(-1)
        split_out = torch.einsum("ths,shd->thd", split_prob, scaled_v)
        partial_out[:, :, split_idx, :].copy_(split_out)
        partial_lse[split_idx].copy_(split_lse.transpose(0, 1))


def materialize_packed_factorized_stage_buffers(
    *,
    spec: Phase2HotShapeSpec,
    hotshape: dict[str, object],
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, torch.Tensor]:
    plan, partial_out, partial_lse, full_out, full_lse = (
        materialize_packed_factorized_partials(
            spec=spec,
            hotshape=hotshape,
            num_kv_splits=num_kv_splits,
            query_tile_size_hint=query_tile_size_hint,
        )
    )
    return plan, pack_stage_buffer(partial_out, partial_lse), full_out, full_lse


def run_reference_splitkv_reduce(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor]:
    plan, mid_o, _, _ = materialize_reference_stage_buffers(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
    )
    return plan, *reduce_reference_stage_buffer(mid_o)


def run_reference_splitkv_parity(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[float, float]:
    _, mid_o, full_out, full_lse = materialize_reference_stage_buffers(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
    )
    merged, merged_lse = reduce_reference_stage_buffer(mid_o)
    out_max_abs = (merged - full_out).abs().max().item()
    lse_max_abs = (merged_lse - full_lse).abs().max().item()
    return out_max_abs, lse_max_abs


def run_reference_pipeline(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, float, float]:
    plan, mid_o, full_out, full_lse = materialize_reference_stage_buffers(
        spec=spec,
        dtype=dtype,
        device=device,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
    )
    merged, merged_lse = reduce_reference_stage_buffer(mid_o)
    out_max_abs = (merged - full_out).abs().max().item()
    lse_max_abs = (merged_lse - full_lse).abs().max().item()
    return plan, merged, merged_lse, out_max_abs, lse_max_abs


def run_packed_factorized_pipeline(
    *,
    spec: Phase2HotShapeSpec,
    hotshape: dict[str, object],
    num_kv_splits: int,
    query_tile_size_hint: int = 16,
) -> tuple[StageBufferPlan, torch.Tensor, torch.Tensor, float, float]:
    plan, mid_o, full_out, full_lse = materialize_packed_factorized_stage_buffers(
        spec=spec,
        hotshape=hotshape,
        num_kv_splits=num_kv_splits,
        query_tile_size_hint=query_tile_size_hint,
    )
    merged, merged_lse = reduce_reference_stage_buffer(mid_o)
    out_max_abs = (merged - full_out).abs().max().item()
    lse_max_abs = (merged_lse - full_lse).abs().max().item()
    return plan, merged, merged_lse, out_max_abs, lse_max_abs
