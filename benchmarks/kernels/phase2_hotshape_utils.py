# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.triton_packed_int_kv import (
    PackedIntAttentionKernelConfig,
    _unpack_signed_values,
    build_packed_int_attention_kernel_config,
    kernel_packed_int_attention,
    resolve_packed_int_attention_launch_config,
)
from vllm.v1.kv_cache_interface import PackedIntPerTokenHeadLayout


@dataclass(frozen=True)
class Phase2HotShapeSpec:
    seq_len: int = 17037
    query_len: int = 1024
    num_query_heads: int = 16
    num_kv_heads: int = 2
    head_size: int = 512
    head_size_v: int = 512
    block_size: int = 16
    sliding_window: int = -1
    k_bits: int = 5
    v_bits: int = 4

    @property
    def num_blocks(self) -> int:
        return math.ceil(self.seq_len / self.block_size)

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def softmax_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_size)


@dataclass(frozen=True)
class CompiledKernelMetadata:
    n_regs: int
    n_spills: int
    num_warps: int
    num_stages: int
    shared: int


@dataclass(frozen=True)
class PackedFactorizedReferenceResult:
    output: torch.Tensor  # [T, H, Dv], float32
    lse: torch.Tensor  # [H, T], float32


def hotshape_spec_to_dict(
    spec: Phase2HotShapeSpec,
    *,
    dtype: torch.dtype,
) -> dict[str, object]:
    return {
        "seq_len": spec.seq_len,
        "query_len": spec.query_len,
        "num_query_heads": spec.num_query_heads,
        "num_kv_heads": spec.num_kv_heads,
        "head_size": spec.head_size,
        "head_size_v": spec.head_size_v,
        "block_size": spec.block_size,
        "sliding_window": spec.sliding_window,
        "k_bits": spec.k_bits,
        "v_bits": spec.v_bits,
        "dtype": str(dtype),
    }


def compiled_metadata_to_dict(
    metadata: CompiledKernelMetadata,
) -> dict[str, object]:
    return {
        "compiled_n_regs": metadata.n_regs,
        "compiled_n_spills": metadata.n_spills,
        "compiled_num_warps": metadata.num_warps,
        "compiled_num_stages": metadata.num_stages,
        "compiled_shared": metadata.shared,
    }


def packed_launch_config_to_dict(
    config: PackedIntAttentionKernelConfig,
    *,
    prefix: str = "packed_launch",
) -> dict[str, object]:
    return {
        f"{prefix}_block_q": config.block_q,
        f"{prefix}_block_m": config.block_m,
        f"{prefix}_tile_size": config.tile_size,
        f"{prefix}_num_warps": config.num_warps,
        f"{prefix}_num_stages": config.num_stages,
        f"{prefix}_sliding_window_val": config.sliding_window_val,
        f"{prefix}_head_size_padded": config.head_size_padded,
        f"{prefix}_head_size_v_padded": config.head_size_v_padded,
    }


def benchmark_cuda(fn, num_warmup_iters: int, num_iters: int) -> float:
    for _ in range(num_warmup_iters):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters


def build_phase2_launch_config(
    *,
    q_element_size: int,
    spec: Phase2HotShapeSpec,
    packed_tile_size: int | None = None,
    packed_num_warps: int | None = None,
    packed_block_m: int | None = None,
    packed_num_stages: int | None = None,
) -> PackedIntAttentionKernelConfig:
    launch = build_packed_int_attention_kernel_config(
        q_element_size=q_element_size,
        head_size=spec.head_size,
        head_size_v=spec.head_size_v,
        num_queries_per_kv=spec.num_queries_per_kv,
        sliding_window=(spec.sliding_window, -1),
    )
    if (
        packed_tile_size is None
        and packed_num_warps is None
        and packed_block_m is None
        and packed_num_stages is None
    ):
        return launch

    block_m = launch.block_m if packed_block_m is None else packed_block_m
    return PackedIntAttentionKernelConfig(
        block_q=block_m // spec.num_queries_per_kv,
        block_m=block_m,
        sliding_window_val=launch.sliding_window_val,
        tile_size=launch.tile_size if packed_tile_size is None else packed_tile_size,
        head_size_padded=launch.head_size_padded,
        head_size_v_padded=launch.head_size_v_padded,
        num_warps=launch.num_warps if packed_num_warps is None else packed_num_warps,
        num_stages=launch.num_stages if packed_num_stages is None else packed_num_stages,
    )


def make_phase2_packed_inputs(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor | PackedIntPerTokenHeadLayout]:
    query = torch.randn(
        spec.query_len,
        spec.num_query_heads,
        spec.head_size,
        dtype=dtype,
        device=device,
    )
    out = torch.empty(
        spec.query_len,
        spec.num_query_heads,
        spec.head_size_v,
        dtype=dtype,
        device=device,
    )
    layout = PackedIntPerTokenHeadLayout.create(
        k_bits=spec.k_bits,
        v_bits=spec.v_bits,
        head_size=spec.head_size,
        head_size_v=spec.head_size_v,
    )
    key_cache = torch.randint(
        0,
        256,
        (spec.num_blocks, spec.block_size, spec.num_kv_heads, layout.k_data_bytes),
        dtype=torch.uint8,
        device=device,
    )
    value_cache = torch.randint(
        0,
        256,
        (spec.num_blocks, spec.block_size, spec.num_kv_heads, layout.v_data_bytes),
        dtype=torch.uint8,
        device=device,
    )
    k_scale = (
        torch.rand((spec.num_blocks, spec.block_size, spec.num_kv_heads), device=device)
        + 0.01
    ).to(torch.float32)
    v_scale = (
        torch.rand((spec.num_blocks, spec.block_size, spec.num_kv_heads), device=device)
        + 0.01
    ).to(torch.float32)
    query_start_loc = torch.tensor([0, spec.query_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([spec.seq_len], device=device, dtype=torch.int32)
    block_table = torch.arange(spec.num_blocks, device=device, dtype=torch.int32).view(
        1, -1
    )
    return {
        "query": query,
        "out": out,
        "layout": layout,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "block_table": block_table,
    }


def extract_compiled_kernel():
    return next(iter(kernel_packed_int_attention.device_caches[0][0].values()))


def extract_triton_jit_kernel_compiled(jit_kernel):
    return next(iter(jit_kernel.device_caches[0][0].values()))


def extract_compiled_metadata() -> CompiledKernelMetadata:
    compiled = extract_compiled_kernel()
    return CompiledKernelMetadata(
        n_regs=compiled.n_regs,
        n_spills=compiled.n_spills,
        num_warps=compiled.metadata.num_warps,
        num_stages=compiled.metadata.num_stages,
        shared=compiled.metadata.shared,
    )


def extract_triton_jit_metadata(jit_kernel) -> CompiledKernelMetadata:
    compiled = extract_triton_jit_kernel_compiled(jit_kernel)
    return CompiledKernelMetadata(
        n_regs=compiled.n_regs,
        n_spills=compiled.n_spills,
        num_warps=compiled.metadata.num_warps,
        num_stages=compiled.metadata.num_stages,
        shared=compiled.metadata.shared,
    )


def extract_triton_jit_shared_diagnostics(
    jit_kernel,
) -> dict[str, object]:
    compiled = extract_triton_jit_kernel_compiled(jit_kernel)
    payload: dict[str, object] = {}
    if "ptx" in compiled.asm:
        payload["shared_range_groups"] = summarize_shared_ranges(compiled.asm["ptx"])
    if "ttgir" in compiled.asm:
        payload["ttgir_shared_allocs"] = extract_ttgir_shared_allocs(
            compiled.asm["ttgir"]
        )
    return payload


def summarize_shared_ranges(ptx: str) -> list[tuple[int, int]]:
    offs = []
    for match in re.finditer(r"\[(?:%r\d+|global_smem)(?:\+(\d+))?\]", ptx):
        offs.append(int(match.group(1) or 0))
    offs = sorted(set(offs))
    groups: list[tuple[int, int]] = []
    if not offs:
        return groups
    start = prev = offs[0]
    for off in offs[1:]:
        if off - prev > 2048:
            groups.append((start, prev))
            start = off
        prev = off
    groups.append((start, prev))
    return groups


def extract_ttgir_shared_allocs(ttgir: str) -> list[str]:
    allocs = []
    for line in ttgir.splitlines():
        stripped = line.strip()
        if "ttg.local_alloc" in stripped and "#smem" in stripped:
            allocs.append(stripped)
    return allocs


def compute_packed_factorized_reference(
    *,
    spec: Phase2HotShapeSpec,
    hotshape: dict[str, torch.Tensor | PackedIntPerTokenHeadLayout],
) -> PackedFactorizedReferenceResult:
    query = hotshape["query"]
    key_cache = hotshape["key_cache"]
    value_cache = hotshape["value_cache"]
    k_scale = hotshape["k_scale"]
    v_scale = hotshape["v_scale"]
    block_table = hotshape["block_table"]
    layout = hotshape["layout"]

    pos = torch.arange(spec.seq_len, device=query.device, dtype=torch.int64)
    blocks = block_table[0][pos // spec.block_size]
    slots = pos % spec.block_size
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
    k_int = k_int.repeat_interleave(q_group, dim=1).contiguous()
    v_int = v_int.repeat_interleave(q_group, dim=1).contiguous()
    k_scales = k_scales.repeat_interleave(q_group, dim=1).contiguous()
    v_scales = v_scales.repeat_interleave(q_group, dim=1).contiguous()

    output = torch.empty(
        spec.query_len,
        spec.num_query_heads,
        spec.head_size_v,
        device=query.device,
        dtype=torch.float32,
    )
    lse = torch.empty(
        spec.num_query_heads,
        spec.query_len,
        device=query.device,
        dtype=torch.float32,
    )
    context_len = spec.seq_len - spec.query_len
    query_abs = context_len + torch.arange(
        spec.query_len,
        device=query.device,
        dtype=torch.int64,
    )
    causal_mask = pos.view(1, spec.seq_len) <= query_abs.view(spec.query_len, 1)
    neg_inf = torch.tensor(-math.inf, device=query.device, dtype=torch.float32)
    for head_idx in range(spec.num_query_heads):
        q_head = query[:, head_idx].to(torch.float32)
        scores = torch.matmul(q_head, k_int[:, head_idx].transpose(0, 1))
        scores = scores * (spec.softmax_scale * k_scales[:, head_idx])[None, :]
        scores = torch.where(causal_mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        output[:, head_idx] = torch.matmul(
            probs * v_scales[:, head_idx][None, :],
            v_int[:, head_idx],
        )
        lse[head_idx] = torch.logsumexp(scores, dim=-1)
    return PackedFactorizedReferenceResult(output=output, lse=lse)
