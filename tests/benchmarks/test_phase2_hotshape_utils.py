# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from benchmarks.kernels.phase2_hotshape_utils import (
    CompiledKernelMetadata,
    PackedFactorizedReferenceResult,
    Phase2HotShapeSpec,
    compute_packed_factorized_reference,
    compiled_metadata_to_dict,
    hotshape_spec_to_dict,
    packed_launch_config_to_dict,
)
from vllm.v1.attention.ops.triton_packed_int_kv import (
    PackedIntAttentionKernelConfig,
    _pack_signed_values,
    resolve_packed_int_attention_launch_config,
)
from vllm.v1.kv_cache_interface import PackedIntPerTokenHeadLayout


def test_phase2_hotshape_spec_properties():
    spec = Phase2HotShapeSpec(
        seq_len=17037,
        query_len=1024,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        block_size=16,
        sliding_window=-1,
        k_bits=5,
        v_bits=4,
    )
    assert spec.num_blocks == math.ceil(17037 / 16)
    assert spec.num_queries_per_kv == 8
    assert spec.softmax_scale == 1.0 / math.sqrt(512)


def test_phase2_hotshape_serializers():
    spec = Phase2HotShapeSpec(
        seq_len=128,
        query_len=16,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        block_size=16,
        sliding_window=-1,
        k_bits=5,
        v_bits=4,
    )
    spec_payload = hotshape_spec_to_dict(spec, dtype=torch.float16)
    assert spec_payload == {
        "seq_len": 128,
        "query_len": 16,
        "num_query_heads": 4,
        "num_kv_heads": 1,
        "head_size": 64,
        "head_size_v": 64,
        "block_size": 16,
        "sliding_window": -1,
        "k_bits": 5,
        "v_bits": 4,
        "dtype": "torch.float16",
    }

    metadata = CompiledKernelMetadata(
        n_regs=208,
        n_spills=0,
        num_warps=4,
        num_stages=3,
        shared=32768,
    )
    metadata_payload = compiled_metadata_to_dict(metadata)
    assert metadata_payload == {
        "compiled_n_regs": 208,
        "compiled_n_spills": 0,
        "compiled_num_warps": 4,
        "compiled_num_stages": 3,
        "compiled_shared": 32768,
    }

    launch = PackedIntAttentionKernelConfig(
        block_q=2,
        block_m=16,
        sliding_window_val=0,
        tile_size=16,
        head_size_padded=512,
        head_size_v_padded=512,
        num_warps=8,
        num_stages=3,
    )
    launch_payload = packed_launch_config_to_dict(launch)
    assert launch_payload == {
        "packed_launch_block_q": 2,
        "packed_launch_block_m": 16,
        "packed_launch_tile_size": 16,
        "packed_launch_num_warps": 8,
        "packed_launch_num_stages": 3,
        "packed_launch_sliding_window_val": 0,
        "packed_launch_head_size_padded": 512,
        "packed_launch_head_size_v_padded": 512,
    }

    effective_payload = packed_launch_config_to_dict(launch, prefix="packed_effective_launch")
    assert effective_payload == {
        "packed_effective_launch_block_q": 2,
        "packed_effective_launch_block_m": 16,
        "packed_effective_launch_tile_size": 16,
        "packed_effective_launch_num_warps": 8,
        "packed_effective_launch_num_stages": 3,
        "packed_effective_launch_sliding_window_val": 0,
        "packed_effective_launch_head_size_padded": 512,
        "packed_effective_launch_head_size_v_padded": 512,
    }


def test_resolve_packed_int_attention_launch_config_overrides():
    layout_512 = PackedIntPerTokenHeadLayout.create(
        k_bits=5,
        v_bits=4,
        head_size=512,
        head_size_v=512,
    )
    requested_512 = PackedIntAttentionKernelConfig(
        block_q=1,
        block_m=8,
        sliding_window_val=0,
        tile_size=8,
        head_size_padded=512,
        head_size_v_padded=512,
        num_warps=4,
        num_stages=3,
    )
    effective_512 = resolve_packed_int_attention_launch_config(
        q_element_size=2,
        head_size=512,
        head_size_v=512,
        num_queries_per_kv=8,
        sliding_window=(-1, -1),
        layout=layout_512,
        launch_config=requested_512,
        max_query_len=1024,
        allow_single_query_override=True,
    )
    assert effective_512 == PackedIntAttentionKernelConfig(
        block_q=2,
        block_m=16,
        sliding_window_val=0,
        tile_size=16,
        head_size_padded=512,
        head_size_v_padded=512,
        num_warps=8,
        num_stages=3,
    )

    layout_256 = PackedIntPerTokenHeadLayout.create(
        k_bits=5,
        v_bits=4,
        head_size=256,
        head_size_v=256,
    )
    requested_256 = PackedIntAttentionKernelConfig(
        block_q=8,
        block_m=16,
        sliding_window_val=1024,
        tile_size=16,
        head_size_padded=256,
        head_size_v_padded=256,
        num_warps=4,
        num_stages=3,
    )
    effective_256 = resolve_packed_int_attention_launch_config(
        q_element_size=2,
        head_size=256,
        head_size_v=256,
        num_queries_per_kv=2,
        sliding_window=(1023, -1),
        layout=layout_256,
        launch_config=requested_256,
        max_query_len=1,
        allow_single_query_override=True,
    )
    assert effective_256 == PackedIntAttentionKernelConfig(
        block_q=4,
        block_m=8,
        sliding_window_val=1024,
        tile_size=16,
        head_size_padded=256,
        head_size_v_padded=256,
        num_warps=4,
        num_stages=3,
    )


def test_packed_factorized_reference_matches_dense_when_scales_are_one():
    spec = Phase2HotShapeSpec(
        seq_len=4,
        query_len=2,
        num_query_heads=2,
        num_kv_heads=1,
        head_size=2,
        head_size_v=2,
        block_size=2,
        sliding_window=-1,
        k_bits=8,
        v_bits=8,
    )
    layout = PackedIntPerTokenHeadLayout.create(
        k_bits=spec.k_bits,
        v_bits=spec.v_bits,
        head_size=spec.head_size,
        head_size_v=spec.head_size_v,
    )
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    key_dense = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[-1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    value_dense = torch.tensor(
        [
            [[1.0, 2.0]],
            [[0.0, 1.0]],
            [[-1.0, 0.0]],
            [[2.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    packed_k = _pack_signed_values(key_dense.to(torch.int32), layout.k_bits)
    packed_v = _pack_signed_values(value_dense.to(torch.int32), layout.v_bits)
    key_cache = torch.empty(
        spec.num_blocks,
        spec.block_size,
        spec.num_kv_heads,
        layout.k_data_bytes,
        dtype=torch.uint8,
    )
    value_cache = torch.empty(
        spec.num_blocks,
        spec.block_size,
        spec.num_kv_heads,
        layout.v_data_bytes,
        dtype=torch.uint8,
    )
    k_scale = torch.ones(
        spec.num_blocks,
        spec.block_size,
        spec.num_kv_heads,
        dtype=torch.float32,
    )
    v_scale = torch.ones_like(k_scale)
    block_table = torch.arange(spec.num_blocks, dtype=torch.int64).view(1, -1)
    key_cache.copy_(packed_k.view(spec.num_blocks, spec.block_size, spec.num_kv_heads, -1))
    value_cache.copy_(
        packed_v.view(spec.num_blocks, spec.block_size, spec.num_kv_heads, -1)
    )
    result = compute_packed_factorized_reference(
        spec=spec,
        hotshape={
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "block_table": block_table,
            "layout": layout,
        },
    )
    assert isinstance(result, PackedFactorizedReferenceResult)

    expanded_key = key_dense.repeat_interleave(spec.num_queries_per_kv, dim=1)
    expanded_value = value_dense.repeat_interleave(spec.num_queries_per_kv, dim=1)
    context_len = spec.seq_len - spec.query_len
    query_abs = context_len + torch.arange(spec.query_len)
    key_pos = torch.arange(spec.seq_len)
    causal_mask = key_pos.view(1, 1, spec.seq_len) <= query_abs.view(spec.query_len, 1, 1)
    full_scores = torch.einsum("thd,shd->ths", query, expanded_key) * spec.softmax_scale
    full_scores = full_scores.masked_fill(~causal_mask, float("-inf"))
    full_prob = torch.softmax(full_scores, dim=-1)
    dense_out = torch.einsum("ths,shd->thd", full_prob, expanded_value)
    dense_lse = torch.logsumexp(full_scores, dim=-1).transpose(0, 1).contiguous()
    assert torch.allclose(result.output, dense_out, atol=1e-6, rtol=0.0)
    assert torch.allclose(result.lse, dense_lse, atol=1e-6, rtol=0.0)
