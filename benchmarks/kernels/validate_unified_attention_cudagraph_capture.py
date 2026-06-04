# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention.attention import _encode_layer_name
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_packed_int_kv import get_packed_int_cache_views
from vllm.v1.kv_cache_interface import PackedIntPerTokenHeadLayout


class _DummyLayer:
    def __init__(
        self,
        *,
        impl: TritonAttentionImpl,
        kv_cache: torch.Tensor,
        layer_name: str,
        head_size_v: int,
    ) -> None:
        self.impl = impl
        self.kv_cache = kv_cache
        self.layer_name = layer_name
        self.head_size_v = head_size_v


def _make_metadata(
    *,
    seq_len: int,
    query_len: int,
    num_blocks: int,
    device: str,
    dtype: torch.dtype,
    slot_mapping: torch.Tensor,
) -> TritonAttentionMetadata:
    return TritonAttentionMetadata(
        num_actual_tokens=query_len,
        max_query_len=query_len,
        query_start_loc=torch.tensor([0, query_len], device=device, dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        block_table=torch.arange(num_blocks, device=device, dtype=torch.int32).view(
            1, -1
        ),
        slot_mapping=slot_mapping,
        seq_threshold_3D=0,
        num_par_softmax_segments=1,
        softmax_segm_output=torch.empty((1, 1, 1, 1), device=device, dtype=dtype),
        softmax_segm_max=torch.empty((1, 1, 1), device=device, dtype=torch.float32),
        softmax_segm_expsum=torch.empty((1, 1, 1), device=device, dtype=torch.float32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        mm_prefix_range=None,
        mm_prefix_range_tensor=None,
    )


def _make_fake_vllm_config(layer_name: str, layer: _DummyLayer):
    compilation_config = SimpleNamespace(
        static_forward_context={layer_name: layer},
        static_all_moe_layers=[],
        fast_moe_cold_start=False,
    )
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        is_moe_model=False,
    )
    return SimpleNamespace(
        compilation_config=compilation_config,
        parallel_config=parallel_config,
    )


def _make_packed_kv_cache(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    k_bits: int,
    v_bits: int,
    device: str,
) -> tuple[torch.Tensor, PackedIntPerTokenHeadLayout]:
    layout = PackedIntPerTokenHeadLayout.create(
        k_bits=k_bits,
        v_bits=v_bits,
        head_size=head_size,
        head_size_v=head_size,
    )
    kv_cache = torch.empty(
        (num_blocks, block_size, num_kv_heads, layout.raw_bytes_per_token_head),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache, k_scale_cache, v_scale_cache = get_packed_int_cache_views(
        kv_cache, layout
    )
    key_cache.random_(0, 256)
    value_cache.random_(0, 256)
    k_scale_cache.copy_((torch.rand_like(k_scale_cache) + 0.01).to(torch.float32))
    v_scale_cache.copy_((torch.rand_like(v_scale_cache) + 0.01).to(torch.float32))
    return kv_cache, layout


def _make_int8_kv_cache(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    device: str,
) -> torch.Tensor:
    kv_cache = torch.empty(
        (num_blocks, 2, block_size, num_kv_heads, head_size + 4),
        dtype=torch.int8,
        device=device,
    )
    kv_cache[..., :head_size].random_(-127, 128)
    return kv_cache


def _run_capture(
    *,
    impl: TritonAttentionImpl,
    kv_cache: torch.Tensor,
    metadata: TritonAttentionMetadata,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    do_kv_update: bool,
    num_warmup_iters: int,
    num_replays: int,
) -> tuple[bool, str | None]:
    layer = _DummyLayer(
        impl=impl,
        kv_cache=kv_cache,
        layer_name=layer_name,
        head_size_v=value.shape[-1],
    )
    vllm_config = _make_fake_vllm_config(layer_name, layer)
    encoded = _encode_layer_name(layer_name)
    slot_mapping = {layer_name: metadata.slot_mapping}
    attn_metadata = {layer_name: metadata}

    def _step() -> None:
        kv_cache_dummy_dep = None
        if do_kv_update:
            kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                key,
                value,
                encoded,
            )
        torch.ops.vllm.unified_attention_with_output(
            query,
            key,
            value,
            out,
            encoded,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )

    try:
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            num_tokens=query.shape[0],
            slot_mapping=slot_mapping,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        ):
            for _ in range(num_warmup_iters):
                _step()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _step()
            torch.cuda.synchronize()

            for _ in range(num_replays):
                graph.replay()
            torch.cuda.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return False, repr(exc)


@torch.inference_mode()
def main(
    *,
    path: str,
    seq_len: int,
    query_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    sliding_window: int,
    k_bits: int,
    v_bits: int,
    dtype: torch.dtype,
    seed: int,
    num_warmup_iters: int,
    num_replays: int,
) -> None:
    set_random_seed(seed)
    device = "cuda"

    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    num_blocks = math.ceil(seq_len / block_size)
    slot_mapping = torch.zeros((query_len,), device=device, dtype=torch.int64)
    metadata = _make_metadata(
        seq_len=seq_len,
        query_len=query_len,
        num_blocks=num_blocks,
        device=device,
        dtype=dtype,
        slot_mapping=slot_mapping,
    )
    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    key = torch.randn(
        query_len,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value = torch.randn_like(key)
    packed_out = torch.empty_like(query)
    int8_out = torch.empty_like(query)
    scale = 1.0 / math.sqrt(head_size)
    packed_sliding_window = sliding_window if sliding_window >= 0 else None
    layer_name = "layer.0"

    packed_kv_cache, packed_layout = _make_packed_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        k_bits=k_bits,
        v_bits=v_bits,
        device=device,
    )
    int8_kv_cache = _make_int8_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        device=device,
    )

    packed_impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=packed_sliding_window,
        kv_cache_dtype="intx_k_inty_v_per_token_head",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_cache_k_bits=k_bits,
        kv_cache_v_bits=v_bits,
    )
    int8_impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=packed_sliding_window,
        kv_cache_dtype="int8_per_token_head",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
    )

    do_kv_update = path == "decode_step"
    packed_ok, packed_err = _run_capture(
        impl=packed_impl,
        kv_cache=packed_kv_cache,
        metadata=metadata,
        layer_name=layer_name,
        query=query,
        key=key,
        value=value,
        out=packed_out,
        do_kv_update=do_kv_update,
        num_warmup_iters=num_warmup_iters,
        num_replays=num_replays,
    )
    int8_ok, int8_err = _run_capture(
        impl=int8_impl,
        kv_cache=int8_kv_cache,
        metadata=metadata,
        layer_name=layer_name,
        query=query,
        key=key,
        value=value,
        out=int8_out,
        do_kv_update=do_kv_update,
        num_warmup_iters=num_warmup_iters,
        num_replays=num_replays,
    )

    print(
        "shape:",
        f"path={path}",
        f"seq_len={seq_len}",
        f"query_len={query_len}",
        f"q_heads={num_query_heads}",
        f"kv_heads={num_kv_heads}",
        f"head_size={head_size}",
        f"block_size={block_size}",
        f"sliding_window={sliding_window}",
        f"k_bits={k_bits}",
        f"v_bits={v_bits}",
        f"dtype={dtype}",
        f"packed_slot_bytes={packed_layout.slot_bytes}",
    )
    print(f"packed_int_unified_capture_ok={packed_ok}")
    if packed_err is not None:
        print(f"packed_int_unified_capture_error={packed_err}")
    print(f"int8_per_token_head_unified_capture_ok={int8_ok}")
    if int8_err is not None:
        print(f"int8_per_token_head_unified_capture_error={int8_err}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=(
            "Validate CUDA-graph capture through vLLM unified attention custom "
            "ops on packed-int and int8 decode paths."
        )
    )
    parser.add_argument(
        "--path",
        type=str,
        choices=["attention_only", "decode_step"],
        default="decode_step",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=3)
    parser.add_argument("--num-replays", type=int, default=3)
    args = parser.parse_args()
    print(args)

    main(
        path=args.path,
        seq_len=args.seq_len,
        query_len=args.query_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        sliding_window=args.sliding_window,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        num_warmup_iters=args.num_warmup_iters,
        num_replays=args.num_replays,
    )
