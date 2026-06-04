# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_packed_int_kv import (
    PackedIntAttentionKernelConfig,
    build_packed_int_attention_kernel_config,
    get_packed_int_cache_views,
    paged_attention_packed_int,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode, PackedIntPerTokenHeadLayout


def _benchmark_cuda(fn, num_warmup_iters: int, num_iters: int) -> float:
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


class _DummyLayer:
    def __init__(self, head_size_v: int) -> None:
        self.head_size_v = head_size_v


def _make_triton_metadata(
    *,
    seq_len: int,
    query_len: int,
    num_blocks: int,
    device: str,
    dtype: torch.dtype,
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
        slot_mapping=torch.tensor([0], device=device, dtype=torch.int64),
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


def _benchmark_kernel_mode(
    *,
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
    num_warmup_iters: int,
    num_iters: int,
    device: str,
    packed_tile_size: int | None = None,
    packed_num_warps: int | None = None,
    packed_block_m: int | None = None,
) -> tuple[float, float]:
    num_blocks = math.ceil(seq_len / block_size)
    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    packed_out = torch.empty_like(query)
    int8_out = torch.empty_like(query)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
    query_start_loc = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_size)

    packed_layout = PackedIntPerTokenHeadLayout.create(
        k_bits=k_bits,
        v_bits=v_bits,
        head_size=head_size,
        head_size_v=head_size,
    )
    packed_key_cache = torch.randint(
        0,
        256,
        (num_blocks, block_size, num_kv_heads, packed_layout.k_data_bytes),
        dtype=torch.uint8,
        device=device,
    )
    packed_value_cache = torch.randint(
        0,
        256,
        (num_blocks, block_size, num_kv_heads, packed_layout.v_data_bytes),
        dtype=torch.uint8,
        device=device,
    )
    packed_k_scale = (
        torch.rand((num_blocks, block_size, num_kv_heads), device=device) + 0.01
    ).to(torch.float32)
    packed_v_scale = (
        torch.rand((num_blocks, block_size, num_kv_heads), device=device) + 0.01
    ).to(torch.float32)

    int8_key_cache = torch.randint(
        -127,
        128,
        (num_blocks, block_size, num_kv_heads, head_size),
        dtype=torch.int8,
        device=device,
    )
    int8_value_cache = torch.randint(
        -127,
        128,
        (num_blocks, block_size, num_kv_heads, head_size),
        dtype=torch.int8,
        device=device,
    )
    int8_k_scale = (
        torch.rand((num_blocks, block_size, num_kv_heads), device=device) + 0.01
    ).to(torch.float32)
    int8_v_scale = (
        torch.rand((num_blocks, block_size, num_kv_heads), device=device) + 0.01
    ).to(torch.float32)

    packed_launch_config = build_packed_int_attention_kernel_config(
        q_element_size=query.element_size(),
        head_size=head_size,
        head_size_v=head_size,
        num_queries_per_kv=num_query_heads // num_kv_heads,
        sliding_window=(sliding_window, -1),
    )
    if (
        packed_tile_size is not None
        or packed_num_warps is not None
        or packed_block_m is not None
    ):
        block_m = (
            packed_launch_config.block_m if packed_block_m is None else packed_block_m
        )
        block_q = block_m // (num_query_heads // num_kv_heads)
        packed_launch_config = PackedIntAttentionKernelConfig(
            block_q=block_q,
            block_m=block_m,
            sliding_window_val=packed_launch_config.sliding_window_val,
            tile_size=(
                packed_launch_config.tile_size
                if packed_tile_size is None
                else packed_tile_size
            ),
            head_size_padded=packed_launch_config.head_size_padded,
            head_size_v_padded=packed_launch_config.head_size_v_padded,
            num_warps=(
                packed_launch_config.num_warps
                if packed_num_warps is None
                else packed_num_warps
            ),
        )

    packed_kwargs = dict(
        q=query,
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        k_scale_cache=packed_k_scale,
        v_scale_cache=packed_v_scale,
        out=packed_out,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        layout=packed_layout,
        softmax_scale=softmax_scale,
        softcap=0.0,
        num_queries_per_kv=num_query_heads // num_kv_heads,
        sliding_window=(sliding_window, -1),
        launch_config=packed_launch_config,
        max_query_len=query_len,
        allow_single_query_override=(
            packed_tile_size is None
            and packed_num_warps is None
            and packed_block_m is None
        ),
    )
    int8_kwargs = dict(
        q=query,
        k=int8_key_cache,
        v=int8_value_cache,
        out=int8_out,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=query_len,
        seqused_k=seq_lens,
        max_seqlen_k=seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(sliding_window, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=int8_k_scale,
        v_scale_cache=int8_v_scale,
    )

    packed_ms = _benchmark_cuda(
        lambda: paged_attention_packed_int(**packed_kwargs),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    int8_ms = _benchmark_cuda(
        lambda: unified_attention(**int8_kwargs),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    return packed_ms, int8_ms


def _benchmark_impl_mode(
    *,
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
    num_warmup_iters: int,
    num_iters: int,
    device: str,
) -> tuple[float, float]:
    num_blocks = math.ceil(seq_len / block_size)
    layer = _DummyLayer(head_size)
    metadata = _make_triton_metadata(
        seq_len=seq_len,
        query_len=query_len,
        num_blocks=num_blocks,
        device=device,
        dtype=dtype,
    )
    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    key = torch.empty(
        query_len,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value = torch.empty_like(key)
    packed_out = torch.empty_like(query)
    int8_out = torch.empty_like(query)
    scale = 1.0 / math.sqrt(head_size)

    packed_layout = PackedIntPerTokenHeadLayout.create(
        k_bits=k_bits,
        v_bits=v_bits,
        head_size=head_size,
        head_size_v=head_size,
    )
    packed_kv_cache = torch.empty(
        (num_blocks, block_size, num_kv_heads, packed_layout.raw_bytes_per_token_head),
        dtype=torch.uint8,
        device=device,
    )
    packed_key_cache, packed_value_cache, packed_k_scale, packed_v_scale = (
        get_packed_int_cache_views(packed_kv_cache, packed_layout)
    )
    packed_key_cache.copy_(
        torch.randint(0, 256, packed_key_cache.shape, device=device, dtype=torch.uint8)
    )
    packed_value_cache.copy_(
        torch.randint(
            0, 256, packed_value_cache.shape, device=device, dtype=torch.uint8
        )
    )
    packed_k_scale.copy_(
        (torch.rand(packed_k_scale.shape, device=device) + 0.01).to(torch.float32)
    )
    packed_v_scale.copy_(
        (torch.rand(packed_v_scale.shape, device=device) + 0.01).to(torch.float32)
    )

    int8_kv_cache = torch.empty(
        (num_blocks, 2, block_size, num_kv_heads, head_size + 4),
        dtype=torch.int8,
        device=device,
    )
    int8_kv_cache[..., :head_size].copy_(
        torch.randint(
            -127,
            128,
            int8_kv_cache[..., :head_size].shape,
            device=device,
            dtype=torch.int8,
        )
    )

    packed_sliding_window = sliding_window if sliding_window >= 0 else None
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

    packed_ms = _benchmark_cuda(
        lambda: packed_impl.forward(
            layer, query, key, value, packed_kv_cache, metadata, packed_out
        ),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    int8_ms = _benchmark_cuda(
        lambda: int8_impl.forward(
            layer, query, key, value, int8_kv_cache, metadata, int8_out
        ),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    return packed_ms, int8_ms


@torch.inference_mode()
def main(
    mode: str,
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
    num_iters: int,
    packed_tile_size: int | None,
    packed_num_warps: int | None,
    packed_block_m: int | None,
) -> None:
    set_random_seed(seed)
    device = "cuda"

    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    print(
        "shape:",
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
    )
    if mode in ("kernel", "both"):
        packed_ms, int8_ms = _benchmark_kernel_mode(
            seq_len=seq_len,
            query_len=query_len,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            sliding_window=sliding_window,
            k_bits=k_bits,
            v_bits=v_bits,
            dtype=dtype,
            num_warmup_iters=num_warmup_iters,
            num_iters=num_iters,
            device=device,
            packed_tile_size=packed_tile_size,
            packed_num_warps=packed_num_warps,
            packed_block_m=packed_block_m,
        )
        print(f"kernel_packed_int_ms={packed_ms:.6f}")
        print(f"kernel_int8_per_token_head_ms={int8_ms:.6f}")
        print(f"kernel_packed_vs_int8_ratio={packed_ms / int8_ms:.4f}")
    if mode in ("impl", "both"):
        packed_ms, int8_ms = _benchmark_impl_mode(
            seq_len=seq_len,
            query_len=query_len,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            sliding_window=sliding_window,
            k_bits=k_bits,
            v_bits=v_bits,
            dtype=dtype,
            num_warmup_iters=num_warmup_iters,
            num_iters=num_iters,
            device=device,
        )
        print(f"impl_packed_int_ms={packed_ms:.6f}")
        print(f"impl_int8_per_token_head_ms={int8_ms:.6f}")
        print(f"impl_packed_vs_int8_ratio={packed_ms / int8_ms:.4f}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark packed-int paged attention against int8 baseline."
    )
    parser.add_argument(
        "--mode", type=str, choices=["impl", "kernel", "both"], default="impl"
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
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--packed-tile-size", type=int, default=None)
    parser.add_argument("--packed-num-warps", type=int, default=None)
    parser.add_argument("--packed-block-m", type=int, default=None)
    args = parser.parse_args()
    print(args)

    main(
        mode=args.mode,
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
        num_iters=args.num_iters,
        packed_tile_size=args.packed_tile_size,
        packed_num_warps=args.packed_num_warps,
        packed_block_m=args.packed_block_m,
    )
