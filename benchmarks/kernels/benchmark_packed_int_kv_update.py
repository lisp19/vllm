# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.kv_cache_interface import PackedIntPerTokenHeadLayout


class _DummyLayer:
    def __init__(self, head_size_v: int) -> None:
        self.head_size_v = head_size_v


def _benchmark(fn, num_warmup_iters: int, num_iters: int) -> float:
    for _ in range(num_warmup_iters):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e6 / num_iters


@torch.inference_mode()
def main(
    num_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    sliding_window: int,
    k_bits: int,
    v_bits: int,
    dtype: torch.dtype,
    seed: int,
    num_warmup_iters: int,
    num_iters: int,
) -> None:
    set_random_seed(seed)
    device = "cuda"
    layer = _DummyLayer(head_size)

    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_size,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)

    packed_layout = PackedIntPerTokenHeadLayout.create(
        k_bits=k_bits,
        v_bits=v_bits,
        head_size=head_size,
        head_size_v=head_size,
    )
    packed_kv = torch.empty(
        (num_blocks, block_size, num_kv_heads, packed_layout.raw_bytes_per_token_head),
        device=device,
        dtype=torch.uint8,
    )
    int8_kv = torch.empty(
        (num_blocks, 2, block_size, num_kv_heads, head_size + 4),
        device=device,
        dtype=torch.int8,
    )

    packed_impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=1.0 / (head_size**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window if sliding_window >= 0 else None,
        kv_cache_dtype="intx_k_inty_v_per_token_head",
        attn_type=AttentionType.DECODER,
        kv_cache_k_bits=k_bits,
        kv_cache_v_bits=v_bits,
    )
    int8_impl = TritonAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=1.0 / (head_size**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window if sliding_window >= 0 else None,
        kv_cache_dtype="int8_per_token_head",
        attn_type=AttentionType.DECODER,
    )

    packed_ms = _benchmark(
        lambda: packed_impl.do_kv_cache_update(
            layer, key, value, packed_kv, slot_mapping
        ),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    int8_ms = _benchmark(
        lambda: int8_impl.do_kv_cache_update(layer, key, value, int8_kv, slot_mapping),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )

    print(
        "shape:",
        f"num_tokens={num_tokens}",
        f"q_heads={num_query_heads}",
        f"kv_heads={num_kv_heads}",
        f"head_size={head_size}",
        f"block_size={block_size}",
        f"num_blocks={num_blocks}",
        f"sliding_window={sliding_window}",
        f"k_bits={k_bits}",
        f"v_bits={v_bits}",
        f"dtype={dtype}",
    )
    print(f"packed_kv_update_us={packed_ms:.6f}")
    print(f"int8_kv_update_us={int8_ms:.6f}")
    print(f"packed_vs_int8_ratio={packed_ms / int8_ms:.4f}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark packed-int KV update against int8 per-token-head."
    )
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=1000)
    args = parser.parse_args()
    print(args)

    main(
        num_tokens=args.num_tokens,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        sliding_window=args.sliding_window,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
    )
