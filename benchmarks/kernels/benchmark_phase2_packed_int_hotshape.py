# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch

try:
    from .phase2_hotshape_utils import (
    Phase2HotShapeSpec,
    benchmark_cuda,
    build_phase2_launch_config,
    compiled_metadata_to_dict,
    extract_compiled_kernel,
    extract_compiled_metadata,
    extract_ttgir_shared_allocs,
    hotshape_spec_to_dict,
    make_phase2_packed_inputs,
    packed_launch_config_to_dict,
    resolve_packed_int_attention_launch_config,
    summarize_shared_ranges,
)
except ImportError:
    from phase2_hotshape_utils import (
        Phase2HotShapeSpec,
        benchmark_cuda,
        build_phase2_launch_config,
        compiled_metadata_to_dict,
        extract_compiled_kernel,
        extract_compiled_metadata,
        extract_ttgir_shared_allocs,
        hotshape_spec_to_dict,
        make_phase2_packed_inputs,
        packed_launch_config_to_dict,
        resolve_packed_int_attention_launch_config,
        summarize_shared_ranges,
    )
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode, PackedIntPerTokenHeadLayout
from vllm.v1.attention.ops.triton_packed_int_kv import (
    kernel_packed_int_attention,
    paged_attention_packed_int,
)


def _emit_json(payload: dict[str, object], *, enabled: bool) -> bool:
    if not enabled:
        return False
    print(json.dumps(payload, sort_keys=True))
    return True


def _run_packed_probe(
    *,
    spec: Phase2HotShapeSpec,
    dtype: torch.dtype,
    device: str,
    hotshape: dict[str, torch.Tensor | PackedIntPerTokenHeadLayout],
    query: torch.Tensor,
    packed_out: torch.Tensor,
    packed_layout: PackedIntPerTokenHeadLayout,
    packed_key_cache: torch.Tensor,
    packed_value_cache: torch.Tensor,
    packed_k_scale: torch.Tensor,
    packed_v_scale: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    packed_tile_size: int | None,
    packed_num_warps: int | None,
    packed_block_m: int | None,
    packed_num_stages: int | None,
    suppress_max_query_len: bool,
    num_warmup_iters: int,
    num_iters: int,
    dump_shared_summary: bool,
    dump_ttgir_shared_allocs: bool,
) -> dict[str, object]:
    packed_launch = build_phase2_launch_config(
        q_element_size=query.element_size(),
        spec=spec,
        packed_tile_size=packed_tile_size,
        packed_num_warps=packed_num_warps,
        packed_block_m=packed_block_m,
        packed_num_stages=packed_num_stages,
    )
    allow_single_query_override = (
        packed_tile_size is None
        and packed_num_warps is None
        and packed_block_m is None
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
        softmax_scale=spec.softmax_scale,
        softcap=0.0,
        num_queries_per_kv=spec.num_queries_per_kv,
        sliding_window=(spec.sliding_window, -1),
        launch_config=packed_launch,
        max_query_len=None if suppress_max_query_len else spec.query_len,
        allow_single_query_override=allow_single_query_override,
    )
    effective_packed_launch = resolve_packed_int_attention_launch_config(
        q_element_size=query.element_size(),
        head_size=spec.head_size,
        head_size_v=spec.head_size_v,
        num_queries_per_kv=spec.num_queries_per_kv,
        sliding_window=(spec.sliding_window, -1),
        layout=packed_layout,
        launch_config=packed_launch,
        max_query_len=None if suppress_max_query_len else spec.query_len,
        allow_single_query_override=allow_single_query_override,
    )
    kernel_packed_int_attention.device_caches.clear()
    packed_ms = benchmark_cuda(
        lambda: paged_attention_packed_int(**packed_kwargs),
        num_warmup_iters=num_warmup_iters,
        num_iters=num_iters,
    )
    metadata = extract_compiled_metadata()
    compiled = extract_compiled_kernel()
    payload = {
        "kernel_packed_int_ms": packed_ms,
        **packed_launch_config_to_dict(packed_launch, prefix="packed_requested_launch"),
        **packed_launch_config_to_dict(effective_packed_launch, prefix="packed_effective_launch"),
        **compiled_metadata_to_dict(metadata),
    }
    if dump_shared_summary:
        payload["shared_range_groups"] = summarize_shared_ranges(compiled.asm["ptx"])
    if dump_ttgir_shared_allocs:
        payload["ttgir_shared_allocs"] = extract_ttgir_shared_allocs(
            compiled.asm["ttgir"]
        )
    return payload


def main() -> None:
    parser = FlexibleArgumentParser(
        description=(
            "Phase-2 Gemma4 packed-int hot-shape benchmark with compiled "
            "kernel metadata."
        )
    )
    parser.add_argument("--seq-len", type=int, default=17037)
    parser.add_argument("--query-len", type=int, default=1024)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=512)
    parser.add_argument("--head-size-v", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sliding-window", type=int, default=-1)
    parser.add_argument("--k-bits", type=int, default=5)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--dtype", choices=sorted(STR_DTYPE_TO_TORCH_DTYPE), default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--packed-tile-size", type=int, default=None)
    parser.add_argument("--packed-num-warps", type=int, default=None)
    parser.add_argument("--packed-block-m", type=int, default=None)
    parser.add_argument("--packed-num-stages", type=int, default=None)
    parser.add_argument("--compare-secondary-packed-tile-size", type=int, default=None)
    parser.add_argument("--compare-secondary-packed-num-warps", type=int, default=None)
    parser.add_argument("--compare-secondary-packed-block-m", type=int, default=None)
    parser.add_argument("--compare-secondary-packed-num-stages", type=int, default=None)
    parser.add_argument(
        "--compare-secondary-suppress-max-query-len",
        action="store_true",
        help=(
            "Apply suppress-max-query-len only to the secondary packed launch "
            "when compare-secondary flags are used."
        ),
    )
    parser.add_argument(
        "--compare-secondary-label",
        type=str,
        default="secondary",
        help="Label to use for the secondary packed launch in compare output.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--dump-shared-summary",
        action="store_true",
        help="Print PTX shared-address range summary for the compiled kernel.",
    )
    parser.add_argument(
        "--dump-ttgir-shared-allocs",
        action="store_true",
        help="Print TTGIR shared-memory local_alloc lines for the compiled kernel.",
    )
    parser.add_argument(
        "--suppress-max-query-len",
        action="store_true",
        help=(
            "Pass max_query_len=None to bypass wrapper-level long-prefill "
            "launch rewrites."
        ),
    )
    args = parser.parse_args()

    set_random_seed(args.seed)
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    device = args.device
    spec = Phase2HotShapeSpec(
        seq_len=args.seq_len,
        query_len=args.query_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        head_size_v=args.head_size if args.head_size_v is None else args.head_size_v,
        block_size=args.block_size,
        sliding_window=args.sliding_window,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
    )
    hotshape = make_phase2_packed_inputs(spec=spec, dtype=dtype, device=device)
    query = hotshape["query"]
    packed_out = hotshape["out"]
    packed_layout = hotshape["layout"]
    packed_key_cache = hotshape["key_cache"]
    packed_value_cache = hotshape["value_cache"]
    packed_k_scale = hotshape["k_scale"]
    packed_v_scale = hotshape["v_scale"]
    query_start_loc = hotshape["query_start_loc"]
    seq_lens = hotshape["seq_lens"]
    block_table = hotshape["block_table"]
    head_size_v = spec.head_size_v
    int8_out = torch.empty_like(packed_out)

    int8_key_cache = torch.randint(
        -127,
        128,
        (spec.num_blocks, args.block_size, args.num_kv_heads, args.head_size),
        dtype=torch.int8,
        device=device,
    )
    int8_value_cache = torch.randint(
        -127,
        128,
        (spec.num_blocks, args.block_size, args.num_kv_heads, head_size_v),
        dtype=torch.int8,
        device=device,
    )
    int8_k_scale = (
        torch.rand((spec.num_blocks, args.block_size, args.num_kv_heads), device=device)
        + 0.01
    ).to(torch.float32)
    int8_v_scale = (
        torch.rand((spec.num_blocks, args.block_size, args.num_kv_heads), device=device)
        + 0.01
    ).to(torch.float32)

    softmax_scale = spec.softmax_scale

    int8_kwargs = dict(
        q=query,
        k=int8_key_cache,
        v=int8_value_cache,
        out=int8_out,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=args.query_len,
        seqused_k=seq_lens,
        max_seqlen_k=args.seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(args.sliding_window, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=int8_k_scale,
        v_scale_cache=int8_v_scale,
    )

    packed_payload = _run_packed_probe(
        spec=spec,
        dtype=dtype,
        device=device,
        hotshape=hotshape,
        query=query,
        packed_out=packed_out,
        packed_layout=packed_layout,
        packed_key_cache=packed_key_cache,
        packed_value_cache=packed_value_cache,
        packed_k_scale=packed_k_scale,
        packed_v_scale=packed_v_scale,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        packed_tile_size=args.packed_tile_size,
        packed_num_warps=args.packed_num_warps,
        packed_block_m=args.packed_block_m,
        packed_num_stages=args.packed_num_stages,
        suppress_max_query_len=args.suppress_max_query_len,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
        dump_shared_summary=args.dump_shared_summary,
        dump_ttgir_shared_allocs=args.dump_ttgir_shared_allocs,
    )
    packed_ms = packed_payload["kernel_packed_int_ms"]
    int8_ms = benchmark_cuda(
        lambda: unified_attention(**int8_kwargs),
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
    )
    payload = {
        "schema_version": 1,
        "tool": "benchmark_phase2_packed_int_hotshape",
        "mode": "kernel_hotshape",
        "shape": hotshape_spec_to_dict(spec, dtype=dtype),
        "kernel_packed_int_ms": packed_ms,
        "kernel_int8_per_token_head_ms": int8_ms,
        "kernel_packed_vs_int8_ratio": packed_ms / int8_ms,
        **packed_payload,
    }
    compare_secondary_requested = (
        args.compare_secondary_packed_tile_size is not None
        or args.compare_secondary_packed_num_warps is not None
        or args.compare_secondary_packed_block_m is not None
        or args.compare_secondary_packed_num_stages is not None
        or args.compare_secondary_suppress_max_query_len
    )
    if compare_secondary_requested:
        secondary_out = torch.empty_like(packed_out)
        secondary_payload = _run_packed_probe(
            spec=spec,
            dtype=dtype,
            device=device,
            hotshape=hotshape,
            query=query,
            packed_out=secondary_out,
            packed_layout=packed_layout,
            packed_key_cache=packed_key_cache,
            packed_value_cache=packed_value_cache,
            packed_k_scale=packed_k_scale,
            packed_v_scale=packed_v_scale,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            packed_tile_size=args.compare_secondary_packed_tile_size,
            packed_num_warps=args.compare_secondary_packed_num_warps,
            packed_block_m=args.compare_secondary_packed_block_m,
            packed_num_stages=args.compare_secondary_packed_num_stages,
            suppress_max_query_len=args.compare_secondary_suppress_max_query_len,
            num_warmup_iters=args.num_warmup_iters,
            num_iters=args.num_iters,
            dump_shared_summary=args.dump_shared_summary,
            dump_ttgir_shared_allocs=args.dump_ttgir_shared_allocs,
        )
        payload = {
            "schema_version": 1,
            "tool": "benchmark_phase2_packed_int_hotshape",
            "mode": "kernel_hotshape_compare",
            "shape": hotshape_spec_to_dict(spec, dtype=dtype),
            "primary_label": "primary",
            "secondary_label": args.compare_secondary_label,
            "kernel_int8_per_token_head_ms": int8_ms,
            "primary_kernel_packed_int_ms": packed_payload["kernel_packed_int_ms"],
            "secondary_kernel_packed_int_ms": secondary_payload["kernel_packed_int_ms"],
            "secondary_vs_primary_kernel_packed_ratio": (
                secondary_payload["kernel_packed_int_ms"] / packed_payload["kernel_packed_int_ms"]
            ),
        }
        for key, value in packed_payload.items():
            if key == "kernel_packed_int_ms":
                continue
            payload[f"primary_{key}"] = value
        for key, value in secondary_payload.items():
            if key == "kernel_packed_int_ms":
                continue
            payload[f"secondary_{key}"] = value
    if _emit_json(payload, enabled=args.json):
        return

    print(
        "shape:"
        f" seq_len={args.seq_len}"
        f" query_len={args.query_len}"
        f" q_heads={args.num_query_heads}"
        f" kv_heads={args.num_kv_heads}"
        f" head_size={args.head_size}"
        f" head_size_v={head_size_v}"
        f" block_size={args.block_size}"
        f" sliding_window={args.sliding_window}"
        f" k_bits={args.k_bits}"
        f" v_bits={args.v_bits}"
        f" dtype={dtype}"
    )
    if compare_secondary_requested:
        print(f"kernel_int8_per_token_head_ms={int8_ms:.6f}")
        print(f"primary_kernel_packed_int_ms={packed_payload['kernel_packed_int_ms']:.6f}")
        print(f"secondary_kernel_packed_int_ms={secondary_payload['kernel_packed_int_ms']:.6f}")
        print(
            "secondary_vs_primary_kernel_packed_ratio="
            f"{secondary_payload['kernel_packed_int_ms'] / packed_payload['kernel_packed_int_ms']:.4f}"
        )
        return

    print(f"kernel_packed_int_ms={packed_ms:.6f}")
    print(f"kernel_int8_per_token_head_ms={int8_ms:.6f}")
    print(f"kernel_packed_vs_int8_ratio={packed_ms / int8_ms:.4f}")
    for key, value in packed_payload.items():
        if key == "kernel_packed_int_ms":
            continue
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
