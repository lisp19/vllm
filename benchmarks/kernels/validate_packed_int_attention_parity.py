# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.ops.triton_packed_int_kv import (
    _materialize_sequence_kv,
    _per_token_head_scale,
    _quant_bounds,
    _unpack_signed_values,
    _value_scale_headroom,
    get_packed_int_cache_views,
    paged_attention_packed_int,
    reshape_and_cache_packed_int_per_token_head,
)
from vllm.v1.kv_cache_interface import PackedIntPerTokenHeadLayout


def _make_reference_quantized(
    values: torch.Tensor,
    bits: int,
    *,
    symmetric_3bit: bool = False,
    apply_value_headroom: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax, qmin = _quant_bounds(bits, symmetric_3bit=symmetric_3bit)
    scales = _per_token_head_scale(values.to(torch.float32), qmax, bits)
    if apply_value_headroom:
        scales = scales * _value_scale_headroom(bits)
    q = torch.clamp(
        torch.round(values.to(torch.float32) / scales.unsqueeze(-1)),
        qmin,
        qmax,
    )
    dequant = q * scales.unsqueeze(-1)
    return dequant, scales


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    num_queries_per_kv: int,
    softmax_scale: float,
    sliding_window: int,
) -> torch.Tensor:
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_size_v = v.shape[2]
    context_len = seq_len - query_len
    query_abs = context_len + torch.arange(
        query_len, device=q.device, dtype=torch.int64
    )
    key_pos = torch.arange(seq_len, device=q.device, dtype=torch.int64)
    base_mask = key_pos[None, :] <= query_abs[:, None]
    if sliding_window > 0:
        base_mask = base_mask & (
            (query_abs[:, None] - key_pos[None, :]) < sliding_window
        )

    out = torch.empty(
        (query_len, num_query_heads, head_size_v),
        device=q.device,
        dtype=torch.float32,
    )
    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)

    for kv_head_idx in range(num_kv_heads):
        q_lo = kv_head_idx * num_queries_per_kv
        q_hi = q_lo + num_queries_per_kv
        q_group = q[:, q_lo:q_hi].to(torch.float32)
        k_group = k[:, kv_head_idx].to(torch.float32)
        v_group = v[:, kv_head_idx].to(torch.float32)
        scores = torch.einsum("qhd,kd->qhk", q_group, k_group) * softmax_scale
        scores = torch.where(base_mask[:, None, :], scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        out[:, q_lo:q_hi] = torch.einsum("qhk,kd->qhd", probs, v_group)

    return out


def _reference_attention_streaming(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    num_queries_per_kv: int,
    softmax_scale: float,
    sliding_window: int,
    k_bits: int,
    v_bits: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_size_v = v.shape[2]
    context_len = seq_len - query_len
    query_abs = context_len + torch.arange(
        query_len, device=q.device, dtype=torch.int64
    )
    start_pos = (
        max(0, int(query_abs[0].item()) - sliding_window + 1)
        if sliding_window > 0
        else 0
    )
    out = torch.empty(
        (query_len, num_query_heads, head_size_v),
        device=q.device,
        dtype=torch.float32,
    )

    for kv_head_idx in range(num_kv_heads):
        q_lo = kv_head_idx * num_queries_per_kv
        q_hi = q_lo + num_queries_per_kv
        q_group = q[:, q_lo:q_hi].to(torch.float32)
        m = torch.full((query_len, num_queries_per_kv), float("-inf"), device=q.device)
        l = torch.zeros(
            (query_len, num_queries_per_kv), device=q.device, dtype=torch.float32
        )
        acc = torch.zeros(
            (query_len, num_queries_per_kv, head_size_v),
            device=q.device,
            dtype=torch.float32,
        )

        for chunk_lo in range(start_pos, seq_len, chunk_size):
            chunk_hi = min(chunk_lo + chunk_size, seq_len)
            key_pos = torch.arange(
                chunk_lo, chunk_hi, device=q.device, dtype=torch.int64
            )
            k_chunk, _ = _make_reference_quantized(
                k[chunk_lo:chunk_hi, kv_head_idx : kv_head_idx + 1], k_bits
            )
            v_chunk, _ = _make_reference_quantized(
                v[chunk_lo:chunk_hi, kv_head_idx : kv_head_idx + 1],
                v_bits,
                symmetric_3bit=True,
                apply_value_headroom=True,
            )
            k_chunk = k_chunk[:, 0].to(torch.float32)
            v_chunk = v_chunk[:, 0].to(torch.float32)
            scores = torch.einsum("qhd,kd->qhk", q_group, k_chunk) * softmax_scale
            mask = key_pos[None, :] <= query_abs[:, None]
            if sliding_window > 0:
                mask = mask & ((query_abs[:, None] - key_pos[None, :]) < sliding_window)
            scores = torch.where(
                mask[:, None, :],
                scores,
                torch.tensor(float("-inf"), device=q.device, dtype=torch.float32),
            )
            m_j = torch.maximum(m, scores.max(dim=2).values)
            p = torch.exp(scores - m_j[:, :, None])
            p = torch.where(torch.isneginf(m_j)[:, :, None], torch.zeros_like(p), p)
            l_j = p.sum(dim=2)
            alpha = torch.exp(m - m_j)
            alpha = torch.where(torch.isneginf(m), torch.zeros_like(alpha), alpha)
            acc = acc * alpha[:, :, None] + torch.einsum("qhk,kd->qhd", p, v_chunk)
            l = l * alpha + l_j
            m = m_j

        out[:, q_lo:q_hi] = acc / l[:, :, None]

    return out


def _sample_roundtrip_stats(
    *,
    seq_len: int,
    block_size: int,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    layout: PackedIntPerTokenHeadLayout,
) -> dict[str, float]:
    sample_pos = sorted({0, seq_len // 2, seq_len - 1})
    pos = torch.tensor(sample_pos, device=key.device, dtype=torch.int64)
    blocks = pos // block_size
    slots = pos % block_size
    packed_k = key_cache[blocks, slots]
    packed_v = value_cache[blocks, slots]
    mat_k = _unpack_signed_values(
        packed_k, bits=layout.k_bits, num_elements=layout.head_size
    ).to(torch.float32) * k_scale_cache[blocks, slots].to(torch.float32).unsqueeze(-1)
    mat_v = _unpack_signed_values(
        packed_v, bits=layout.v_bits, num_elements=layout.head_size_v
    ).to(torch.float32) * v_scale_cache[blocks, slots].to(torch.float32).unsqueeze(-1)
    ref_k, ref_k_scales = _make_reference_quantized(key[pos], layout.k_bits)
    ref_v, ref_v_scales = _make_reference_quantized(
        value[pos],
        layout.v_bits,
        symmetric_3bit=True,
        apply_value_headroom=True,
    )
    return {
        "k_scale_max_abs": (k_scale_cache[blocks, slots] - ref_k_scales)
        .abs()
        .max()
        .item(),
        "v_scale_max_abs": (v_scale_cache[blocks, slots] - ref_v_scales)
        .abs()
        .max()
        .item(),
        "k_roundtrip_max_abs": (mat_k - ref_k).abs().max().item(),
        "k_roundtrip_mean_abs": (mat_k - ref_k).abs().mean().item(),
        "v_roundtrip_max_abs": (mat_v - ref_v).abs().max().item(),
        "v_roundtrip_mean_abs": (mat_v - ref_v).abs().mean().item(),
    }


@torch.inference_mode()
def run_case(
    *,
    seq_len: int,
    query_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    block_size: int,
    sliding_window: int,
    k_bits: int,
    v_bits: int,
    dtype: torch.dtype,
    seed: int,
    softmax_scale: float | None,
    padded_bytes_per_token_head: int | None,
) -> None:
    set_random_seed(seed)
    device = "cuda"
    num_blocks = math.ceil(seq_len / block_size)
    num_queries_per_kv = num_query_heads // num_kv_heads
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_size)

    layout = PackedIntPerTokenHeadLayout.create(
        k_bits=k_bits,
        v_bits=v_bits,
        head_size=head_size,
        head_size_v=head_size_v,
        padded_bytes_per_token_head=padded_bytes_per_token_head,
    )
    kv_cache = torch.empty(
        (num_blocks, block_size, num_kv_heads, layout.raw_bytes_per_token_head),
        device=device,
        dtype=torch.uint8,
    )
    key_cache, value_cache, k_scale_cache, v_scale_cache = get_packed_int_cache_views(
        kv_cache, layout
    )

    key = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn(seq_len, num_kv_heads, head_size_v, device=device, dtype=dtype)
    query = torch.randn(
        query_len, num_query_heads, head_size, device=device, dtype=dtype
    )
    if query_len == seq_len:
        query.copy_(
            key[:, torch.arange(num_query_heads, device=device) // num_queries_per_kv]
        )

    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int64)
    reshape_and_cache_packed_int_per_token_head(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        layout,
    )

    block_table_row = torch.arange(num_blocks, device=device, dtype=torch.int32)
    use_streaming_ref = seq_len > 4096 and query_len <= 32
    if use_streaming_ref:
        roundtrip_stats = _sample_roundtrip_stats(
            seq_len=seq_len,
            block_size=block_size,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            layout=layout,
        )
    else:
        ref_k, ref_k_scales = _make_reference_quantized(key, k_bits)
        ref_v, ref_v_scales = _make_reference_quantized(
            value,
            v_bits,
            symmetric_3bit=True,
            apply_value_headroom=True,
        )
        mat_k, mat_v = _materialize_sequence_kv(
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            block_table_row,
            seq_len,
            block_size,
            layout,
        )
        seq_pos = torch.arange(seq_len, device=device, dtype=torch.int64)
        mat_k_scales = k_scale_cache[seq_pos // block_size, seq_pos % block_size]
        mat_v_scales = v_scale_cache[seq_pos // block_size, seq_pos % block_size]
        roundtrip_stats = {
            "k_scale_max_abs": (mat_k_scales - ref_k_scales).abs().max().item(),
            "v_scale_max_abs": (mat_v_scales - ref_v_scales).abs().max().item(),
            "k_roundtrip_max_abs": (mat_k - ref_k).abs().max().item(),
            "k_roundtrip_mean_abs": (mat_k - ref_k).abs().mean().item(),
            "v_roundtrip_max_abs": (mat_v - ref_v).abs().max().item(),
            "v_roundtrip_mean_abs": (mat_v - ref_v).abs().mean().item(),
        }

    query_start_loc = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    block_table = block_table_row.view(1, -1)
    packed_out = torch.empty(
        query_len, num_query_heads, head_size_v, device=device, dtype=dtype
    )
    paged_attention_packed_int(
        q=query,
        key_cache=key_cache,
        value_cache=value_cache,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        out=packed_out,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        layout=layout,
        softmax_scale=softmax_scale,
        softcap=0.0,
        num_queries_per_kv=num_queries_per_kv,
        sliding_window=(sliding_window, -1),
    )
    torch.cuda.synchronize()

    if use_streaming_ref:
        ref_out = _reference_attention_streaming(
            query,
            key,
            value,
            seq_len=seq_len,
            query_len=query_len,
            num_queries_per_kv=num_queries_per_kv,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            k_bits=k_bits,
            v_bits=v_bits,
        )
    else:
        ref_out = _reference_attention(
            query,
            ref_k,
            ref_v,
            seq_len=seq_len,
            query_len=query_len,
            num_queries_per_kv=num_queries_per_kv,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
        )
    out_diff = (packed_out.to(torch.float32) - ref_out).abs()

    print(
        "case:",
        f"seq_len={seq_len}",
        f"query_len={query_len}",
        f"q_heads={num_query_heads}",
        f"kv_heads={num_kv_heads}",
        f"head_size={head_size}",
        f"head_size_v={head_size_v}",
        f"block_size={block_size}",
        f"sliding_window={sliding_window}",
        f"k_bits={k_bits}",
        f"v_bits={v_bits}",
        f"dtype={dtype}",
        f"seed={seed}",
    )
    effective_sliding_window = 1 + sliding_window if sliding_window >= 0 else 0
    print(f"effective_sliding_window={effective_sliding_window}")
    print(f"reference_mode={'streaming' if use_streaming_ref else 'full'}")
    print(f"k_scale_max_abs={roundtrip_stats['k_scale_max_abs']:.6e}")
    print(f"v_scale_max_abs={roundtrip_stats['v_scale_max_abs']:.6e}")
    print(f"k_roundtrip_max_abs={roundtrip_stats['k_roundtrip_max_abs']:.6e}")
    print(f"k_roundtrip_mean_abs={roundtrip_stats['k_roundtrip_mean_abs']:.6e}")
    print(f"v_roundtrip_max_abs={roundtrip_stats['v_roundtrip_max_abs']:.6e}")
    print(f"v_roundtrip_mean_abs={roundtrip_stats['v_roundtrip_mean_abs']:.6e}")
    print(f"attn_out_max_abs={out_diff.max().item():.6e}")
    print(f"attn_out_mean_abs={out_diff.mean().item():.6e}")
    print(f"attn_out_last_token_max_abs={out_diff[-1].max().item():.6e}")
    print(f"attn_out_last_token_mean_abs={out_diff[-1].mean().item():.6e}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Validate packed-int KV roundtrip and attention parity."
    )
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[704, 768])
    parser.add_argument("--query-len", type=int, default=0)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--head-size-v", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=1024,
        help=(
            "Packed-int kernel semantics: pass -1 for full attention. "
            "Non-negative values are interpreted as an effective window of "
            "1 + sliding_window."
        ),
    )
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--softmax-scale", type=float, default=0.0)
    parser.add_argument("--padded-bytes-per-token-head", type=int, default=0)
    args = parser.parse_args()
    print(args)

    for seq_len in args.seq_lens:
        run_case(
            seq_len=seq_len,
            query_len=args.query_len or seq_len,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            head_size_v=args.head_size_v,
            block_size=args.block_size,
            sliding_window=args.sliding_window,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            seed=args.seed,
            softmax_scale=args.softmax_scale or None,
            padded_bytes_per_token_head=args.padded_bytes_per_token_head or None,
        )
