# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import random

import torch

try:
    from .phase2_hotshape_utils import Phase2HotShapeSpec, hotshape_spec_to_dict
    from .phase2_staged_kernel_experiment import (
        list_discovered_impl_modules,
        list_stage1_impls,
        list_stage2_impls,
        pipeline_artifacts_to_dict,
        registry_selfcheck_to_dict,
        run_pipeline,
        run_registry_selfcheck,
    )
except ImportError:
    from phase2_hotshape_utils import Phase2HotShapeSpec, hotshape_spec_to_dict
    from phase2_staged_kernel_experiment import (
        list_discovered_impl_modules,
        list_stage1_impls,
        list_stage2_impls,
        pipeline_artifacts_to_dict,
        registry_selfcheck_to_dict,
        run_pipeline,
        run_registry_selfcheck,
    )
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed


def _seed_all(seed: int) -> None:
    try:
        set_random_seed(seed)
    except NotImplementedError:
        random.seed(seed)
        torch.manual_seed(seed)


def main() -> None:
    parser = FlexibleArgumentParser(
        description=(
            "Validate the phase-2 staged/split-KV scaffold via direct library "
            "APIs on a small smoke shape."
        )
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--query-len", type=int, default=16)
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--head-size-v", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sliding-window", type=int, default=-1)
    parser.add_argument("--k-bits", type=int, default=5)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=sorted(STR_DTYPE_TO_TORCH_DTYPE),
        default="half",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-kv-splits", type=int, default=4)
    parser.add_argument("--query-tile-size-hint", type=int, default=16)
    parser.add_argument("--stage1-impl", type=str, default="reference")
    parser.add_argument("--stage2-impl", type=str, default="reference")
    parser.add_argument("--out-tol", type=float, default=1e-5)
    parser.add_argument("--lse-tol", type=float, default=1e-5)
    args = parser.parse_args()

    _seed_all(args.seed)
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    spec = Phase2HotShapeSpec(
        seq_len=args.seq_len,
        query_len=args.query_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        head_size_v=args.head_size_v,
        block_size=args.block_size,
        sliding_window=args.sliding_window,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
    )

    pipeline = run_pipeline(
        spec=spec,
        dtype=dtype,
        device=args.device,
        num_kv_splits=args.num_kv_splits,
        query_tile_size_hint=args.query_tile_size_hint,
        stage1_impl=args.stage1_impl,
        stage2_impl=args.stage2_impl,
    )
    selfcheck = run_registry_selfcheck(
        spec=spec,
        dtype=dtype,
        device=args.device,
        num_kv_splits=args.num_kv_splits,
        query_tile_size_hint=args.query_tile_size_hint,
        out_tol=args.out_tol,
        lse_tol=args.lse_tol,
    )

    payload = {
        "schema_version": 1,
        "tool": "validate_phase2_scaffold",
        "mode": "smoke",
        "shape": hotshape_spec_to_dict(spec, dtype=dtype),
        "impl_modules": list_discovered_impl_modules(),
        "stage1_impls": list_stage1_impls(),
        "stage2_impls": list_stage2_impls(),
        "selected_stage1_impl": args.stage1_impl,
        "selected_stage2_impl": args.stage2_impl,
        **pipeline_artifacts_to_dict(pipeline),
        "pipeline_ok": pipeline.out_max_abs < args.out_tol
        and pipeline.lse_max_abs < args.lse_tol,
        **registry_selfcheck_to_dict(selfcheck),
        "selfcheck_ok": selfcheck.passed_pairs == selfcheck.total_pairs,
    }
    print(json.dumps(payload, sort_keys=True))

    if not payload["pipeline_ok"] or not payload["selfcheck_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
