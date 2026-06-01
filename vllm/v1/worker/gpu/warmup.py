# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.distributed.parallel_state import get_pp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@torch.inference_mode()
def warmup_slot_mapping_kernel(
    model_runner: GPUModelRunner,
    prompt_len: int,
) -> None:
    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is None:
        return

    multi_group_block_table = getattr(input_batch, "block_table", None)
    if multi_group_block_table is None:
        return

    direct_block_ids = tuple(
        list(range(1, cdiv(prompt_len, bt.block_size) + 1))
        for bt in multi_group_block_table.block_tables
    )
    multi_group_block_table.add_row(direct_block_ids, 0)
    multi_group_block_table.commit_block_table(1)

    # Reuse the same long-lived buffers that the real request path slices from,
    # rather than allocating fresh tensors. This keeps the warmup closer to the
    # serving path that still triggers the live Triton JIT miss.
    query_start_loc = getattr(model_runner, "query_start_loc", None)
    positions = getattr(model_runner, "positions", None)
    if query_start_loc is None or positions is None:
        multi_group_block_table.clear_row(0)
        multi_group_block_table.commit_block_table(1)
        return

    query_start_loc.np[0] = 0
    query_start_loc.np[1] = prompt_len
    direct_query_start_loc = query_start_loc.copy_to_gpu(2)
    positions[:prompt_len] = torch.arange(
        prompt_len, dtype=torch.int64, device=model_runner.device
    )
    direct_positions = positions[:prompt_len]

    multi_group_block_table.compute_slot_mapping(
        1, direct_query_start_loc, direct_positions
    )
    multi_group_block_table.clear_row(0)
    multi_group_block_table.commit_block_table(1)


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    2 + num_spec_steps prompt tokens each. The second iteration simulates
    a decode step with all requests generating 1 + num_spec_steps tokens.
    """
    # V2 uses `num_speculative_steps`, while the legacy V1 runner stores the
    # same concept as `num_spec_tokens`.
    num_spec_steps = getattr(
        model_runner,
        "num_speculative_steps",
        getattr(model_runner, "num_spec_tokens", 0),
    )
    # Use 1 + num_spec_steps + 1 tokens so the prefill batch's per-request
    # query length exceeds decode_query_len (= 1 + num_spec_steps), preventing
    # it from being misclassified as a uniform decode batch.
    base_prompt_len = 2 + num_spec_steps

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]

    def _get_decode_block_counts(
        prompt_len: int,
    ) -> tuple[list[int], list[int], list[int]]:
        # After prefill, decode generates 1 verified + num_spec_steps draft
        # tokens.
        decode_len = prompt_len + 1 + num_spec_steps
        prefill_block_counts = [cdiv(prompt_len, bs) for bs in group_block_sizes]
        decode_block_counts = [cdiv(decode_len, bs) for bs in group_block_sizes]
        decode_block_deltas = [
            d - p for d, p in zip(decode_block_counts, prefill_block_counts)
        ]
        return prefill_block_counts, decode_block_counts, decode_block_deltas

    _, base_decode_block_counts, _ = _get_decode_block_counts(base_prompt_len)
    max_blocks_per_req = sum(base_decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(base_prompt_len, 1 + num_spec_steps),
        # Reserve block 0 (null block) and ensure we have enough blocks.
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign distinct block IDs per request per group. 0 null block, start from 1.
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    def _run_warmup_batch(req_ids: list[str], prompt_len: int) -> None:
        prompt_token_ids = list(range(prompt_len))
        (
            prefill_block_counts,
            _decode_block_counts,
            decode_block_deltas,
        ) = _get_decode_block_counts(prompt_len)

        new_reqs = [
            NewRequestData.from_request(
                Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
                block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
                prefill_token_ids=prompt_token_ids,
            )
            for i in range(len(req_ids))
        ]

        prefill_output = SchedulerOutput.make_empty()
        prefill_output.scheduled_new_reqs = new_reqs
        prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
        prefill_output.total_num_scheduled_tokens = prompt_len * len(req_ids)
        prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups
        worker_execute_model(prefill_output)

        if not model_runner.is_pooling_model:
            grammar_output = None
            if get_pp_group().is_last_rank:
                # Build a GrammarOutput to exercise the structured output
                # bitmask kernel during the prefill step.
                vocab_size = model_runner.model_config.get_vocab_size()
                bitmask_width = (vocab_size + 31) // 32
                grammar_bitmask = np.full(
                    (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
                )
                grammar_output = GrammarOutput(
                    structured_output_request_ids=req_ids,
                    grammar_bitmask=grammar_bitmask,
                )

            worker_sample_tokens(grammar_output)

            cached_req_data = CachedRequestData.make_empty()
            cached_req_data.req_ids = list(req_ids)
            cached_req_data.num_computed_tokens = [prompt_len] * len(req_ids)
            cached_req_data.num_output_tokens = [1] * len(req_ids)
            new_block = any(decode_block_deltas)
            cached_req_data.new_block_ids = [
                tuple(_alloc_blocks(n) for n in decode_block_deltas)
                if new_block
                else None
                for _ in range(len(req_ids))
            ]

            decode_output = SchedulerOutput.make_empty()
            decode_output.scheduled_cached_reqs = cached_req_data
            decode_output.num_scheduled_tokens = {
                req_id: 1 + num_spec_steps for req_id in req_ids
            }
            if num_spec_steps > 0:
                decode_output.scheduled_spec_decode_tokens = {
                    req_id: [0] * num_spec_steps for req_id in req_ids
                }
            decode_output.total_num_scheduled_tokens = sum(
                decode_output.num_scheduled_tokens.values()
            )
            decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

            worker_execute_model(decode_output)
            worker_sample_tokens(None)

        cleanup_output = SchedulerOutput.make_empty()
        cleanup_output.finished_req_ids = set(req_ids)
        worker_execute_model(cleanup_output)

    # Disable KV connector for warmup runs when the model runner exposes a
    # direct connector handle (V2). Legacy V1 runners route KV transfer
    # through the global transfer group and do not expose `kv_connector`.
    kv_connector = getattr(model_runner, "kv_connector", None)
    if kv_connector is not None:
        kv_connector.set_disabled(True)
    _run_warmup_batch(req_ids, base_prompt_len)
    # Also cover the common serving case of a single short prompt followed by
    # single-token decode. The tiny synthetic batch above is enough to avoid
    # decode misclassification, but it can still miss first-request Triton
    # specializations observed on real OpenAI-style traffic.
    short_prompt_len = max(base_prompt_len, 32)
    _run_warmup_batch(["_warmup_short_0_"], short_prompt_len)
    # Directly warm the slot-mapping kernel as well. In practice the
    # request-shaped execute_model path above was sufficient to cover packed-int
    # attention, but the Triton slot-mapping kernel could still JIT on the
    # first real single-request prompt.
    warmup_slot_mapping_kernel(model_runner, short_prompt_len)
    if kv_connector is not None:
        kv_connector.set_disabled(False)
    torch.accelerator.synchronize()
