# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shutil
import tempfile
from contextlib import contextmanager

import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.workspace import init_workspace_manager


@contextmanager
def _maybe_force_packed_int_decode_cg(force: bool):
    if not force:
        yield
        return

    original = TritonAttentionMetadataBuilder.get_cudagraph_support.__func__

    @classmethod
    def _forced_support(cls, vllm_config, kv_cache_spec):
        if getattr(kv_cache_spec, "packed_int_layout", None) is not None:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return original(cls, vllm_config, kv_cache_spec)

    TritonAttentionMetadataBuilder.get_cudagraph_support = _forced_support
    try:
        yield
    finally:
        TritonAttentionMetadataBuilder.get_cudagraph_support = classmethod(original)


def _make_vllm_config(
    *,
    model: str,
    dtype: str,
    cache_dtype: str,
    k_bits: int | None,
    v_bits: int | None,
    cudagraph_mode: CUDAGraphMode,
    capture_size: int,
) -> VllmConfig:
    model_config = ModelConfig(
        model=model,
        tokenizer=None,
        dtype=dtype,
        seed=0,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=capture_size,
        max_model_len=128,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.5,
        cache_dtype=cache_dtype,
        kv_cache_k_bits=k_bits,
        kv_cache_v_bits=v_bits,
    )
    parallel_config = ParallelConfig()
    attention_config = AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN)
    compilation_config = CompilationConfig(
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=[capture_size],
        max_cudagraph_capture_size=capture_size,
        cudagraph_num_of_warmups=0,
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        attention_config=attention_config,
        compilation_config=compilation_config,
    )


def _summarize_capture_descs(
    runner: GPUModelRunner,
) -> list[tuple[str, int, int, bool]]:
    summary: list[tuple[str, int, int, bool]] = []
    for mode, descs in runner.cudagraph_dispatcher.get_capture_descs():
        for desc in descs:
            summary.append(
                (mode.name, desc.num_tokens, desc.num_reqs or -1, desc.uniform)
            )
    return summary


@contextmanager
def _maybe_make_synthetic_gemma4_tiny(enable: bool, disable_moe: bool):
    if not enable:
        yield None
        return

    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4TextConfig,
    )

    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    text_config = Gemma4TextConfig(
        architectures=["Gemma4ForCausalLM"],
        vocab_size=4096,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        max_position_embeddings=8192,
        sliding_window=1024,
        layer_types=layer_types,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=not disable_moe,
        num_experts=None if disable_moe else 8,
        top_k_experts=None if disable_moe else 2,
        moe_intermediate_size=None if disable_moe else 512,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=4096,
        dtype="float32",
    )
    config = Gemma4Config(
        text_config=text_config,
        tie_word_embeddings=True,
    )
    config.architectures = ["Gemma4ForCausalLM"]

    temp_dir = tempfile.mkdtemp(prefix="vllm_gemma4_tiny_")
    try:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f)
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@torch.inference_mode()
def main(
    *,
    model: str,
    dtype: str,
    cache_dtype: str,
    k_bits: int | None,
    v_bits: int | None,
    force_packed_int_decode_cg: bool,
    capture_size: int,
    synthetic_gemma4_tiny: bool,
    synthetic_gemma4_disable_moe: bool,
) -> None:
    set_random_seed(0)
    update_environment_variables(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
    )

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)

    runner: GPUModelRunner | None = None
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)

        with _maybe_make_synthetic_gemma4_tiny(
            synthetic_gemma4_tiny, synthetic_gemma4_disable_moe
        ) as synthetic_dir:
            resolved_model = synthetic_dir or model
            vllm_config = _make_vllm_config(
                model=resolved_model,
                dtype=dtype,
                cache_dtype=cache_dtype,
                k_bits=k_bits,
                v_bits=v_bits,
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                capture_size=capture_size,
            )

            with _maybe_force_packed_int_decode_cg(force_packed_int_decode_cg):
                with set_current_vllm_config(vllm_config):
                    torch.set_default_dtype(torch.float16)
                    num_ubatches = 2 if vllm_config.parallel_config.enable_dbo else 1
                    init_workspace_manager(
                        torch.device(current_platform.device_type),
                        num_ubatches,
                    )
                    runner = GPUModelRunner(vllm_config, current_platform.device_type)
                    runner.load_model(load_dummy_weights=True)
                    current_platform.update_block_size_for_backend(vllm_config)

                    result = {
                        "success": True,
                        "error": None,
                        "resolved_cudagraph_mode": None,
                        "capture_descs": None,
                        "memory_bytes": None,
                    }

                    try:
                        memory_bytes = runner.profile_cudagraph_memory()
                        result["memory_bytes"] = memory_bytes
                    except Exception as exc:  # noqa: BLE001
                        result["success"] = False
                        result["error"] = repr(exc)
                    finally:
                        result["resolved_cudagraph_mode"] = (
                            runner.compilation_config.cudagraph_mode.name
                            if runner.compilation_config.cudagraph_mode is not None
                            else None
                        )
                        result["capture_descs"] = _summarize_capture_descs(runner)

        print(
            "case:",
            f"model={model}",
            f"synthetic_gemma4_tiny={synthetic_gemma4_tiny}",
            f"synthetic_gemma4_disable_moe={synthetic_gemma4_disable_moe}",
            f"dtype={dtype}",
            f"cache_dtype={cache_dtype}",
            f"k_bits={k_bits}",
            f"v_bits={v_bits}",
            f"force_packed_int_decode_cg={force_packed_int_decode_cg}",
            f"capture_size={capture_size}",
        )
        print(f"resolved_cudagraph_mode={result['resolved_cudagraph_mode']}")
        print(f"capture_descs={result['capture_descs']}")
        print(f"gpu_model_runner_capture_success={result['success']}")
        if result["memory_bytes"] is not None:
            print(f"gpu_model_runner_capture_memory_bytes={result['memory_bytes']}")
        if result["error"] is not None:
            print(f"gpu_model_runner_capture_error={result['error']}")
    finally:
        if runner is not None:
            with torch.no_grad():
                try:
                    runner.shutdown()
                except Exception:
                    pass
        cleanup_dist_env_and_memory()
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=(
            "Validate CUDA-graph capture at GPUModelRunner level with dummy "
            "weights, and optionally force packed-int decode cudagraph support."
        )
    )
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--synthetic-gemma4-tiny",
        action="store_true",
    )
    parser.add_argument(
        "--synthetic-gemma4-disable-moe",
        action="store_true",
    )
    parser.add_argument(
        "--cache-dtype",
        type=str,
        choices=["int8_per_token_head", "intx_k_inty_v_per_token_head"],
        required=True,
    )
    parser.add_argument("--k-bits", type=int, default=None)
    parser.add_argument("--v-bits", type=int, default=None)
    parser.add_argument("--capture-size", type=int, default=1)
    parser.add_argument(
        "--force-packed-int-decode-cg",
        action="store_true",
    )
    args = parser.parse_args()
    print(args)

    main(
        model=args.model,
        dtype=args.dtype,
        cache_dtype=args.cache_dtype,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        force_packed_int_decode_cg=args.force_packed_int_decode_cg,
        capture_size=args.capture_size,
        synthetic_gemma4_tiny=args.synthetic_gemma4_tiny,
        synthetic_gemma4_disable_moe=args.synthetic_gemma4_disable_moe,
    )
