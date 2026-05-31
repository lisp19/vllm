# Packed-int Per-Token-Head KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add three lower-memory packed-int per-token-head KV cache modes and verify all three with the original Gemma4 runtime arguments.

**Architecture:** build one generic packed-int metadata and allocation chain, then hang `8/4` and `4/4` specialized runtime paths off that shared chain. Use a dedicated Triton tiled packed-int attention kernel for the read path so long-context requests avoid dense whole-sequence K/V materialization and full score-matrix allocation. Keep Gemma4 padding logic hard-coded and keep all public signature changes optional.

**Tech Stack:** Python, PyTorch, Triton, vLLM v1 attention backend, Docker CUDA build.

**Execution constraints:** finish implementation for all three branches before starting runtime validation; use staged commits during development; reuse a built image until code changes require rebuilding; during debugging, overlay-style replacement is allowed for faster iteration, but any final report must be backed by a fresh build; kernel work may use static validation code before full runtime startup.

---

## Task 1: Persist design and implementation artifacts

**Files:**

- Create: `docs/superpowers/specs/2026-05-31-packed-int-per-token-head-design.md`
- Create: `docs/superpowers/plans/2026-05-31-packed-int-per-token-head.md`

- [ ] Save the approved design in the spec file.
- [ ] Save this implementation plan in the plan file.

## Task 2: Add config and metadata chain

**Files:**

- Modify: `vllm/config/cache.py`
- Modify: `vllm/engine/arg_utils.py`
- Modify: `vllm/utils/torch_utils.py`
- Modify: `vllm/v1/kv_cache_interface.py`
- Modify: `vllm/model_executor/layers/attention/attention.py`

- [ ] Add new cache dtype strings and K/V bit config fields.
- [ ] Parse and validate preset vs generic bit selections.
- [ ] Introduce packed-int layout metadata and propagate it through attention specs.
- [ ] Keep all shared interface changes backward-compatible via optional params.

## Task 3: Add page-size accounting and reshape support

**Files:**

- Modify: `vllm/v1/core/kv_cache_utils.py`
- Modify: `vllm/v1/worker/gpu/attn_utils.py`
- Modify: `vllm/v1/worker/kv_cache_shape_utils.py`
- Modify: `vllm/v1/attention/backend.py`
- Modify: `vllm/v1/attention/backends/triton_attn.py`

- [ ] Make packed-int modes allocate by packed slot size.
- [ ] Preserve strict startup memory accounting.
- [ ] Preserve worker-side padded-page reshape restoration.

## Task 4: Add Gemma4-specific padding table

**Files:**

- Modify: `vllm/model_executor/models/gemma4.py`

- [ ] Add hard-coded `(k_bits, v_bits, head_dim)` padding lookup for Gemma4.
- [ ] Keep model-local logic hard-coded rather than generalized.

## Task 5: Add packed-int runtime kernels and dispatch

**Files:**

- Modify: `vllm/v1/attention/backends/triton_attn.py`
- Modify: `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- Modify: `vllm/v1/attention/ops/triton_unified_attention.py`
- Create: `vllm/v1/attention/ops/triton_packed_int_kv.py`

- [ ] Add packed-int cache view extraction from raw byte slots.
- [ ] Add specialized `8/4` cache-write and decode path.
- [ ] Add specialized `4/4` cache-write and decode path.
- [ ] Add generic `x/y` cache-write and decode path for `2..8` bits.
- [ ] Replace the packed-int Python whole-sequence attention fallback with a
  dedicated Triton tiled kernel that unpacks/dequantizes in-kernel and uses
  online softmax accumulation.

## Task 6: Build and runtime-verify all three chains

**Files:**

- Reuse existing docker build and runtime flow.

- [ ] Build image from the modified repo.
- [ ] Validate `int8_k_int4_v_per_token_head` using the original runtime args, changing only docker auto-restart and KV cache type.
- [ ] Validate `int4_per_token_head` using the original runtime args, changing only docker auto-restart and KV cache type.
- [ ] Validate `intx_k_inty_v_per_token_head` using the original runtime args, changing only docker auto-restart, KV cache type, and the new K/V bit flags.
- [ ] For the generic branch, validate multiple combinations covering even/even, odd/odd, divisible, and non-divisible packing.
- [ ] Re-run the prepared long engineering prompt that previously OOMed on
  `k4v3` and confirm the kernelized path returns a valid answer.

## Task 7: Summarize verified results

**Files:**

- Update local notes if needed.

- [ ] Report only after all three chains start and return valid model/completion responses.
- [ ] Include which bit combinations were used for generic-path verification.
