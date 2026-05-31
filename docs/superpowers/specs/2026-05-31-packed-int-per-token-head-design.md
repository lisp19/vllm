# Packed-int Per-Token-Head KV Cache Design

**Goal:** add three lower-memory KV cache modes for Gemma4 on the Triton v1
attention path: `int8_k_int4_v_per_token_head`, `int4_per_token_head`, and
`intx_k_inty_v_per_token_head`.

## Scope

This design only targets the current working `int8_per_token_head + Triton`
baseline and keeps the existing hybrid full/sliding attention, mm-prefix, and
strict startup KV-capacity rules.

## Architecture

Public config, metadata, page-size accounting, and padding propagation are all
designed around the generic `intx_k_inty_v_per_token_head` model. The 8/4 and
4/4 paths are presets that reuse the same metadata but may use specialized
store/unpack implementations.

Packed-int modes use `torch.uint8` raw-byte storage with one combined slot per
token/head. Backend-side views split that slot into key bytes, value bytes, K
scale, and V scale. Memory accounting uses packed byte counts rather than the
existing `dtype * head_size` formula.

## Public configuration

Expose three user-visible cache dtypes:

- `int8_k_int4_v_per_token_head`
- `int4_per_token_head`
- `intx_k_inty_v_per_token_head`

and two new optional config fields / CLI flags:

- `kv_cache_k_bits`
- `kv_cache_v_bits`

Rules:

- 8/4 preset resolves to `(8, 4)`.
- int4 preset resolves to `(4, 4)`.
- generic mode requires both K/V bit fields.
- allowed bit range is `2..8`.

## Metadata propagation

The lower layers must receive enough structured metadata to avoid repeated
string parsing. The propagated layout object carries at least:

- `k_bits`
- `v_bits`
- `head_size`
- `head_size_v`
- `k_data_bytes`
- `v_data_bytes`
- `k_scale_bytes`
- `v_scale_bytes`
- `raw_bytes_per_token_head`
- `padded_bytes_per_token_head`
- `kernel_kind`

Public chain signature changes must keep backward compatibility by adding only
optional parameters.

## Backend/runtime design

The Triton backend adds a packed-int path alongside the existing
`int8_per_token_head` path.

- cache allocation shape is derived from the packed slot size
- backend view extraction reconstructs packed key/value views and scale views
- cache-update dispatch selects specialized 8/4, specialized 4/4, or generic
  packed-int writer
- attention forward uses a dedicated Triton tiled packed-int attention kernel
  instead of a Python path that materializes dense K/V or a full score matrix

## Long-context production requirement

The packed-int read path must be production-usable under long engineering
prompts. That requires avoiding a fallback implementation that:

- fully materializes dense K/V for a whole sequence
- allocates a full `Q @ K^T` score matrix

Instead, the packed-int path must unpack/dequantize tile-by-tile inside a
Triton kernel and accumulate with an online softmax loop.

## Gemma4 padding

Gemma4 uses hard-coded padding rules keyed by `(k_bits, v_bits, head_dim)`.
This is intentionally model-specific and does not need to be generalized.

The table is allowed to cover `k_bits=8..2` and `v_bits=8..2`. For full vs
sliding layers, global padded page size is chosen so hybrid page-size
unification remains valid.

## Verification

Validation uses the original successful docker run shape and only changes:

- docker auto-restart disabled for log viewing
- KV cache dtype
- generic mode K/V bit arguments

Required runtime checks for each path:

- container starts successfully
- startup KV capacity checks remain strict
- `/v1/models` returns 200
- `/v1/chat/completions` returns 200 with non-empty content

Validation order:

1. `int8_k_int4_v_per_token_head`
2. `int4_per_token_head`
3. `intx_k_inty_v_per_token_head`

The generic path must be validated with multiple combinations, including odd
and non-byte-aligned cases.

## Execution notes

- Implement all three branches in one integrated development pass before
  starting runtime validation.
- Use staged commits to keep config/metadata, runtime implementation, and
  verification-related changes traceable.
- Reuse a built image across validation steps unless code changes require a new
  rebuild.
