# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4-specific Rotary Positional Embeddings (proportional scaling).

Gemma4 uses "proportional" RoPE which computes inv_freq frequencies scaled
by head_dim (not rotary_dim), and zero-pads for non-rotated dimensions when
partial_rotary_factor < 1. The actual rotation uses standard neox-style
rotate_half, matching HF transformers' apply_rotary_pos_emb.
"""

import torch

from .base import RotaryEmbedding
from .common import yarn_find_correction_range, yarn_get_mscale, yarn_linear_ramp_mask


class Gemma4RotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE.

    Extends RotaryEmbedding (which provides standard neox-style rotation
    via ops.rotary_embedding CUDA kernel) but overrides the inv_freq
    computation to match HF's _compute_proportional_rope_parameters:
    - Frequency exponents use head_dim (not rotary_dim) as denominator
    - Non-rotated dims are zero-padded (cos=1, sin=0 = identity rotation)

    When partial_rotary_factor=1.0 (the default for some variants), ALL dims are
    rotated and this is equivalent to standard RotaryEmbedding with
    head_dim-scaled frequencies.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        # Number of rotation angle pairs (from partial_rotary_factor)
        self.rope_angles = rotary_dim // 2
        # Non-rotated angle pairs per half
        self.nope_angles = (head_size // 2) - self.rope_angles

        # Important: set rotary_dim = head_size so the base class's
        # forward_static applies rotation to ALL dims of the cos/sin cache.
        # The non-rotated dims will have cos=1, sin=0 (identity) thanks
        # to our _compute_inv_freq zero-padding.
        super().__init__(
            head_size,
            head_size,  # rotary_dim = head_size (full application)
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute frequencies matching HF proportional RoPE.

        Key difference from base: exponent denominator is head_size (not
        rotary_dim), and non-rotated dims are zero-padded.
        """
        # HF formula: base ** (arange(0, 2*rope_angles, 2) / head_dim)
        freq_exponents = (
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float) / self.head_size
        )
        inv_freq = 1.0 / (base**freq_exponents)

        # Zero-pad for non-rotated dims (identity rotation: cos=1, sin=0)
        if self.nope_angles > 0:
            inv_freq = torch.cat(
                [
                    inv_freq,
                    torch.zeros(self.nope_angles, dtype=torch.float),
                ]
            )
        return inv_freq

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", rope_angles={self.rope_angles}, nope_angles={self.nope_angles}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


class Gemma4YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE with YaRN scaling.

    This preserves Gemma4's full-head proportional semantics:
    - frequency exponents are normalized by the full head size
    - only the first `partial_rotary_factor` slice carries non-zero frequencies
    - the remaining dimensions stay zero-padded, so they rotate as identity

    The YaRN correction mask is evaluated in the full-head frequency space and
    then sliced to the active Gemma4 rope frequencies. This keeps the
    proportional frequency layout instead of compressing it into `rotary_dim`.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        apply_yarn_scaling: bool = True,
        truncate: bool = True,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.truncate = truncate
        self.mscale = (
            float(yarn_get_mscale(self.scaling_factor) * attn_factor)
            if apply_yarn_scaling
            else float(attn_factor)
        )

        self.rope_angles = rotary_dim // 2
        self.nope_angles = (head_size // 2) - self.rope_angles

        super().__init__(
            head_size,
            head_size,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        # Gemma4 proportional RoPE uses the full head size in the exponent
        # denominator, even when only a fraction of the head is rotated.
        freq_exponents = (
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float) / self.head_size
        )
        pos_freqs = self.base**freq_exponents
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.head_size,
            self.base,
            self.max_position_embeddings,
            self.truncate,
        )
        full_head_mask = (
            1
            - yarn_linear_ramp_mask(
                low,
                high,
                self.head_size // 2,
                dtype=torch.float,
            )
        ) * self.extrapolation_factor
        inv_freq_mask = full_head_mask[: self.rope_angles]
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        if self.nope_angles > 0:
            inv_freq = torch.cat(
                [
                    inv_freq,
                    torch.zeros(self.nope_angles, dtype=torch.float),
                ]
            )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            int(self.max_position_embeddings * self.scaling_factor),
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", rope_angles={self.rope_angles}, nope_angles={self.nope_angles}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        s += f", scaling_factor={self.scaling_factor}, mscale={self.mscale}"
        return s
