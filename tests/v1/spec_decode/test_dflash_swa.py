# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.transformers_utils.configs.speculators import SpeculatorsConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer


class _FakeBuilder:
    def build_for_drafting(self, common_attn_metadata, draft_index):
        return SimpleNamespace(causal=common_attn_metadata.causal)


class _FakeAttentionGroup:
    def __init__(self, layer_names):
        self.layer_names = layer_names
        self._builder = _FakeBuilder()

    def get_metadata_builder(self):
        return self._builder


def test_dflash_speculators_preserves_swa_config():
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    config = {
        "speculators_model_type": "dflash",
        "transformer_layer_config": {
            "num_hidden_layers": len(layer_types),
            "sliding_window": None,
        },
        "draft_vocab_size": 100,
        "target_hidden_size": 64,
        "aux_hidden_state_layer_ids": [0, 1, 2],
        "mask_token_id": 99,
        "layer_types": layer_types,
        "use_sliding_window": True,
        "sliding_window": 2048,
        "max_window_layers": len(layer_types),
    }

    hf_config = SpeculatorsConfig.extract_transformers_pre_trained_config(config)

    assert hf_config["layer_types"] == layer_types
    assert hf_config["use_sliding_window"] is True
    assert hf_config["sliding_window"] == 2048
    assert hf_config["max_window_layers"] == len(layer_types)


def test_dflash_swa_layers_use_causal_metadata():
    proposer = object.__new__(DFlashProposer)
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer.draft_attn_groups = [_FakeAttentionGroup(["layer.sw", "layer.full"])]
    cad = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.empty(1, 1, dtype=torch.int32),
        slot_mapping=torch.empty(2, dtype=torch.int64),
        causal=False,
    )

    _, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, cad
    )

    assert per_layer["layer.sw"].causal is True
    assert per_layer["layer.full"].causal is False
