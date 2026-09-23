# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Reference document validation and HF configuration for the JT model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from transformers import DeepseekV32Config


class JTDeepseekV3Config(DeepseekV32Config):
    """Keep HF configuration fields with an independently discoverable family identity."""

    model_type = "jt_deepseek_v3"


def load_reference(path: str | Path) -> dict[str, Any]:
    """Read and validate the reference topology and compute policy.

    Args:
        path: Reference YAML path.
    """
    with Path(path).open(encoding="utf-8") as source:
        document = yaml.safe_load(source)
    model = document["model"]["model_config"]
    parallel = document["parallel_config"]
    expected_parallel = {
        "data_parallel": 1,
        "model_parallel": 8,
        "expert_parallel": 8,
        "pipeline_stage": 1,
        "context_parallel": 1,
        "micro_batch_num": 1,
        "use_seq_parallel": True,
    }
    for name, value in expected_parallel.items():
        if parallel[name] != value:
            raise ValueError(f"Reference recipe requires {name}={value}, got {parallel[name]}")
    expected_model = {
        "seq_length": 262144,
        "params_dtype": "float32",
        "compute_dtype": "bfloat16",
        "multi_latent_attention": True,
        "use_flash_attention": True,
        "mla_qkv_concat": False,
        "add_bias_linear": False,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "rotary_scaling_factor": 1,
        "n_group": 1,
        "topk_group": 1,
        "scoring_func": "sigmoid",
    }
    for name, value in expected_model.items():
        if model[name] != value:
            raise ValueError(f"Unsupported reference {name}={model[name]}; expected {value}")
    if document["runner_config"]["batch_size"] != 1:
        raise ValueError("Reference conversion currently requires batch size one")
    for name in ("vocab_size", "num_attention_heads", "intermediate_size", "n_routed_experts"):
        if model[name] % 8:
            raise ValueError(f"{name} must be divisible by eight")
    return document


def make_hf_config(document: dict[str, Any]) -> DeepseekV32Config:
    """Map every reference model dimension without taking HF defaults.

    Args:
        document: Validated reference configuration.
    """
    reference = document["model"]["model_config"]
    names = (
        "vocab_size", "hidden_size", "intermediate_size", "moe_intermediate_size",
        "num_hidden_layers", "num_attention_heads", "n_shared_experts", "n_routed_experts",
        "routed_scaling_factor", "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim",
        "v_head_dim", "qk_nope_head_dim", "n_group", "topk_group", "num_experts_per_tok",
        "norm_topk_prob", "hidden_act", "max_position_embeddings", "initializer_range",
        "rms_norm_eps", "first_k_dense_replace", "attention_dropout",
    )
    config = JTDeepseekV3Config(
        **{name: reference[name] for name in names},
        num_key_value_heads=reference["num_attention_heads"],
        rope_parameters={"rope_type": "default", "rope_theta": reference["rope_theta"]},
        tie_word_embeddings=False,
        use_cache=False,
        attention_bias=False,
        mlp_bias=False,
        layer_types=["full_attention"] * reference["num_hidden_layers"],
    )
    config.architectures = ["JTDeepseekV3ForCausalLM"]
    config.jt_config = dict(reference)
    config.rope_interleave = True
    return config


__all__ = ["JTDeepseekV3Config", "load_reference", "make_hf_config"]
