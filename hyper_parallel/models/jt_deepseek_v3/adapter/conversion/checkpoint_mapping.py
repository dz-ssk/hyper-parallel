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
"""Offline conversion of reference shards into recipe-ready JT NPZ weights."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import DeepseekV32Config

from hyper_parallel.models.jt_deepseek_v3.configuration import load_reference


@dataclass(frozen=True)
class MappingRule:
    """One reference logical tensor and its reversible HF storage transformation."""

    source: str
    targets: tuple[str, ...]
    shard_axis: int | None = None
    operation: str = "identity"

    def encode(self, array: np.ndarray, model: DeepseekV32Config) -> tuple[np.ndarray, ...]:
        """Convert a global reference tensor into HF storage without arithmetic."""
        if self.operation == "interleaved":
            return array[0::2].copy(), array[1::2].copy()
        if self.operation == "expert_up":
            return (array.reshape(model.n_routed_experts, model.hidden_size, -1)
                    .transpose(0, 2, 1).copy(),)
        if self.operation == "expert_down":
            return (array.reshape(model.n_routed_experts, model.moe_intermediate_size, -1)
                    .transpose(0, 2, 1).copy(),)
        if self.operation != "identity":
            raise ValueError(f"Unknown mapping operation: {self.operation}")
        return (array.copy(),)

    def decode(self, arrays: tuple[np.ndarray, ...]) -> np.ndarray:
        """Invert the mapping for round-trip validation."""
        if self.operation == "interleaved":
            return np.stack(arrays, axis=1).reshape(-1, arrays[0].shape[-1])
        if self.operation in ("expert_up", "expert_down"):
            value = arrays[0].transpose(0, 2, 1)
            return value.reshape(-1, value.shape[-1])
        return arrays[0]


def mapping_rules(model: DeepseekV32Config) -> list[MappingRule]:
    """Declare reference-to-model mappings for the native JT configuration."""
    rules = [
        MappingRule("embedding.word_embeddings.weight", ("model.embed_tokens.weight",), 0),
        MappingRule("output_layer.weight", ("lm_head.weight",), 0),
        MappingRule("decoder.final_layernorm.weight", ("model.norm.weight",)),
    ]
    layers = [
        (f"decoder.layers.{index}", f"model.layers.{index}", model.mlp_layer_types[index] == "sparse")
        for index in range(model.num_hidden_layers)
    ]
    for index in range(model.num_nextn_predict_layers):
        prefix = f"mtp.layers.{index}"
        for name in ("enorm", "hnorm", "eh_proj", "final_layernorm"):
            rules.append(MappingRule(f"{prefix}.{name}.weight", (f"{prefix}.{name}.weight",)))
        layers.append((f"{prefix}.transformer_layer", f"{prefix}.transformer_layer", True))
    attention = (
        ("linear_proj", "o_proj", 1),
        ("linear_q_down_proj", "q_a_proj", None),
        ("linear_q_up_proj", "q_b_proj", 0),
        ("linear_kv_down_proj", "kv_a_proj_with_mqa", None),
        ("linear_kv_up_proj", "kv_b_proj", 0),
        ("q_layernorm", "q_a_layernorm", None),
        ("kv_layernorm", "kv_a_layernorm", None),
    )
    for source, target, moe in layers:
        for old, new in (("input_layernorm", "input_layernorm"), ("pre_mlp_layernorm", "post_attention_layernorm")):
            rules.append(MappingRule(f"{source}.{old}.weight", (f"{target}.{new}.weight",)))
        for old, new, axis in attention:
            rules.append(MappingRule(f"{source}.self_attention.{old}.weight", (f"{target}.self_attn.{new}.weight",), axis))
        if moe:
            rules.extend([
                MappingRule(f"{source}.mlp.router.weight", (f"{target}.mlp.gate.weight",)),
                MappingRule(f"{source}.mlp.router.expert_bias", (f"{target}.mlp.gate.e_score_correction_bias",)),
                MappingRule(f"{source}.mlp.experts.weight1", (f"{target}.mlp.experts.gate_up_proj",), 0, "expert_up"),
                MappingRule(f"{source}.mlp.experts.weight2", (f"{target}.mlp.experts.down_proj",), 0, "expert_down"),
            ])
        dense_source = f"{source}.mlp" + (".shared_experts" if moe else "")
        dense_target = f"{target}.mlp" + (".shared_experts" if moe else "")
        rules.append(MappingRule(
            f"{dense_source}.linear_fc1.weight",
            (f"{dense_target}.gate_proj.weight", f"{dense_target}.up_proj.weight"),
            None if moe else 0,
            "interleaved",
        ))
        rules.append(MappingRule(
            f"{dense_source}.linear_fc2.weight",
            (f"{dense_target}.down_proj.weight",),
            None if moe else 1,
        ))
    return rules


def _join_reference_parts(rule: MappingRule, parts: list[np.ndarray]) -> np.ndarray:
    """Join sharded tensors or require replicated tensors to match bitwise."""
    if rule.shard_axis is not None:
        return np.concatenate(parts, axis=rule.shard_axis)
    if any(part.tobytes() != parts[0].tobytes() or part.shape != parts[0].shape for part in parts):
        raise ValueError(f"Replicated reference state differs across shards: {rule.source}")
    return parts[0]


def convert_reference(directory: str | Path, document: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Reconstruct and convert reference shards into recipe-ready arrays."""
    model = DeepseekV32Config(**document["model"]["model_config"])
    world = document["parallel_config"]["model_parallel"]
    shards = []
    for rank in range(world):
        with np.load(Path(directory) / f"initial_rank_{rank}.npz", allow_pickle=False) as archive:
            shards.append({name: archive[name] for name in archive.files})
    if any(set(shard) != set(shards[0]) for shard in shards):
        raise ValueError("Reference rank archives have different keys")

    converted, consumed, entries = {}, set(), []
    for rule in mapping_rules(model):
        parts = [shard[rule.source] for shard in shards]
        global_array = _join_reference_parts(rule, parts)
        values = rule.encode(global_array, model)
        restored = rule.decode(values)
        if restored.shape != global_array.shape or restored.tobytes() != global_array.tobytes():
            raise ValueError(f"Non-invertible weight mapping: {rule.source}")
        for name, value in zip(rule.targets, values):
            if name in converted:
                raise ValueError(f"Duplicate destination: {name}")
            converted[name] = value
        consumed.add(rule.source)
        entries.append({
            "source": rule.source,
            "targets": list(rule.targets),
            "shard_axis": rule.shard_axis,
            "operation": rule.operation,
            "shape": list(global_array.shape),
            "roundtrip_exact": True,
        })

    auxiliary = {}
    allowed = (".seed", ".offset", ".core_attention.max_logits_val", ".router.expert_load", ".router.fi_accu")
    for name in sorted(set(shards[0]) - consumed):
        if not name.endswith(allowed):
            raise ValueError(f"Unmapped reference tensor: {name}")
        for rank, shard in enumerate(shards):
            auxiliary[f"rank_{rank}/{name}"] = shard[name]

    report = {
        "world_size": world,
        "source_keys_per_rank": len(shards[0]),
        "rules": entries,
        "hf_tensors": len(converted),
        "preserved_auxiliary_shards": len(auxiliary),
    }
    report["logical_optimizer_groups"] = _fuse_mla_projections(converted)
    report["loaded_state_tensors"] = len(converted)
    return converted, auxiliary, report


def _fuse_mla_projections(arrays: dict[str, np.ndarray]) -> list[dict]:
    """Fuse trunk and MTP Q/KV projections for the JT MLA replacement."""
    logical_groups = []
    for name in tuple(arrays):
        suffix = "q_a_proj.weight"
        if not name.endswith(f".{suffix}"):
            continue
        prefix = name[:-len(suffix)]
        source_names = [name, f"{prefix}kv_a_proj_with_mqa.weight"]
        target = f"{prefix}linear_qkv.weight"
        if target in arrays:
            raise ValueError(f"Duplicate destination: {target}")
        values = [arrays.pop(source) for source in source_names]
        arrays[target] = np.concatenate(values, axis=0)
        restored = np.split(arrays[target], [values[0].shape[0]], axis=0)
        if any(left.tobytes() != right.tobytes() for left, right in zip(values, restored)):
            raise ValueError(f"Non-invertible MLA fusion: {target}")
        logical_groups.append({
            "storage": target,
            "logical_parameters": source_names,
            "sections": [value.shape[0] for value in values],
        })
    return logical_groups


def main() -> None:
    """Convert reference initial_rank_*.npz shards once into model.npz."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-yaml", type=Path, required=True)
    parser.add_argument("--reference-weights", type=Path, required=True,
                        help="Directory containing the original initial_rank_*.npz shards")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory used by model.reference_weights in the JT recipe")
    args = parser.parse_args()
    output_path = args.output / "model.npz"
    if output_path.exists():
        raise FileExistsError(output_path)
    arrays, _, report = convert_reference(args.reference_weights, load_reference(args.reference_yaml))
    args.output.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output:
        np.savez(output, **arrays)
    print(f"Converted {report['world_size']} reference shards into {len(arrays)} tensors: {output_path}")


if __name__ == "__main__":
    main()
