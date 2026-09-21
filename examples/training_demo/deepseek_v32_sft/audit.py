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
"""Audit real HF parameter structure and Hyper replacement without training.

The temporary MTP containers below inspect parameter coverage only. They do not
implement MTP forward or claim numerical equivalence of the replaced modules.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import transformers
from transformers import DeepseekV32ForCausalLM
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32RMSNorm,
)

from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)

from .config import load_reference, make_hf_config
from .parallel import plan_overrides
from .weights import convert_reference


def inspect_structure(document: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[nn.Module, dict]:
    """Check exact state coverage using real HF modules on meta then CPU.

    Args:
        document: Validated reference configuration.
        arrays: Global HF-layout arrays reconstructed from reference rank shards.
    """
    cfg = make_hf_config(document)
    reference = document["model"]["model_config"]
    with torch.device("meta"):
        model = DeepseekV32ForCausalLM(cfg)
        model.mtp = nn.Module()
        model.mtp.layers = nn.ModuleList()
        mtp_config = copy.deepcopy(cfg)
        mtp_config.mlp_layer_types = ["sparse"] * reference["num_nextn_predict_layers"]
        for index in range(reference["num_nextn_predict_layers"]):
            layer = nn.Module()
            layer.enorm = DeepseekV32RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            layer.hnorm = DeepseekV32RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            layer.eh_proj = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
            layer.transformer_layer = DeepseekV32DecoderLayer(mtp_config, index)
            layer.final_layernorm = DeepseekV32RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            model.mtp.layers.append(layer)
        spec = ModuleReplacementSpec(
            match=("model.layers.*.self_attn", "mtp.layers.*.transformer_layer.self_attn"),
            factory=MLAAttention, module_type=DeepseekV32Attention, exact_type=True,
        )
        replacement_plan = compile_module_replacements(model, [spec])
        model, transforms = apply_module_replacements(model, replacement_plan, weights_mapping=[])
    fused = dict(arrays)
    logical_groups = []
    for name, module in model.named_modules():
        if not isinstance(module, MLAAttention):
            continue
        source_names = [f"{name}.q_a_proj.weight", f"{name}.kv_a_proj_with_mqa.weight"]
        values = [fused.pop(key) for key in source_names]
        target = f"{name}.linear_qkv.weight"
        fused[target] = np.concatenate(values, axis=0)
        restored = np.split(fused[target], [values[0].shape[0]], axis=0)
        if any(left.tobytes() != right.tobytes() for left, right in zip(values, restored)):
            raise ValueError(f"Non-invertible MLA fusion: {target}")
        logical_groups.append({"storage": target, "logical_parameters": source_names,
                               "sections": [value.shape[0] for value in values]})
    expected = model.state_dict()
    if set(expected) != set(fused):
        raise ValueError(f"State coverage mismatch: missing={set(expected) - set(fused)}, "
                         f"unexpected={set(fused) - set(expected)}")
    for name, value in fused.items():
        if tuple(expected[name].shape) != value.shape:
            raise ValueError(f"Shape mismatch: {name}: {expected[name].shape} != {value.shape}")
    model.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in fused.items()},
                          strict=True, assign=True)
    for name, value in model.state_dict().items():
        if value.detach().numpy().tobytes() != fused[name].tobytes():
            raise ValueError(f"Loaded tensor differs: {name}")
    if model.model.embed_tokens.weight is model.lm_head.weight:
        raise ValueError("Reference embedding and LM head must not be tied")
    return model, {"model_class": type(model).__name__, "loaded_state_tensors": len(fused),
                   "replacement_count": len(replacement_plan.targets), "transform_count": len(transforms),
                   "logical_optimizer_groups": logical_groups, "all_loaded_values_exact": True}


def main() -> None:
    """Write conversion evidence and optionally derive a non-executing TP/EP plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-yaml", required=True)
    parser.add_argument("--reference-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    document = load_reference(args.reference_yaml)
    arrays, auxiliary, report = convert_reference(args.reference_weights, document)
    model, structure = inspect_structure(document, arrays)
    report.update(structure)
    report["transformers_version"] = transformers.__version__
    report["sequence_length"] = document["model"]["model_config"]["seq_length"]
    report["reference_yaml_sha256"] = hashlib.sha256(Path(args.reference_yaml).read_bytes()).hexdigest()
    report["validation_scope"] = "weight structure and lossless conversion only; no forward or distributed run"
    np.savez(output / "preserved_reference_state.npz", **auxiliary)
    (output / "weight_mapping.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.plan:
        mesh = init_device_mesh("npu", (1, 1, 8), mesh_dim_names=("dp", "cp", "tp"),
                                rank_list=list(range(8)), init_backend=False)
        overrides = plan_overrides(document["model"]["model_config"]["num_nextn_predict_layers"])
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            model, mesh, tp_size=8, ep_size=8, cp_size=1, sequence_parallel=True, loss_parallel=True)
        (output / "parallel_plan.txt").write_text(plan.explain(), encoding="utf-8")
    print(json.dumps({name: value for name, value in report.items() if name != "rules"}, indent=2))


if __name__ == "__main__":
    main()
