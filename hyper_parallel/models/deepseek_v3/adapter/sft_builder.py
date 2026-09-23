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
"""Reference configuration and weights adapted to Hyper's model build pipeline."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel

from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models._transformers.model_builder import (
    _build_replacement_context,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.models.deepseek_v3.configuration import load_reference, make_hf_config
from hyper_parallel.models.deepseek_v3.modeling_deepseek_v32_sft import (
    DeepseekV32SFTForCausalLM,
)
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.models.deepseek_v3.adapter.distributed.sft_expert_parallel import (
    deepseek_v32_sft_ep_compute,
)
from hyper_parallel.models.deepseek_v3.adapter.conversion.checkpoint_mapping import convert_reference


def _reference_moe_fqns(reference: dict[str, Any]) -> tuple[str, ...]:
    """Return the PR946 routed-MoE parents, excluding dense trunk layers."""
    trunk = tuple(
        f"model.layers.{index}.mlp"
        for index in range(reference["first_k_dense_replace"], reference["num_hidden_layers"])
    )
    mtp = tuple(
        f"mtp.layers.{index}.transformer_layer.mlp"
        for index in range(reference["num_nextn_predict_layers"])
    )
    return trunk + mtp


def _has_explicit_ep_override(overrides: dict[str, Any], fqn: str) -> bool:
    """Let a user-selected EP compute factory win over the reference default."""
    return any(
        (key == fqn or (any(char in key for char in "*?[") and fnmatchcase(fqn, key)))
        and getattr(spec, "local_compute_fn", None) is not None
        for key, spec in overrides.items()
    )


def _with_reference_ep_overrides(distributed_setup: Any, reference: dict[str, Any]) -> Any:
    """Add only missing PR946 MoE EP factories as explicit model FQNs."""
    overrides = dict(getattr(distributed_setup, "plan_overrides", None) or {})
    for fqn in _reference_moe_fqns(reference):
        if _has_explicit_ep_override(overrides, fqn):
            continue
        if fqn in overrides:
            overrides[fqn] = replace(
                overrides[fqn],
                local_compute_fn=deepseek_v32_sft_ep_compute,
                region_dispatch=(
                    False if overrides[fqn].region_dispatch is None
                    else overrides[fqn].region_dispatch
                ),
            )
            continue
        overrides[fqn] = ModuleShardingSpec(
            local_compute_fn=deepseek_v32_sft_ep_compute,
            region_dispatch=False,
        )
    return replace(distributed_setup, plan_overrides=overrides)


def build_sft_model(*, reference_yaml: str | Path, reference_weights: str | Path,
                    distributed_setup: Any, **infrastructure_options: Any) -> PreTrainedModel:
    """Build PR946 reference state before applying Hyper's parallel infrastructure."""
    if infrastructure_options.get("model_init_dtype") not in (None, "float32"):
        raise ValueError("SFT precision requires FP32 master parameters")
    infrastructure_options["model_init_dtype"] = "float32"

    document = load_reference(reference_yaml)
    reference = document["model"]["model_config"]
    arrays, _, conversion = convert_reference(reference_weights, document)
    setup = _with_reference_ep_overrides(distributed_setup, reference)
    mesh = setup.mesh_context
    if (mesh.tp_size, mesh.ep_size, mesh.cp_size, mesh.dp_size, mesh.pp_size) != (8, 8, 1, 1, 1):
        raise ValueError("SFT recipe requires TP8/EP8 and DP/CP/PP1")
    if not mesh.sequence_parallel or not mesh.loss_parallel:
        raise ValueError("SFT recipe requires sequence_parallel and loss_parallel")

    config = make_hf_config(document)
    with torch.device("meta"):
        model = DeepseekV32SFTForCausalLM(config)
        model, _ = _apply_module_replacement_actions(
            model,
            getattr(setup, "module_replacements", None),
            weights_mapping=get_model_conversion_mapping(model),
            context=_build_replacement_context(setup, None),
        )
    model.to_empty(device="cpu")
    # Rotary buffers are nonpersistent; restore their deterministic reference state after to_empty().
    model.model.rotary_emb = type(model.model.rotary_emb)(config)

    logical_groups = []
    for target in model.state_dict():
        suffix = "linear_qkv.weight"
        if not target.endswith(f".{suffix}"):
            continue
        prefix = target[:-len(suffix)]
        source_names = [f"{prefix}q_a_proj.weight", f"{prefix}kv_a_proj_with_mqa.weight"]
        values = [arrays.pop(name) for name in source_names]
        arrays[target] = np.concatenate(values, axis=0)
        restored = np.split(arrays[target], [values[0].shape[0]], axis=0)
        if any(left.tobytes() != right.tobytes() for left, right in zip(values, restored)):
            raise ValueError(f"Non-invertible MLA fusion: {target}")
        logical_groups.append({
            "storage": target,
            "logical_parameters": source_names,
            "sections": [value.shape[0] for value in values],
        })

    expected = model.state_dict()
    if set(expected) != set(arrays):
        raise ValueError(
            f"State coverage mismatch: missing={set(expected) - set(arrays)}, "
            f"unexpected={set(arrays) - set(expected)}",
        )
    for name, value in arrays.items():
        if tuple(expected[name].shape) != value.shape:
            raise ValueError(f"Shape mismatch for {name}: {expected[name].shape} != {value.shape}")
        if torch.from_numpy(value).dtype != expected[name].dtype:
            raise ValueError(
                f"Reference dtype mismatch for {name}: "
                f"{torch.from_numpy(value).dtype} != {expected[name].dtype}",
            )
    model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in arrays.items()},
        strict=True,
    )
    for name, value in model.state_dict().items():
        if value.detach().numpy().tobytes() != arrays[name].tobytes():
            raise ValueError(f"Loaded tensor differs: {name}")
    if model.model.embed_tokens.weight is model.lm_head.weight:
        raise ValueError("SFT embedding and LM head must not be tied")

    framework_setup = replace(setup, module_replacements=())

    # Replacements have already shaped the reference state, so do not apply them a second time.
    planner, fsdp = instantiate_infrastructure(distributed_setup=framework_setup)
    device = torch.device(mesh.device_mesh.device_type, torch.distributed.get_rank() % mesh.tp_size)
    model.to(device)
    model.loss_group = mesh.device_mesh["tp"].get_group()
    model = apply_model_infrastructure(
        model,
        mesh=mesh,
        sharding_planner=planner,
        fsdp2_manager=fsdp,
        distributed_setup=framework_setup,
        device=device,
        is_meta_device=False,
        is_hf_model=True,
        **infrastructure_options,
    )
    model.build_report = {
        **conversion,
        "model_class": type(model).__name__,
        "loaded_state_tensors": len(expected),
        "logical_optimizer_groups": logical_groups,
        "all_loaded_values_exact": True,
    }
    return model
