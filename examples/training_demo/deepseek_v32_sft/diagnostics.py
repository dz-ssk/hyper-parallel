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
"""Optional tensor diagnostics against the archived, independently loaded model."""

import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.functional.rotary_embedding import apply_rotary_pos_emb_interleave


def attach_reference_comparison(model: Any, document: dict, reference_code: str,
                                yaml_path: str, weights_path: str, tokens: torch.Tensor,
                                labels: torch.Tensor, mask: torch.Tensor,
                                reference_model_class: str = "ReferenceSFTModel") -> tuple[list, list, dict]:
    """Capture archived outputs and compare matching HF module boundaries.

    The archived model is diagnostic only; it is not part of the HF execution
    graph and is never modified. The returned handles must be removed after use.
    """
    path = Path(reference_code)
    package = "deepseek_v32_sft_archived_diagnostic"
    spec = importlib.util.spec_from_file_location(package, path / "__init__.py",
                                                submodule_search_locations=[str(path)])
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load archived reference package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    config_class = importlib.import_module(f"{package}.config").SFTConfig
    parallel_class = importlib.import_module(f"{package}.parallel").ParallelContext
    model_class = getattr(importlib.import_module(f"{package}.model"), reference_model_class)
    reference = model_class(config_class(yaml_path), parallel_class(dist.get_world_size())).to(tokens.device)
    reference.load_reference(Path(weights_path) / f"initial_rank_{dist.get_rank()}.npz")
    mapping = {"embedding.word_embeddings": "model.embed_tokens", "decoder.final_layernorm": "model.norm",
               "output_layer": "lm_head"}
    cfg = document["model"]["model_config"]
    layers = [(f"decoder.layers.{index}", f"model.layers.{index}")
              for index in range(cfg["num_hidden_layers"])]
    for index in range(cfg["num_nextn_predict_layers"]):
        prefix = f"mtp.layers.{index}"
        for suffix in ("enorm", "hnorm", "eh_proj", "final_layernorm"):
            mapping[f"{prefix}.{suffix}"] = f"{prefix}.{suffix}"
        layers.append((f"{prefix}.transformer_layer", f"{prefix}.transformer_layer"))
    for source, target in layers:
        mapping[source] = target
        for old, new in (("input_layernorm", "input_layernorm"), ("self_attention", "self_attn"),
                         ("pre_mlp_layernorm", "post_attention_layernorm"), ("mlp", "mlp")):
            mapping[f"{source}.{old}"] = f"{target}.{new}"
    for old, new in (("q_layernorm", "q_a_layernorm"), ("kv_layernorm", "kv_a_layernorm"),
                     ("linear_q_up_proj", "q_b_proj"), ("linear_kv_up_proj", "kv_b_proj"),
                     ("linear_proj", "o_proj")):
        mapping[f"decoder.layers.0.self_attention.{old}"] = f"model.layers.0.self_attn.{new}"
    mapping["decoder.layers.0.self_attention.rotary"] = "model.layers.0.self_attn.explicit_rotary"
    captured, cursors, results = {}, {}, []

    def capture(name: str):
        def hook(owner: Any, inputs: tuple, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured.setdefault(name, []).append(value.detach().float().cpu())
        return hook

    handles = [reference.get_submodule(source).register_forward_hook(capture(target))
               for source, target in mapping.items()]
    with torch.no_grad():
        reference_losses = reference(tokens[0], labels[0], mask[0])
    reference_values = {name: value.item() for name, value in reference_losses.items()}
    for handle in handles:
        handle.remove()
    del reference

    def compare(name: str):
        def hook(owner: Any, inputs: tuple, output: Any) -> None:
            index = cursors.get(name, 0)
            cursors[name] = index + 1
            value = output[0] if isinstance(output, tuple) else output
            actual = value.detach().float().cpu().reshape(-1, value.shape[-1])
            expected = captured[name][index].reshape(-1, actual.shape[-1])
            if actual.shape[0] == expected.shape[0] * dist.get_world_size():
                actual = actual.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]
            if actual.shape != expected.shape:
                raise ValueError(f"Diagnostic shape mismatch at {name}: {actual.shape} != {expected.shape}")
            delta = (actual - expected).abs()
            results.append({"module": name, "call": index, "shape": list(actual.shape),
                            "different": int(torch.count_nonzero(delta)), "max_abs": delta.max().item()})
            if name.endswith(".explicit_rotary"):
                values, cos, sin = inputs
                fused, _ = apply_rotary_pos_emb_interleave(values, values, cos, sin)
                rotary_delta = (fused.float() - value.float()).abs()
                results[-1]["generic_fused_rope_different"] = int(torch.count_nonzero(rotary_delta).item())
                results[-1]["generic_fused_rope_max_abs"] = rotary_delta.max().item()
            captured[name][index] = None
        return hook

    handles = [model.get_submodule(target).register_forward_hook(compare(target)) for target in mapping.values()]
    return handles, results, reference_values
