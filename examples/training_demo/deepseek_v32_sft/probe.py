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
"""Eight-rank layout/MLA probe and optional reference SFT first-loss evaluator.

The first-loss mode uses the reference EP factory and runs the complete forward.
This entry does not execute backward or optimizer updates. Archived-model
tensor diagnostics are optional and excluded from the normal evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.expert_parallel.recipes import deepseekv3_ep_compute_fn
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec

from .audit import inspect_structure
from .config import load_reference
from .parallel import plan_overrides
from .weights import convert_reference
from .ep import deepseek_v32_sft_ep_compute
from .diagnostics import attach_reference_comparison


def main() -> None:
    """Apply eight-rank meshes and optionally evaluate MLA or the 256K first loss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-yaml", required=True)
    parser.add_argument("--reference-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mla-forward", action="store_true")
    parser.add_argument("--first-loss-data")
    parser.add_argument("--reference-code")
    parser.add_argument("--reference-model-class", default="ReferenceSFTModel")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("hccl")
    try:
        if dist.get_world_size() != 8:
            raise ValueError("Probe requires exactly eight ranks")
        document = load_reference(args.reference_yaml)
        config = document["model"]["model_config"]
        arrays, _, _ = convert_reference(args.reference_weights, document)
        model, report = inspect_structure(document, arrays, sft_forward=args.first_loss_data is not None)
        global_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        device = torch.device(f"npu:{rank}")
        # Nonpersistent meta RoPE buffers are unused: this probe supplies frequencies explicitly.
        for parameter in model.parameters():
            parameter.data = parameter.data.to(device)
        for module in model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if not buffer.is_meta:
                    setattr(module, name, buffer.to(device))
        mesh = MeshContext(tp_size=8, ep_size=8, sequence_parallel=True, loss_parallel=True)
        mesh.build_meshs("npu", 8)
        overrides = plan_overrides(config["num_nextn_predict_layers"])
        for name, module in model.named_modules():
            if hasattr(module, "experts") and hasattr(module, "shared_experts"):
                overrides[name] = ModuleShardingSpec(
                    local_compute_fn=deepseek_v32_sft_ep_compute if args.first_loss_data else deepseekv3_ep_compute_fn,
                    region_dispatch=False)
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            model, mesh.device_mesh, tp_size=8, ep_size=8, cp_size=1,
            sequence_parallel=True, loss_parallel=True,
        )
        model, source_info = apply_sharding_plan(model, plan, mesh)
        for name, parameter in model.named_parameters():
            expected = global_parameters[name]
            placements, source_mesh = source_info[name]
            coordinates = source_mesh.get_coordinate()
            for axis, placement in enumerate(placements):
                if placement.is_shard():
                    expected = expected.chunk(source_mesh.size(axis), dim=placement.dim)[coordinates[axis]]
            actual = parameter.detach().cpu()
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError(f"Hyper parameter shard differs from mapped reference: {name}")
        report = {"rank": rank, "stage": "apply_only", "parameters": {
            name: list(parameter.shape) for name, parameter in model.named_parameters()},
                  "source_layout_count": len(source_info), "sequence_length": config["seq_length"],
                  "all_parameter_shards_exact": True}
        if args.mla_forward:
            length = config["seq_length"]
            torch.manual_seed(42 + rank)
            hidden = torch.randn(1, length // 8, config["hidden_size"], device=device, dtype=torch.bfloat16)
            dim = config["qk_rope_head_dim"]
            inverse = 1.0 / (config["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
            inverse = torch.from_numpy(inverse.astype(np.float32)).to(device)
            frequency = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inverse
            frequency = torch.cat((frequency, frequency), dim=-1).unsqueeze(0)
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                result, _ = model.model.layers[0].self_attn(
                    hidden, position_embeddings=(frequency.cos(), frequency.sin()),
                    actual_seq_len=(length,),
                )
            torch_npu.npu.synchronize()
            report.update(stage="mla_forward", output_shape=list(result.shape),
                          output_finite=bool(torch.isfinite(result).all().item()))
        if args.first_loss_data:
            with np.load(args.first_loss_data, allow_pickle=False) as archive:
                tokens = torch.from_numpy(archive["input_ids"][0].astype(np.int64)).unsqueeze(0).to(device)
                labels = torch.from_numpy(archive["labels"][0].astype(np.int64)).unsqueeze(0).to(device)
                loss_mask = torch.from_numpy(archive["loss_mask"][0].astype(np.float32)).unsqueeze(0).to(device)
            handles = []
            if args.reference_code:
                handles, comparisons, reference_losses = attach_reference_comparison(
                    model, document, args.reference_code, args.reference_yaml, args.reference_weights,
                    tokens, labels, loss_mask, args.reference_model_class)
                report.update(tensor_comparisons=comparisons, reference_losses=reference_losses)
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                losses = model(tokens, labels, loss_mask)
            for handle in handles:
                handle.remove()
            torch_npu.npu.synchronize()
            report.update(stage="first_loss", losses={name: value.item() for name, value in losses.items()},
                          dataset_sha256=hashlib.sha256(Path(args.first_loss_data).read_bytes()).hexdigest())
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"rank_{rank}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if rank == 0:
            (destination / "parallel_plan.txt").write_text(plan.explain(), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
