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
"""Eight-rank Hyper apply/MLA probe; this is not a reference SFT training entry.

The stock DeepSeek EP factory is used only to check apply-time compatibility.
No MoE forward is called, and reference routing/loss equivalence is not claimed.
"""

import argparse
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


def main() -> None:
    """Apply actual eight-rank meshes and optionally execute one 256K MLA layer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-yaml", required=True)
    parser.add_argument("--reference-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mla-forward", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        if dist.get_world_size() != 8:
            raise ValueError("Probe requires exactly eight ranks")
        document = load_reference(args.reference_yaml)
        config = document["model"]["model_config"]
        arrays, _, _ = convert_reference(args.reference_weights, document)
        model, report = inspect_structure(document, arrays)
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
                    local_compute_fn=deepseekv3_ep_compute_fn, region_dispatch=False)
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
