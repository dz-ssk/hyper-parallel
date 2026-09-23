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
"""Reference routing semantics on Hyper's EP dispatcher and local expert binder."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.nn import functional as F

from hyper_parallel.core.dtensor._utils import differentiable_all_reduce
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute


@local_compute
def deepseek_v32_sft_ep_compute(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any) -> Callable:
    """Bind reference sigmoid routing and auxiliary loss to Hyper EP communication.

    Args:
        module: HF module being adapted.
        mesh: Mesh.
        tp_mesh: Tp mesh.
        cp_mesh: Cp mesh.
        ep_mesh: Ep mesh.
    """
    del mesh, tp_mesh, cp_mesh
    if ep_mesh is None:
        raise ValueError("DeepSeek V3.2 SFT requires an EP mesh")
    group = ep_mesh.get_group("ep")
    world = ep_mesh["ep"].size()
    padding = module.reference_config["n_routed_experts"] if module.reference_config["use_pad_tokens"] else 0

    def router(owner: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select sigmoid experts and compute sequence-balancing loss.

        Args:
            owner: MoE module owning router state.
            hidden: Trunk hidden states.
        """
        config = owner.reference_config
        hidden = hidden[:, padding:]
        with torch.autocast("npu", enabled=False):
            logits = F.linear(hidden.reshape(-1, hidden.shape[-1]).float(), owner.gate.weight)
            scores = logits.sigmoid()
            selection = scores + owner.gate.e_score_correction_bias
            indices = selection.topk(config["num_experts_per_tok"], dim=-1).indices
            frequency = torch.bincount(indices.flatten(), minlength=config["n_routed_experts"]).float()
            frequency = frequency / indices.numel()
            dist.all_reduce(frequency, group=group)
            frequency = frequency / world
            selected = scores.gather(-1, indices)
            if config["norm_topk_prob"] and config["num_experts_per_tok"] > 1:
                selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
            normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
            auxiliary = (normalized.mean(0) * frequency).sum() * scores.shape[-1] * config["moe_aux_loss_coeff"]
            owner.auxiliary_loss = differentiable_all_reduce(auxiliary, "sum", group) / world
        selected = selected * config["routed_scaling_factor"]
        if padding:
            pad_ids = torch.arange(padding * config["num_experts_per_tok"], device=indices.device)
            pad_ids = pad_ids.reshape(padding, config["num_experts_per_tok"]) % padding
            indices = torch.cat((pad_ids, indices))
            selected = torch.cat((selected.new_zeros(padding, selected.shape[-1]), selected))
        return indices, selected

    def combine(owner: Any, hidden: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """Join routed and shared expert outputs at the configured dtype boundary.

        Args:
            owner: MoE module owning router state.
            hidden: Trunk hidden states.
            routed: Routed expert result.
        """
        shared = owner.shared_experts(hidden.to(torch.bfloat16))
        return routed.to(torch.bfloat16).float() + shared.float()

    compute = build_ep_compute(module, ep_mesh, router_fn=router, archetype_key="deepseek_v32_sft_hf",
                               expected_attrs=["gate", "experts", "shared_experts", "reference_config"],
                               combine=combine, use_grouped_gemm=True)

    def forward(owner: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Preserve dummy expert padding around Hyper token dispatch.

        Args:
            owner: MoE module owning router state.
            hidden_states: Input hidden states.
        """
        # Keep routing probabilities and weighted aggregation in FP32 inside the dispatcher.
        hidden = hidden_states.float()
        if padding:
            hidden = torch.cat((hidden.new_zeros(1, padding, hidden.shape[-1]), hidden), dim=1)
        return compute(owner, hidden)[:, padding:]

    return forward
