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
"""JT model hooks around the public Muon optimizer."""

import math
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3MLAAttention


@torch.no_grad()
def clip_qk(model: torch.nn.Module, threshold: float) -> dict[str, torch.Tensor]:
    """Clip coupled query/key projections and return detached global maxima.

    Args:
        model: JT model with MLA statistics.
        threshold: Positive clipping threshold from the optimizer adapter configuration.
    """
    metrics = {}
    for name, module in model.named_modules():
        if not isinstance(module, JTDeepseekV3MLAAttention):
            continue
        maximum = module.max_logits_val
        observed = maximum.detach().amax().clone()
        dist.all_reduce(observed, op=dist.ReduceOp.MAX, group=model.loss_group)
        metrics[f"optimizer/qkclip_maxlogits/{name}"] = observed
        scale = threshold / maximum.clamp_min(threshold)
        query = module.q_b_proj.weight.view(
            module.num_heads, module.qk_nope_head_dim + module.qk_rope_head_dim, -1)
        query[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        query[:, module.qk_nope_head_dim:].mul_(scale[:, None, None])
        key_value = module.kv_b_proj.weight.view(module.num_heads, module.qk_nope_head_dim + module.v_head_dim, -1)
        key_value[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        maximum.zero_()
    if metrics:
        metrics["optimizer/qkclip_maxlogits"] = torch.stack(list(metrics.values())).amax()
    return metrics


def reshape_gate_up_projection(parameter_name: str, update: torch.Tensor) -> list[torch.Tensor]:
    """Expose fused Gate/Up projections as independent logical Muon matrices.

    The model stores Gate and Up together for the grouped expert kernel. Muon
    receives two transposed views so it normalizes and orthogonalizes each
    projection independently while writing updates into the same storage.

    Args:
        parameter_name: Fully qualified parameter name assigned by the optimizer.
        update: Local Muon update matrix for one parameter.

    Returns:
        The logical matrices to be processed by the public Muon implementation.
    """
    if parameter_name.endswith("experts.gate_up_proj"):
        return [projection.mT for projection in update.chunk(2, dim=1)]
    return [update]


@torch.no_grad()
def _after_update(model: torch.nn.Module, threshold: float, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Apply model-owned updates after all public optimizer leaves complete."""
    del optimizer, args, kwargs
    model.jt_optimizer_metrics = clip_qk(model, threshold)
    config = model.config
    if config.moe_router_enable_expert_bias:
        for module in model.modules():
            if getattr(module, "expert_load", None) is not None:
                direction = (1 / config.n_routed_experts - module.expert_load).sign()
                module.gate.e_score_correction_bias.add_(direction, alpha=config.moe_router_bias_update_rate)
                module.expert_load.zero_()


def _take_metrics(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return QK clipping metrics collected after the previous optimizer step."""
    result = model.jt_optimizer_metrics
    model.jt_optimizer_metrics = {}
    return result


def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, **kwargs: Any) -> Muon:
    """Build public Muon/AdamW and attach the JT-specific post-update hooks.

    Args:
        model: Model whose final FSDP parameter layouts are already prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        **kwargs: Public Muon Builder options from the training recipe.

    Returns:
        The unmodified public Muon Builder.
    """
    if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
        raise ValueError("qk_clip_threshold must be finite and positive")
    muon_config = {**kwargs["muon_config"], "reshape_fn": reshape_gate_up_projection}
    builder = Muon(model=model, muon_config=muon_config, **{
        name: value for name, value in kwargs.items() if name != "muon_config"
    })
    optimizer = builder.get_optimizer()
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    model.jt_optimizer_metrics = {}
    optimizer.get_logging_metrics = partial(_take_metrics, model)
    return builder
