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
"""Model-owned post-update hooks around the unmodified public Muon builder."""

import math
from functools import partial
from typing import Any

import torch

from hyper_parallel.components.optim.builders import Muon
from .jt_optimizer import clip_qk


@torch.no_grad()
def _after_update(model: torch.nn.Module, threshold: float, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Apply coupled model updates once after all public optimizer leaves."""
    del optimizer, args, kwargs
    model.jt_optimizer_metrics = clip_qk(model, threshold)
    cfg = model.config
    if cfg.moe_router_enable_expert_bias:
        for module in model.modules():
            if getattr(module, "expert_load", None) is not None:
                direction = (1 / cfg.n_routed_experts - module.expert_load).sign()
                module.gate.e_score_correction_bias.add_(direction, alpha=cfg.moe_router_bias_update_rate)
                module.expert_load.zero_()


def _take_metrics(model: torch.nn.Module) -> dict:
    result = model.jt_optimizer_metrics
    model.jt_optimizer_metrics = {}
    return result


def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, **kwargs: Any) -> Muon:
    """Build public Muon/AdamW and attach QK clipping and router-bias updates.

    Args:
        model: Model managed by Hyper's source-layout FSDP.
        qk_clip_threshold: Finite positive bound for per-head attention logits.
        **kwargs: Options passed unchanged to the public Muon builder.
    """
    if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
        raise ValueError("qk_clip_threshold must be finite and positive")
    builder = Muon(model=model, **kwargs)
    optimizer = builder.get_optimizer()
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    model.jt_optimizer_metrics = {}
    optimizer.get_logging_metrics = partial(_take_metrics, model)
    return builder
