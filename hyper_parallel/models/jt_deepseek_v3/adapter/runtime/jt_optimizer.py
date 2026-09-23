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
"""JT training policy around Hyper's Muon/AdamW, using model parameters directly."""

# The JT model adapter runs in the Torch backend.
# pylint: disable=forbidden-backend-import
from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.components.optim.builders import Muon as MuonBuilder
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3MLAAttention


class JTOptimizer(ChainedOptimizer):
    """Apply JT synchronization and clipping around the native core optimizers."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Build core Muon and AdamW over the existing FP32 model parameters."""
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.use_deterministic_algorithms(True)
        document = model.jt_document
        self.config = document["model"]["model_config"]
        self.schedule = document["lr_schedule"]
        self.optimizer_config = document["optimizer"]
        self.group = model.loss_group
        self.world = dist.get_world_size(self.group)
        self.replicated_names = frozenset(model.jt_replicated_names)
        self.last_global_norm = None
        self.last_max_logits = {}
        runtime = MuonBuilder(
            model=model,
            muon_config={
                "lr": self.schedule["learning_rate"],
                "weight_decay": self.optimizer_config["weight_decay"],
                "momentum": self.optimizer_config.get("momentum", .95),
                "nesterov": self.optimizer_config.get("nesterov", True),
                "matched_adamw_rms": self.optimizer_config["matched_adamw_rms"],
                "ns_steps": 5,
                "ns_variant": "legacy",
                "ns_epsilon": 1e-7,
            },
            adamw_config={
                "lr": self.schedule["learning_rate"],
                "adamw_weight_decay": self.optimizer_config["weight_decay"],
                "betas": self.optimizer_config["adamw_betas"],
                "eps": self.optimizer_config["adamw_eps"],
            },
        ).get_optimizer()
        super().__init__(model, runtime.optimizers_dict, flatten=runtime.flatten)

    def _learning_rate(self) -> float:
        """Retain the zero-based reference warmup/cosine schedule."""
        group = self.optimizers_dict["muon"].param_groups[0]
        step_number = group.get("step") or 0
        device = group["params"][0].device
        step = torch.tensor(step_number, device=device, dtype=torch.float32)
        start = torch.tensor(self.schedule["learning_rate"], device=device, dtype=torch.float32)
        end = torch.tensor(self.schedule["lr_end"], device=device, dtype=torch.float32)
        warmup = self.schedule["warmup_steps"]
        if step_number < warmup:
            return (start * step / warmup).item()
        if step_number >= self.schedule["total_steps"]:
            return end.item()
        progress = (step - warmup) / (self.schedule["total_steps"] - warmup)
        percent = .5 * (1 + torch.cos(progress * math.pi))
        return (end + (start - end) * percent).item()

    @torch.no_grad()
    def step(self, closure: Any = None) -> float:
        """Synchronize and clip model gradients, run core updates, then apply JT post-updates."""
        if closure is not None:
            raise ValueError("This JT recipe does not support closures")
        named = list(self.model.named_parameters())
        for name, parameter in named:
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            if name in self.replicated_names:
                dist.all_reduce(parameter.grad, group=self.group)
        ordered = sorted(named, key=lambda item: item[1].ndim == 1)
        terms = [parameter.grad.float().square().sum() /
                 (self.world if name in self.replicated_names else 1)
                 for name, parameter in ordered]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        dist.all_reduce(total, group=self.group)
        norm = total.sqrt()
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite global gradient norm")
        self.last_global_norm = norm.detach()
        coefficient = (1 / (norm.clamp_min(1.0) + 1e-6)).clamp_max(1.0)
        for _, parameter in named:
            parameter.grad.mul_(coefficient)
        rate = self._learning_rate()
        for group in self.param_groups:
            group["lr"] = rate
        super().step()
        self._clip_qk()
        for module in self.model.modules():
            if hasattr(module, "expert_load") and self.config["moe_router_enable_expert_bias"]:
                direction = (1 / self.config["n_routed_experts"] - module.expert_load).sign()
                module.gate.e_score_correction_bias.add_(
                    direction, alpha=self.config["moe_router_bias_update_rate"])
                module.expert_load.zero_()
        return rate

    def _clip_qk(self) -> None:
        """Retain post-update QK clipping and capture its pre-reset statistics."""
        threshold = self.optimizer_config["qk_clip_threshold"]
        self.last_max_logits = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, JTDeepseekV3MLAAttention):
                continue
            maximum = module.max_logits_val
            self.last_max_logits[name] = maximum.detach().amax()
            scale = torch.where(maximum >= threshold, threshold / maximum.clamp_min(threshold),
                                torch.ones_like(maximum))
            query = module.q_b_proj.weight.view(
                module.num_heads, self.config["qk_nope_head_dim"] + self.config["qk_rope_head_dim"], -1)
            query[:, :self.config["qk_nope_head_dim"]].mul_(scale.sqrt()[:, None, None])
            query[:, self.config["qk_nope_head_dim"]:].mul_(scale[:, None, None])
            key_value = module.kv_b_proj.weight.view(
                module.num_heads, self.config["qk_nope_head_dim"] + self.config["v_head_dim"], -1)
            key_value[:, :self.config["qk_nope_head_dim"]].mul_(scale.sqrt()[:, None, None])
            maximum.zero_()

    def get_logging_metrics(self) -> dict[str, torch.Tensor]:
        """Consume pre-reset QK maxima, reducing head shards over TP."""
        if not self.last_max_logits:
            return {}
        names = sorted(self.last_max_logits)
        values = torch.stack([self.last_max_logits[name] for name in names])
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=self.group)
        self.last_max_logits = {}
        metrics = {f"optimizer/qkclip_maxlogits/{name}": value for name, value in zip(names, values.unbind())}
        metrics["optimizer/qkclip_maxlogits"] = values.amax()
        return metrics


class JTOptimizerBuilder:
    """Construct the JT policy wrapper consumed by the standard Trainer."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.optimizer = JTOptimizer(model)

    def get_optimizer(self) -> JTOptimizer:
        """Return the runtime over core Muon and AdamW."""
        return self.optimizer


class JTOptimizerSchedule:
    """Keep schedule ownership in the optimizer's existing step lifecycle."""

    def get_lr_scheduler(self) -> None:
        """No additional Trainer scheduler is needed."""
        return None
