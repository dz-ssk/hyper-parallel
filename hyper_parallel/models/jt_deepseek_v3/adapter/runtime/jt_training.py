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
"""Local ten-step extension: optimizer lifecycle adapter for TextTrainer."""

# The HF model adapter runs in the Torch backend.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from .jt_logical_parameters import LogicalParameters
from .jt_optimizer import ReferenceMuon


class JTAlignmentOptimizer(torch.optim.Optimizer):
    """Keep reference update boundaries inside the configured optimizer lifecycle."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize the configured components and retained parameter state.

        Args:
            model: Model.
        """
        # Preserve the archived NPU execution policy without the Trainer's
        # additional environment overrides for generic full-determinism mode.
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.use_deterministic_algorithms(True)
        document = model.jt_document
        self.model = model
        self.config = document["model"]["model_config"]
        self.logical = LogicalParameters(model, self.config)
        self.parallel = SimpleNamespace(world=dist.get_world_size(), rank=dist.get_rank(), group=model.loss_group)
        config = SimpleNamespace(**self.config, optimizer=document["optimizer"], schedule=document["lr_schedule"])
        self.reference_optimizer = ReferenceMuon(self.logical, config, self.parallel)
        self.last_global_norm = None
        super().__init__(list(model.parameters()), {"lr": config.schedule["learning_rate"]})

    @torch.no_grad()
    def step(self, closure: Any = None) -> float:
        """Synchronize and update parameters through the configured optimizer.

        Args:
            closure: Closure.
        """
        if closure is not None:
            raise ValueError("This alignment optimizer does not support closures")
        named = dict(self.model.named_parameters())
        for name, value in named.items():
            if value.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
        for name in self.model.jt_replicated_names:
            dist.all_reduce(named[name].grad)
        self.logical.read_gradients()
        ordered = list(self.logical.named_parameters())
        ordered.sort(key=lambda item: item[1].ndim == 1)
        terms = [value.grad.float().square().sum() / (self.parallel.world if value.jt_shard_dim < 0 else 1)
                 for _, value in ordered]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        dist.all_reduce(total, group=self.parallel.group)
        norm = total.sqrt()
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite global gradient norm")
        self.last_global_norm = norm.detach()
        coefficient = (1 / (norm.clamp_min(1.0) + 1e-6)).clamp_max(1.0)
        for _, value in self.logical.named_parameters():
            value.grad.mul_(coefficient)
        rate = self.reference_optimizer.step()
        for module in self.model.modules():
            if hasattr(module, "expert_load") and self.config["moe_router_enable_expert_bias"]:
                direction = (1 / self.config["n_routed_experts"] - module.expert_load).sign()
                module.gate.e_score_correction_bias.add_(direction, alpha=self.config["moe_router_bias_update_rate"])
                module.expert_load.zero_()
        for group in self.param_groups:
            group["lr"] = rate
        return rate

    def state_dict(self) -> dict:
        """Export optimizer state for the configured checkpoint lifecycle."""
        return {"physical": super().state_dict(), "logical": self.reference_optimizer.state_dict(),
                "step_number": self.reference_optimizer.step_number}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state and refresh the logical parameter views.

        Args:
            state_dict: State dict.
        """
        super().load_state_dict(state_dict["physical"])
        self.reference_optimizer.load_state_dict(state_dict["logical"])
        self.reference_optimizer.step_number = state_dict["step_number"]
        self.logical.read_weights()


class JTAlignmentOptimizerBuilder:
    """Trainer target providing the locally retained ten-step optimizer."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize the configured components and retained parameter state.

        Args:
            model: Model.
        """
        self.optimizer = JTAlignmentOptimizer(model)

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer consumed by the Trainer."""
        return self.optimizer


class JTAlignmentSchedule:
    """The optimizer owns the captured schedule, so Trainer has no extra scheduler."""

    def get_lr_scheduler(self) -> None:
        """Return no extra scheduler because the optimizer owns its schedule."""
        return None
