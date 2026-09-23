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
"""Masked, pre-shifted vocabulary-parallel objective and Trainer input contract."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn

class JTDeepseekV3Loss(nn.Module):
    """Pass already-shifted labels and masks through Trainer's loss protocol."""

    def __init__(self) -> None:
        """Keep local execution independent of any initialized process group."""
        super().__init__()
        self.loss_group = None

    def bind_model(self, model: nn.Module, distributed_setup: Any = None) -> None:
        """Bind vocabulary parallelism after the Trainer has built its mesh.

        A direct model construction deliberately leaves this unset so the local
        full-vocabulary loss never creates or queries a process group.

        Args:
            model: Complete model with an optional vocabulary process group.
            distributed_setup: Resolved Trainer mesh and loss-parallel settings.
        """
        mesh = getattr(distributed_setup, "mesh_context", None)
        if (
            mesh is None
            or getattr(mesh, "tp_size", 1) <= 1
            or not getattr(mesh, "loss_parallel", False)
        ):
            self.loss_group = None
            model.loss_group = None
            return
        self.loss_group = mesh.device_mesh["tp"].get_group()
        model.loss_group = self.loss_group

    @staticmethod
    def prepare_model_inputs(model_inputs: dict, loss_inputs: dict) -> dict:
        """Keep model semantics independent of the task and data-source selection.

        Args:
            model_inputs: Model fields emitted by the batch adapter.
            loss_inputs: Pre-shifted labels and masks.
        """
        return {"input_ids": model_inputs["input_ids"],
                "labels": loss_inputs["shift_labels"], "loss_mask": loss_inputs["loss_mask"]}

    def forward(self, *, model_output: Any, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Return the already combined objective without summing auxiliary values twice.

        Args:
            model_output: Model output.
            labels: Already-shifted target token IDs.
        """
        del labels
        return _ReplicatedLossGradient.apply(model_output.loss, self.loss_group)


class _ReplicatedLossGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group: Any) -> torch.Tensor:
        """Execute the declared forward computation.

        Args:
            ctx: Ctx.
            value: Value.
        """
        ctx.world = dist.get_world_size(group) if group is not None else 1
        return value

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Propagate gradients with the specified precision and replication rules.

        Args:
            ctx: Ctx.
            gradient: Gradient.
        """
        return gradient / ctx.world, None
