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
"""JT-specific backward scaling for the model-computed training objective."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from hyper_parallel.components.losses.model_output import ModelComputedLoss

class JTDeepseekV3Loss(ModelComputedLoss):
    """Add only JT's replicated-objective backward scaling to the shared adapter."""

    def forward(self, *, model_output: Any, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Return the already combined objective without summing auxiliary values twice.

        Args:
            model_output: Model output.
            labels: Already-shifted target token IDs.
        """
        del labels
        loss = super().forward(model_output=model_output)
        return _ReplicatedLossGradient.apply(loss, self.loss_group)


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
