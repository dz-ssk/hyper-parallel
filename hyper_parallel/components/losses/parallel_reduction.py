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
"""Reductions of partitioned losses to a single model-parallel objective."""

from typing import Any

# These components implement the Torch Trainer objective contract.
# pylint: disable=forbidden-backend-import
import torch
import torch.distributed as dist


class _ModelParallelMean(torch.autograd.Function):
    """Replicate a global mean while differentiating each local contribution once."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group: Any) -> torch.Tensor:
        """Reduce equally weighted local objectives over an explicit group."""
        ctx.world_size = dist.get_world_size(group)
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result / ctx.world_size

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Differentiate one logical replicated result, not independent consumers."""
        return gradient / ctx.world_size, None


def model_parallel_mean(value: torch.Tensor, group: Any = None) -> torch.Tensor:
    """Average partitioned losses with replicated-output gradient semantics.

    Each rank owns a disjoint, equally weighted contribution to one objective.
    All ranks receive the same forward mean and must supply the same upstream
    derivative. Backward returns that derivative divided by group size to each
    local contribution, with no second collective over identical loss replicas.
    This is a model-parallel boundary, not a replacement for DDP averaging or
    an all-reduce with independently consumed outputs. Unequal token partitions
    require explicit token weighting before this operation.

    Args:
        value: Local contribution to the global objective.
        group: Explicit model-parallel group; None keeps the operation local.

    Returns:
        Replicated global mean, or the original tensor for local execution.
    """
    if group is None:
        return value
    return _ModelParallelMean.apply(value, group)


__all__ = ["model_parallel_mean"]
