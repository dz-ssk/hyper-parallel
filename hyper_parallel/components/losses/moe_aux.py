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
"""MoE auxiliary (load-balancing) loss objectives."""

from typing import Any, Optional

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
# pylint: disable-next=forbidden-backend-import
import torch
# pylint: disable-next=forbidden-backend-import
import torch.distributed as dist


class _ReplicatedGroupMean(torch.autograd.Function):
    """Replicate one group mean while differentiating each local contribution once."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group: Any) -> torch.Tensor:
        """Average equally weighted contributions across ``group``."""
        ctx.world_size = dist.get_world_size(group)
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result / ctx.world_size

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Scale the local derivative without summing identical output replicas."""
        return gradient / ctx.world_size, None


def _replicated_group_mean(value: torch.Tensor, group: Any = None) -> torch.Tensor:
    """Reduce a partitioned objective; absent an explicit group, keep it local.

    Contributions must be equally weighted, with the same upstream derivative
    on all ranks. This does not implement DDP averaging or independently
    consumed all-reduce outputs; uneven token partitions need explicit weights.
    """
    return value if group is None else _ReplicatedGroupMean.apply(value, group)


def calculate_seq_aux_loss(
    scores: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    coeff: float,
    sequence_partition_group: Optional[Any] = None,
) -> torch.Tensor:
    """DeepSeek-V3 complementary sequence-wise aux loss ``coeff * E * sum_i(f_i * P_i)``.

    For each sequence, ``f_i`` is the fraction of routed slots assigned to expert
    ``i`` and ``P_i`` is the mean over its tokens of each token's affinities
    normalized across all ``E`` experts; the loss is averaged over the sequences.
    Unlike the top-k-weight load-balance loss of
    :class:`~hyper_parallel.components.modules.moe.MoE`, ``P_i`` uses the affinities
    of every expert.

    When the sequences are partitioned across ``sequence_partition_group``, ``f_i``
    is averaged over the group and the returned loss is the group mean replicated
    on every rank, each rank differentiating only its own tokens. Unlike the
    per-rank partial value of the MoE load-balance loss, this value is complete on
    every rank, so it can be reported and added to a replicated objective.

    Args:
        scores: Router affinities of every expert, ``[..., tokens, num_experts]``,
            for example sigmoid scores before top-k selection; leading dimensions
            index independent sequences, and a 2-D input is one sequence.
        selected_experts: Selected expert indices, ``[..., tokens, top_k]``, with the
            same leading dimensions as ``scores``. For the DeepSeek-V3 sequence
            objective, select from the raw affinities without dispatch bias or
            group filtering; dispatch statistics remain a separate objective.
        coeff: Aux loss coefficient.
        sequence_partition_group: Optional process group whose ranks hold different
            token shards of the same sequences; ``None`` keeps the statistics local.

    Returns:
        The 0-d aux loss.
    """
    num_experts = scores.shape[-1]
    num_sequences = scores.shape[:-2].numel()
    slots = selected_experts.reshape(num_sequences, -1)
    offsets = torch.arange(num_sequences, device=slots.device).unsqueeze(-1) * num_experts
    index = (slots + offsets).flatten()
    # scatter_add_ counts on device; torch.bincount first reads the largest index back to the host.
    counts = index.new_zeros(num_sequences * num_experts, dtype=torch.float32).scatter_add_(
        0, index, index.new_ones(index.shape, dtype=torch.float32))
    load = counts.view(num_sequences, num_experts) / slots.shape[-1]
    if sequence_partition_group is not None:
        dist.all_reduce(load, group=sequence_partition_group)
        load = load / dist.get_world_size(sequence_partition_group)
    normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
    affinity = normalized.reshape(num_sequences, -1, num_experts).mean(-2)
    loss = (affinity * load).sum(-1).mean() * num_experts * coeff
    return _replicated_group_mean(loss, sequence_partition_group)


__all__ = ["calculate_seq_aux_loss"]
