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
"""Sequence alignment for MTP on contiguous context-parallel token shards."""

# Native Torch is required; the legacy platform-agnostic checker predates its removal.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
import torch.distributed as dist


def shift_mtp_tokens(tokens: torch.Tensor, offset: int, *, group: dist.ProcessGroup | None = None,
                     pad_value: int = 0) -> torch.Tensor:
    """Shift integer metadata left, exchanging CP prefixes without wrapping rows.

    Args:
        tokens: Equal-length contiguous [B, S_local] CP shards in group-rank order.
            This operation is for IDs, positions, labels and masks, not activations.
        offset: Number of future positions to read, possibly larger than a shard.
        group: Explicit CP process group. None means a local, unsharded sequence;
            it never implicitly selects the distributed world group.
        pad_value: Value used beyond the last global sequence position.

    Returns:
        A tensor with the same shape/dtype/device; row boundaries never wrap.
    """
    if tokens.ndim != 2 or tokens.shape[1] == 0 or tokens.is_floating_point() or tokens.is_complex():
        raise ValueError("shift_mtp_tokens requires nonempty integer/bool [B, S] metadata")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if offset == 0:
        return tokens.clone()
    size = 1 if group is None else dist.get_world_size(group)
    rank = 0 if group is None else dist.get_rank(group)
    length = tokens.shape[1]
    result = torch.full_like(tokens, pad_value)
    if offset >= length * size:
        return result
    if size == 1:
        result[:, :length - offset] = tokens[:, offset:]
        return result
    # At most offset tokens from each neighbor can be needed. No activations or
    # parameters are exchanged, and the collectives stay outside autograd/recompute.
    prefix = tokens[:, :min(offset, length)].contiguous()
    prefixes = [torch.empty_like(prefix) for _ in range(size)]
    dist.all_gather(prefixes, prefix, group=group)
    start = rank * length + offset
    end = min(start + length, size * length)
    while start < end:
        source_rank, source_offset = divmod(start, length)
        count = min(length - source_offset, end - start)
        source = tokens if source_rank == rank else prefixes[source_rank]
        destination = start - (rank * length + offset)
        result[:, destination:destination + count] = source[:, source_offset:source_offset + count]
        start += count
    return result


def prepare_mtp_inputs(input_ids: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor,
                       depth: int, *, cp_group: dist.ProcessGroup | None = None,
                       attention_mask: torch.Tensor | None = None) -> dict[str, Any]:
    """Align fixed-length MTP inputs before embedding and TP sequence scattering.

    Sequences are contiguous CP shards of independent rows. Targets must already
    be causal-next-token labels, including ignore_index at document boundaries.
    """
    shifted_targets = [shift_mtp_tokens(targets, k, group=cp_group, pad_value=-100)
                       for k in range(1, depth + 1)]
    if attention_mask is not None and attention_mask.ndim == 2:
        rank = 0 if cp_group is None else dist.get_rank(cp_group)
        start, length = rank * input_ids.shape[1], input_ids.shape[1]
        valid = attention_mask[:, start:start + length].bool()
        shifted_targets = [target.masked_fill(~valid, -100) for target in shifted_targets]
    if attention_mask is not None and attention_mask.ndim == 4:
        allowed = attention_mask if attention_mask.dtype == torch.bool else attention_mask.eq(0)
        rank = 0 if cp_group is None else dist.get_rank(cp_group)
        start, length = rank * input_ids.shape[1], input_ids.shape[1]
        for k in range(1, depth + 1):
            valid = F.pad(allowed[:, 0, k:, :].diagonal(dim1=-2, dim2=-1), (0, k))
            shifted_targets[k - 1] = shifted_targets[k - 1].masked_fill(~valid[:, start:start + length], -100)
    return {
        "mtp_input_ids": tuple(shift_mtp_tokens(input_ids, k, group=cp_group) for k in range(1, depth + 1)),
        "mtp_position_ids": tuple(shift_mtp_tokens(positions, k, group=cp_group) for k in range(1, depth + 1)),
        "mtp_targets": tuple(shifted_targets),
    }
