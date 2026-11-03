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

import torch
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
