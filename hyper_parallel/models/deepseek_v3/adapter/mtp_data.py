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
"""MTP token alignment and per-objective counts for the normal text Trainer."""

# Native Torch is required; the legacy platform-agnostic checker predates its removal.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F

from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.distributed.mtp import shift_mtp_tokens


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


class MTPTextParallelBatch(TextParallelBatch):
    """Supply MTP-specific inputs and denominators through the existing batch API.

    YAML should share mtp_num_layers/normalization with the model using YAML
    anchors. Per-depth *_labels fields are consumed by count_loss_token, so DP,
    CP and gradient accumulation use each objective's own denominator.
    """

    def __init__(self, *, mtp_num_layers: int, mtp_loss_normalization: str = "sequence", **kwargs: Any) -> None:
        """Build the ordinary text runtime plus the MTP depth contract."""
        if isinstance(mtp_num_layers, bool) or not isinstance(mtp_num_layers, int) or mtp_num_layers < 1:
            raise ValueError("MTPTextParallelBatch requires a positive mtp_num_layers")
        if mtp_loss_normalization not in ("sequence", "valid_tokens"):
            raise ValueError("Unknown MTP normalization")
        if kwargs.get("attention_mode", "dense") != "dense":
            raise ValueError("MTPTextParallelBatch requires dense attention metadata to preserve document boundaries")
        self.mtp_num_layers = mtp_num_layers
        self.mtp_loss_normalization = mtp_loss_normalization
        mesh = kwargs["mesh_context"]
        self.cp_group = mesh.cp_mesh.get_group() if mesh.cp_size > 1 else None
        super().__init__(**kwargs)

    def __call__(self, data_iterator: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return aligned inputs plus independent per-depth loss-count fields."""
        model_inputs, loss_inputs = super().__call__(data_iterator)
        model_inputs, loss_inputs = dict(model_inputs), dict(loss_inputs)
        targets = loss_inputs.get("shift_labels", loss_inputs["labels"])
        targets = targets.masked_fill(~loss_inputs["loss_mask"].bool(), -100)
        if "shift_labels" not in loss_inputs:
            targets = shift_mtp_tokens(targets, 1, group=self.cp_group, pad_value=-100)
        model_inputs["shift_labels"] = targets
        loss_inputs["shift_labels"] = targets
        loss_inputs["loss_mask"] = targets.ne(-100)
        prepared = prepare_mtp_inputs(model_inputs["input_ids"], model_inputs["position_ids"], targets,
                                      self.mtp_num_layers, cp_group=self.cp_group,
                                      attention_mask=model_inputs.get("attention_mask"))
        model_inputs.update(prepared)
        for depth, labels in enumerate(prepared["mtp_targets"], 1):
            loss_inputs[f"mtp_depth{depth}_labels"] = (
                torch.zeros_like(labels) if self.mtp_loss_normalization == "sequence" else labels
            )
        return model_inputs, loss_inputs
