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

from hyper_parallel.data.batching.get_batch import TextParallelBatch
from hyper_parallel.distributed.mtp import shift_mtp_tokens, prepare_mtp_inputs


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
