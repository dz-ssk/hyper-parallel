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
"""Explicit stage partitioning for DeepSeek-V3 MTP pipeline schedules."""

# Native Torch is required; the legacy platform-agnostic checker predates its removal.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any
import math

import torch
from torch import nn

from hyper_parallel.core.pipeline_parallel.stage import PipelineStage, SharedParameterInfo
from hyper_parallel.models.deepseek_v3.adapter.mtp import DeepseekV3MTPForCausalLM


class MTPPipelineModule(nn.Module):
    """Adapt token/activation stage inputs and scalar loss to PipelineSchedule.

    Every stage receives the same token metadata as keyword arguments. The
    first positional tensor is input IDs on the first stage, and received
    activations on subsequent stages. Loss scaling is explicit because the
    schedule accumulates microbatch gradients without averaging them.
    """

    def __init__(self, model: DeepseekV3MTPForCausalLM, loss_scale: float = 1.0) -> None:
        """Wrap a partitioned model; use 1 / microbatch_count for equal batches."""
        super().__init__()
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive")
        self.model = model
        self.loss_scale = loss_scale
        self.last_loss_metrics: dict[str, torch.Tensor] = {}

    def forward(self, stage_input: torch.Tensor, *, input_ids: torch.Tensor | None = None,
                **metadata: Any) -> torch.Tensor:
        """Return activations for a successor or the final scaled scalar loss."""
        if self.model.mtp_pipeline_first:
            output = self.model(input_ids=stage_input, **metadata)
        else:
            output = self.model(input_ids=input_ids, hidden_states=stage_input, **metadata)
        if not self.model.mtp_pipeline_last:
            return output
        self.last_loss_metrics = {name: value.detach() for name, value in output.loss.items()}
        self.last_loss_metrics["mtp_loss"] = sum(
            value for name, value in self.last_loss_metrics.items() if name.startswith("mtp_depth"))
        return sum(output.loss.values()) * self.loss_scale


def partition_mtp_model(model: DeepseekV3MTPForCausalLM, stage_index: int,
                        stage_count: int) -> list[SharedParameterInfo]:
    """Prune an unwrapped model to one PP stage before sharding and optimization.

    The trunk is evenly partitioned (earlier stages receive any remainder), and
    all MTP depths plus the shared output head reside on the final stage. First
    and final stages own embedding replicas synchronized by PipelineStage.
    Decoder dictionary keys retain their global layer indices for checkpointing.
    A separate model instance is required on each PP rank. After materialization,
    the schedule must synchronize shared weights before forward and shared grads
    after all microbatch backwards, before the optimizer step.
    """
    if not isinstance(model, DeepseekV3MTPForCausalLM) or model.mtp is None:
        raise ValueError("partition_mtp_model requires an enabled MTP model")
    total = len(model.model.layers)
    if not 1 <= stage_count <= total or not 0 <= stage_index < stage_count:
        raise ValueError("PP stages must partition the trunk into nonempty consecutive layer ranges")
    if not isinstance(model.model.layers, nn.ModuleList):
        raise ValueError("The model is already partitioned")
    quotient, remainder = divmod(total, stage_count)
    start = stage_index * quotient + min(stage_index, remainder)
    stop = start + quotient + int(stage_index < remainder)
    model.model.layers = nn.ModuleDict({str(index): model.model.layers[index] for index in range(start, stop)})
    model.mtp_pipeline_first = stage_index == 0
    model.mtp_pipeline_last = stage_index == stage_count - 1
    shared = []
    if stage_count > 1 and (model.mtp_pipeline_first or model.mtp_pipeline_last):
        embedding = model.model.embed_tokens
        shared.append(SharedParameterInfo(embedding.weight, [0, stage_count - 1],
                                           owner_module=embedding, param_name="weight"))
    if not model.mtp_pipeline_last:
        model.mtp = None
        model.lm_head = None
        model.model.norm = None
    if not model.mtp_pipeline_first and not model.mtp_pipeline_last:
        model.model.embed_tokens = None
    return shared


def build_mtp_pipeline_stage(model: DeepseekV3MTPForCausalLM, *, stage_index: int,
                             stage_count: int, device: Any, mesh: Any = None,
                             loss_scale: float = 1.0) -> PipelineStage:
    """Build a PP-runtime stage for an explicit PipelineSchedule caller.

    This is not a monolithic TextTrainer factory. Callers supply input_ids,
    labels/shift_labels and attention metadata to every stage, and route the
    previous stage's tensor as the positional input. The final stage returns
    a scalar sum of the objectives, multiplied by loss_scale. Use
    partition_mtp_model directly when TP/CP/EP/FSDP wrapping must occur between
    partitioning and PipelineStage construction.
    """
    shared = partition_mtp_model(model, stage_index, stage_count)
    return PipelineStage(MTPPipelineModule(model, loss_scale), stage_index, stage_count, device=device, mesh=mesh,
                         shared_parameters=shared)
