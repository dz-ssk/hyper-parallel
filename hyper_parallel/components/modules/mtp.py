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
"""Composable multi-token prediction blocks, independent of any model family."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MultiTokenPredictionLayer(nn.Module):
    """Fuse next-token embeddings and trunk states using injected components.

    The caller owns token shifting, embedding/head sharing, loss weighting and
    parallel placement. Norms, decoder and projection are supplied by the model
    adapter; this module imposes no DeepSeek, dtype or distributed defaults.
    """

    def __init__(self, *, embedding_norm: nn.Module, hidden_norm: nn.Module,
                 projection: nn.Module, decoder: nn.Module, output_norm: nn.Module,
                 fusion_dtype: torch.dtype | None = None) -> None:
        """Register caller-provided components without changing their parameters.

        Args:
            embedding_norm: Normalization for future-token embeddings.
            hidden_norm: Normalization for trunk states.
            projection: Embedding/hidden fusion projection.
            decoder: Model-provided decoder.
            output_norm: Prediction output normalization.
            fusion_dtype: Optional dtype at the concatenation boundary.
        """
        super().__init__()
        self.enorm = embedding_norm
        self.hnorm = hidden_norm
        self.eh_proj = projection
        self.transformer_layer = decoder
        self.final_layernorm = output_norm
        self.fusion_dtype = fusion_dtype

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor,
                **decoder_kwargs: Any) -> torch.Tensor:
        """Return the next prediction state without creating logits or losses.

        Args:
            hidden: Trunk hidden states.
            embedding: Next-token embedding states.
        """
        hidden = self.hnorm(hidden)
        embedding = self.enorm(embedding)
        if self.fusion_dtype is not None:
            hidden = hidden.to(self.fusion_dtype)
            embedding = embedding.to(self.fusion_dtype)
        combined = torch.cat((hidden, embedding), dim=-1)
        return self.final_layernorm(self.transformer_layer(self.eh_proj(combined), **decoder_kwargs))


class MultiTokenPrediction(nn.Module):
    """Register independent prediction depths; the model controls their schedule."""

    def __init__(self, layers: list[MultiTokenPredictionLayer]) -> None:
        """Register caller-provided components without changing their parameters.

        Args:
            layers: Independent prediction depth modules.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
