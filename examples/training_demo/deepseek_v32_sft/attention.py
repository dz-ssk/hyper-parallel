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
"""Local reference RoPE adaptation around Hyper's MLA projection and attention path."""

from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.models.replacement import module_replacement


class ExplicitFP32RotaryEmbedding(nn.Module):
    """Preserve separate FP32 products and addition before the BF16 RoPE cast."""

    def forward(self, values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Reorder even/odd channels and rotate using the reference operation sequence."""
        ordered = torch.cat((values[..., ::2], values[..., 1::2]), dim=-1).float()
        first, second = ordered.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (ordered * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).to(values.dtype)


@module_replacement
class DeepseekV32SFTMLAAttention(MLAAttention):
    """Reuse Hyper MLA and specialize only its rotary computation boundary."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.explicit_rotary = ExplicitFP32RotaryEmbedding()

    def _project_attention_inputs(self, hidden_states: torch.Tensor, position_embeddings: Any,
                                  past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Reference MLA requires explicit positions and no KV cache")
        query, key, value = super()._project_attention_inputs(hidden_states, None, None)
        cos, sin = position_embeddings
        query = torch.cat((query[..., :self.qk_nope_head_dim],
                           self.explicit_rotary(query[..., self.qk_nope_head_dim:], cos, sin)), dim=-1)
        key = torch.cat((key[..., :self.qk_nope_head_dim],
                         self.explicit_rotary(key[..., self.qk_nope_head_dim:], cos, sin)), dim=-1)
        return query, key, value
