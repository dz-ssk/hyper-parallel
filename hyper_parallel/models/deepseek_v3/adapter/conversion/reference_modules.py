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
"""DeepSeek V3.2 SFT precision replacements through the Hyper replacement API."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch_npu

from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Experts, DeepseekV32MLP, DeepseekV32RMSNorm,
)

from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.models.deepseek_v3.adapter.conversion.mla_attention import MLAAttention
from hyper_parallel.models.deepseek_v3.modeling_deepseek_v32_sft import (
    DeepseekV32SFTDecoder as BaseSFTDecoder,
    ExplicitFP32RotaryEmbedding,
)
from hyper_parallel.models.replacement import module_replacement


def _retain_state(target: nn.Module, source: nn.Module) -> None:
    """Preserve the source module registry and every checkpoint identity."""
    for name, value in vars(source).items():
        if not name.startswith("_"):
            setattr(target, name, value)
    for name, child in source.named_children():
        target.add_module(name, child)
    for name, parameter in source.named_parameters(recurse=False):
        target.register_parameter(name, parameter)
    for name, value in source.named_buffers(recurse=False):
        target.register_buffer(name, value)
    target.train(source.training)


@module_replacement
class DeepseekV32SFTRMSNorm(DeepseekV32RMSNorm):
    """Keep the HF scale and use the reference fused NPU normalization."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        del module_fqn, context
        nn.Module.__init__(self)
        _retain_state(self, module)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Match the reference fused FP32 norm's output dtype boundary."""
        normalized = torch_npu.npu_rms_norm(hidden_states.float(), self.weight, self.variance_epsilon)[0]
        return normalized.to(hidden_states.dtype)


@module_replacement
class DeepseekV32SFTMLAAttention(MLAAttention):
    """Reuse Hyper MLA with PR FP32 latent norms and interleaved RoPE."""
    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        """Reuse Hyper MLA while selecting the explicit FP32 rotary boundary.

        Args:
            module: HF module being adapted.
            module_fqn: Module fqn.
            context: Context.
        """
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.q_a_layernorm = DeepseekV32SFTRMSNorm(
            module=module.q_a_layernorm, module_fqn=f"{module_fqn}.q_a_layernorm", context=context)
        self.kv_a_layernorm = DeepseekV32SFTRMSNorm(
            module=module.kv_a_layernorm, module_fqn=f"{module_fqn}.kv_a_layernorm", context=context)
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


@module_replacement
class DeepseekV32SFTMLP(DeepseekV32MLP):
    """Retain HF projections and replace only the SwiGLU computation."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        del module_fqn, context
        nn.Module.__init__(self)
        _retain_state(self, module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the reference single fused SwiGLU operation."""
        values = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(torch_npu.npu_swiglu(values, dim=-1))


@module_replacement
class DeepseekV32SFTExperts(DeepseekV32Experts):
    """Keep packed experts and expose the reference grouped-compute hook."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        del module_fqn, context
        nn.Module.__init__(self)
        _retain_state(self, module)

    def forward_expert_major(self, inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Use Hyper grouped SwiGLU with BF16 compute and FP32 master parameters."""
        return npu_grouped_swiglu(inputs.to(torch.bfloat16), self.gate_up_proj.to(torch.bfloat16),
                                 self.down_proj.to(torch.bfloat16), counts)


def _replace_mlp(module: nn.Module, module_fqn: str, context: Any) -> nn.Module:
    """Adapt Dense directly, or compose the original MoE with adapted children."""
    if not (hasattr(module, "experts") and hasattr(module, "shared_experts")):
        return DeepseekV32SFTMLP(module=module, module_fqn=module_fqn, context=context)

    target = type(module).__new__(type(module))
    nn.Module.__init__(target)
    _retain_state(target, module)
    target.experts = DeepseekV32SFTExperts(
        module=module.experts, module_fqn=f"{module_fqn}.experts", context=context)
    target.shared_experts = DeepseekV32SFTMLP(
        module=module.shared_experts, module_fqn=f"{module_fqn}.shared_experts", context=context)
    return target


@module_replacement
class DeepseekV32SFTDecoder(BaseSFTDecoder):
    """Replace precision-sensitive children while inheriting the base forward."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        nn.Module.__init__(self)
        _retain_state(self, module)
        self.input_layernorm = DeepseekV32SFTRMSNorm(
            module=module.input_layernorm, module_fqn=f"{module_fqn}.input_layernorm", context=context)
        self.post_attention_layernorm = DeepseekV32SFTRMSNorm(
            module=module.post_attention_layernorm,
            module_fqn=f"{module_fqn}.post_attention_layernorm", context=context)
        # Keep the attention conversion inside this parent replacement.  The generic
        # replacement executor deliberately rejects overlapping parent/child targets.
        self.self_attn = DeepseekV32SFTMLAAttention(
            module=module.self_attn, module_fqn=f"{module_fqn}.self_attn", context=context)
        self.mlp = _replace_mlp(module.mlp, f"{module_fqn}.mlp", context)

    def make_transforms(self):
        """Scope the MLA's fused Q/KV checkpoint conversion to this decoder."""
        transforms = self.self_attn.make_transforms()
        return [
            type(transform)(
                source_patterns=[f"self_attn.{pattern}" for pattern in transform.source_patterns],
                target_patterns=[f"self_attn.{pattern}" for pattern in transform.target_patterns],
                operations=transform.operations,
            )
            for transform in transforms
        ]

