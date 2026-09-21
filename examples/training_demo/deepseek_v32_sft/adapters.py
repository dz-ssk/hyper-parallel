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
"""Opt-in forward precision adapters retaining actual HF parameters and children."""

from typing import Any

import torch
from torch import nn
import torch_npu
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32DecoderLayer, DeepseekV32Experts, DeepseekV32MLP, DeepseekV32RMSNorm,
)

from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements, module_replacement,
)


def _retain_state(target: nn.Module, source: nn.Module) -> None:
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
    """Keep the HF scale and use FP32 reference normalization with an explicit cast."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        nn.Module.__init__(self)
        _retain_state(self, module)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Match the reference fused FP32 norm's output dtype boundary."""
        return torch_npu.npu_rms_norm(hidden_states.float(), self.weight, self.variance_epsilon)[0].to(hidden_states.dtype)


@module_replacement
class DeepseekV32SFTMLP(DeepseekV32MLP):
    """Retain HF gate/up/down projections and replace only the SwiGLU computation."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        nn.Module.__init__(self)
        _retain_state(self, module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Avoid materializing a rounded SiLU result before the gate/up product."""
        values = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(torch_npu.npu_swiglu(values, dim=-1))


@module_replacement
class DeepseekV32SFTExperts(DeepseekV32Experts):
    """Keep HF packed expert tensors and expose Hyper's grouped-compute hook."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        nn.Module.__init__(self)
        _retain_state(self, module)

    def forward_expert_major(self, inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Use Hyper grouped SwiGLU with BF16 compute and FP32 master parameters."""
        return npu_grouped_swiglu(inputs.to(torch.bfloat16), self.gate_up_proj.to(torch.bfloat16),
                                 self.down_proj.to(torch.bfloat16), counts)


@module_replacement
class DeepseekV32SFTDecoder(DeepseekV32DecoderLayer):
    """Preserve HF submodules while matching FP32 residual accumulation boundaries."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        nn.Module.__init__(self)
        _retain_state(self, module)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Any = None,
                position_ids: Any = None, past_key_values: Any = None, use_cache: bool = False,
                position_embeddings: Any = None, **kwargs: Any) -> torch.Tensor:
        """Run the original HF children with the reference conversion order."""
        if past_key_values is not None or use_cache:
            raise ValueError("DeepSeek V3.2 SFT does not support cached decoding")
        residual = hidden_states.float()
        branch, _ = self.self_attn(
            self.input_layernorm(residual).to(torch.bfloat16), attention_mask=attention_mask,
            position_ids=position_ids, position_embeddings=position_embeddings, **kwargs,
        )
        hidden_states = (residual + branch.float()).to(torch.bfloat16)
        residual = hidden_states.float()
        branch = self.mlp(self.post_attention_layernorm(residual).to(torch.bfloat16))
        return (residual + branch.float()).to(torch.bfloat16)


def apply_precision_adapters(model: nn.Module) -> nn.Module:
    """Replace only selected HF module instances; never alter global HF classes."""
    for source, replacement in ((DeepseekV32RMSNorm, DeepseekV32SFTRMSNorm), (DeepseekV32MLP, DeepseekV32SFTMLP),
                                (DeepseekV32Experts, DeepseekV32SFTExperts), (DeepseekV32DecoderLayer, DeepseekV32SFTDecoder)):
        names = tuple(name for name, module in model.named_modules() if type(module) is source)
        if not names:
            raise ValueError(f"Expected HF modules absent: {source.__name__}")
        plan = compile_module_replacements(model, [ModuleReplacementSpec(names, replacement, source, True)])
        model, _ = apply_module_replacements(model, plan, weights_mapping=[])
    return model
