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
"""Expose logical reference matrices without changing HF parameter ownership."""

# The HF model adapter runs in the Torch backend.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Iterator

import torch
from torch import nn

from ..conversion.checkpoint_mapping import MappingRule, mapping_rules


class LogicalParameters:
    """Map local HF master parameters and gradients to optimizer matrix layouts."""

    def __init__(self, model: nn.Module, config: dict) -> None:
        """Create local logical views in the reference optimizer matrix layout."""
        self.hf_model = model
        self.config = config
        self.rules = [rule for rule in mapping_rules(config) if not rule.source.endswith('.expert_bias')]
        self.parameters = {}
        with torch.no_grad():
            for rule in self.rules:
                value = self._decode(rule, gradient=False)
                parameter = nn.Parameter(value.clone())
                parameter.jt_shard_dim = -1 if rule.shard_axis is None else rule.shard_axis
                parameter.jt_expert = '.experts.' in rule.source
                self.parameters[rule.source] = parameter
        self.order = self._reference_order()
        if set(self.order) != set(self.parameters):
            raise ValueError('Logical parameter order does not cover every mapped parameter')

    def _reference_order(self) -> list[str]:
        names = ['embedding.word_embeddings.weight']
        attention = ['linear_proj', 'linear_q_down_proj', 'linear_q_up_proj', 'linear_kv_down_proj',
                     'linear_kv_up_proj', 'q_layernorm', 'kv_layernorm']

        def layer(prefix: str, moe: bool) -> None:
            """Append one decoder layer to the reference parameter order.

            Args:
                prefix: Prefix.
                moe: Moe.
            """
            names.append(f'{prefix}.input_layernorm.weight')
            names.extend(f'{prefix}.self_attention.{key}.weight' for key in attention)
            names.append(f'{prefix}.pre_mlp_layernorm.weight')
            if moe:
                names.extend([f'{prefix}.mlp.router.weight', f'{prefix}.mlp.experts.weight1',
                              f'{prefix}.mlp.experts.weight2'])
            dense = f'{prefix}.mlp' + ('.shared_experts' if moe else '')
            names.extend([f'{dense}.linear_fc1.weight', f'{dense}.linear_fc2.weight'])

        for index in range(self.config['num_hidden_layers']):
            layer(f'decoder.layers.{index}', index >= self.config['first_k_dense_replace'])
        names.append('decoder.final_layernorm.weight')
        for index in range(self.config['num_nextn_predict_layers']):
            prefix = f'mtp.layers.{index}'
            names.extend(f'{prefix}.{key}.weight' for key in ['enorm', 'hnorm', 'eh_proj'])
            layer(f'{prefix}.transformer_layer', True)
            names.append(f'{prefix}.final_layernorm.weight')
        return names + ['output_layer.weight']

    def _tensor(self, name: str, gradient: bool) -> torch.Tensor:
        if name.endswith(('.q_a_proj.weight', '.kv_a_proj_with_mqa.weight')):
            prefix = name.rsplit('.', 2)[0]
            parameter = self.hf_model.get_parameter(prefix + '.linear_qkv.weight')
            value = parameter.grad if gradient else parameter
            boundary = self.config['q_lora_rank']
            return value[:boundary] if name.endswith('.q_a_proj.weight') else value[boundary:]
        parameter = self.hf_model.get_parameter(name)
        return parameter.grad if gradient else parameter

    def _decode(self, rule: MappingRule, gradient: bool) -> torch.Tensor:
        values = [self._tensor(name, gradient) for name in rule.targets]
        if any(value is None for value in values):
            raise RuntimeError(f'Missing logical gradient for {rule.source}')
        if rule.operation == 'interleaved':
            return torch.stack(values, dim=1).reshape(-1, values[0].shape[-1])
        if rule.operation in ('expert_up', 'expert_down'):
            value = values[0].transpose(1, 2)
            return value.reshape(-1, value.shape[-1])
        return values[0]

    @torch.no_grad()
    def read_gradients(self) -> None:
        """Copy independently accumulated HF gradients into logical matrices."""
        for rule in self.rules:
            self.parameters[rule.source].grad = self._decode(rule, gradient=True).clone()

    @torch.no_grad()
    def read_weights(self) -> None:
        """Refresh logical matrices after post-update HF QK clipping."""
        for rule in self.rules:
            self.parameters[rule.source].copy_(self._decode(rule, gradient=False))

    @torch.no_grad()
    def write_weights(self) -> None:
        """Scatter updated logical matrices back into the original HF parameters."""
        for rule in self.rules:
            source = self.parameters[rule.source]
            targets = [self._tensor(name, False) for name in rule.targets]
            if rule.operation == 'interleaved':
                targets[0].copy_(source[0::2])
                targets[1].copy_(source[1::2])
            elif rule.operation in ('expert_up', 'expert_down'):
                target = targets[0]
                target.copy_(source.reshape(target.shape[0], target.shape[2], target.shape[1]).transpose(1, 2))
            else:
                targets[0].copy_(source)

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield logical matrices in reference norm-reduction order."""
        return iter((name, self.parameters[name]) for name in self.order)
