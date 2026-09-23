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
"""Reference DeepSeek V3.2 SFT model with an optional MLA replacement path."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import DeepseekV32ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32DecoderLayer, DeepseekV32Experts,
)

from hyper_parallel.components.modules.mtp import MultiTokenPrediction, MultiTokenPredictionLayer


def shift_left(value: torch.Tensor) -> torch.Tensor:
    """Shift global tokens, labels or masks without wrapping the final token."""
    return torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), dim=1)


class _MaskedVocabLoss(torch.autograd.Function):
    """Masked vocabulary-parallel NLL with an eager collective backward."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        group: Any,
    ) -> torch.Tensor:
        """Compute one stable masked mean while preserving local-logit gradients."""
        values = logits.float()
        maximum = values.amax(-1)
        if group is not None:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        stable = values - maximum.unsqueeze(-1)
        exponential = stable.exp()
        denominator = exponential.sum(-1)
        if group is None:
            if torch.any(labels < 0) or torch.any(labels >= logits.shape[-1]):
                raise ValueError("Single-device vocabulary loss requires local vocabulary labels")
            indices = labels
            owned = torch.ones_like(labels, dtype=torch.bool)
        else:
            dist.all_reduce(denominator, group=group)
            local_labels = labels - dist.get_rank(group) * logits.shape[-1]
            owned = (local_labels >= 0) & (local_labels < logits.shape[-1])
            indices = local_labels.clamp(0, logits.shape[-1] - 1)
        selected = stable.gather(-1, indices.unsqueeze(-1)).squeeze(-1) * owned
        if group is not None:
            dist.all_reduce(selected, group=group)
        normalizer = mask.sum() + 1e-8
        ctx.save_for_backward(exponential / denominator.unsqueeze(-1), indices, owned, mask, normalizer)
        ctx.input_dtype = logits.dtype
        return ((denominator.log() - selected) * mask).sum() / normalizer

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        """Differentiate local logits without differentiating collectives."""
        probabilities, indices, owned, mask, normalizer = ctx.saved_tensors
        gradient = probabilities.clone()
        gradient.scatter_add_(-1, indices.unsqueeze(-1), -owned.to(gradient.dtype).unsqueeze(-1))
        gradient = gradient * (mask * grad_output / normalizer).unsqueeze(-1)
        return gradient.to(ctx.input_dtype), None, None, None


def masked_vocab_parallel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    group: Any,
) -> torch.Tensor:
    """Compute the model's masked NLL on either a local or vocabulary-sharded head."""
    return _MaskedVocabLoss.apply(logits, labels, mask, group)


class ExplicitFP32RotaryEmbedding(nn.Module):
    """Apply the reference interleaved RoPE with FP32 products and sum."""

    def forward(self, values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate interleaved channels while preserving the reference cast boundary."""
        ordered = torch.cat((values[..., ::2], values[..., 1::2]), dim=-1).float()
        first, second = ordered.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (ordered * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).to(values.dtype)


class DeepseekV32SFTRMSNorm(nn.Module):
    """RMSNorm with FP32 reduction and the reference BF16 output boundary."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @classmethod
    def from_hf(cls, module: nn.Module) -> "DeepseekV32SFTRMSNorm":
        """Adopt a HF RMSNorm parameter without changing its checkpoint key."""
        target = cls.__new__(cls)
        nn.Module.__init__(target)
        target.weight = module.weight
        target.variance_epsilon = module.variance_epsilon
        target.train(module.training)
        return target

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32, then return at the caller's activation dtype."""
        values = hidden_states.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (normalized * self.weight.float()).to(hidden_states.dtype)


def _reference_swiglu(gate: torch.Tensor, up: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """Preserve a FP32 SiLU/product intermediate before the fused-output cast."""
    with torch.autocast(gate.device.type, enabled=False):
        return (F.silu(gate.float()) * up.float()).to(output_dtype)


class DeepseekV32SFTMLP(nn.Module):
    """Dense SwiGLU MLP retaining the HF projection names and parameters."""

    @classmethod
    def from_hf(cls, module: nn.Module) -> "DeepseekV32SFTMLP":
        """Adopt an existing HF MLP without allocating or copying parameters."""
        target = cls()
        target.config = module.config
        target.hidden_size = module.hidden_size
        target.intermediate_size = module.intermediate_size
        target.gate_proj = module.gate_proj
        target.up_proj = module.up_proj
        target.down_proj = module.down_proj
        target.train(module.training)
        return target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU without materializing its SiLU intermediate in BF16."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(_reference_swiglu(gate, up, gate.dtype))


class DeepseekV32SFTExperts(DeepseekV32Experts):
    """Packed routed-expert weights with the reference local aggregation path."""

    @classmethod
    def from_hf(cls, module: nn.Module) -> "DeepseekV32SFTExperts":
        """Adopt existing packed expert parameters without changing their layout."""
        target = cls.__new__(cls)
        nn.Module.__init__(target)
        target.num_experts = module.num_experts
        target.hidden_dim = module.hidden_dim
        target.intermediate_dim = module.intermediate_dim
        target.gate_up_proj = module.gate_up_proj
        target.down_proj = module.down_proj
        target.act_fn = module.act_fn
        target.train(module.training)
        return target

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros(
            hidden_states.shape, dtype=torch.float32, device=hidden_states.device
        )
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_index in expert_mask.sum(dim=(-1, -2)).nonzero().flatten():
            top_k_position, token_index = torch.where(expert_mask[expert_index])
            values = hidden_states[token_index]
            gate, up = F.linear(values, self.gate_up_proj[expert_index]).chunk(2, dim=-1)
            values = F.linear(
                _reference_swiglu(gate, up, gate.dtype),
                self.down_proj[expert_index],
            )
            values = values.float() * top_k_weights[token_index, top_k_position, None].float()
            result.index_add_(0, token_index, values)
        return result


class DeepseekV32SFTMoE(nn.Module):
    """Reference sigmoid routing, routed experts, shared experts, and aux loss."""

    @classmethod
    def from_hf(cls, module: nn.Module) -> "DeepseekV32SFTMoE":
        """Adopt the HF router and all checkpoint-bearing MoE children."""
        target = cls()
        target.config = module.config
        target.gate = module.gate
        target.experts = DeepseekV32SFTExperts.from_hf(module.experts)
        target.shared_experts = DeepseekV32SFTMLP.from_hf(module.shared_experts)
        target.reference_config: dict[str, Any] | None = None
        target.auxiliary_loss: torch.Tensor | None = None
        target.train(module.training)
        return target

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the reference group-one sigmoid/top-k selection in FP32."""
        if self.reference_config is None:
            raise RuntimeError("DeepseekV32SFTMoE requires reference_config before forward")
        config = self.reference_config
        with torch.autocast(hidden_states.device.type, enabled=False):
            values = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
            scores = F.linear(values, self.gate.weight.float()).sigmoid()
            selection = scores + self.gate.e_score_correction_bias.float()
            indices = selection.topk(config["num_experts_per_tok"], dim=-1).indices
            weights = scores.gather(-1, indices)
            if config["norm_topk_prob"] and config["num_experts_per_tok"] > 1:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            frequency = torch.bincount(indices.flatten(), minlength=config["n_routed_experts"]).float()
            frequency = frequency / indices.numel()
            normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
            self.auxiliary_loss = (
                (normalized.mean(0) * frequency).sum() * scores.shape[-1] * config["moe_aux_loss_coeff"]
            )
            return indices, weights * config["routed_scaling_factor"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine FP32 weighted routed output with the shared SwiGLU branch."""
        original_shape = hidden_states.shape
        indices, weights = self._route(hidden_states)
        routed = self.experts(hidden_states.reshape(-1, hidden_states.shape[-1]), indices, weights)
        routed = routed.view(original_shape)
        shared = self.shared_experts(hidden_states.to(torch.bfloat16))
        return routed.to(torch.bfloat16).float() + shared.float()


class DeepseekV32SFTAttention(nn.Module):
    """Dense MLA reference path; Hyper may replace this with fused MLA later."""

    @classmethod
    def from_hf(cls, module: nn.Module) -> "DeepseekV32SFTAttention":
        """Adopt MLA parameters while deliberately omitting the HF DSA indexer."""
        if getattr(module, "q_lora_rank", None) is None:
            raise ValueError("DeepseekV32SFTAttention requires the Q-LoRA MLA path")
        target = cls()
        for name in (
            "config", "layer_idx", "num_key_value_groups", "attention_dropout", "num_heads", "q_lora_rank",
            "qk_rope_head_dim", "kv_lora_rank", "v_head_dim", "qk_nope_head_dim", "qk_head_dim",
            "is_causal", "scaling",
        ):
            setattr(target, name, getattr(module, name))
        target.q_a_proj = module.q_a_proj
        target.q_a_layernorm = DeepseekV32SFTRMSNorm.from_hf(module.q_a_layernorm)
        target.q_b_proj = module.q_b_proj
        target.kv_a_proj_with_mqa = module.kv_a_proj_with_mqa
        target.kv_a_layernorm = DeepseekV32SFTRMSNorm.from_hf(module.kv_a_layernorm)
        target.kv_b_proj = module.kv_b_proj
        target.o_proj = module.o_proj
        target.rotary = ExplicitFP32RotaryEmbedding()
        target.train(module.training)
        return target

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run dense causal MLA without DSA, cache, or backend-specific kernels."""
        del position_ids, kwargs
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Reference MLA requires explicit positions and no KV cache")
        batch_size, sequence_length = hidden_states.shape[:-1]
        query_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(query_residual).view(
            batch_size, sequence_length, self.num_heads, self.qk_head_dim
        ).transpose(1, 2)
        query_nope, query_rope = torch.split(
            query, (self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1
        )
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        key_latent, key_rope = torch.split(compressed, (self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        key_value = self.kv_b_proj(self.kv_a_layernorm(key_latent)).view(
            batch_size, sequence_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        key_nope, value = torch.split(key_value, (self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        cos, sin = position_embeddings
        query_rope = self.rotary(query_rope, cos, sin)
        key_rope = self.rotary(key_rope.view(batch_size, 1, sequence_length, self.qk_rope_head_dim), cos, sin)
        key = torch.cat((key_nope, key_rope.expand(-1, self.num_heads, -1, -1)), dim=-1)
        query = torch.cat((query_nope, query_rope), dim=-1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
            scale=self.scaling,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1).contiguous()
        return self.o_proj(output), None


class DeepseekV32SFTDecoder(nn.Module):
    """Decoder with reference FP32 residual accumulation and SFT children."""

    @classmethod
    def from_hf(cls, module: nn.Module) -> "DeepseekV32SFTDecoder":
        """Adopt a HF decoder and substitute only its semantic implementation."""
        target = cls()
        target.hidden_size = module.hidden_size
        target.self_attn = DeepseekV32SFTAttention.from_hf(module.self_attn)
        target.mlp = (
            DeepseekV32SFTMoE.from_hf(module.mlp)
            if hasattr(module.mlp, "experts") and hasattr(module.mlp, "shared_experts")
            else DeepseekV32SFTMLP.from_hf(module.mlp)
        )
        target.input_layernorm = DeepseekV32SFTRMSNorm.from_hf(module.input_layernorm)
        target.post_attention_layernorm = DeepseekV32SFTRMSNorm.from_hf(module.post_attention_layernorm)
        target.train(module.training)
        return target

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run attention and MLP branches with the reference BF16 residual boundary."""
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


class DeepseekV32SFTForCausalLM(DeepseekV32ForCausalLM):
    """Complete single-device SFT model with optional post-construction MLA fusion."""

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "DeepseekV32SFTForCausalLM":
        """Construct through Transformers' standard configuration path."""
        return cls._from_config(config, **kwargs)

    def __init__(self, config: Any) -> None:
        """Construct the HF model and replace its semantic children in-place."""
        super().__init__(config)
        if not hasattr(config, "sft_config"):
            raise ValueError("DeepseekV32SFTForCausalLM requires an explicit sft_config")
        self.reference_config = config.sft_config
        self.model.layers = nn.ModuleList(
            DeepseekV32SFTDecoder.from_hf(layer) for layer in self.model.layers
        )
        self.model.norm = DeepseekV32SFTRMSNorm.from_hf(self.model.norm)
        mtp_config = copy.deepcopy(config)
        depth = self.reference_config["num_nextn_predict_layers"]
        mtp_config.mlp_layer_types = ["sparse"] * depth
        self.mtp = MultiTokenPrediction([
            MultiTokenPredictionLayer(
                embedding_norm=DeepseekV32SFTRMSNorm(config.hidden_size, config.rms_norm_eps),
                hidden_norm=DeepseekV32SFTRMSNorm(config.hidden_size, config.rms_norm_eps),
                projection=nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False),
                decoder=DeepseekV32SFTDecoder.from_hf(DeepseekV32DecoderLayer(mtp_config, index)),
                output_norm=DeepseekV32SFTRMSNorm(config.hidden_size, config.rms_norm_eps),
                fusion_dtype=torch.bfloat16,
            ) for index in range(depth)
        ])
        # HF initialized the trunk before the additional MTP subtree existed.
        self.mtp.apply(self._initialize_weights)
        for module in self.modules():
            if isinstance(module, DeepseekV32SFTMoE):
                module.reference_config = self.reference_config
        self.loss_group = None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> CausalLMOutputWithPast:
        """Return the combined LM, MTP, and router objective for Trainer backward."""
        if use_cache:
            raise ValueError("SFT does not support cached decoding")
        with torch.autocast(input_ids.device.type, dtype=torch.bfloat16):
            losses = self.compute_sft_losses(input_ids, labels, loss_mask)
        return CausalLMOutputWithPast(loss=losses["loss"])

    def compute_sft_losses(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the reference LM, MTP, and router losses on pre-shifted labels."""
        config = self.reference_config
        if input_ids.shape != (1, config["seq_length"]):
            raise ValueError("Expected a complete batch-one input sequence")
        if labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("Tokens, pre-shifted labels and loss mask must have identical shapes")
        dimension = config["qk_rope_head_dim"]
        inverse = 1.0 / (config["rope_theta"] ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension))
        inverse = torch.from_numpy(inverse.astype(np.float32)).to(input_ids.device)
        frequency = torch.arange(config["seq_length"], device=input_ids.device, dtype=torch.float32)[:, None] * inverse
        frequency = torch.cat((frequency, frequency), dim=-1).unsqueeze(0)
        attention_kwargs = {
            "position_embeddings": (frequency.cos(), frequency.sin()),
            "actual_seq_len": (config["seq_length"],),
        }
        hidden = self.model.embed_tokens(input_ids).to(torch.bfloat16)
        auxiliary = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden, **attention_kwargs)
            if isinstance(layer.mlp, DeepseekV32SFTMoE):
                auxiliary = auxiliary + layer.mlp.auxiliary_loss
        hidden = hidden.float()
        lm_loss = masked_vocab_parallel_loss(
            self.lm_head(self.model.norm(hidden).to(torch.bfloat16)), labels, loss_mask, self.loss_group
        )
        mtp_loss = torch.zeros_like(lm_loss)
        for layer in self.mtp.layers:
            input_ids, labels, loss_mask = shift_left(input_ids), shift_left(labels), shift_left(loss_mask)
            embedding = self.model.embed_tokens(input_ids).to(torch.bfloat16)
            hidden = layer(hidden, embedding, **attention_kwargs)
            auxiliary = auxiliary + layer.transformer_layer.mlp.auxiliary_loss
            mtp_loss = mtp_loss + masked_vocab_parallel_loss(
                self.lm_head(hidden), labels, loss_mask, self.loss_group
            ) * (config["mtp_loss_factor"] / len(self.mtp.layers))
        return {"loss": lm_loss + mtp_loss + auxiliary, "lm_loss": lm_loss,
                "mtp_loss": mtp_loss, "aux_loss": auxiliary}
