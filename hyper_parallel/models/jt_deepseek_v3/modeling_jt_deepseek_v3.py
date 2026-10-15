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
"""JT DeepSeek-V3 causal language model built from Transformers DeepSeek-V3.2 components."""

# This model uses the Torch/HF runtime, like the existing Trainer model families.
# pylint: disable=forbidden-backend-import
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, DeepseekV32Config, DeepseekV32ForCausalLM
from transformers.utils import ModelOutput
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention, DeepseekV32DecoderLayer, DeepseekV32Experts, DeepseekV32MLP,
    DeepseekV32MoE, DeepseekV32Model, DeepseekV32PreTrainedModel, DeepseekV32RMSNorm,
    DeepseekV32RotaryEmbedding, DeepseekV32TopkRouter,
)

from hyper_parallel.components.losses import calculate_mtp_loss, calculate_seq_aux_loss
from hyper_parallel.components.losses.mtp import iter_mtp_targets
from hyper_parallel.components.losses.chunked_cross_entropy import projected_cross_entropy
from hyper_parallel.components.modules.mtp import DeepseekV3MTP, shift_mtp_sequence
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.components.functional.npu_fusion_attention import (
    _attention_options, _prepare_attention_inputs, resolve_packed_sequence_lengths,
)
from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.components.functional.rotary_embedding import apply_rotary_pos_emb_single
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS


class JTDeepseekV3Config(DeepseekV32Config):
    """DeepSeek-V3.2 configuration registered as the ``jt_deepseek_v3`` model type."""

    model_type = "jt_deepseek_v3"


AutoConfig.register(JTDeepseekV3Config.model_type, JTDeepseekV3Config, exist_ok=True)


@dataclass
class JTDeepseekV3Output(ModelOutput):
    """Model output holding the named JT training losses."""

    # Keep logits first and optional so ModelOutput preserves the named loss
    # mapping instead of interpreting it as a field iterator.
    logits: torch.Tensor | None = None
    loss: dict[str, torch.Tensor] | None = None


class JTDeepseekV3Experts(DeepseekV32Experts):
    """HF packed experts with the expert-major grouped-GEMM entry of both routed paths."""

    def forward_expert_major(self, inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Run the SwiGLU experts on tokens grouped by local expert.

        The single-device routed path and expert parallelism both compute the experts here: two
        grouped GEMMs around the fused ``npu_swiglu``.

        Args:
            inputs: Tokens sorted by local expert, ``[tokens, hidden]``.
            counts: Number of tokens of each local expert.

        Returns:
            Expert outputs in the order of ``inputs``.
        """
        return npu_grouped_swiglu(inputs, self.gate_up_proj, self.down_proj, counts)


class JTDeepseekV3RotaryEmbedding(nn.Module):
    """Rotary position embedding for the interleaved (``rope_interleave``) channel layout."""

    def __init__(self, use_fused_rope: bool = False) -> None:
        """Select the optional NPU kernel without changing the FP32 rotary contract.

        Args:
            use_fused_rope: Use the shared half-split NPU kernel after de-interleaving.
        """
        super().__init__()
        if not isinstance(use_fused_rope, bool):
            raise TypeError("use_fused_rope must be a bool")
        self.use_fused_rope = use_fused_rope

    def forward(self, values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                *, unsqueeze_dim: int = 1) -> torch.Tensor:
        """De-interleave ``values`` and rotate them by the position angles in FP32.

        As MindFormers does with ``rotary_dtype: float32``, the channels and tables are upcast to
        FP32 and the rotated channels are rounded to the dtype of ``values`` once.

        Args:
            values: Rotary channels in interleaved order, with BNSD or BSND axes.
            cos: Cosines of the position angles, ``[batch, sequence, rope_dim]``.
            sin: Sines of the position angles, ``[batch, sequence, rope_dim]``.
            unsqueeze_dim: Head axis, 1 for BNSD or 2 for BSND inputs.

        Returns:
            The rotated channels in half-split order, in the dtype of ``values``.
        """
        # Avoid strided-slice gradient scatter while preserving the rotary channel order.
        ordered = values.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2).float()
        if self.use_fused_rope:
            return apply_rotary_pos_emb_single(
                ordered, cos.float(), sin.float(), unsqueeze_dim=unsqueeze_dim).to(values.dtype)
        first, second = ordered.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        cosine_term = ordered * cos.unsqueeze(unsqueeze_dim).float()
        del ordered, first, second
        rotated = rotated * sin.unsqueeze(unsqueeze_dim).float()
        output = cosine_term + rotated
        return output.to(values.dtype)


def observed_fusion_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                              value: torch.Tensor, attention_mask: Any, dropout: float = 0.0,
                              scaling: float | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run NPU fused attention and record per-head maxima of the attention logits.

    Inputs are prepared like the framework's ``npu_fusion_attention_forward``; the
    kernel's softmax maxima are accumulated into ``module.max_logits_val`` for QK
    clipping.

    Args:
        module: Attention module owning the kernel options and ``max_logits_val``.
        query: Query states, ``[batch, heads, sequence, head_dim]``.
        key: Key states, ``[batch, heads, sequence, head_dim]``.
        value: Value states, ``[batch, heads, sequence, v_head_dim]``.
        attention_mask: Optional attention mask.
        dropout: Attention dropout probability.
        scaling: Attention score scale; ``head_dim ** -0.5`` when ``None``.
        **kwargs: Packed sequence lengths and attention window options.

    Returns:
        The attention output, ``[batch, sequence, heads, v_head_dim]``, and ``None``.
    """
    batch_size, _, query_length, head_dim = query.shape
    query_lengths, key_lengths = resolve_packed_sequence_lengths(
        kwargs, batch_size * query_length, key.shape[0] * key.shape[2])
    pre_tokens, next_tokens, sparse_mode, window, causal = _attention_options(module, kwargs)
    query, key, value, layout, mask, sparse_mode = _prepare_attention_inputs(
        query, key, value, attention_mask, is_packed=query_lengths is not None,
        is_causal=causal, sliding_window=window, sparse_mode=sparse_mode)
    result = torch_npu.npu_fusion_attention(
        query, key, value, query.shape[1], layout,
        pse=None, padding_mask=None, atten_mask=mask,
        scale=head_dim**-0.5 if scaling is None else scaling,
        pre_tockens=pre_tokens, next_tockens=next_tokens,
        keep_prob=1.0 - dropout, inner_precise=0, sparse_mode=sparse_mode,
        actual_seq_qlen=query_lengths, actual_seq_kvlen=key_lengths,
        softmax_layout="TND" if layout == "TND" else "")
    with torch.no_grad():
        # TND kernels otherwise return statistics in NTD order.
        maximum = result[1].amax(dim=(0, 2) if layout == "TND" else (0, 2, 3))
        if module.max_logits_val is None:
            module.max_logits_val = torch.zeros_like(maximum)
        module.max_logits_val.copy_(torch.maximum(module.max_logits_val, maximum))
    if query_lengths is not None:
        return result[0].reshape(batch_size, query_length, *result[0].shape[1:]), None
    return result[0].transpose(1, 2), None


@module_replacement
class JTDeepseekV3MLAAttention(MLAAttention):
    """MLA attention on the framework module with JT's rotary embedding and QK-clip statistics.

    Replaces :class:`JTDeepseekV3Attention`; ``q_a_proj`` and ``kv_a_proj_with_mqa``
    are fused into ``linear_qkv``.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        """Build the replacement from the attention module it replaces.

        Args:
            module: The :class:`JTDeepseekV3Attention` being replaced.
            module_fqn: Fully qualified name of ``module``.
            context: Replacement context of the build pipeline.
        """
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        if not self.linear_qkv.weight.is_meta:
            with torch.no_grad():
                self.linear_qkv.weight.copy_(torch.cat((module.q_a_proj.weight, module.kv_a_proj_with_mqa.weight)))
        self.explicit_rotary = JTDeepseekV3RotaryEmbedding(
            use_fused_rope=getattr(module.config, "use_fused_rope", False))
        self.key_rope_gather = nn.Identity()
        self.register_buffer("max_logits_val", None, persistent=False)
        self.attention_interface = observed_fusion_attention

    def project_latent_inputs(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized Q/KV latents and the shared key at their existing SP boundaries.

        Args:
            hidden_states: Input activations in the layout selected by the model recipe.
        """
        latent_states = self.linear_qkv(hidden_states)
        query_local, kv_local = latent_states.split(
            (self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim), dim=-1
        )
        kv_local, rope_local = kv_local.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        query_latent = self.q_a_layernorm(query_local)
        kv_latent = self.kv_a_layernorm(kv_local)
        key_rope = self.key_rope_gather(rope_local)
        return query_latent, kv_latent, key_rope

    def _project_attention_inputs(self, hidden_states: torch.Tensor, position_embeddings: Any,
                                  past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to queries, keys and values.

        Unlike the parent, the rotary key passes through ``key_rope_gather``, where
        the recipe gathers the sequence-parallel shards, and both rotary parts use
        :class:`JTDeepseekV3RotaryEmbedding`.

        Args:
            hidden_states: Input hidden states, ``[batch, sequence, hidden]``.
            position_embeddings: Rotary ``(cos, sin)`` tables.
            past_key_values: Must be ``None``; cached decoding is unsupported.

        Returns:
            Query, key and value states, each ``[batch, heads, sequence, head_dim]``.

        Raises:
            ValueError: If ``position_embeddings`` is missing or a cache is given.
        """
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("JT MLA attention requires position embeddings and no KV cache")
        query_latent, kv_latent, key_rope = self.project_latent_inputs(hidden_states)
        batch, sequence = query_latent.shape[:2]
        query = self.q_b_proj(query_latent).reshape(batch, sequence, self.num_heads, self.qk_head_dim)
        query_pass, query_rope = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        cos, sin = position_embeddings
        # Assemble Q before expanding KV so its old projection views can be released.
        query = torch.cat((query_pass, self.explicit_rotary(query_rope, cos, sin, unsqueeze_dim=2)), dim=-1)
        del query_pass, query_rope
        kv_latent = kv_latent.reshape(batch, 1, sequence, self.kv_lora_rank)
        kv_states = self.kv_b_proj(kv_latent).view(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        key_pass, value = kv_states.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        key_rope = self.explicit_rotary(key_rope.unsqueeze(2), cos, sin, unsqueeze_dim=2)
        key = torch.cat((key_pass, key_rope.expand(-1, -1, self.num_heads, -1)), dim=-1)
        # Preserve the BNSD interface with token-major storage for packed attention.
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                attention_mask: Any = None, past_key_values: Any = None,
                actual_seq_len: Any = None, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        """Run MLA attention with the NPU fused attention kernel.

        Args:
            hidden_states: Input hidden states, ``[batch, sequence, hidden]``.
            position_embeddings: Rotary ``(cos, sin)`` tables.
            attention_mask: Optional attention mask.
            past_key_values: Must be ``None``; cached decoding is unsupported.
            actual_seq_len: Packed sequence lengths.
            **kwargs: Additional attention kernel options.

        Returns:
            The attention output and ``None`` attention weights.
        """
        query, key, value = self._project_attention_inputs(hidden_states, position_embeddings, past_key_values)
        # Replace each owning reference before FA so old projection storage can be released.
        query = query.transpose(1, 2).contiguous().transpose(1, 2)
        key = key.transpose(1, 2).contiguous().transpose(1, 2)
        value = value.transpose(1, 2).contiguous().transpose(1, 2)
        output, weights = self.attention_interface(
            self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0, scaling=self.scaling,
            sliding_window=self.sliding_window, actual_seq_len=actual_seq_len, **kwargs)
        output = output.reshape(query.shape[0], query.shape[2], -1).contiguous()
        return self.o_proj(output), weights


class JTDeepseekV3Attention(DeepseekV32Attention):
    """Causal MLA attention with HF DeepSeek-V3 parameter names and no DSA indexer."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Build the MLA projections and latent norms.

        Args:
            config: JT model configuration.
            layer_idx: Index of the decoder layer.
        """
        nn.Module.__init__(self)
        self.config, self.layer_idx = config, layer_idx
        self.num_heads = config.num_attention_heads
        self.q_lora_rank, self.kv_lora_rank = config.q_lora_rank, config.kv_lora_rank
        self.qk_rope_head_dim, self.qk_nope_head_dim = config.qk_rope_head_dim, config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.scaling = self.qk_head_dim ** -0.5
        self.attention_dropout, self.is_causal, self.sliding_window = config.attention_dropout, True, None
        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV32RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepseekV32RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.explicit_rotary = JTDeepseekV3RotaryEmbedding(
            use_fused_rope=getattr(config, "use_fused_rope", False))
        self.register_buffer("max_logits_val", None, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                past_key_values: Any = None, attention_mask: Any = None, **kwargs: Any) -> tuple:
        """Run causal MLA attention: NPU fused attention on NPU, SDPA elsewhere.

        Args:
            hidden_states: Input hidden states, ``[batch, sequence, hidden]``.
            position_embeddings: Rotary ``(cos, sin)`` tables.
            past_key_values: Must be ``None``; cached decoding is unsupported.
            attention_mask: Optional attention mask; attention is causal when ``None``.
            **kwargs: Packed sequence lengths for NPU fused attention.

        Returns:
            The attention output and ``None`` attention weights.

        Raises:
            ValueError: If ``position_embeddings`` is missing or a cache is given.
        """
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("JT attention requires position embeddings and no KV cache")
        batch, sequence = hidden_states.shape[:2]
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query = query.reshape(batch, sequence, self.num_heads, self.qk_head_dim).transpose(1, 2)
        kv, rope = self.kv_a_proj_with_mqa(hidden_states).split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv)).reshape(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        key, value = kv.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        cos, sin = position_embeddings
        query = torch.cat((query[..., :self.qk_nope_head_dim],
                           self.explicit_rotary(query[..., self.qk_nope_head_dim:], cos, sin)), dim=-1)
        rope = self.explicit_rotary(rope.unsqueeze(1), cos, sin).expand(-1, self.num_heads, -1, -1)
        key = torch.cat((key, rope), dim=-1)
        if hidden_states.device.type == "npu":
            output, _ = observed_fusion_attention(self, query, key, value, attention_mask,
                                                  scaling=self.scaling, **kwargs)
        else:
            lengths, _ = resolve_packed_sequence_lengths(kwargs, batch * sequence, batch * sequence)
            if lengths is not None:
                if batch != 1 or attention_mask is not None:
                    raise ValueError("JT packed attention requires batch one and no explicit mask")
                starts = (0, *lengths[:-1])
                output = torch.cat([
                    # Pylint cannot infer this Torch C-extension callable.
                    F.scaled_dot_product_attention(  # pylint: disable=not-callable
                        query[:, :, begin:end], key[:, :, begin:end], value[:, :, begin:end],
                        is_causal=True, scale=self.scaling)
                    for begin, end in zip(starts, lengths)
                ], dim=2)
            else:
                output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                                        is_causal=attention_mask is None, scale=self.scaling)
            output = output.transpose(1, 2)
        return self.o_proj(output.reshape(batch, sequence, -1)), None


class JTDeepseekV3Gate(DeepseekV32TopkRouter):
    """HF router projection computing FP32 logits, kept for the sequence auxiliary loss."""

    def __init__(self, config: Any) -> None:
        """Create the HF router weight and expert bias.

        Args:
            config: JT model configuration.
        """
        super().__init__(config)
        self.router_logits = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the FP32 router logits, ``[tokens, num_experts]``, and keep them.

        As in the HF router, the input and the weight are cast to FP32, so routing and the
        auxiliary loss use FP32 logits instead of logits rounded to the BF16 activation dtype.

        Args:
            hidden_states: Router inputs, ``[..., hidden]``.
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        self.router_logits = F.linear(hidden_states.float(), self.weight.float())  # pylint: disable=not-callable
        return self.router_logits

    def pop_router_logits(self) -> torch.Tensor:
        """Return the logits of the last forward and release them.

        Callers release the logits through this method: a checkpoint wrapper around
        the router forwards attribute reads but keeps attribute writes, so assigning
        ``router_logits`` through the wrapper would hide the logits of later steps.
        """
        logits, self.router_logits = self.router_logits, None
        return logits


class JTDeepseekV3MoE(DeepseekV32MoE):
    """Sigmoid-routed MoE with shared experts, the sequence auxiliary loss and expert-load counts."""

    def __init__(self, config: Any) -> None:
        """Build the routed experts, router and shared experts.

        Args:
            config: JT model configuration.
        """
        nn.Module.__init__(self)
        self.config = config
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.experts = JTDeepseekV3Experts(config)
        self.gate = JTDeepseekV3Gate(config)
        self.shared_experts = DeepseekV32MLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
        # Ranks holding shards of the same sequences (TP with sequence parallelism); the JT builder sets it.
        self.sequence_partition_group = None
        # Auxiliary gradients also average TP replicas when sequence parallelism is off.
        self.auxiliary_loss_group = None
        self.ep_compute = self.local_routed_forward
        self.auxiliary_loss = None
        # Routed-token counts since the last bias update; the optimizer hook sums them over the batch.
        self.tokens_per_expert = None

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts for the real tokens, record the auxiliary loss and count routed tokens.

        Args:
            hidden: Hidden states led by ``padding`` pad tokens, ``[1, padding + tokens, hidden]``.

        Returns:
            Expert indices and routing weights, ``[padding + tokens, top_k]``; the pad
            tokens cover every expert with zero weight.
        """
        padding, config = self.padding, self.config
        hidden = hidden[:, padding:]
        indices, selected = MOE_ROUTER_ADAPTERS["deepseekv3"](self, hidden)
        scores = self.gate.pop_router_logits().sigmoid()
        auxiliary_loss = self.routing_statistics(indices, scores)
        # Replay must keep the same tensor operations without retaining its graph on the module.
        if not is_recomputing():
            self.auxiliary_loss = auxiliary_loss
        flat_indices = indices.reshape(-1)
        # scatter_add_ counts on device; torch.bincount first reads the largest index back to the host.
        counts = flat_indices.new_zeros(config.n_routed_experts, dtype=torch.float32).scatter_add_(
            0, flat_indices, flat_indices.new_ones(flat_indices.shape, dtype=torch.float32))
        self.tokens_per_expert = counts if self.tokens_per_expert is None else self.tokens_per_expert + counts
        if padding:
            pad_ids = torch.arange(padding * config.num_experts_per_tok, device=indices.device)
            pad_ids = pad_ids.reshape(padding, config.num_experts_per_tok) % padding
            indices = torch.cat((pad_ids, indices))
            selected = torch.cat((selected.new_zeros(padding, selected.shape[-1]), selected))
        return indices, selected

    def pop_auxiliary_loss(self) -> torch.Tensor:
        """Transfer the forward loss to its consumer without retaining its graph.

        Returns:
            The differentiable auxiliary loss from the preceding forward.

        Raises:
            RuntimeError: If no unconsumed forward loss is available.
        """
        loss, self.auxiliary_loss = self.auxiliary_loss, None
        if loss is None:
            raise RuntimeError("No auxiliary loss is available; run the MoE forward before consuming it")
        return loss

    def routing_statistics(self, indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Compute the sequence objective separately from step-level router counts.

        Args:
            indices: Selected expert indices for the real, unpadded tokens.
            scores: Router affinities before top-k normalization.
        """
        return calculate_seq_aux_loss(scores, indices, coeff=self.config.moe_aux_loss_coeff,
                                      sequence_partition_group=self.auxiliary_loss_group)

    @torch.no_grad()
    def update_expert_bias(self, lr: float, num_recomputations: int = 1) -> None:
        """Move the routing bias toward balanced load once ``tokens_per_expert`` holds global counts.

        :func:`~hyper_parallel.core.utils.moe_utils.sync_and_update_expert_bias` sums the
        counts over the batch before calling this. Unlike
        :meth:`hyper_parallel.components.modules.moe.MoE.update_expert_bias`, the step is
        not re-centered, matching MindFormers.

        Args:
            lr: Bias update rate.
            num_recomputations: Forward executions per optimizer step; counting every
                token the same number of times leaves the sign update unchanged.
        """
        del num_recomputations
        counts = self.tokens_per_expert
        self.gate.e_score_correction_bias.add_((counts.mean() - counts).sign(), alpha=lr)
        self.tokens_per_expert = None

    def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the routed experts on this rank without expert parallelism.

        Mirrors the EP pipeline on one rank: tokens are sorted by expert and computed by
        :meth:`JTDeepseekV3Experts.forward_expert_major`, the routing weights are cast to the
        activation dtype, and the weighted outputs are summed per token with ``index_add_``. The
        routed output therefore does not depend on whether expert parallelism is enabled.

        Args:
            hidden: Pad-prefixed hidden states, ``[1, padding + tokens, hidden]``.

        Returns:
            The weighted routed-expert output, shaped like ``hidden``.
        """
        indices, probabilities = self.route(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        expert_ids = indices.reshape(-1)
        order = expert_ids.argsort()
        sources = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(indices.shape[1])[order]
        counts = expert_ids.new_zeros(self.experts.num_experts).scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
        outputs = self.experts.forward_expert_major(flat[sources], counts)
        weights = probabilities.reshape(-1).to(flat.dtype)[order].unsqueeze(-1)
        return torch.zeros_like(flat).index_add_(0, sources, outputs * weights).reshape(hidden.shape)

    @staticmethod
    def combine_routed(owner: Any, hidden: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """Return the routed branch; :meth:`forward` adds the shared experts.

        Args:
            owner: MoE module owning the execution.
            hidden: Hidden states.
            routed: Routed expert outputs.

        Returns:
            ``routed`` unchanged.
        """
        del owner, hidden
        return routed

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the routed experts on pad-prefixed tokens and add the shared experts.

        Args:
            hidden_states: Hidden states, ``[1, tokens, hidden]``.

        Returns:
            The MoE output, shaped like ``hidden_states``.
        """
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.ep_compute(hidden)
        return routed[:, self.padding:] + self.shared_experts(hidden_states)


class JTDeepseekV3Decoder(DeepseekV32DecoderLayer):
    """Pre-norm decoder layer with MLA attention and a dense or MoE MLP."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Build the attention, MLP and norms of one layer.

        Args:
            config: JT model configuration.
            layer_idx: Layer index; ``config.mlp_layer_types[layer_idx]`` selects the MLP.
        """
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = JTDeepseekV3Attention(config, layer_idx)
        self.mlp = (JTDeepseekV3MoE(config)
                    if config.mlp_layer_types[layer_idx] == "sparse" else DeepseekV32MLP(config))
        self.input_layernorm = DeepseekV32RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV32RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Any = None,
                position_ids: Any = None, past_key_values: Any = None, use_cache: bool = False,
                position_embeddings: Any = None, **kwargs: Any) -> torch.Tensor:
        """Apply the attention and MLP sublayers, each with a residual connection.

        Args:
            hidden_states: Input hidden states, ``[batch, sequence, hidden]``.
            attention_mask: Optional attention mask.
            position_ids: Sequence positions.
            past_key_values: Must be ``None``; cached decoding is unsupported.
            use_cache: Must be ``False``.
            position_embeddings: Rotary ``(cos, sin)`` tables.
            **kwargs: Attention kernel options.

        Returns:
            The layer output, shaped like ``hidden_states``.

        Raises:
            ValueError: If cached decoding is requested.
        """
        if past_key_values is not None or use_cache:
            raise ValueError("JT does not support cached decoding")
        residual = hidden_states
        branch, _ = self.self_attn(
            self.input_layernorm(residual),
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + branch
        residual = hidden_states
        branch = self.mlp(self.post_attention_layernorm(residual))
        return residual + branch


class JTDeepseekV3Model(DeepseekV32Model):
    """DeepSeek-V3.2 base model built from JT decoder layers."""

    def __init__(self, config: Any) -> None:
        """Build the token embedding, decoder layers, final norm and HF rotary embedding.

        Args:
            config: JT model configuration.
        """
        DeepseekV32PreTrainedModel.__init__(self, config)
        self.padding_idx, self.vocab_size = config.pad_token_id, config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([JTDeepseekV3Decoder(config, index) for index in range(config.num_hidden_layers)])
        self.norm = DeepseekV32RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DeepseekV32RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class JTDeepseekV3ForCausalLM(DeepseekV32ForCausalLM):
    """JT DeepSeek-V3 causal LM whose forward returns the LM, MTP and router auxiliary losses."""

    config_class = JTDeepseekV3Config

    def __init__(self, config: Any) -> None:
        """Build the base model, LM head and multi-token-prediction depths.

        Args:
            config: JT model configuration.

        Raises:
            ValueError: If the configuration needs an unsupported activation, routing
                group or RoPE type.
        """
        DeepseekV32PreTrainedModel.__init__(self, config)
        if config.hidden_act != "silu":
            raise ValueError("JT fused experts require hidden_act='silu'")
        if config.n_group != 1 or config.topk_group != 1:
            raise ValueError("JT routing currently requires n_group=topk_group=1")
        if config.rope_parameters["rope_type"] != "default":
            raise ValueError("JT rotary computation currently requires rope_type='default'")
        self.model = JTDeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        mtp_config = copy.deepcopy(config)
        depth = config.num_nextn_predict_layers
        mtp_config.mlp_layer_types = ["sparse"] * depth
        self.mtp = DeepseekV3MTP(
            hidden_size=config.hidden_size, num_layers=depth,
            decoder_factory=lambda index: JTDeepseekV3Decoder(mtp_config, index),
            norm_factory=lambda size: DeepseekV32RMSNorm(size, eps=config.rms_norm_eps),
            output_norm_factory=lambda size: DeepseekV32RMSNorm(size, eps=config.rms_norm_eps),
        )
        self.loss_tp_mesh = None
        self.loss_sequence_parallel_size = 1
        self.enable_chunk_loss = getattr(config, "enable_chunk_loss", False)
        self.chunk_loss_size = getattr(config, "chunk_loss_size", 1024)
        if not isinstance(self.enable_chunk_loss, bool):
            raise ValueError("enable_chunk_loss must be a bool")
        if (isinstance(self.chunk_loss_size, bool) or not isinstance(self.chunk_loss_size, int)
                or self.chunk_loss_size <= 0):
            raise ValueError("chunk_loss_size must be a positive integer; use enable_chunk_loss to disable chunking")
        # Ranks holding the same attention heads (DP+CP); the JT builder sets it for QK clipping.
        self.qk_clip_group = None
        # Ranks holding different batch data (DP+CP); the JT builder sets it to sum expert token counts.
        self.expert_load_group = None
        self.post_init()

    def forward(self, input_ids: torch.Tensor, shift_labels: torch.Tensor | None = None, *,
                labels: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, use_cache: bool = False,
                actual_seq_len: tuple[int, ...] | None = None, sequence_start: int = 0) -> JTDeepseekV3Output:
        """Compute the JT training losses of one batch.

        Args:
            input_ids: Token IDs, ``[1, sequence]``.
            shift_labels: Targets already shifted by one token; negative targets are ignored.
            labels: Unused; ``shift_labels`` carries the supervision.
            position_ids: Sequence positions for RoPE.
            attention_mask: Must be ``None``; cumulative lengths define independent causal documents.
            use_cache: Must be ``False``.
            actual_seq_len: Global cumulative document ends, without a leading zero.
            sequence_start: Global offset of the local input interval.

        Returns:
            Output whose ``loss`` maps ``foundation_loss/lm``, ``foundation_loss/mtp`` and
            ``foundation_loss/aux`` to 0-d losses.

        Raises:
            ValueError: If ``attention_mask`` is given, ``shift_labels`` is missing or cached
                decoding is requested.
        """
        del labels
        if attention_mask is not None:
            raise ValueError("JT requires attention_mask=None for its full causal sequence")
        if shift_labels is None:
            raise ValueError("JT training requires explicit shift_labels")
        if use_cache:
            raise ValueError("JT does not support cached decoding")
        # Same rule as the shared text batch's loss mask; JT data folds its 0/1 mask into the labels.
        losses = self.compute_jt_losses(input_ids, shift_labels, shift_labels >= 0, position_ids=position_ids,
                                        actual_seq_len=actual_seq_len, sequence_start=sequence_start)
        # ``<token-domain>_loss[/<name>]`` keys: every JT objective is weighted by foundation tokens.
        return JTDeepseekV3Output(loss={
            "foundation_loss/lm": losses["lm_loss"],
            "foundation_loss/mtp": losses["mtp_loss"],
            "foundation_loss/aux": losses["aux_loss"],
        })

    def _token_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean causal-LM loss over targets with a nonzero mask; zero when there are none.

        Uses the framework-selected ``loss_function``.
        """
        valid = (mask != 0) & (labels >= 0)
        loss = self.loss_function(logits=logits, labels=None, vocab_size=self.config.vocab_size,
                                  shift_labels=labels.masked_fill(~valid, -100),
                                  num_items_in_batch=valid.sum().clamp_min(1))
        # causal_lm_loss_parallel returns shape [1]; the Trainer stacks named losses, so keep each one 0-d.
        return loss.reshape(())

    def _projection_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Evaluate head and CE in checkpointed chunks of the CP-local sequence."""
        return projected_cross_entropy(
            hidden, targets, head=self.lm_head, loss_fn=self.loss_function,
            vocab_size=self.config.vocab_size, chunk_size=self.chunk_loss_size,
            sequence_parallel_size=self.loss_sequence_parallel_size, tp_mesh=self.loss_tp_mesh,
        )

    @staticmethod
    def shift_mtp_inputs(value: torch.Tensor, *, pad_value: int = 0) -> torch.Tensor:
        """Shift future inputs or targets; adapters may supply partition boundary values."""
        return shift_mtp_sequence(value, pad_value=pad_value)

    def _mtp_token_loss(self, *, logits: torch.Tensor, shift_labels: torch.Tensor,
                        **kwargs: Any) -> torch.Tensor:
        """Use the same objective normalization for LM and each future prediction depth."""
        del kwargs
        return self._token_loss(logits, shift_labels, shift_labels >= 0)

    def compute_jt_losses(self, input_ids: torch.Tensor, labels: torch.Tensor,
                          loss_mask: torch.Tensor, *, position_ids: torch.Tensor | None = None,
                          actual_seq_len: tuple[int, ...] | None = None, sequence_start: int = 0
                          ) -> dict[str, torch.Tensor]:
        """Compute the LM, MTP and router auxiliary losses on pre-shifted labels.

        Args:
            input_ids: Token IDs, ``[1, sequence]``.
            labels: Targets already shifted by one token.
            loss_mask: Mask of the supervised targets.
            position_ids: Optional sequence positions for RoPE.
            actual_seq_len: Global cumulative ends of independent documents.
            sequence_start: Global offset of the local input interval.

        Returns:
            ``lm_loss``, ``mtp_loss`` and ``aux_loss`` as 0-d tensors.

        Raises:
            ValueError: If the inputs are not one sequence with matching shapes.
        """
        cfg = self.config
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("JT sequence loss currently requires a two-dimensional batch-one input")
        sequence_length = input_ids.shape[1]
        if labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("Tokens, pre-shifted labels and loss mask must have identical shapes")
        ends = (sequence_length,) if actual_seq_len is None else tuple(actual_seq_len)
        if (not ends or ends[0] <= 0 or any(left >= right for left, right in zip(ends, ends[1:]))
                or sequence_start < 0 or sequence_start + sequence_length > ends[-1]):
            raise ValueError("JT cumulative document ends must increase and cover the local token interval")
        indices = torch.arange(sequence_start, sequence_start + sequence_length, device=input_ids.device)
        sequence_end_mask = None
        if len(ends) > 1:
            boundaries = indices.new_tensor((0, *ends))
            documents = torch.bucketize(indices, boundaries[1:], right=True)
            sequence_end_mask = (indices + 1 == boundaries[documents + 1]).unsqueeze(0)
            default_positions = indices - boundaries[documents]
        else:
            default_positions = indices
        # NumPy FP32 inverse frequencies match MindFormers bitwise; torch.pow rounds some entries differently.
        dim = cfg.qk_rope_head_dim
        inverse = 1.0 / (cfg.rope_parameters["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        inverse = torch.from_numpy(inverse.astype(np.float32)).to(input_ids.device)
        if position_ids is None:
            position_ids = default_positions.unsqueeze(0)
        if position_ids.shape != input_ids.shape:
            raise ValueError("JT position_ids must match input_ids")
        frequency = position_ids.to(device=input_ids.device, dtype=torch.float32).unsqueeze(-1) * inverse
        frequency = torch.cat((frequency, frequency), dim=-1)
        hidden = self.model.embed_tokens(input_ids)
        # FP32 tables; JTDeepseekV3RotaryEmbedding rotates in FP32 and rounds to the activation dtype once.
        attention_kwargs = {"position_embeddings": (frequency.cos(), frequency.sin()), "actual_seq_len": ends}
        auxiliary = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden, **attention_kwargs)
            if hasattr(layer.mlp, "pop_auxiliary_loss"):
                auxiliary = auxiliary + layer.mlp.pop_auxiliary_loss()
        valid = (loss_mask != 0) & (labels >= 0)
        targets = labels.masked_fill(~valid, -100)
        normalized = self.model.norm(hidden)
        lm_loss = (self._projection_loss(normalized, targets) if self.enable_chunk_loss
                   else self._token_loss(self.lm_head(normalized), labels, loss_mask))
        mtp_output = self.mtp(hidden, input_ids, embedding=self.model.embed_tokens,
                              head=None if self.enable_chunk_loss else self.lm_head,
                              decoder_kwargs=attention_kwargs, shift_fn=self.shift_mtp_inputs,
                              sequence_end_mask=sequence_end_mask)
        for layer in self.mtp.layers:
            auxiliary = auxiliary + layer.transformer_layer.mlp.pop_auxiliary_loss()
        shift_targets = partial(self.shift_mtp_inputs, pad_value=-100)
        if self.enable_chunk_loss:
            mtp_loss = torch.zeros_like(lm_loss)
            depths = len(mtp_output.prediction_hidden_states)
            targets_by_depth = iter_mtp_targets(targets, depths, shift_fn=shift_targets,
                                                sequence_end_mask=sequence_end_mask)
            for prediction, future_targets in zip(mtp_output.prediction_hidden_states, targets_by_depth):
                mtp_loss = mtp_loss + self._projection_loss(prediction, future_targets) * (cfg.mtp_loss_factor / depths)
        else:
            mtp_loss = calculate_mtp_loss(mtp_output.logits, targets, self._mtp_token_loss, vocab_size=cfg.vocab_size,
                                          loss_factor=cfg.mtp_loss_factor, shift_fn=shift_targets,
                                          sequence_end_mask=sequence_end_mask)
        return {"lm_loss": lm_loss, "mtp_loss": mtp_loss, "aux_loss": auxiliary}
