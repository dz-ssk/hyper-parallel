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
"""Configuration-driven DeepSeek-V3 training with the public MTP components.

The fixed-length training path preserves TP/SP and CP layouts. Model-owned
recipes apply the same attention and expert boundaries to trunk and MTP blocks.
"""

# Native Torch is required; the legacy platform-agnostic checker predates its removal.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import copy
import math
from functools import partial
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import DeepseekV3Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3TopkRouter,
)

from hyper_parallel.components.modules import MTPLayer, build_mtp
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.models.materialization import register_rebuildable_buffer, register_materialized_state_hook
from hyper_parallel.models.registry import register_custom_model
from hyper_parallel.models.deepseek_v3.adapter.mtp_data import prepare_mtp_inputs
from hyper_parallel.distributed.mtp import shift_mtp_tokens
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def _validate_options(depth: int | None, factor: float, normalization: str) -> None:
    """Validate configuration before allocating a model."""
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 0):
        raise ValueError("mtp_num_layers must be None or a nonnegative integer")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 0:
        raise ValueError("mtp_loss_factor must be a finite nonnegative number")
    if normalization not in ("sequence", "valid_tokens"):
        raise ValueError("mtp_loss_normalization must be 'sequence' or 'valid_tokens'")


def _rope_buffer(rotary, context):
    """Recreate deterministic frequencies after meta allocation, never from a checkpoint."""
    with torch.device(context.device):
        return type(rotary)(rotary.config).inv_freq


def _rope_alias(rotary, _context):
    """Refresh HF's non-registered alias after materialization replaces storage."""
    rotary.original_inv_freq = rotary.inv_freq


class DeepseekV3MTPForCausalLM(DeepseekV3ForCausalLM):
    """Keep the HF trunk/head and add independent, embedding-first MTP layers.

    Training returns named LM and weighted MTP losses for ModelOutputLoss and
    the normal Trainer metric pipeline. Inference uses the original LM path.
    """

    def __init__(self, config: DeepseekV3Config) -> None:
        """Construct all trainable MTP parameters before sharding/optimizer setup."""
        depth = getattr(config, "mtp_num_layers", 0)
        factor = getattr(config, "mtp_loss_factor", 0.3)
        normalization = getattr(config, "mtp_loss_normalization", "sequence")
        _validate_options(depth, factor, normalization)
        super().__init__(config)
        rotary = self.model.rotary_emb
        register_rebuildable_buffer(rotary, "inv_freq", factory=partial(_rope_buffer, rotary))
        register_materialized_state_hook(rotary, _rope_alias)
        self.mtp = build_mtp(depth, layer_factory=self._build_mtp_layer)
        self.mtp_pipeline_first = True
        self.mtp_pipeline_last = True
        self.mtp_cp_group = None
        self.mtp_cp_size = 1
        self.mtp_loss_factor = factor
        self.mtp_loss_normalization = normalization
        if self.mtp is not None:
            self.mtp.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize router checkpoint state as well as weights on from-config builds."""
        super()._init_weights(module)
        if isinstance(module, DeepseekV3TopkRouter):
            module.e_score_correction_bias.zero_()

    def _build_mtp_layer(self, depth: int) -> MTPLayer:
        """Create a fresh decoder at the position following the trunk layers."""
        return MTPLayer(
            self.config.hidden_size,
            DeepseekV3DecoderLayer(self.config, self.config.num_hidden_layers + depth),
            rms_norm_eps=self.config.rms_norm_eps,
        )

    @classmethod
    def from_config(cls, config: DeepseekV3Config, **kwargs: Any) -> DeepseekV3MTPForCausalLM:
        """Honor the standard HF dtype and initialization context."""
        return cls._from_config(config, **kwargs)

    def _decoder_inputs(self, hidden: torch.Tensor, positions: torch.Tensor,
                        attention_mask: torch.Tensor | None, depth: int = 0) -> dict[str, Any]:
        """Build per-depth metadata, retaining the global attention mask under CP."""
        length = positions.shape[1] * self.mtp_cp_size
        allowed = torch.ones((length, length), dtype=torch.bool, device=hidden.device).tril()
        allowed = allowed[None, None].expand(hidden.shape[0], 1, length, length)
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape[-1] == length:
                allowed = allowed & attention_mask[:, None, None, :].bool()
            elif attention_mask.ndim == 4 and attention_mask.shape[-2:] == (length, length):
                allowed = allowed & (attention_mask if attention_mask.dtype == torch.bool else attention_mask.eq(0))
            else:
                raise ValueError("MTP attention_mask must be a global [B,S] padding or [B,1,S,S] attention mask")
        if depth:
            shifted = F.pad(allowed[..., depth:, depth:], (0, depth, 0, depth), value=False)
            same_document = F.pad(allowed[:, :, depth:, :].diagonal(dim1=-2, dim2=-1), (0, depth))
            allowed = allowed & shifted & same_document.unsqueeze(-1) & same_document.unsqueeze(-2)
        layers = self.model.layers.values() if isinstance(self.model.layers, torch.nn.ModuleDict) else self.model.layers
        attention = next(iter(layers)).self_attn
        # The public MLA replacement consumes True=allowed; native HF consumes
        # additive masks. Both paths keep the same causal/document boundaries.
        uses_mla = hasattr(attention, "project_latent_inputs")
        mask = allowed if uses_mla else hidden.new_zeros(allowed.shape).masked_fill(
            ~allowed, torch.finfo(hidden.dtype).min)
        return {"attention_mask": mask, "position_ids": positions,
                "position_embeddings": self.model.rotary_emb(hidden, positions), "use_cache": False}

    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, labels: torch.Tensor | None = None,
                shift_labels: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None,
                use_cache: bool | None = None, hidden_states: torch.Tensor | None = None,
                mtp_input_ids: Sequence[torch.Tensor] | None = None,
                mtp_position_ids: Sequence[torch.Tensor] | None = None,
                mtp_targets: Sequence[torch.Tensor] | None = None,
                **kwargs: Any) -> CausalLMOutputWithPast | torch.Tensor:
        """Return independently normalized LM and per-depth MTP training losses.

        MTPTextParallelBatch supplies per-depth denominators to the Trainer.
        Direct calls may omit the prepared MTP fields for an unsharded batch.
        TP vocabulary outputs must be gathered (loss_parallel=false).
        """
        if self.mtp_pipeline_first and self.mtp_pipeline_last and (
                self.mtp is None or (labels is None and shift_labels is None)):
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                   position_ids=position_ids, labels=labels, use_cache=use_cache,
                                   **({"shift_labels": shift_labels} if shift_labels is not None else {}), **kwargs)
        if use_cache or kwargs:
            raise ValueError(f"MTP training requires use_cache=False; unsupported kwargs: {sorted(kwargs)}")
        if (input_ids is None or input_ids.ndim != 2
                or input_ids.shape[1] * self.mtp_cp_size <= self.config.mtp_num_layers):
            raise ValueError("MTP requires input_ids [B,S_local] and global S greater than mtp_num_layers")
        rank = 0 if self.mtp_cp_group is None else dist.get_rank(self.mtp_cp_group)
        padding = None
        if attention_mask is not None and attention_mask.ndim == 2:
            start = rank * input_ids.shape[1]
            padding = attention_mask[:, start:start + input_ids.shape[1]]
        targets = self._targets(input_ids, labels, shift_labels, loss_mask, padding)
        positions = (torch.arange(input_ids.shape[1], device=input_ids.device)[None] + rank * input_ids.shape[1]
                     if position_ids is None else position_ids)
        if self.mtp_pipeline_first:
            if hidden_states is not None:
                raise ValueError("The first pipeline stage takes input_ids, not received activations")
            hidden = self.model.embed_tokens(input_ids)
        else:
            if hidden_states is None:
                raise ValueError("A non-first pipeline stage requires hidden_states from its predecessor")
            hidden = hidden_states
        decoder_inputs = self._decoder_inputs(hidden, positions, attention_mask)
        layers = self.model.layers.values() if isinstance(self.model.layers, torch.nn.ModuleDict) else self.model.layers
        for layer in layers:
            hidden = layer(hidden, **decoder_inputs)
        if not self.mtp_pipeline_last:
            return hidden
        supplied = (mtp_input_ids, mtp_position_ids, mtp_targets)
        if all(value is None for value in supplied):
            prepared = prepare_mtp_inputs(input_ids, positions, targets, len(self.mtp.layers),
                                          cp_group=self.mtp_cp_group, attention_mask=attention_mask)
            mtp_input_ids, mtp_position_ids, mtp_targets = prepared.values()
        elif any(value is None or len(value) != len(self.mtp.layers) for value in supplied):
            raise ValueError("All prepared MTP fields must contain exactly mtp_num_layers entries")
        logits = self.lm_head(self.model.norm(hidden))
        lm_sum = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1),
                                 ignore_index=-100, reduction="sum")
        losses = {"foundation_loss/lm_loss": lm_sum / targets.ne(-100).sum().clamp_min(1)}
        predictions = self.mtp.forward_aligned(
            hidden, [self.model.embed_tokens(ids) for ids in mtp_input_ids], head=self.lm_head,
            decoder_kwargs=[self._decoder_inputs(hidden, pos, attention_mask, depth)
                            for depth, pos in enumerate(mtp_position_ids, 1)],
        )
        for depth, (prediction, target) in enumerate(zip(predictions.logits, mtp_targets), 1):
            numerator = F.cross_entropy(prediction.float().reshape(-1, self.config.vocab_size), target.reshape(-1),
                                        ignore_index=-100, reduction="sum")
            denominator = (target.numel() if self.mtp_loss_normalization == "sequence"
                           else target.ne(-100).sum().clamp_min(1))
            losses[f"mtp_depth{depth}_loss"] = numerator / denominator * self.mtp_loss_factor / len(self.mtp.layers)
        output = CausalLMOutputWithPast(loss=losses, logits=logits)
        output.loss_metric_groups = {"mtp_loss": tuple(key for key in losses if key.startswith("mtp_depth"))}
        return output

    def _targets(self, input_ids: torch.Tensor, labels: torch.Tensor | None, shifted: torch.Tensor | None,
                 loss_mask: torch.Tensor | None, padding: torch.Tensor | None) -> torch.Tensor:
        """Align causal labels and exclusion masks without modifying the input batch."""
        targets = shifted if shifted is not None else labels
        if targets is None or targets.shape != input_ids.shape:
            raise ValueError("MTP labels must match input_ids")
        if loss_mask is not None:
            if loss_mask.shape != targets.shape:
                raise ValueError("loss_mask must match labels")
            targets = targets.masked_fill(~loss_mask.bool(), -100)
        if padding is not None:
            if padding.shape != targets.shape:
                raise ValueError("The local padding mask must match labels")
            targets = targets.masked_fill(~padding.bool(), -100)
        if shifted is None:
            targets = shift_mtp_tokens(targets, 1, group=self.mtp_cp_group, pad_value=-100)
        if padding is not None:
            targets = targets.masked_fill(~padding.bool(), -100)
        return targets


register_custom_model("DeepseekV3MTPForCausalLM", __name__, "DeepseekV3MTPForCausalLM")


def build_model(config: Mapping[str, Any] | DeepseekV3Config, *, mtp_num_layers: int | None = None,
                mtp_loss_factor: float = 0.3, mtp_loss_normalization: str = "sequence",
                **build_options: Any) -> DeepseekV3ForCausalLM:
    """Build a DeepSeek-V3 model from YAML, including optional MTP before FSDP.

    Args:
        config: HF DeepSeek-V3 model dimensions, or an existing HF config.
        mtp_num_layers: Number of extra MTP layers; None/0 preserves the native LM.
        mtp_loss_factor: Total weight shared across MTP depths.
        mtp_loss_normalization: Per-depth sequence or valid-token denominator.
        **build_options: Normal HyperAutoModel construction options from Trainer.

    Note:
        CP uses the MLA CP recipe on both trunk and MTP attention.
        MTPTextParallelBatch supplies independent loss denominators to Trainer.
    """
    _validate_options(mtp_num_layers, mtp_loss_factor, mtp_loss_normalization)
    resolved = DeepseekV3Config(**dict(config)) if isinstance(config, Mapping) else copy.deepcopy(config)
    if not isinstance(resolved, DeepseekV3Config):
        raise ValueError("build_model requires a DeepseekV3Config or mapping")
    resolved.architectures = ["DeepseekV3MTPForCausalLM" if mtp_num_layers else "DeepseekV3ForCausalLM"]
    resolved.mtp_num_layers = mtp_num_layers
    resolved.mtp_loss_factor = mtp_loss_factor
    resolved.mtp_loss_normalization = mtp_loss_normalization
    mesh = None
    if mtp_num_layers:
        setup = build_options.get("distributed_setup")
        mesh = getattr(setup, "mesh_context", None)
        if getattr(mesh, "loss_parallel", False):
            raise ValueError("MTP requires loss_parallel=false to use the shared full-vocabulary loss")
        if getattr(mesh, "pp_size", 1) > 1:
            raise ValueError("Use the explicit MTP pipeline stage builder; a monolithic model cannot select PP stages")
        if setup is not None:
            setup = copy.copy(setup)
            setup.plan_overrides = dict(setup.plan_overrides or {})
            placement = {"tp": Shard(1) if getattr(mesh, "sequence_parallel", False) else Replicate(),
                         "cp": Shard(1), "ep": Replicate()}
            setup.plan_overrides.setdefault("mtp.layers.*.eh_proj", ModuleShardingSpec(
                params={"weight": {"tp": Replicate(), "cp": Replicate(), "ep": Replicate()}},
                in_src={"input": placement}, in_dst={"input": placement},
                out_src={"output": placement}, out_dst={"output": placement},
            ))
            build_options["distributed_setup"] = setup
        if build_options.get("codegen") or build_options.get("attn_implementation", "sdpa") not in ("eager", "sdpa"):
            raise ValueError("MTP training currently requires native eager/SDPA attention")
    model = HyperAutoModelForCausalLM.from_config(resolved, **build_options)
    if mtp_num_layers and mesh is not None:
        model.mtp_cp_size = mesh.cp_size
        model.mtp_cp_group = mesh.cp_mesh.get_group() if mesh.cp_size > 1 else None
    return model
