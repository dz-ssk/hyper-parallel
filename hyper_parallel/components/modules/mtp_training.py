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
"""Attach public MTP training to decoder-only models before parallel wrapping.

The structural contract is get_decoder().{layers,norm,rotary_emb} plus the
standard get_input_embeddings/get_output_embeddings methods. No model family
is imported or selected here. Checkpoints retain the original parameter names.
"""
# Native Torch is required by the current framework contract.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from types import MethodType
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from hyper_parallel.components.modules.mtp import MTPLayer, build_mtp
from hyper_parallel.distributed.mtp import shift_mtp_tokens, prepare_mtp_inputs
from hyper_parallel.models.materialization import register_rebuildable_buffer, register_materialized_state_hook


def _rope_buffer(rotary: torch.nn.Module, context: Any) -> torch.Tensor:
    """Restore deterministic HF rotary frequencies after meta materialization."""
    with torch.device(context.device):
        return type(rotary)(rotary.config).inv_freq


def _rope_alias(rotary: torch.nn.Module, _context: Any) -> None:
    """Refresh the non-registered HF alias after its registered buffer changes."""
    rotary.original_inv_freq = rotary.inv_freq


def attach_mtp(model: torch.nn.Module, mtp_num_layers: int | None = None, *,
               mtp_loss_factor: float = 0.3, mtp_loss_normalization: str = "sequence",
               mtp_boolean_mask: bool = False, mesh: Any = None,
               layer_factory: Callable[[int], torch.nn.Module] | None = None) -> torch.nn.Module:
    """Create MTP in the public build lifecycle, before sharding and optimizer setup.

    The default uses independently initialized copies of the last trunk block;
    a custom layer_factory may supply MTPLayers instead. The trunk must be
    a homogeneous causal decoder with layers, norm and rotary_emb. Unsupported
    structures fail here, rather than silently enabling a different forward.
    Zero/None is a no-op. Inference continues using the original model forward.
    """
    if mtp_num_layers is not None and (
            isinstance(mtp_num_layers, bool) or not isinstance(mtp_num_layers, int) or mtp_num_layers < 0):
        raise ValueError("mtp_num_layers must be None or a nonnegative integer")
    if not mtp_num_layers:
        if mtp_num_layers == 0:
            model.config.mtp_num_layers = 0
        return model
    if hasattr(model, "mtp"):
        raise ValueError("MTP is already attached")
    if not isinstance(mtp_loss_factor, (float, int)) or isinstance(mtp_loss_factor, bool) or (
            not math.isfinite(mtp_loss_factor) or mtp_loss_factor < 0):
        raise ValueError("mtp_loss_factor must be finite and nonnegative")
    if mtp_loss_normalization not in ("sequence", "valid_tokens"):
        raise ValueError("mtp_loss_normalization must be sequence or valid_tokens")
    if not isinstance(mtp_boolean_mask, bool):
        raise ValueError("mtp_boolean_mask must be a bool")
    if not all(callable(getattr(model, name, None)) for name in (
            "get_decoder", "get_input_embeddings", "get_output_embeddings", "_init_weights")):
        raise ValueError("MTP requires the standard causal-LM decoder, embedding, head and initialization interfaces")
    decoder = model.get_decoder()
    if not isinstance(getattr(decoder, "layers", None), torch.nn.ModuleList) or not decoder.layers or (
            not hasattr(decoder, "norm") or not hasattr(decoder, "rotary_emb")):
        raise ValueError("MTP requires a decoder with nonempty layers, final norm and rotary_emb")
    if getattr(mesh, "loss_parallel", False):
        raise ValueError("MTP currently requires loss_parallel=false")
    if getattr(mesh, "pp_size", 1) > 1:
        raise ValueError("PP requires the explicit public MTP pipeline stage interface")
    config = model.config
    factory = layer_factory or (lambda _k: MTPLayer(
        config.hidden_size, deepcopy(decoder.layers[-1]),
        rms_norm_eps=config.rms_norm_eps))
    model.mtp = build_mtp(mtp_num_layers, layer_factory=factory)
    # Native initialization also runs again after meta materialization, where
    # required. Norms declare their own reset contract in the public component.
    model.mtp.apply(model._init_weights)
    embedding = model.get_input_embeddings().weight
    model.mtp.to(device=embedding.device, dtype=embedding.dtype)
    rotary = decoder.rotary_emb
    register_rebuildable_buffer(rotary, "inv_freq", factory=partial(_rope_buffer, rotary))
    register_materialized_state_hook(rotary, _rope_alias)
    model.mtp_pipeline_first = model.mtp_pipeline_last = True
    model.mtp_cp_size = getattr(mesh, "cp_size", 1)
    model.mtp_cp_group = mesh.cp_mesh.get_group() if model.mtp_cp_size > 1 else None
    for name, value in (("mtp_num_layers", mtp_num_layers), ("mtp_loss_factor", mtp_loss_factor),
                        ("mtp_loss_normalization", mtp_loss_normalization), ("mtp_boolean_mask", mtp_boolean_mask)):
        setattr(config, name, value)
        setattr(model, name, value)
    model._mtp_native_forward = model.forward
    model.forward = MethodType(_forward, model)
    return model


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
    mask = allowed if self.mtp_boolean_mask else hidden.new_zeros(allowed.shape).masked_fill(
        ~allowed, torch.finfo(hidden.dtype).min)
    return {"attention_mask": mask, "position_ids": positions,
            "position_embeddings": self.get_decoder().rotary_emb(hidden, positions), "use_cache": False}


def _forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
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
        return self._mtp_native_forward(input_ids=input_ids, attention_mask=attention_mask,
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
    targets = _targets(self, input_ids, labels, shift_labels, loss_mask, padding)
    positions = (torch.arange(input_ids.shape[1], device=input_ids.device)[None] + rank * input_ids.shape[1]
                 if position_ids is None else position_ids)
    if self.mtp_pipeline_first:
        if hidden_states is not None:
            raise ValueError("The first pipeline stage takes input_ids, not received activations")
        hidden = self.get_input_embeddings()(input_ids)
    else:
        if hidden_states is None:
            raise ValueError("A non-first pipeline stage requires hidden_states from its predecessor")
        hidden = hidden_states
    decoder_inputs = _decoder_inputs(self, hidden, positions, attention_mask)
    layers = self.get_decoder().layers
    layers = layers.values() if isinstance(layers, torch.nn.ModuleDict) else layers
    for layer in layers:
        hidden = layer(hidden, **decoder_inputs)
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
    if not self.mtp_pipeline_last:
        return hidden
    supplied = (mtp_input_ids, mtp_position_ids, mtp_targets)
    if all(value is None for value in supplied):
        prepared = prepare_mtp_inputs(input_ids, positions, targets, len(self.mtp.layers),
                                      cp_group=self.mtp_cp_group, attention_mask=attention_mask)
        mtp_input_ids, mtp_position_ids, mtp_targets = prepared.values()
    elif any(value is None or len(value) != len(self.mtp.layers) for value in supplied):
        raise ValueError("All prepared MTP fields must contain exactly mtp_num_layers entries")
    logits = self.get_output_embeddings()(self.get_decoder().norm(hidden))
    lm_sum = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1),
                             ignore_index=-100, reduction="sum")
    losses = {"foundation_loss/lm_loss": lm_sum / targets.ne(-100).sum().clamp_min(1)}
    predictions = self.mtp.forward_aligned(
        hidden, [self.get_input_embeddings()(ids) for ids in mtp_input_ids], head=self.get_output_embeddings(),
        decoder_kwargs=[_decoder_inputs(self, hidden, pos, attention_mask, depth)
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
