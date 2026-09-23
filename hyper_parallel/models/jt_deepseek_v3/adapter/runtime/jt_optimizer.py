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
"""Reference Muon/AdamW partition, projection-wise NS and post-update QK clipping."""

# The HF model adapter runs in the Torch backend.
# pylint: disable=forbidden-backend-import
from __future__ import annotations
import math
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn import functional
from torch import nn

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3MLAAttention


def _reference_fused_matrix_norm(values: torch.Tensor) -> torch.Tensor:
    """Reproduce the captured large-matrix DVM reduction layouts on 910B.

    These two layouts cover the reduced recipe's fused Q-down and MTP merge
    projections. Other sizes retain the ordinary norm; this is not a claim of
    universal graph-compiler equivalence for arbitrary model shapes.
    """
    if values.numel() not in (32768, 131072):
        return values.norm(dim=(-2, -1), keepdim=True)
    tile_count = 40
    tile_size = math.ceil(values.numel() / tile_count / 8) * 8
    square = functional.pad(values.square().flatten(), (0, tile_size * tile_count - values.numel()))
    square = square.reshape(tile_count, tile_size)
    partial = square.sum(-1)
    # Reassociating the tile join changes the norm at BF16 rounding boundaries.
    total = torch.zeros((), device=values.device, dtype=torch.float32)
    for index in range(tile_count):
        total = total + partial[index]
    return total.sqrt().reshape(1, 1)


def reference_newton_schulz(tensor: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Match the reference 2D/3D NS casts, scalar precision and matrix layouts.

    Args:
        tensor: Logical projection, with its reference pre-normalization dtype.
        steps: Number of polynomial iterations.
    """
    transposed = tensor.shape[-2] > tensor.shape[-1]
    values = tensor.mT if transposed else tensor
    values = values.float()
    if tensor.dtype == torch.float32 and tensor.ndim == 2:
        norm = _reference_fused_matrix_norm(values)
    else:
        norm = values.norm(dim=(-2, -1), keepdim=True)
    epsilon = torch.tensor(1e-7, device=tensor.device, dtype=torch.bfloat16)
    if tensor.ndim == 3:
        denominator = (norm + epsilon.float()).to(torch.bfloat16).float()
    else:
        # The reference graph promotes a BF16 scalar when fusing the FP32 norm.
        denominator = norm + epsilon.float()
    values = (values / denominator).to(torch.bfloat16)
    coeff_a, coeff_b, coeff_c = [
        torch.tensor(value, device=tensor.device, dtype=torch.bfloat16)
        for value in (3.4445, -4.7750, 2.0315)]
    for _ in range(steps):
        # Materialized transposes preserve the reference GEMM accumulation path.
        gram = values.contiguous() @ values.mT.contiguous()
        if tensor.ndim == 2:
            second = ((coeff_c * gram) @ gram).float()
        else:
            second = coeff_c.float() * (gram @ gram).float()
        polynomial = (coeff_b.float() * gram.float() + second).to(torch.bfloat16)
        values = (coeff_a.float() * values.float() +
                  (polynomial.contiguous() @ values.contiguous()).float()).to(torch.bfloat16)
    return values.mT if transposed else values


def split_projection(name: str, tensor: torch.Tensor, cfg: SimpleNamespace) -> tuple[list[torch.Tensor], Callable]:
    """Split logical gate/query/key/value matrices in the reference storage order.

    Args:
        name: Reference parameter name identifying its projection layout.
        tensor: Input tensor to transform.
        cfg: Validated reference model configuration.
    """
    shape = tensor.shape
    if name.endswith("mlp.experts.weight1"):
        expanded = tensor.reshape(-1, cfg.hidden_size, 2 * cfg.moe_intermediate_size)
        return list(expanded.chunk(2, -1)), lambda parts: torch.cat(parts, -1).reshape(shape)
    if name.endswith("mlp.experts.weight2"):
        expanded = tensor.reshape(-1, cfg.moe_intermediate_size, cfg.hidden_size)
        return [expanded], lambda parts: parts[0].reshape(shape)
    if name.endswith("linear_fc1.weight"):
        return [tensor[::2], tensor[1::2]], lambda parts: torch.stack(parts, dim=1).reshape(shape)
    if name.endswith("linear_q_up_proj.weight"):
        dims = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim)
    elif name.endswith("linear_kv_up_proj.weight"):
        dims = (cfg.qk_nope_head_dim, cfg.v_head_dim)
    elif name.endswith("linear_kv_down_proj.weight"):
        dims = (cfg.kv_lora_rank, cfg.qk_rope_head_dim)
    else:
        return [tensor], lambda parts: parts[0]
    expanded = tensor.reshape(-1, sum(dims), shape[-1])
    pieces = [piece.reshape(-1, shape[-1]) for piece in expanded.split(dims, dim=1)]

    def _merge(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([part.reshape(-1, dim, shape[-1]) for part, dim in zip(parts, dims)], dim=1).reshape(shape)

    return pieces, _merge


class ReferenceMuon(torch.optim.Optimizer):
    """Match reference Muon defaults and static-graph numerical boundaries."""

    def __init__(self, model: nn.Module, config: SimpleNamespace, parallel: SimpleNamespace) -> None:
        """Route vocabulary/norm parameters to AdamW and matrices to Muon."""
        self.named = list(model.named_parameters())
        self.last_max_logits = {}
        self.config = config
        self.parallel = parallel
        self.model = model
        self.step_number = 0
        defaults = {"lr": config.schedule["learning_rate"]}
        super().__init__([value for _, value in self.named], defaults)

    def learning_rate(self) -> float:
        """Evaluate the reference zero-based cosine schedule, including warmup."""
        schedule = self.config.schedule
        device = self.named[0][1].device
        step = torch.tensor(self.step_number, device=device, dtype=torch.float32)
        start = torch.tensor(schedule["learning_rate"], device=device, dtype=torch.float32)
        end = torch.tensor(schedule["lr_end"], device=device, dtype=torch.float32)
        warmup = schedule["warmup_steps"]
        if self.step_number < warmup:
            return (start * step / warmup).item()
        if self.step_number >= schedule["total_steps"]:
            return end.item()
        progress = (step - warmup) / (schedule["total_steps"] - warmup)
        percent = .5 * (1 + torch.cos(progress * math.pi))
        return (end + (start - end) * percent).item()

    def _muon_update(self, name: str, value: nn.Parameter, gradient: torch.Tensor,
                     rate: float = 1.0) -> torch.Tensor:
        state = self.state[value]
        momentum = torch.tensor(self.config.optimizer.get("momentum", .95), device=value.device, dtype=torch.float32)
        buffer = state.setdefault("momentum", torch.zeros_like(value))
        buffer.copy_(torch.addcmul(gradient, buffer, momentum))
        update = torch.addcmul(gradient, buffer, momentum) if self.config.optimizer.get("nesterov", True) else buffer
        pieces, merge = split_projection(name, update, self.config)
        shard = getattr(value, "jt_shard_dim", -1)
        gather = shard >= 0 and not getattr(value, "jt_expert", False) and self.parallel.world > 1
        # Only an unsplit, untransposed local matrix can fuse the initial cast
        # into normalization; reshape/split/communication materialize BF16 first.
        fuse_normalization = (len(pieces) == 1 and pieces[0].ndim == 2 and
                              not gather and pieces[0].shape[-2] <= pieces[0].shape[-1])
        updates = []
        last_shape = pieces[-1].shape[-2:]
        for piece in pieces:
            if not fuse_normalization:
                piece = piece.to(torch.bfloat16)
            if gather:
                shards = [torch.empty_like(piece) for _ in range(self.parallel.world)]
                dist.all_gather(shards, piece.contiguous(), group=self.parallel.group)
                piece = torch.cat(shards, dim=shard)
            last_shape = piece.shape[-2:]
            result = reference_newton_schulz(piece)
            if gather:
                result = result.chunk(self.parallel.world, dim=shard)[self.parallel.rank]
            updates.append(result)
        ratio = torch.tensor(max(last_shape), device=value.device, dtype=torch.float32).sqrt()
        ratio = ratio * self.config.optimizer["matched_adamw_rms"]
        adjusted_rate = torch.tensor(rate, device=value.device, dtype=torch.float32) * ratio
        return merge(updates).float() * adjusted_rate

    def _adamw_update(self, value: nn.Parameter, gradient: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        state = self.state[value]
        first = state.setdefault("exp_avg", torch.zeros_like(value))
        second = state.setdefault("exp_avg_sq", torch.zeros_like(value))
        beta1, beta2 = [torch.tensor(beta, device=value.device, dtype=torch.float32)
                        for beta in self.config.optimizer["adamw_betas"]]
        first.copy_(first * beta1 + gradient * (1 - beta1))
        second.copy_(torch.addcmul(second * beta2, gradient, gradient, value=(1 - beta2).item()))
        step = self.step_number + 1
        denominator = (second / (1-beta2**step)).sqrt() + self.config.optimizer["adamw_eps"]
        step_size = torch.tensor(rate, device=value.device, dtype=torch.float32) / (1 - beta1**step)
        return (first / denominator) * step_size

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float:
        """Apply one reference update and return its learning rate.

        Args:
            closure: Unsupported optimizer closure; must be None.
        """
        if closure is not None:
            raise ValueError("Closures are not supported by this alignment recipe")
        rate = self.learning_rate()
        for name, value in self.named:
            if value.grad is None:
                raise RuntimeError(f"Missing gradient for reference parameter {name}")
            gradient = value.grad.float()
            use_muon = value.ndim >= 2 and "word_embeddings" not in name and "output_layer" not in name
            if use_muon:
                update = self._muon_update(name, value, gradient, rate)
            else:
                update = self._adamw_update(value, gradient, rate)
            decay = 0.0 if value.ndim == 1 or name.endswith(".bias") else self.config.optimizer["weight_decay"]
            value.mul_(1 - rate * decay).sub_(update)
        self.model.write_weights()
        self._clip_qk()
        self.model.read_weights()
        self.step_number += 1
        return rate

    def _clip_qk(self) -> None:
        threshold = self.config.optimizer["qk_clip_threshold"]
        self.last_max_logits = {}
        for name, module in self.model.hf_model.named_modules():
            if not isinstance(module, JTDeepseekV3MLAAttention):
                continue
            maximum = module.max_logits_val
            self.last_max_logits[name] = maximum.detach().amax()
            scale = torch.where(maximum >= threshold, threshold / maximum.clamp_min(threshold),
                                torch.ones_like(maximum))
            cfg = self.config
            query = module.q_b_proj.weight.view(module.num_heads, module.qk_nope_head_dim + module.qk_rope_head_dim, -1)
            query[:, :cfg.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
            query[:, cfg.qk_nope_head_dim:].mul_(scale[:, None, None])
            key_value = module.kv_b_proj.weight.view(module.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, -1)
            key_value[:, :cfg.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
            maximum.zero_()
