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
"""HF V3.2 skeleton with reference MTP and first-loss evaluation semantics."""

import numpy as np
import torch
import torch.distributed as dist
from transformers import DeepseekV32ForCausalLM


def shift_left(value: torch.Tensor) -> torch.Tensor:
    """Shift global tokens, labels or masks without wrapping the final token."""
    return torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), dim=1)


def reference_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Evaluate reference masked NLL over Hyper's vocabulary-sharded logits.

    This evaluator is forward-only. Training backward semantics are deliberately
    not claimed until the subsequent gradient-alignment stage.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("First-loss evaluator must run under torch.no_grad()")
    values = logits.float()
    maximum = values.amax(-1)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    stable = values - maximum.unsqueeze(-1)
    denominator = stable.exp().sum(-1)
    dist.all_reduce(denominator)
    local_labels = labels - dist.get_rank() * logits.shape[-1]
    owned = (local_labels >= 0) & (local_labels < logits.shape[-1])
    indices = local_labels.clamp(0, logits.shape[-1] - 1)
    selected = stable.gather(-1, indices.unsqueeze(-1)).squeeze(-1) * owned
    dist.all_reduce(selected)
    return ((denominator.log() - selected) * mask).sum() / (mask.sum() + 1e-8)


class DeepseekV32SFTForCausalLM(DeepseekV32ForCausalLM):
    """Reuse HF construction and children; override the reference training orchestration."""

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor,
                loss_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate LM, MTP and sequence-balancing loss without a dense causal mask."""
        cfg = self.reference_config
        if input_ids.shape != (1, cfg["seq_length"]):
            raise ValueError("Expected a complete batch-one 256K input sequence")
        if labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("Tokens, pre-shifted labels and loss mask must have identical shapes")
        dim = cfg["qk_rope_head_dim"]
        inverse = 1.0 / (cfg["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        inverse = torch.from_numpy(inverse.astype(np.float32)).to(input_ids.device)
        frequency = torch.arange(cfg["seq_length"], device=input_ids.device, dtype=torch.float32)[:, None] * inverse
        frequency = torch.cat((frequency, frequency), dim=-1).unsqueeze(0)
        attention_kwargs = {"position_embeddings": (frequency.cos(), frequency.sin()),
                            "actual_seq_len": (cfg["seq_length"],)}
        hidden = self.model.embed_tokens(input_ids).to(torch.bfloat16)
        auxiliary = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden, **attention_kwargs)
            if hasattr(layer.mlp, "auxiliary_loss"):
                auxiliary = auxiliary + layer.mlp.auxiliary_loss
        hidden = hidden.float()
        lm_loss = reference_loss(self.lm_head(self.model.norm(hidden).to(torch.bfloat16)), labels, loss_mask)
        mtp_loss = torch.zeros_like(lm_loss)
        for layer in self.mtp.layers:
            input_ids, labels, loss_mask = shift_left(input_ids), shift_left(labels), shift_left(loss_mask)
            embedding = self.model.embed_tokens(input_ids).to(torch.bfloat16)
            combined = torch.cat((layer.hnorm(hidden).to(torch.bfloat16),
                                  layer.enorm(embedding).to(torch.bfloat16)), dim=-1)
            hidden = layer.transformer_layer(layer.eh_proj(combined), **attention_kwargs)
            hidden = layer.final_layernorm(hidden)
            auxiliary = auxiliary + layer.transformer_layer.mlp.auxiliary_loss
            mtp_loss = mtp_loss + reference_loss(self.lm_head(hidden), labels, loss_mask) * (
                cfg["mtp_loss_factor"] / len(self.mtp.layers))
        return {"loss": lm_loss + mtp_loss + auxiliary, "lm_loss": lm_loss,
                "mtp_loss": mtp_loss, "aux_loss": auxiliary}
