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
"""JT entry-point initialization and strict reference-loading contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.jt_deepseek_v3.adapter import jt_builder
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config,
    JTDeepseekV3ForCausalLM,
    JTDeepseekV3MLAAttention,
)
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.common.jt_config import jt_test_config
from tests.common.mark_utils import arg_mark


def _cpu_grouped_swiglu(inputs: torch.Tensor, gate_up_weight: torch.Tensor,
                       down_weight: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Replace only the NPU kernel boundary with differentiable CPU expert math."""
    outputs = []
    for index, tokens in enumerate(inputs.split(counts.tolist())):
        gate, up = F.linear(tokens, gate_up_weight[index]).chunk(2, dim=-1)
        outputs.append(F.linear(F.silu(gate) * up, down_weight[index]))
    return torch.cat(outputs)


class TestJTBuilder(unittest.TestCase):
    """Use real model construction and initialization with CPU-only infrastructure."""

    def setUp(self) -> None:
        """Isolate random state and replace accelerator and distributed setup on CPU."""
        self.config = {
            "vocab_size": 32, "hidden_size": 64, "intermediate_size": 128, "moe_intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 32, "num_key_value_heads": 32,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1,
            "q_lora_rank": 8, "kv_lora_rank": 8, "qk_rope_head_dim": 4, "qk_nope_head_dim": 4, "v_head_dim": 4,
            "mlp_layer_types": ["dense", "sparse"], "max_position_embeddings": 32, "tie_word_embeddings": False,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "architectures": ["JTDeepseekV3ForCausalLM"], "rope_interleave": True,
            "num_nextn_predict_layers": 1, "use_pad_tokens": True, "norm_topk_prob": True,
            "routed_scaling_factor": 1.0, "moe_aux_loss_coeff": 0.01, "mtp_loss_factor": 0.3,
        }
        mesh = SimpleNamespace(
            device_mesh=SimpleNamespace(device_type="cpu"), dp_cp_mesh=None,
            tp_size=1, cp_size=1, ep_size=1, pp_size=1, sequence_parallel=False, loss_parallel=False,
        )
        self.setup = DistributedSetup(mesh_context=mesh)
        self.enterContext(torch.random.fork_rng(devices=[]))
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        self.enterContext(patch.object(jt_builder.torch_npu.npu, "set_compile_mode"))
        self.enterContext(patch.object(jt_builder, "get_device_id", return_value=0))
        self.enterContext(patch.object(jt_builder, "instantiate_infrastructure", return_value=(None, None)))

    def build(self, seed: int = 42, **options: Any) -> JTDeepseekV3ForCausalLM:
        """Build through the production entry with a caller-owned seed."""
        torch.manual_seed(seed)
        return jt_builder.build_jt_model(config=self.config, distributed_setup=self.setup, **options)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_optional_reference_uses_native_initialization(self) -> None:
        """Feature: Optional reference weights.

        Description: Build a 32-head model with trunk/MTP MLA replacements, omitted/null references and two seeds.
        Expectation: Initialization is reproducible, preserves native constants and buffers, and never reads weights.
        """
        torch.manual_seed(42)
        expected = JTDeepseekV3ForCausalLM(JTDeepseekV3Config(**self.config))
        self.setup.module_replacements = entries_to_module_replacements(jt_test_config().plan_overrides)
        with patch.object(jt_builder.np, "load", side_effect=AssertionError("Unexpected weight read")):
            omitted = self.build()
            explicit_null = self.build(reference_weights=None)
            changed_seed = self.build(seed=43)
        self.assertEqual(omitted.model.layers[0].self_attn.o_proj.weight.shape, (64, 128))
        torch.testing.assert_close(omitted.state_dict(), explicit_null.state_dict(), rtol=0, atol=0)
        for name, value in omitted.state_dict().items():
            self.assertTrue(torch.isfinite(value).all(), f"Non-finite initialized tensor: {name}")
            if "norm" in name or name.endswith("e_score_correction_bias"):
                torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
        for name, value in expected.named_buffers():
            torch.testing.assert_close(dict(omitted.named_buffers())[name], value, rtol=0, atol=0)
        attention = [module for module in omitted.modules() if isinstance(module, JTDeepseekV3MLAAttention)]
        self.assertEqual(len(attention), 3)
        for module in attention:
            self.assertGreater(torch.count_nonzero(module.linear_qkv.weight).item(), 0)
        self.assertFalse(torch.equal(omitted.model.embed_tokens.weight, changed_seed.model.embed_tokens.weight))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_random_model_trains(self) -> None:
        """Feature: Training from scratch.

        Description: Run forward, backward and an optimizer step with only the NPU grouped kernel replaced on CPU.
        Expectation: All three losses and gradients are finite and the embedding changes.
        """
        model = self.build()
        tokens = torch.arange(8).unsqueeze(0)
        with patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu",
                   side_effect=_cpu_grouped_swiglu):
            output = model(tokens, (tokens + 1) % self.config["vocab_size"])
        self.assertEqual(set(output.loss), {"foundation_loss/lm", "foundation_loss/mtp", "foundation_loss/aux"})
        loss = sum(output.loss.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ("model.embed_tokens.weight", "model.layers.1.mlp.gate.weight", "mtp.layers.0.eh_proj.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, f"Missing gradient: {name}")
            self.assertTrue(torch.isfinite(gradient).all(), f"Non-finite gradient: {name}")
            self.assertGreater(gradient.abs().sum().item(), 0, f"Zero gradient: {name}")
        before = model.model.embed_tokens.weight.detach().clone()
        torch.optim.SGD(model.parameters(), lr=0.01, foreach=False).step()
        self.assertFalse(torch.equal(before, model.model.embed_tokens.weight))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_matching_reference_loads_exact_values(self) -> None:
        """Feature: Explicit reference loading.

        Description: Load a matching archive through both string and Path arguments.
        Expectation: Every tensor equals the archive; initialization is never used as a fallback.
        """
        expected = JTDeepseekV3ForCausalLM(JTDeepseekV3Config(**self.config)).state_dict()
        with TemporaryDirectory() as directory, patch.object(
                jt_builder, "_initialize_model_weights", create=True,
                side_effect=AssertionError("Unexpected initialization")):
            np.savez(Path(directory) / "model.npz", **{name: value.numpy() for name, value in expected.items()})
            for reference in (directory, Path(directory)):
                with self.subTest(reference_type=type(reference).__name__):
                    model = self.build(reference_weights=reference)
                    for name, value in expected.items():
                        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_reference_does_not_fall_back(self) -> None:
        """Feature: Strict reference validation.

        Description: Supply a missing file or incompatible names, shapes and dtypes.
        Expectation: Explicit invalid references fail instead of silently training from scratch.
        """
        expected = JTDeepseekV3ForCausalLM(JTDeepseekV3Config(**self.config)).state_dict()
        arrays = {name: value.numpy() for name, value in expected.items()}
        projection = "model.layers.0.self_attn.o_proj.weight"
        cases = (
            ("State coverage mismatch", {name: value for name, value in arrays.items() if name != projection}),
            ("Shape mismatch", {**arrays, projection: arrays[projection][:, :32]}),
            ("Reference dtype mismatch", {**arrays, projection: arrays[projection].astype(np.float64)}),
        )
        with TemporaryDirectory() as directory, patch.object(
                jt_builder, "_initialize_model_weights", create=True,
                side_effect=AssertionError("Unexpected fallback")):
            with self.assertRaises(FileNotFoundError):
                self.build(reference_weights=directory)
            for message, invalid in cases:
                with self.subTest(error=message):
                    np.savez(Path(directory) / "model.npz", **invalid)
                    with self.assertRaisesRegex(ValueError, message):
                        self.build(reference_weights=directory)
