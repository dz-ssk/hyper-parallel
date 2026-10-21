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
"""Complete standalone model construction and optional acceleration boundaries."""
# pylint: disable=protected-access

import copy
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config,
    JTDeepseekV3ForCausalLM, JTDeepseekV3Decoder, JTDeepseekV3Gate, JTDeepseekV3MoE,
    JTDeepseekV3Attention, JTDeepseekV3MLAAttention, JTDeepseekV3RotaryEmbedding,
)
from hyper_parallel.components.losses import calculate_seq_aux_loss
from hyper_parallel.components.modules.mtp import MultiTokenPredictionLayer
from hyper_parallel.core.activation_memory import checkpoint_wrapper
from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.replacement import compile_module_replacements, apply_module_replacements
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _load_reference_state
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.common.jt_config import jt_test_config
from tests.common.mark_utils import arg_mark


def small_config() -> JTDeepseekV3Config:
    """Build a CPU-sized fixture without changing the production validation recipe."""
    config = JTDeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        mlp_layer_types=["dense", "sparse"], max_position_embeddings=32, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        architectures=["JTDeepseekV3ForCausalLM"],
        num_nextn_predict_layers=1, use_pad_tokens=True, norm_topk_prob=True,
        routed_scaling_factor=1.0, moe_aux_loss_coeff=0.01, mtp_loss_factor=0.3)
    config.rope_interleave = True
    return config


class TestCompleteModel(unittest.TestCase):
    """Exercise model semantics without a builder, EP adapter or replacement pass."""

    def test_auxiliary_loss_pop_preserves_router_and_input_gradients(self):
        """Consuming the loss clears its owner but preserves the differentiable objective."""
        torch.manual_seed(17)
        model = JTDeepseekV3ForCausalLM(small_config())
        moe = model.model.layers[1].mlp
        hidden = torch.randn(1, moe.padding + 8, 16, requires_grad=True)
        moe.route(hidden)
        reference = moe.auxiliary_loss
        loss = moe.pop_auxiliary_loss()
        self.assertIs(loss, reference)
        self.assertIsNone(moe.auxiliary_loss)
        parameters = (hidden, moe.gate.weight)
        expected = torch.autograd.grad(reference, parameters, retain_graph=True)
        actual = torch.autograd.grad(loss, parameters)
        for gradient, baseline in zip(actual, expected, strict=True):
            torch.testing.assert_close(gradient, baseline, rtol=0, atol=0)
            self.assertGreater(gradient.abs().sum().item(), 0)
        with self.assertRaisesRegex(RuntimeError, "No auxiliary loss"):
            moe.pop_auxiliary_loss()

    def test_sequence_auxiliary_loss_ignores_dispatch_bias_and_group_filter(self) -> None:
        """The sequence objective uses raw affinities while bias counts track actual dispatch."""
        logits = torch.tensor([[2, 1, -1, -2], [1.8, .9, -.8, -1.5], [-.5, 1.3, 2, -1]])
        scores = logits.double().sigmoid()
        # These fixed logits select {0, 1}, {0, 1}, {1, 2} without dispatch corrections.
        load = torch.tensor([2, 3, 1, 0], dtype=torch.float64) / 6
        denominator = scores.sum(-1, keepdim=True)
        projection = (scores * load).sum(-1, keepdim=True)
        coefficient = 1e-4
        expected_loss = coefficient * 4 * (load * (scores / denominator).mean(0)).sum()
        expected_gradient = coefficient * 4 / 3 * scores * (1 - scores) * (
            load * denominator - projection) / denominator.square()

        for groups in (1, 2):
            for use_padding in (False, True):
                with self.subTest(groups=groups, use_padding=use_padding):
                    config = small_config()
                    config.n_group = groups
                    config.topk_group = 1
                    config.use_pad_tokens = use_padding
                    config.moe_aux_loss_coeff = coefficient
                    module = JTDeepseekV3MoE(config)
                    with torch.no_grad():
                        module.gate.weight.zero_()
                        module.gate.weight[:, :4].copy_(torch.eye(4))
                    dispatches = []
                    loads = []
                    for bias in (torch.zeros(4), torch.tensor([-1., -1., 1., 1.])):
                        module.tokens_per_expert = None
                        with torch.no_grad():
                            module.gate.e_score_correction_bias.copy_(bias)
                        hidden = torch.zeros(1, module.padding + 3, 16, requires_grad=True)
                        with torch.no_grad():
                            hidden[0, module.padding:, :4].copy_(logits)
                        indices, weights = module.route(hidden)
                        loss = module.pop_auxiliary_loss()
                        hidden_gradient, weight_gradient = torch.autograd.grad(loss, (hidden, module.gate.weight))
                        torch.testing.assert_close(loss.double(), expected_loss, rtol=0, atol=1e-10)
                        torch.testing.assert_close(hidden_gradient[0, module.padding:, :4].double(),
                                                   expected_gradient, rtol=0, atol=1e-11)
                        expected_weight = expected_gradient.T @ hidden.detach()[0, module.padding:].double()
                        torch.testing.assert_close(weight_gradient.double(), expected_weight, rtol=0, atol=5e-11)
                        padding_gradient = hidden_gradient[0, :module.padding]
                        torch.testing.assert_close(padding_gradient, torch.zeros_like(padding_gradient), rtol=0, atol=0)
                        selected = indices[module.padding:]
                        counts = torch.bincount(selected.flatten(), minlength=4).float()
                        torch.testing.assert_close(module.tokens_per_expert, counts, rtol=0, atol=0)
                        torch.testing.assert_close(weights[:module.padding],
                                                   torch.zeros_like(weights[:module.padding]), rtol=0, atol=0)
                        uncorrected = logits.sigmoid().gather(-1, selected)
                        expected_weights = uncorrected / uncorrected.sum(-1, keepdim=True)
                        torch.testing.assert_close(weights[module.padding:], expected_weights, rtol=1e-6, atol=1e-7)
                        module.update_expert_bias(lr=.001)
                        torch.testing.assert_close(module.gate.e_score_correction_bias,
                                                   bias + (counts.mean() - counts).sign() * .001, rtol=0, atol=0)
                        dispatches.append(selected.sort(-1).values)
                        loads.append(counts)
                    self.assertFalse(torch.equal(*dispatches),
                                     f"Dispatch must respond to bias: first={dispatches[0]}, second={dispatches[1]}")
                    self.assertFalse(torch.equal(*loads),
                                     f"Bias counts must track dispatch: first={loads[0]}, second={loads[1]}")

    @patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu")
    def test_checkpoint_auxiliary_loss_consumption_matches_eager_gradients(self, grouped_swiglu):
        """Trunk and MTP replay preserve accumulated gradients without leaving loss references."""
        # Use differentiable CPU experts to exercise graph lifetime without the NPU-only kernel.
        def cpu_experts(inputs: torch.Tensor, gate_up_weight: torch.Tensor,
                        down_weight: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            """Evaluate the expert branches with differentiable CPU operations."""
            outputs = []
            for index, tokens in enumerate(inputs.split(counts.tolist())):
                gate, up = F.linear(tokens, gate_up_weight[index]).chunk(2, dim=-1)
                outputs.append(F.linear(F.silu(gate) * up, down_weight[index]))
            return torch.cat(outputs)

        grouped_swiglu.side_effect = cpu_experts
        torch.manual_seed(19)
        reference = JTDeepseekV3ForCausalLM(small_config())
        candidate = copy.deepcopy(reference)
        for index, layer in enumerate(candidate.model.layers):
            candidate.model.layers[index] = checkpoint_wrapper(layer)
        for layer in candidate.mtp.layers:
            layer.transformer_layer = checkpoint_wrapper(layer.transformer_layer)
        owners = [module for module in candidate.modules() if isinstance(module, JTDeepseekV3MoE)]
        self.assertEqual(len(owners), 2)
        for objective in ("aux", "total"):
            with self.subTest(objective=objective):
                reference.zero_grad(set_to_none=True)
                candidate.zero_grad(set_to_none=True)
                for micro_step in range(3):
                    tokens = (torch.arange(8).unsqueeze(0) + micro_step) % 32
                    expected = reference(tokens, (tokens + 1) % 32).loss
                    actual = candidate(tokens, (tokens + 1) % 32).loss
                    for name, value in expected.items():
                        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
                    self.assertTrue(all(module.auxiliary_loss is None for module in owners))
                    expected_loss = expected["foundation_loss/aux"] if objective == "aux" else sum(expected.values())
                    actual_loss = actual["foundation_loss/aux"] if objective == "aux" else sum(actual.values())
                    (expected_loss / 3).backward()
                    (actual_loss / 3).backward()
                    self.assertTrue(all(module.auxiliary_loss is None for module in owners))
                for original, replayed in zip(reference.parameters(), candidate.parameters(), strict=True):
                    if original.grad is None:
                        self.assertIsNone(replayed.grad)
                    else:
                        self.assertIsNotNone(replayed.grad)
                        torch.testing.assert_close(replayed.grad, original.grad, rtol=1e-5, atol=1e-7)
                self.assertGreater(candidate.model.embed_tokens.weight.grad.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standalone_forward_backward(self):
        """Feature: Complete model.

        Description: Construct directly while replacement compilation is forbidden.
        Expectation: Trunk, MTP and router execute and backpropagate without an EP adapter.
        """
        torch.manual_seed(11)
        with patch("hyper_parallel.models.replacement.compile_module_replacements", side_effect=AssertionError):
            model = JTDeepseekV3ForCausalLM(small_config())
        self.assertIsInstance(model.model.layers[0], JTDeepseekV3Decoder)
        self.assertIsInstance(model.model.layers[1].mlp, JTDeepseekV3MoE)
        self.assertEqual(model.model.layers[1].mlp.ep_compute, model.model.layers[1].mlp.local_routed_forward)
        self.assertIsInstance(model.mtp.layers[0], MultiTokenPredictionLayer)
        tokens = torch.arange(8).unsqueeze(0)
        output = model(tokens, (tokens + 1) % 32)
        self.assertEqual(
            set(output.loss),
            {"foundation_loss/lm", "foundation_loss/mtp", "foundation_loss/aux"},
        )
        total_loss = sum(output.loss.values())
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()
        for name in ["model.embed_tokens.weight", "model.layers.1.mlp.gate.weight", "mtp.layers.0.eh_proj.weight"]:
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standard_policy_bfloat16_forward_closes_activations(self):
        """Feature: Framework BF16 policy.

        Description: Emulate FSDP param_dtype=bfloat16 compute copies over FP32 storage; buffers stay as stored.
        Expectation: Activations stay BF16 without FSDP input casts; losses and the expert bias stay FP32. Trunk
            and MTP router logits and auxiliary-loss scores are FP32, computed from the upcast BF16 router input
            and BF16 weight copy.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        biases = [
            module.e_score_correction_bias
            for module in model.modules()
            if hasattr(module, "e_score_correction_bias")
        ]
        self.assertTrue(biases)
        self.assertTrue(all(bias.dtype == torch.float32 for bias in biases))

        activations = {}
        routers = {}

        def capture(name: str) -> Any:
            """Record a selected module output by name."""

            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                """Store the tensor output after tuple unwrapping."""
                output = output[0] if isinstance(output, tuple) else output
                activations[name] = output

            return hook

        def capture_router(name: str) -> Any:
            """Record a router input and its logits by router name."""

            def hook(_module: Any, inputs: Any, output: Any) -> None:
                """Store the router input and the returned logits."""
                routers[name] = (inputs[0], output)

            return hook

        gates = {name: module for name, module in model.named_modules() if isinstance(module, JTDeepseekV3Gate)}
        handles = [
            model.model.layers[0].self_attn.register_forward_hook(capture("attention")),
            model.model.layers[1].mlp.register_forward_hook(capture("moe")),
            model.model.layers[1].register_forward_hook(capture("decoder")),
            *(gate.register_forward_hook(capture_router(name)) for name, gate in gates.items()),
        ]
        try:
            tokens = torch.arange(8).unsqueeze(0)
            with torch.no_grad(), patch(f"{JTDeepseekV3Gate.__module__}.calculate_seq_aux_loss",
                                        wraps=calculate_seq_aux_loss) as auxiliary_loss:
                output = model(tokens, (tokens + 1) % 32)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual({value.dtype for value in activations.values()}, {torch.bfloat16})
        self.assertTrue(all(value.dtype == torch.float32 for value in output.loss.values()))
        self.assertEqual(list(routers), ["model.layers.1.mlp.gate", "mtp.layers.0.transformer_layer.mlp.gate"])
        scores = [call.args[0] for call in auxiliary_loss.call_args_list]
        for (name, (router_input, logits)), router_scores in zip(routers.items(), scores, strict=True):
            with self.subTest(router=name):
                weight = gates[name].weight
                self.assertEqual((router_input.dtype, weight.dtype), (torch.bfloat16, torch.bfloat16))
                self.assertEqual(logits.dtype, torch.float32)
                expected = F.linear(  # pylint: disable=not-callable
                    router_input.reshape(-1, router_input.shape[-1]).float(), weight.float())
                self.assertTrue(torch.equal(logits, expected), f"{name} logits differ from the FP32 product")
                self.assertEqual(router_scores.dtype, torch.float32)
                self.assertTrue(torch.equal(router_scores, logits.sigmoid()),
                                f"{name} auxiliary-loss scores differ from the router logits")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standard_policy_bfloat16_rotates_rope_in_fp32(self):
        """Feature: RoPE precision under the framework BF16 policy.

        Description: Emulate FSDP param_dtype=bfloat16 compute copies; capture every trunk and MTP rotary call.
        Expectation: The position tables are FP32, and every rotated output equals the FP32 rotation of the
            BF16 channel pairs rounded to BF16 once, as MindFormers rotates with rotary_dtype=float32.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        calls = []

        def capture(_module: Any, inputs: Any, output: torch.Tensor) -> None:
            """Store the rotary channels, the position tables and the rotated output."""
            calls.append((*inputs, output))

        handles = [module.register_forward_hook(capture) for module in model.modules()
                   if isinstance(module, JTDeepseekV3RotaryEmbedding)]
        try:
            tokens = torch.arange(8).unsqueeze(0)
            with torch.no_grad():
                model(tokens, (tokens + 1) % 32)
        finally:
            for handle in handles:
                handle.remove()
        # Query and key rotations in both trunk layers and in the MTP layer.
        self.assertEqual(len(calls), 6)
        for values, cos, sin, output in calls:
            self.assertEqual((values.dtype, cos.dtype, sin.dtype, output.dtype),
                             (torch.bfloat16, torch.float32, torch.float32, torch.bfloat16))
            even, odd = values[..., ::2].float(), values[..., 1::2].float()
            half = even.shape[-1]
            cos, sin = cos[..., :half].unsqueeze(1), sin[..., :half].unsqueeze(1)
            expected = torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
            self.assertTrue(torch.equal(output, expected.to(torch.bfloat16)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replacement_selects_attention(self):
        """Feature: Acceleration boundary.

        Description: Apply the optional high-performance attention replacement.
        Expectation: Attention changes while decoder, MLP, norm and MTP weights survive.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        previous = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3Attention) for m in previous.values()), 3)
        original_q = model.model.layers[0].self_attn.q_a_proj.weight.detach().clone()
        original_kv = model.model.layers[0].self_attn.kv_a_proj_with_mqa.weight.detach().clone()
        recipe = jt_test_config()
        rules = entries_to_module_replacements(recipe.plan_overrides)
        self.assertEqual(len(rules), 1)
        plan = compile_module_replacements(model, rules)
        apply_module_replacements(model, plan, weights_mapping=[])
        self.assertTrue(torch.equal(model.model.layers[0].self_attn.linear_qkv.weight,
                                    torch.cat((original_q, original_kv))))
        current = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3MLAAttention) for m in current.values()), 3)
        for name in ["model.layers.0", "model.layers.1.mlp", "model.norm", "mtp.layers.0"]:
            self.assertIs(current[name], previous[name])
        self.assertIs(model.model.layers[0].self_attn.q_a_layernorm,
                      previous["model.layers.0.self_attn.q_a_layernorm"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_replacement_loads_converted_reference_state(self):
        """Feature: Offline checkpoint loading.

        Description: Load arrays that already use the model's final parameter layout and differ from its state.
        Expectation: Every materialized parameter and buffer holds the loaded values exactly.
        """
        config = small_config()
        rules = entries_to_module_replacements(jt_test_config().plan_overrides)
        with torch.device("meta"):
            candidate = JTDeepseekV3ForCausalLM(config)
            plan = compile_module_replacements(candidate, rules)
            apply_module_replacements(candidate, plan, weights_mapping=[])
        candidate.to_empty(device="cpu")
        generator = torch.Generator().manual_seed(0)
        arrays = {name: torch.randn(value.shape, generator=generator).to(value.dtype).numpy()
                  for name, value in candidate.state_dict().items()}
        _load_reference_state(candidate, arrays)
        for name, value in candidate.state_dict().items():
            self.assertEqual(value.detach().numpy().tobytes(), arrays[name].tobytes(), name)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_family_registration_does_not_replace_standard_deepseek(self):
        """Feature: Independent family registration.

        Description: Resolve both families and the custom architecture identity.
        Expectation: JT owns its spec; the original V3/V2 identities stay registered independently.
        """
        standard = get_model_adapter("deepseek_v3")
        custom = get_model_adapter("jt_deepseek_v3")
        self.assertIs(get_model_adapter("JTDeepseekV3ForCausalLM"), custom)
        self.assertIsNot(standard, custom)
        self.assertEqual(standard.architecture, "DeepseekV3ForCausalLM")
        self.assertEqual(custom.model_type, "jt_deepseek_v3")
        self.assertEqual(small_config().model_type, "jt_deepseek_v3")
        self.assertIs(get_model_adapter(small_config().architectures[0]), custom)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fsdp_discovery_uses_jt_identity_and_declared_mtp_unit(self):
        """Feature: FSDP unit discovery.

        Description: Resolve FSDP units for the JT model through the framework manager.
        Expectation: The JT adapter applies; each MTP transformer layer is one unit, not split leaves.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        self.assertEqual(FSDP2Manager._get_model_adapter_spec(model).model_type, "jt_deepseek_v3")
        manager = FSDP2Manager(FSDP2Config(), SimpleNamespace(fsdp_moe_mesh=None))
        self.assertEqual(
            sorted(unit.fqn for unit in manager._find_wrap_modules(model)),
            ["model.layers.0", "model.layers.1", "mtp.layers.0.transformer_layer"],
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_config_roundtrip(self):
        """Feature: Native model configuration.

        Description: Build and serialize native JT configuration without an example YAML.
        Expectation: Model dimensions, JT options and independent adapter identity survive.
        """
        recipe = jt_test_config()
        config = JTDeepseekV3Config(**recipe.model.config)
        self.assertIs(type(config), JTDeepseekV3Config)
        self.assertFalse(hasattr(recipe.model, "reference_yaml"))
        self.assertFalse(hasattr(config, "jt_config"))
        restored = JTDeepseekV3Config.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.mlp_layer_types, ["dense", "sparse"])
        self.assertEqual(restored.num_nextn_predict_layers, 1)
        self.assertEqual(restored.mtp_loss_factor, 0.3)
        restored.n_group = 2
        with self.assertRaisesRegex(ValueError, "n_group=topk_group=1"):
            JTDeepseekV3ForCausalLM(restored)
        self.assertEqual(restored.moe_aux_loss_coeff, 0.0001)
        self.assertEqual(restored.rope_parameters["rope_theta"], 5000000)
        self.assertIs(get_model_adapter(restored.architectures[0]), get_model_adapter("jt_deepseek_v3"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_batch_inputs_define_lm_and_mtp_objectives(self):
        """Feature: Native Trainer input and objective contract.

        Description: Forward the shared text batch's model inputs unchanged; the final target is the valid ID 0.
        Expectation: LM and MTP losses equal dense CE over the batch mask; MTP has no target past the end.
        """
        torch.manual_seed(5)
        model = JTDeepseekV3ForCausalLM(small_config())
        tokens = torch.arange(8).unsqueeze(0)
        labels = torch.roll(tokens, -1, 1)
        labels[0, [2, 5]] = -100
        mesh = SimpleNamespace(cp_size=1, pp_size=1, tp_size=1, dp_size=1, dp_rank=0, device_mesh=None)
        batch = TextParallelBatch(mesh, torch.device("cpu"), None, {}, False,
                                  source_type="indexed", attention_mode="compressed")
        model_inputs, loss_inputs = batch(iter([{"tokens": tokens, "labels": labels}]))
        head_outputs = []
        handle = model.lm_head.register_forward_hook(lambda _module, _inputs, output: head_outputs.append(output))
        try:
            output = model(**model_inputs, use_cache=False)
        finally:
            handle.remove()
        lm_logits, mtp_logits = (logits[0].detach() for logits in head_outputs)
        targets, mask = loss_inputs["shift_labels"][0], loss_inputs["loss_mask"][0].bool()
        torch.testing.assert_close(output.loss["foundation_loss/lm"], F.cross_entropy(lm_logits[mask], targets[mask]))
        # The single MTP depth predicts each next target, scaled by the MTP loss factor.
        expected_mtp = F.cross_entropy(mtp_logits[:-1][mask[1:]], targets[1:][mask[1:]]) * model.config.mtp_loss_factor
        torch.testing.assert_close(output.loss["foundation_loss/mtp"], expected_mtp)
        with self.assertRaisesRegex(ValueError, "attention_mask=None"):
            model(**(model_inputs | {"attention_mask": torch.ones_like(tokens)}))
        with self.assertRaisesRegex(ValueError, "explicit shift_labels"):
            model(input_ids=tokens, labels=labels)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_row_without_targets_has_zero_lm_and_mtp_losses(self):
        """Feature: Native Trainer objective contract.

        Description: Forward a row whose shifted labels are all ignored, with two MTP depths.
        Expectation: The LM and MTP losses are zero instead of NaN, and every gradient stays finite.
        """
        torch.manual_seed(5)
        config = small_config()
        config.num_nextn_predict_layers = 2
        model = JTDeepseekV3ForCausalLM(config)
        tokens = torch.arange(8).unsqueeze(0)
        losses = model(tokens, torch.full_like(tokens, -100)).loss
        self.assertEqual(losses["foundation_loss/lm"].item(), 0.0)
        self.assertEqual(losses["foundation_loss/mtp"].item(), 0.0)
        sum(losses.values()).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                with self.subTest(parameter=name):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
