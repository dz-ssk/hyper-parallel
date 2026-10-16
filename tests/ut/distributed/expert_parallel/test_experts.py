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
"""CPU contract tests for optional fused EP weighted aggregation.

The kernel stand-in checks layouts and autograd wiring, not NPU rounding.
"""
# pylint: disable=protected-access,forbidden-backend-import

import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel import experts, recipes
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed import ep_compute
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config


_KERNEL_MODULE = "hyper_parallel.components.functional.moe_token_unpermute"


def _kernel_reference(states, inverse, probabilities):
    """Follow the operator's original-slot -> sorted-row gather contract."""
    restored = states[inverse.long()].reshape(*probabilities.shape, states.shape[-1])
    return (restored * probabilities.unsqueeze(-1)).sum(dim=1)


class TestFusedEPAggregation(unittest.TestCase):
    """Exercise token ordering, both gradients, eligibility and API plumbing."""

    def test_output_and_both_gradients(self):
        """Non-self-inverse permutations must restore weights and rows together."""
        generator = torch.Generator().manual_seed(2026)
        for top_k in (1, 2, 6):
            with self.subTest(top_k=top_k):
                tokens, hidden = 6, 4
                routes = tokens * top_k
                # A cyclic shift is deliberately not its own inverse.
                order = torch.arange(routes).roll(1)
                source = torch.arange(tokens).repeat_interleave(top_k)
                states = torch.randn(routes, hidden, dtype=torch.float64, generator=generator, requires_grad=True)
                weights = torch.randn(routes, dtype=torch.float64, generator=generator, requires_grad=True)
                shape = (2, 3, hidden)
                reference = experts._aggregate_ep_outputs(states, weights, source, order, shape)
                gradient = torch.randn(*shape, dtype=torch.float64, generator=generator)
                expected_grads = torch.autograd.grad(reference, (states, weights), gradient)
                kernel = Mock(side_effect=_kernel_reference)
                wrapper = ModuleType(_KERNEL_MODULE)
                wrapper.moe_token_unpermute = kernel
                with patch.object(experts, "_can_fuse_ep_outputs", return_value=True), \
                        patch.dict(sys.modules, {_KERNEL_MODULE: wrapper}):
                    actual = experts._aggregate_ep_outputs(
                        states, weights, source, order, shape, fused_unpermute=True
                    )
                actual_grads = torch.autograd.grad(actual, (states, weights), gradient)
                torch.testing.assert_close(actual, reference)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)
                kernel.assert_called_once()
                _, inverse, probabilities = kernel.call_args.args
                self.assertEqual(inverse.dtype, torch.int32)
                torch.testing.assert_close(order[inverse.long()], torch.arange(routes))
                torch.testing.assert_close(probabilities, weights.reshape(tokens, top_k))

    def test_kernel_eligibility_metadata(self):
        """Device, dtype, empty, top-k and INT32 limits require no device reads."""
        cases = [
            ("npu", torch.bfloat16, torch.bfloat16, 12, 6, True),
            ("cpu", torch.bfloat16, torch.bfloat16, 12, 6, False),
            ("cuda", torch.bfloat16, torch.bfloat16, 12, 6, False),
            ("npu", torch.float16, torch.float16, 12, 6, False),
            ("npu", torch.float32, torch.float32, 12, 6, False),
            ("npu", torch.bfloat16, torch.float32, 12, 6, False),
            ("npu", torch.bfloat16, torch.bfloat16, 0, 0, False),
            ("npu", torch.bfloat16, torch.bfloat16, 12, 0, False),
            ("npu", torch.bfloat16, torch.bfloat16, 13, 6, False),
            ("npu", torch.bfloat16, torch.bfloat16, 512, 1, True),
            ("npu", torch.bfloat16, torch.bfloat16, 513, 1, False),
            ("npu", torch.bfloat16, torch.bfloat16, 2**31 - 1, 2**31 - 1, True),
            ("npu", torch.bfloat16, torch.bfloat16, 2**31, 2**31, False),
        ]
        for device, dtype, weight_dtype, routes, tokens, expected in cases:
            with self.subTest(device=device, dtype=dtype, routes=routes, tokens=tokens):
                states = SimpleNamespace(device=SimpleNamespace(type=device), dtype=dtype)
                weights = SimpleNamespace(dtype=weight_dtype)
                order = SimpleNamespace(numel=lambda: routes)
                actual = experts._can_fuse_ep_outputs(states, weights, order, (1, tokens, 4))
                self.assertEqual(actual, expected)

    def test_cpu_and_empty_fallback(self):
        """Enabling fusion on CPU or empty input preserves the original path."""
        for tokens in (0, 3):
            with self.subTest(tokens=tokens):
                states = torch.ones(tokens * 2, 4, dtype=torch.bfloat16, requires_grad=True)
                weights = torch.full((tokens * 2,), 0.5, dtype=torch.bfloat16, requires_grad=True)
                source = torch.arange(tokens).repeat_interleave(2)
                order = torch.arange(tokens * 2).roll(1)
                with patch.dict(sys.modules, {_KERNEL_MODULE: None}):
                    output = experts._aggregate_ep_outputs(
                        states, weights, source, order, (1, tokens, 4), fused_unpermute=True
                    )
                torch.testing.assert_close(output, torch.ones(1, tokens, 4, dtype=torch.bfloat16))
                state_grad, weight_grad = torch.autograd.grad(output.sum(), (states, weights))
                torch.testing.assert_close(state_grad, torch.full_like(states, 0.5))
                torch.testing.assert_close(weight_grad, torch.full_like(weights, 4))

    def test_disabled_never_checks_or_imports_kernel(self):
        """The default path remains independent of the optional operator."""
        with patch.object(experts, "_can_fuse_ep_outputs") as eligibility, \
                patch.dict(sys.modules, {_KERNEL_MODULE: None}):
            output = experts._aggregate_ep_outputs(
                torch.ones(2, 4), torch.tensor([0.25, 0.75]), torch.zeros(2, dtype=torch.long),
                torch.tensor([1, 0]), (1, 1, 4)
            )
        eligibility.assert_not_called()
        torch.testing.assert_close(output, torch.ones(1, 1, 4))

    def test_config_round_trip_and_validation(self):
        """Old configs default off; explicit bool survives serialization."""
        self.assertFalse(JTDeepseekV3Config().expert_fused_unpermute)
        config = JTDeepseekV3Config(expert_fused_unpermute=True)
        self.assertTrue(JTDeepseekV3Config.from_dict(config.to_dict()).expert_fused_unpermute)
        for value in ("false", 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a bool"):
                JTDeepseekV3Config(expert_fused_unpermute=value)

    def test_recipe_option_and_default_contract(self):
        """Only opted-in recipes add the new routed-forward keyword."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                module, mesh, router, combine = Mock(), MagicMock(), Mock(), Mock()
                hidden = torch.ones(1, 2, 4)
                with patch.object(recipes, "bind_local_expert_forward"), \
                        patch.object(recipes, "ep_routed_forward") as routed:
                    compute = recipes.build_ep_compute(
                        module, mesh, router_fn=router, archetype_key="test", expected_attrs=[],
                        combine=combine, fused_unpermute=enabled
                    )
                    self.assertIs(compute(module, hidden), combine.return_value)
                options = {"fused_unpermute": True} if enabled else {}
                routed.assert_called_once_with(
                    module, hidden, router_fn=router, ep_group=mesh.get_group.return_value, **options
                )
                combine.assert_called_once_with(module, hidden, routed.return_value)
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            recipes.build_ep_compute(
                None, None, router_fn=Mock(), archetype_key="test", expected_attrs=[],
                combine=Mock(), fused_unpermute="true"
            )

    def test_jt_adapter_passes_flag(self):
        """The JT config enables only weighted aggregation in the public recipe."""
        class _MoE:
            route = Mock()
            forward = Mock()
            combine_routed = Mock()

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                module = _MoE()
                module.config = JTDeepseekV3Config(expert_fused_unpermute=enabled)
                with patch.object(ep_compute, "build_ep_compute") as build:
                    ep_compute.jt_deepseek_v3_ep_compute_fn(
                        module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=Mock()
                    )
                self.assertIs(build.call_args.kwargs["fused_unpermute"], enabled)
                self.assertTrue(build.call_args.kwargs["use_grouped_gemm"])

    def test_routed_forward_passes_flag(self):
        """The dispatch/combine pipeline forwards fusion without changing collectives."""
        module, group, router = Mock(), Mock(), Mock()
        group.size.return_value = 2
        module.experts.local_expert_count = 4
        router.return_value = (torch.tensor([[0, 1]]), torch.tensor([[0.25, 0.75]]))
        with patch.object(experts.dist, "get_rank", return_value=0), \
                patch.object(experts, "_prepare_ep_dispatch"), \
                patch.object(experts, "_run_ep_local_experts"), \
                patch.object(experts, "_aggregate_ep_outputs") as aggregate:
            result = experts.ep_routed_forward(
                module, torch.ones(1, 1, 4), router_fn=router, ep_group=group, fused_unpermute=True
            )
        self.assertIs(result, aggregate.return_value)
        self.assertTrue(aggregate.call_args.kwargs["fused_unpermute"])