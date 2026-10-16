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
"""CPU contracts for optional NPU local expert token reordering."""

# Torch is the native backend; private helpers define the permutation contract.
# pylint: disable=forbidden-backend-import,protected-access

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import torch

from hyper_parallel.distributed.expert_parallel import experts, recipes
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed import ep_compute
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config
from tests.common.mark_utils import arg_mark


_KERNEL_MODULE = "hyper_parallel.components.functional.moe_token_unpermute"


def _kernel_reference(states, inverse, probs=None):
    """Emulate the documented no-probability gather; do not simulate a reduction."""
    if probs is not None:
        raise AssertionError("Local reorder must not apply routing probabilities")
    return states.index_select(0, inverse.long())


def _mock_kernel():
    module = ModuleType(_KERNEL_MODULE)
    kernel = Mock(side_effect=_kernel_reference)
    module.moe_token_unpermute = kernel
    return module, kernel


class TestLocalTokenReorder(unittest.TestCase):
    """Verify inverse direction, autograd, fallback, and option propagation."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_inverse_output_and_gradient(self):
        """Feature: inverse permutation.
        Description: Restore a non-self-inverse permutation through the kernel contract.
        Expectation: Exact output/gradient parity, INT32 inverse, no probabilities.
        """
        for dtype in (torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                states = torch.arange(20, dtype=dtype).reshape(5, 4).requires_grad_()
                order = torch.tensor([2, 4, 1, 0, 3])
                gradient = torch.arange(20, dtype=dtype).reshape(5, 4).flip(0)
                reference = experts._restore_local_token_order(states, order)
                reference_grad, = torch.autograd.grad(reference, states, gradient)
                wrapper, kernel = _mock_kernel()
                with patch.object(experts, "_can_unpermute_local_tokens", return_value=True), \
                        patch.dict(sys.modules, {_KERNEL_MODULE: wrapper}):
                    actual = experts._restore_local_token_order(states, order, use_fused=True)
                actual_grad, = torch.autograd.grad(actual, states, gradient)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
                kernel.assert_called_once()
                arguments = kernel.call_args.args
                self.assertEqual(len(arguments), 2)
                self.assertEqual(arguments[1].dtype, torch.int32)
                torch.testing.assert_close(order[arguments[1].long()], torch.arange(5), rtol=0, atol=0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_kernel_eligibility(self):
        """Feature: kernel limits.
        Description: Inspect metadata for device, dtype, shape and INT32 bounds.
        Expectation: Only the supported nonempty BF16 NPU contract is accepted.
        """
        cases = [
            ({}, True), ({"device": "cpu"}, False), ({"device": "cuda"}, False),
            ({"dtype": torch.float16}, False), ({"dtype": torch.float32}, False),
            ({"rows": 0}, False), ({"rows": 2**31 - 1}, True), ({"rows": 2**31}, False),
            ({"width": 0}, False), ({"ndim": 3}, False), ({"order_ndim": 2}, False),
            ({"order_dtype": torch.int32}, False), ({"order_device": "cpu"}, False),
            ({"order_size": 3}, False),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                data = {"device": "npu", "dtype": torch.bfloat16, "rows": 5, "width": 4, "ndim": 2,
                        "order_ndim": 1, "order_dtype": torch.int64, "order_device": "npu"}
                data.update(changes)
                states = SimpleNamespace(device=SimpleNamespace(type=data["device"]), dtype=data["dtype"],
                                         ndim=data["ndim"], shape=(data["rows"], data["width"]))
                order = SimpleNamespace(device=SimpleNamespace(type=data["order_device"]),
                                        dtype=data["order_dtype"], ndim=data["order_ndim"],
                                        numel=lambda: data.get("order_size", data["rows"]))
                self.assertEqual(experts._can_unpermute_local_tokens(states, order), expected)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_default_and_cpu_fallback(self):
        """Feature: reference fallback.
        Description: Disabled, CPU and empty cases must not import the NPU kernel.
        Expectation: Correct order and gradient through the original scatter path.
        """
        for enabled in (False, True):
            for count in (0, 4):
                with self.subTest(enabled=enabled, count=count):
                    states = torch.arange(count * 4, dtype=torch.float64).reshape(count, 4).requires_grad_()
                    order = torch.arange(count).roll(1)
                    with patch.dict(sys.modules, {_KERNEL_MODULE: None}):
                        result = experts._restore_local_token_order(states, order, use_fused=enabled)
                    torch.testing.assert_close(result[order], states, rtol=0, atol=0)
                    gradient, = torch.autograd.grad(result.sum(), states)
                    torch.testing.assert_close(gradient, torch.ones_like(states), rtol=0, atol=0)
        with patch.object(experts, "_can_unpermute_local_tokens") as eligibility:
            experts._restore_local_token_order(torch.ones(1, 4), torch.tensor([0]))
        eligibility.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_local_grouped_output_and_weight_gradients(self):
        """Feature: grouped expert integration.
        Description: Run real grouped-reference arithmetic with original/fused restoration.
        Expectation: Output, input gradient and weight gradient agree exactly.
        """
        indices = torch.tensor([2, 0, 2, 1, 0])
        values = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 20
        results = []
        for enabled in (False, True):
            states = values.clone().requires_grad_()
            weight = torch.arange(48, dtype=torch.float64).reshape(3, 4, 4).requires_grad_()

            def _grouped_forward(sorted_states, counts):
                chunks = sorted_states.split(counts.tolist())
                return torch.cat([chunk @ weight[index] for index, chunk in enumerate(chunks)])

            holder = SimpleNamespace(local_expert_count=3, _ep_use_grouped_gemm=True,
                                     ep_local_reorder=enabled, forward_expert_major=_grouped_forward)
            wrapper, kernel = _mock_kernel()
            with patch.object(experts, "_can_unpermute_local_tokens", return_value=True), \
                    patch.dict(sys.modules, {_KERNEL_MODULE: wrapper}):
                output = experts._local_swiglu_expert_forward(holder, states, indices)
            self.assertEqual(kernel.call_count, int(enabled))
            results.append((output, *torch.autograd.grad(output.square().sum(), (states, weight))))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_npu_gather_selection(self):
        """Feature: local input gather.
        Description: Simulate only NPU device metadata and capture index_select.
        Expectation: Opt-in selects index_select with the same sorted rows and output.
        """
        states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        indices = torch.tensor([2, 0, 2, 1, 0])
        order = indices.argsort()
        original_select = torch.Tensor.index_select
        for enabled in (False, True):
            calls = []

            def _select(tensor, dim, index):
                calls.append((dim, index))
                return original_select(tensor, dim, index)

            def _grouped_forward(sorted_states, counts):
                torch.testing.assert_close(sorted_states, states[order], rtol=0, atol=0)
                torch.testing.assert_close(counts, torch.tensor([2, 1, 2]), rtol=0, atol=0)
                return sorted_states * 2

            holder = SimpleNamespace(local_expert_count=3, _ep_use_grouped_gemm=True,
                                     ep_local_reorder=enabled, forward_expert_major=_grouped_forward)
            with patch.object(torch.Tensor, "device", new_callable=PropertyMock) as device, \
                    patch.object(torch.Tensor, "index_select", new=_select), \
                    patch.object(experts, "_can_unpermute_local_tokens", return_value=False):
                device.return_value = SimpleNamespace(type="npu")
                result = experts._local_swiglu_expert_forward(holder, states, indices)
            self.assertEqual(len(calls), int(enabled))
            torch.testing.assert_close(result, states * 2, rtol=0, atol=0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_config_roundtrip(self):
        """Feature: configuration.
        Description: Load old configs and serialize the explicit new option.
        Expectation: Default false, bool roundtrip, invalid values rejected.
        """
        self.assertFalse(JTDeepseekV3Config().expert_local_reorder)
        config = JTDeepseekV3Config(expert_local_reorder=True)
        self.assertTrue(JTDeepseekV3Config.from_dict(config.to_dict()).expert_local_reorder)
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a bool"):
                JTDeepseekV3Config(expert_local_reorder=value)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_recipe_flag(self):
        """Feature: public EP recipe.
        Description: Build both modes and reject malformed options before mesh access.
        Expectation: Flag reaches local experts while routed collectives keep their contract.
        """
        for enabled in (False, True):
            module, mesh, router, combine = Mock(), MagicMock(), Mock(), Mock()
            hidden = torch.ones(1, 2, 4)
            with patch.object(recipes, "bind_local_expert_forward") as bind, \
                    patch.object(recipes, "ep_routed_forward") as routed:
                compute = recipes.build_ep_compute(
                    module, mesh, router_fn=router, archetype_key="test", expected_attrs=[],
                    combine=combine, use_grouped_gemm=True, local_reorder=enabled,
                )
                self.assertIs(compute(module, hidden), combine.return_value)
            self.assertIs(module.experts.ep_local_reorder, enabled)
            bind.assert_called_once_with(module, mesh["ep"].size(), use_grouped_gemm=True)
            routed.assert_called_once_with(module, hidden, router_fn=router, ep_group=mesh.get_group.return_value)
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            recipes.build_ep_compute(None, None, router_fn=Mock(), archetype_key="test",
                                     expected_attrs=[], combine=Mock(), local_reorder=1)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_jt_adapter_option(self):
        """Feature: JT adapter.
        Description: Pass the model option through the EP recipe.
        Expectation: Only the local-reorder keyword changes; grouped GEMM remains enabled.
        """
        class _MoE:
            route = Mock()
            forward = Mock()
            combine_routed = Mock()

        for enabled in (False, True):
            module = _MoE()
            module.config = JTDeepseekV3Config(expert_local_reorder=enabled)
            with patch.object(ep_compute, "build_ep_compute") as build:
                ep_compute.jt_deepseek_v3_ep_compute_fn(
                    module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=Mock(),
                )
            self.assertIs(build.call_args.kwargs["local_reorder"], enabled)
            self.assertTrue(build.call_args.kwargs["use_grouped_gemm"])