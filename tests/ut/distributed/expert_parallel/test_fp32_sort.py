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
"""CPU contracts for bounded FP32 expert keys and EP option propagation."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.experts import (
    _expert_argsort, _local_swiglu_expert_forward, _prepare_ep_dispatch,
)
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute


class _DeviceKeys:
    """Emulate device metadata while converting and sorting real CPU tensors."""

    def __init__(self, values: torch.Tensor, device: str = "npu") -> None:
        """Store integer keys and record conversion calls."""
        self.values = values
        self.device = SimpleNamespace(type=device)
        self.converted = False

    def float(self) -> torch.Tensor:
        """Use an actual FP32 conversion to check exact integer representation."""
        self.converted = True
        return self.values.float()

    def argsort(self) -> torch.Tensor:
        """Run the original integer sort path on CPU."""
        return self.values.argsort()


class TestFP32ExpertSort(unittest.TestCase):
    """Keep expert membership exact without requiring equal-key permutation identity."""

    def test_exact_keys_empty_duplicates_and_precision_boundary(self) -> None:
        """Every input slot appears once and sorted expert keys retain exact values."""
        for dtype in (torch.int32, torch.int64):
            for values in ([], [0], [4, 4, 4], [2**24 - 1, 0, 159, 5, 159, 0]):
                with self.subTest(dtype=dtype, values=values):
                    integer_keys = torch.tensor(values, dtype=dtype)
                    keys = _DeviceKeys(integer_keys)
                    order = _expert_argsort(keys, 2**24, True)
                    self.assertTrue(keys.converted)
                    self.assertEqual(order.dtype, torch.int64)
                    torch.testing.assert_close(integer_keys[order], integer_keys.sort().values, rtol=0, atol=0)
                    torch.testing.assert_close(order.sort().values, torch.arange(len(values)), rtol=0, atol=0)

    def test_disabled_large_bound_and_other_devices_keep_integer_sort(self) -> None:
        """Do not round keys beyond FP32's contiguous exact integer range."""
        values = torch.tensor([2**24 + 1, 2**24, 3], dtype=torch.int64)
        for enabled, bound, device in ((False, 2**24, "npu"), (True, 2**24 + 2, "npu"),
                                        (True, 2**24, "cpu"), (True, 2**24, "cuda")):
            with self.subTest(enabled=enabled, bound=bound, device=device):
                keys = _DeviceKeys(values, device)
                actual = _expert_argsort(keys, bound, enabled)
                self.assertFalse(keys.converted)
                torch.testing.assert_close(actual, values.argsort(), rtol=0, atol=0)

    def test_dispatch_uses_composite_key_bound(self) -> None:
        """The upper bound covers destination-rank and global-expert components."""
        hidden = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
        indices = torch.tensor([[7, 0], [3, 5], [1, 7]])
        weights = torch.ones(3, 2, dtype=torch.float64)

        def _exchange(output: torch.Tensor, inputs: torch.Tensor, **kwargs: object) -> None:
            del kwargs
            output.copy_(inputs)

        with patch("hyper_parallel.distributed.expert_parallel.experts._expert_argsort",
                   wraps=_expert_argsort) as sort:
            with patch("hyper_parallel.distributed.expert_parallel.experts.dist.all_to_all_single",
                       side_effect=_exchange):
                dispatch = _prepare_ep_dispatch(hidden, indices, weights, local_expert_count=4,
                                                global_expert_count=8, ep_size=2, ep_group=None,
                                                sort_in_fp32=True)
        self.assertEqual(sort.call_args.args[1:], (16, True))
        expected_keys = (indices.flatten() // 4) * 8 + indices.flatten()
        torch.testing.assert_close(sort.call_args.args[0], expected_keys, rtol=0, atol=0)
        self.assertEqual(sum(dispatch.send_counts), 6)
        torch.testing.assert_close(dispatch.expert_indices.flatten(), indices.flatten().sort().values)

    def test_local_grouping_restores_rows_and_gradients(self) -> None:
        """FP32-key ordering does not change which expert processes each token."""
        indices = torch.tensor([2, 0, 2, 1, 0, 1])
        inputs = torch.arange(24, dtype=torch.float64).reshape(6, 4).requires_grad_()
        reference_input = inputs.detach().clone().requires_grad_()
        weights = torch.tensor([2., 3., 5.], dtype=torch.float64, requires_grad=True)
        reference_weights = weights.detach().clone().requires_grad_()

        def _grouped(states: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            return states * weights.repeat_interleave(counts).unsqueeze(-1)

        experts = SimpleNamespace(local_expert_count=3, ep_sort_in_fp32=True,
                                  _ep_use_grouped_gemm=True, forward_expert_major=_grouped)
        with patch("hyper_parallel.distributed.expert_parallel.experts._expert_argsort",
                   return_value=indices.float().argsort()) as sort:
            actual = _local_swiglu_expert_forward(experts, inputs, indices)
        sort.assert_called_once_with(indices, 3, True)
        expected = reference_input * reference_weights[indices].unsqueeze(-1)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(inputs.grad, reference_input.grad, rtol=0, atol=0)
        torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0, atol=0)

    def test_recipe_propagates_flag_and_preserves_default_call_contract(self) -> None:
        """Both dispatch and the local expert entry receive the opt-in selection."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                module = SimpleNamespace(experts=SimpleNamespace())
                mesh = MagicMock()
                router, combine = Mock(), Mock()
                with patch("hyper_parallel.distributed.expert_parallel.recipes.bind_local_expert_forward"):
                    compute = build_ep_compute(module, mesh, router_fn=router, archetype_key="test",
                                               expected_attrs=["experts"], combine=combine, sort_in_fp32=enabled)
                self.assertEqual(module.experts.ep_sort_in_fp32, enabled)
                with patch("hyper_parallel.distributed.expert_parallel.recipes.ep_routed_forward") as routed:
                    result = compute(module, torch.zeros(1, 2, 4))
                self.assertEqual(routed.call_args.kwargs.get("sort_in_fp32", False), enabled)
                if not enabled:
                    self.assertNotIn("sort_in_fp32", routed.call_args.kwargs)
                self.assertIs(result, combine.return_value)
                self.assertIs(combine.call_args.args[2], routed.return_value)
        with self.assertRaisesRegex(ValueError, "sort_in_fp32 must be a boolean"):
            build_ep_compute(None, None, router_fn=Mock(), archetype_key="test", expected_attrs=[],
                             combine=Mock(), sort_in_fp32="true")