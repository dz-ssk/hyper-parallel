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

"""Bounded EP count semantics and integration (NPU selection simulated on CPU)."""

# Torch is the repository backend; CPU tensors emulate NPU branch selection.
# pylint: disable=forbidden-backend-import

from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

import torch

from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from tests.common.mark_utils import arg_mark


_COUNT_TOKENS = ep_experts._expert_token_counts  # pylint: disable=protected-access


def _npu_counts_on_cpu(indices, expert_count):
    """Exercise actual counting ops on CPU, changing only device-type selection."""
    with patch.object(torch.Tensor, "device", new_callable=PropertyMock) as device:
        device.return_value = SimpleNamespace(type="npu")
        return _COUNT_TOKENS(indices, expert_count)


class TestExpertTokenCounts(unittest.TestCase):
    """Check exact counts, device fallback, collective payloads, and gradients."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_count_boundaries_and_skew(self):
        """Feature: bounded counts.
        Description: Count empty, skewed, repeated and boundary expert indices.
        Expectation: Fixed INT64 counts exactly match bincount, with empty bins.
        """
        for values, size in (([], 4), ([0], 1), ([0, 3, 3], 4), ([2] * 8192, 5), ([0, 159, 159], 160)):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(size=size, length=len(values), dtype=dtype):
                    indices = torch.tensor(values, dtype=dtype)
                    expected = torch.bincount(indices, minlength=size)
                    with patch.object(torch, "bincount", side_effect=AssertionError("unexpected dynamic count")):
                        actual = _npu_counts_on_cpu(indices, size)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, torch.int64)
                    self.assertEqual(actual.device, indices.device)
                    self.assertFalse(actual.requires_grad)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_noncontiguous_indices(self):
        """Feature: strided routing indices.
        Description: Count a noncontiguous INT32 view without modifying its source.
        Expectation: Exact counts and unchanged input storage values.
        """
        source = torch.tensor([2, 99, 0, 99, 2, 99, 4, 99], dtype=torch.int32)
        original = source.clone()
        indices = source[::2]
        self.assertFalse(indices.is_contiguous())
        result = _npu_counts_on_cpu(indices, 6)
        torch.testing.assert_close(result, torch.tensor([1, 0, 2, 0, 1, 0]), rtol=0, atol=0)
        torch.testing.assert_close(source, original, rtol=0, atol=0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_other_devices_keep_bincount(self):
        """Feature: device fallback.
        Description: Exercise real CPU and simulated CUDA selection.
        Expectation: Native bincount is called with the declared minimum size.
        """
        indices = torch.tensor([0, 2, 2])
        for device_type in ("cpu", "cuda"):
            with self.subTest(device=device_type):
                with patch.object(torch.Tensor, "device", new_callable=PropertyMock) as device:
                    device.return_value = SimpleNamespace(type=device_type)
                    with patch.object(torch, "bincount", wraps=torch.bincount) as native:
                        result = _COUNT_TOKENS(indices, 4)
                native.assert_called_once_with(indices, minlength=4)
                torch.testing.assert_close(result, torch.tensor([1, 0, 2, 0]), rtol=0, atol=0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_dispatch_collective_counts(self):
        """Feature: EP dispatch counts.
        Description: Prepare skewed and empty routes while mocking count exchange.
        Expectation: Collective sees exact INT64 per-rank counts; dispatch is unchanged.
        """
        for rows in ([[0, 1], [0, 5], [5, 4]], []):
            with self.subTest(rows=rows):
                indices = torch.tensor(rows, dtype=torch.int64).reshape(-1, 2)
                states = torch.arange(len(rows) * 4, dtype=torch.float32).reshape(1, len(rows), 4)
                weights = torch.full(indices.shape, 0.5)
                group = object()
                received = torch.tensor([2, 0, 4], dtype=torch.int64)

                def exchange(output: torch.Tensor, counts: torch.Tensor, **kwargs: object) -> None:
                    """Check the outgoing collective payload and supply peer counts."""
                    self.assertIs(kwargs["group"], group)
                    expected = torch.bincount(indices.flatten() // 2, minlength=3)
                    torch.testing.assert_close(counts, expected, rtol=0, atol=0)
                    output.copy_(received)

                with patch.object(ep_experts.dist, "all_to_all_single", side_effect=exchange):
                    reference = ep_experts._prepare_ep_dispatch(  # pylint: disable=protected-access
                        states, indices, weights, local_expert_count=2,
                        global_expert_count=6, ep_size=3, ep_group=group,
                    )
                    with patch.object(ep_experts, "_expert_token_counts", side_effect=_npu_counts_on_cpu) as counter:
                        actual = ep_experts._prepare_ep_dispatch(  # pylint: disable=protected-access
                            states, indices, weights, local_expert_count=2,
                            global_expert_count=6, ep_size=3, ep_group=group,
                        )
                self.assertEqual(counter.call_count, 1)
                self.assertEqual(counter.call_args.args[1], 3)
                self.assertEqual(actual.send_counts, reference.send_counts)
                self.assertEqual(actual.receive_counts, received.tolist())
                for field in ("source_indices", "expert_weights", "dispatch_order", "states", "expert_indices"):
                    torch.testing.assert_close(getattr(actual, field), getattr(reference, field), rtol=0, atol=0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_local_expert_output_and_gradients(self):
        """Feature: local expert computation.
        Description: Compare bounded/native counts with an empty expert and skewed routing.
        Expectation: Local outputs, input gradients and both weight gradients agree exactly.
        """
        generator = torch.Generator().manual_seed(21)
        values = [torch.randn(shape, generator=generator, dtype=torch.float64)
                  for shape in ((5, 3), (3, 8, 3), (3, 3, 4))]
        indices = torch.tensor([2, 0, 2, 2, 0])
        results = []
        for count_fn in (_COUNT_TOKENS, _npu_counts_on_cpu):
            states, gate_up, down = [value.clone().requires_grad_() for value in values]
            experts = SimpleNamespace(local_expert_count=3, gate_up_proj=gate_up, down_proj=down)
            with patch.object(ep_experts, "_expert_token_counts", side_effect=count_fn) as counter:
                output = ep_experts._local_swiglu_expert_forward(  # pylint: disable=protected-access
                    experts, states, indices,
                )
            counter.assert_called_once_with(indices, 3)
            gradients = torch.autograd.grad(output.square().sum(), (states, gate_up, down))
            results.append((output, *gradients))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_grouped_expert_count_contract(self):
        """Feature: grouped GEMM count input.
        Description: Capture counts and sorted states delivered to grouped experts.
        Expectation: Fixed INT64 counts include empty experts and original token order returns.
        """
        states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        indices = torch.tensor([2, 0, 2, 0])

        def grouped_forward(sorted_states: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            """Validate the grouped-kernel contract without requiring an NPU."""
            torch.testing.assert_close(counts, torch.tensor([2, 0, 2]), rtol=0, atol=0)
            torch.testing.assert_close(sorted_states, states[indices.argsort()], rtol=0, atol=0)
            return sorted_states * 2

        experts = SimpleNamespace(local_expert_count=3, _ep_use_grouped_gemm=True,
                                  forward_expert_major=grouped_forward)
        with patch.object(ep_experts, "_expert_token_counts", side_effect=_npu_counts_on_cpu):
            actual = ep_experts._local_swiglu_expert_forward(  # pylint: disable=protected-access
                experts, states, indices,
            )
        torch.testing.assert_close(actual, states * 2, rtol=0, atol=0)