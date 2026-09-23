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
"""Masked loss forward/backward contract on one emulated vocabulary rank."""

import unittest

from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_loss import (
    JTDeepseekV3Loss,
)


from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    _MaskedVocabLoss, masked_vocab_parallel_loss,
)
from tests.common.mark_utils import arg_mark


class TestJTLoss(unittest.TestCase):
    """Use an independent dense CE oracle and explicit group assertions."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_loss_and_gradient(self):
        """Masked vocabulary CE has the same derivative as the dense objective.

        Feature: jt_loss.
        Description: Masked vocabulary CE has the same derivative as the dense objective.
        Expectation: The asserted values and state transitions hold.
        """
        torch.manual_seed(12)
        values = torch.randn(1, 5, 7, requires_grad=True)
        oracle = values.detach().clone().requires_grad_()
        labels = torch.tensor([[1, 2, 0, 4, 6]])
        mask = torch.tensor([[1., 0., 1., 0., 1.]])
        group = object()
        with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.all_reduce") as reduce, patch("torch.distributed.get_rank", return_value=0):
            actual = _MaskedVocabLoss.apply(values, labels, mask, group)
            actual.backward()
            self.assertEqual(reduce.call_count, 3)
            self.assertTrue(all(call.kwargs["group"] is group for call in reduce.call_args_list))
        expected = (F.cross_entropy(oracle.flatten(0, 1), labels.flatten(), reduction="none") * mask.flatten()).sum()
        expected = expected / mask.sum()
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(values.grad, oracle.grad)
        self.assertEqual(torch.count_nonzero(values.grad[:, 1]).item(), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_all_masked_is_zero(self):
        """An empty objective stays finite and sends no gradient to logits.

        Feature: jt_loss.
        Description: An empty objective stays finite and sends no gradient to logits.
        Expectation: The asserted values and state transitions hold.
        """
        values = torch.randn(1, 2, 4, requires_grad=True)
        with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.all_reduce"), patch("torch.distributed.get_rank", return_value=0):
            loss = _MaskedVocabLoss.apply(values, torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2), None)
            loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(values.grad.abs().sum().item(), 0.)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_pre_shifted_labels_and_mask_are_preserved(self):
        """The loss adapter must neither shift labels again nor reconstruct masks.

        Feature: jt_loss.
        Description: The loss adapter must neither shift labels again nor reconstruct masks.
        Expectation: The asserted values and state transitions hold.
        """
        ids = torch.tensor([[3, 4]])
        labels = torch.tensor([[4, 5]])
        mask = torch.tensor([[0., 1.]])
        result = JTDeepseekV3Loss(input_mapping={"input_ids": "input_ids", "labels": "shift_labels",
                                                 "loss_mask": "loss_mask"}).prepare_model_inputs(
            {"input_ids": ids}, {"shift_labels": labels, "loss_mask": mask})
        self.assertIs(result["labels"], labels)
        self.assertIs(result["loss_mask"], mask)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_unbound_loss_never_uses_the_default_process_group(self):
        """Feature: Explicit vocabulary ownership.

        Description: Run a full-vocabulary objective while a default group exists.
        Expectation: Both forward paths and backward remain local without a bound TP group.
        """
        labels = torch.tensor([[1, 2]])
        mask = torch.ones(1, 2)
        with patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.all_reduce", side_effect=AssertionError), \
                patch("torch.distributed.get_rank", side_effect=AssertionError), \
                patch("torch.distributed.get_world_size", side_effect=AssertionError):
            values = torch.randn(1, 2, 4, requires_grad=True)
            loss = masked_vocab_parallel_loss(values, labels, mask, None)
            loss.backward()
            with torch.no_grad():
                evaluated = masked_vocab_parallel_loss(values, labels, mask, None)
        torch.testing.assert_close(loss, evaluated)
        self.assertTrue(torch.isfinite(values.grad).all())
