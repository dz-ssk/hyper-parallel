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
"""Shared model-computed loss protocol without model-specific numerical policy."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from hyper_parallel.components.losses.model_output import ModelComputedLoss
from tests.common.mark_utils import arg_mark


class TestModelComputedLoss(unittest.TestCase):
    """Check explicit input mapping, auxiliary preservation and group ownership."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_combined_loss_keeps_auxiliary_and_gradient(self):
        """Feature: Model-owned objective.

        Description: Return a combined objective when supervised labels are all masked.
        Expectation: The adapter neither zeros auxiliary terms nor scales gradients.
        """
        value = torch.tensor(2., requires_grad=True)
        adapter = ModelComputedLoss()
        result = adapter(model_output=SimpleNamespace(loss=value), labels=torch.full((1, 2), -100))
        self.assertIs(result, value)
        result.backward()
        self.assertEqual(value.grad.item(), 1.)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mapping_and_explicit_group_binding(self):
        """Feature: Shared Trainer adaptation.

        Description: Translate a configurable supervision field and bind one explicit TP group.
        Expectation: Tensors are preserved by identity and unbound operation uses no default group.
        """
        adapter = ModelComputedLoss({"targets": "shift_labels"}, loss_group_attribute="vocabulary_group")
        labels = torch.tensor([[3, -100]])
        result = adapter.prepare_model_inputs({"unused": 1}, {"shift_labels": labels})
        self.assertEqual(set(result), {"targets"})
        self.assertIs(result["targets"], labels)
        with self.assertRaisesRegex(ValueError, "missing"):
            adapter.prepare_model_inputs({}, {})
        model, group = SimpleNamespace(), object()
        tp = Mock()
        tp.get_group.return_value = group
        setup = SimpleNamespace(mesh_context=SimpleNamespace(tp_size=8, loss_parallel=True, device_mesh={"tp": tp}))
        adapter.bind_model(model, setup)
        self.assertIs(model.vocabulary_group, group)
        self.assertIs(adapter.loss_group, group)
        adapter.bind_model(model)
        self.assertIsNone(model.vocabulary_group)
        self.assertIsNone(adapter.loss_group)
