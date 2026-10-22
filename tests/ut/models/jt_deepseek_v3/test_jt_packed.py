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
"""Packed JT objectives keep document semantics with chunked output projection."""

import copy
from functools import partial
from itertools import product
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.components.losses.chunked_cross_entropy import projected_cross_entropy
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import _JTTrainingContext
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3ForCausalLM, JTDeepseekV3MoE,
)
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


def reference_grouped_swiglu(inputs: torch.Tensor, gate_up: torch.Tensor,
                            down: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Use differentiable CPU expert math to test loss policy independently of NPU kernels."""
    outputs = []
    for index, tokens in enumerate(torch.split(inputs, counts.tolist())):
        gate, up = (tokens @ gate_up[index].t()).chunk(2, dim=-1)
        outputs.append((F.silu(gate) * up) @ down[index].t())
    return torch.cat(outputs)


class TestJTPacked(unittest.TestCase):
    """Compare actual model losses and gradients across projection policies."""

    def setUp(self) -> None:
        """Keep CPU checkpoint recomputation independent of installed accelerators."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self.enterContext(patch(
            "hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu",
            side_effect=reference_grouped_swiglu))

    def test_packed_lm_isolates_documents(self) -> None:
        """Changing the preceding document must not change the supervised document's LM loss or gradients."""
        torch.manual_seed(8)
        model = JTDeepseekV3ForCausalLM(small_config())
        tokens = torch.arange(1, 9).unsqueeze(0)
        changed = tokens.clone()
        changed[:, :4] = torch.tensor([[9, 10, 11, 12]])
        labels = torch.tensor([[-100, -100, -100, -100, 6, 7, 8, -100]])
        expected = model(tokens, labels, actual_seq_len=(4, 8)).loss["foundation_loss/lm"]
        expected.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        actual = model(changed, labels, actual_seq_len=(4, 8)).loss["foundation_loss/lm"]
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            {name: p.grad for name, p in model.named_parameters() if p.grad is not None},
            gradients, rtol=0, atol=0)

    def test_packed_mtp_preserves_reference_target_shift(self) -> None:
        """Both projection paths preserve shifted labels without adding document-tail masks."""
        tokens = torch.arange(1, 9).unsqueeze(0)
        labels = torch.tensor([[2, 3, 4, -100, 6, 7, 8, -100]])
        expected = torch.tensor([[3, 4, -100, 6, 7, 8, -100, -100]])
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                config = small_config()
                config.num_nextn_predict_layers = 1
                config.enable_chunk_loss = chunked
                model = JTDeepseekV3ForCausalLM(config)
                method = "_projection_loss" if chunked else "_token_loss"
                with patch.object(model, method, wraps=getattr(model, method)) as loss:
                    model(tokens, labels, actual_seq_len=(4, 8))
                self.assertEqual(loss.call_count, 2)
                torch.testing.assert_close(loss.call_args_list[1].args[1], expected, rtol=0, atol=0)

    def test_chunked_lm_mtp_preserves_packed_losses_and_gradients(self) -> None:
        """Feature: Chunk loss for packed LM and MTP objectives.

        Description: Compare two supervised MTP depths with unequal counts and a partial chunk.
        Expectation: All objective values and parameter gradients match the eager model.
        """
        torch.manual_seed(9)
        config = small_config()
        config.num_nextn_predict_layers = 2
        eager = JTDeepseekV3ForCausalLM(config)
        chunked = copy.deepcopy(eager)
        inputs = torch.arange(12).unsqueeze(0)
        targets = torch.tensor([[1, 2, 3, -100, 5, -100, 7, -100, -100, 10, 11, -100]])
        expected = eager(inputs, targets, actual_seq_len=(4, 8, 12)).loss
        sum(expected.values()).backward()
        chunked.enable_chunk_loss = True
        chunked.chunk_loss_size = 5
        actual = chunked(inputs, targets, actual_seq_len=(4, 8, 12)).loss
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
        sum(actual.values()).backward()
        torch.testing.assert_close(
            {name: parameter.grad for name, parameter in chunked.named_parameters()},
            {name: parameter.grad for name, parameter in eager.named_parameters()}, rtol=3e-4, atol=2e-6)

    def test_chunked_cp_mean_matches_global_loss_and_gradients(self) -> None:
        """Feature: Checkpointed projection loss with CP token normalization.

        Description: Compare uneven, empty-local and empty-global target shards with global CE.
        Expectation: Loss and gradients agree; backward does not repeat the count collective.
        """
        torch.manual_seed(19)
        context = object.__new__(_JTTrainingContext)
        context.cp_group = object()

        def loss_fn(*, logits: torch.Tensor, shift_labels: torch.Tensor,
                    num_items_in_batch: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            """Use independent summed CE for each checkpointed projection."""
            return F.cross_entropy(logits.flatten(0, 1), shift_labels.flatten(), reduction="sum") / num_items_in_batch

        label_cases = ((1, -100, -100, -100, 2, 3, 4, 5),
                       (-100, -100, -100, -100, 2, 3, 4, 5),
                       (-100,) * 8)
        for labels in label_cases:
            with self.subTest(labels=labels):
                hidden = torch.randn(1, 8, 4, requires_grad=True)
                head = torch.nn.Linear(4, 11, bias=False)
                targets = torch.tensor([labels])
                count = targets.ne(-100).sum()
                expected = F.cross_entropy(head(hidden).flatten(0, 1), targets.flatten(), reduction="sum")
                expected = expected / count.clamp_min(1)
                local_loss = partial(projected_cross_entropy, head=head, loss_fn=loss_fn,
                                     vocab_size=11, chunk_size=3)

                def reduce(value: torch.Tensor, **kwargs: Any) -> None:
                    """Supply the global count without creating a real process group."""
                    self.assertIs(kwargs["group"], context.cp_group)
                    value.copy_(count)

                with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel.dist.all_reduce",
                           side_effect=reduce) as collective:
                    actual = sum(context.projection_loss(local_loss, states, shard)
                                 for states, shard in zip(hidden.chunk(2, 1), targets.chunk(2, 1)))
                    actual_gradients = torch.autograd.grad(actual, (hidden, head.weight))
                    self.assertEqual(collective.call_count, 2)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_gradients, torch.autograd.grad(expected, (hidden, head.weight)))

    def test_layer_checkpoint_and_chunk_loss_gradients(self) -> None:
        """Feature: Layer checkpointing with chunk loss.

        Description: Exercise full/selective layer wrappers with chunk loss enabled and disabled.
        Expectation: Losses and gradients match eager execution without retaining auxiliary graphs.
        """
        torch.manual_seed(19)
        config = small_config()
        reference = JTDeepseekV3ForCausalLM(config)
        self.assertFalse(reference.enable_chunk_loss)
        tokens = torch.arange(9).unsqueeze(0)
        targets = torch.tensor([[-100, -100, 3, 4, -100, 6, 7, 8, -100]])
        expected = reference(tokens, targets, actual_seq_len=(3, 6, 9)).loss
        sum(expected.values()).backward()
        for mode, enabled in product(("full", "selective"), (False, True)):
            with self.subTest(mode=mode, enable_chunk_loss=enabled):
                chunk_config = copy.deepcopy(config)
                chunk_config.chunk_loss_size = 3
                chunk_config.enable_chunk_loss = enabled
                model = JTDeepseekV3ForCausalLM(chunk_config)
                model.load_state_dict(reference.state_dict())
                self.assertEqual(model.enable_chunk_loss, enabled)
                model = _apply_activation_checkpointing(model, mode)
                head_calls = []
                handle = model.lm_head.register_forward_hook(
                    lambda _module, _args, output: head_calls.append(output.shape[1]))
                self.addCleanup(handle.remove)
                actual = model(tokens, targets, actual_seq_len=(3, 6, 9)).loss
                for key, loss in expected.items():
                    torch.testing.assert_close(actual[key], loss, rtol=2e-5, atol=2e-6)
                auxiliary_owners = [module for module in model.modules() if isinstance(module, JTDeepseekV3MoE)]
                self.assertTrue(all(module.auxiliary_loss is None for module in auxiliary_owners))
                forward_calls = len(head_calls)
                sum(actual.values()).backward()
                self.assertEqual(len(head_calls), forward_calls * (2 if enabled else 1))
                self.assertTrue(all(module.auxiliary_loss is None for module in auxiliary_owners))
                for parameter, original in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, original.grad, rtol=3e-4, atol=2e-6)

    def test_chunk_loss_configuration(self) -> None:
        """Feature: Explicit chunk loss configuration.

        Description: Check defaults, enabled configuration and invalid option values.
        Expectation: Only a boolean switch and a positive integer size configure chunk loss.
        """
        config = small_config()
        default = JTDeepseekV3ForCausalLM(config)
        self.assertFalse(default.enable_chunk_loss)
        self.assertEqual(default.chunk_loss_size, 1024)
        config.enable_chunk_loss = True
        config.chunk_loss_size = 7
        enabled = JTDeepseekV3ForCausalLM(config)
        self.assertTrue(enabled.enable_chunk_loss)
        self.assertEqual(enabled.chunk_loss_size, 7)
        invalid = [("enable_chunk_loss", value) for value in ("false", 0, 1, None)]
        invalid += [("chunk_loss_size", value) for value in (0, -1, True, "1024", None)]
        for option, value in invalid:
            with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, option):
                config = small_config()
                setattr(config, option, value)
                JTDeepseekV3ForCausalLM(config)
