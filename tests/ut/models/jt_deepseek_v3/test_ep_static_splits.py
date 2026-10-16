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
"""Fixed-route split arithmetic, topology checks and dispatch equivalence on CPU."""
# pylint: disable=protected-access

from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.distributed.expert_parallel import experts, recipes
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed import ep_compute
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config


def _routes(real_tokens, expert_count, top_k, padding):
    """Explicit cyclic slots serve as the reference, including the separate pad prefix."""
    real = torch.arange(real_tokens * top_k).reshape(real_tokens, top_k) % expert_count
    pad = torch.arange(padding * top_k).reshape(padding, top_k) % expert_count
    return torch.cat((pad, real))


class TestStaticSplits(unittest.TestCase):
    """Check the model-owned proof without requiring a device or process group."""

    def test_counts_match_explicit_routes(self):
        """Handle remainders, empty peers, top-k extremes and both padding modes."""
        for expert_count, ep_size, top_k in [(4, 2, 1), (8, 4, 8), (160, 32, 6)]:
            for padding in (0, expert_count):
                for real_tokens in (0, 1, 7, 159, 160, 8192):
                    with self.subTest(experts=expert_count, ep=ep_size, padding=padding, tokens=real_tokens):
                        routes = _routes(real_tokens, expert_count, top_k, padding)
                        expected = torch.bincount(
                            routes.reshape(-1) // (expert_count // ep_size), minlength=ep_size).tolist()
                        for rank in range(ep_size):
                            send, receive = ep_compute._fixed_route_splits(
                                real_tokens + padding, expert_count=expert_count, top_k=top_k,
                                padding=padding, ep_size=ep_size, ep_rank=rank)
                            self.assertEqual(send, expected)
                            self.assertEqual(receive, [expected[rank]] * ep_size)

    def test_invalid_routing_parameters(self):
        """Reject unsupported counts before producing collective split sizes."""
        valid = {"token_count": 12, "expert_count": 8, "top_k": 2, "padding": 8, "ep_size": 4, "ep_rank": 0}
        for override in [{'expert_count': 0}, {'ep_size': 0}, {'ep_size': 3}, {'ep_rank': 4},
                         {'top_k': 0}, {'top_k': 9}, {'padding': 3}, {'token_count': 7}]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                ep_compute._fixed_route_splits(**(valid | override))

    def test_config_default_validation_and_roundtrip(self):
        """Keep old configurations disabled and serialize the explicit opt-in."""
        self.assertFalse(JTDeepseekV3Config().expert_static_splits)
        config = JTDeepseekV3Config(expert_static_splits=True)
        self.assertTrue(JTDeepseekV3Config.from_dict(config.to_dict()).expert_static_splits)
        for value in ('true', 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                JTDeepseekV3Config(expert_static_splits=value)

    @staticmethod
    def _fixture():
        config = SimpleNamespace(expert_static_splits=True, fix_router=True,
                                 n_routed_experts=8, num_experts_per_tok=2)
        module = SimpleNamespace(config=config, padding=8, fixed_routing=lambda scores: scores)
        cp_mesh, ep_mesh = MagicMock(), MagicMock()
        ep_mesh.__getitem__.return_value.size.return_value = 4
        return module, cp_mesh, ep_mesh

    def test_disabled_provider_does_not_inspect_mesh_or_dependency(self):
        """Default behavior works without fixed routing or a CP mesh."""
        module = SimpleNamespace(config=JTDeepseekV3Config())
        self.assertIsNone(ep_compute._static_split_provider(module, None, None, None))

    def test_provider_rejects_missing_fixed_router(self):
        """A config flag alone must not enable splits on the learned routing implementation."""
        module, cp_mesh, ep_mesh = self._fixture()
        for missing in ('flag', 'implementation'):
            with self.subTest(missing=missing):
                module.config.fix_router = missing != 'flag'
                module.fixed_routing = None if missing == 'implementation' else lambda scores: scores
                with self.assertRaisesRegex(ValueError, 'fix_router support'):
                    ep_compute._static_split_provider(module, None, cp_mesh, ep_mesh)

    def test_provider_rejects_tp_and_missing_cp(self):
        """Static receive counts require equal CP shards with no extra TP partition."""
        module, cp_mesh, ep_mesh = self._fixture()
        tp_mesh = MagicMock()
        tp_mesh.size.return_value = 2
        for tp, cp in [(tp_mesh, cp_mesh), (None, None)]:
            with self.subTest(tp=tp, cp=cp), self.assertRaisesRegex(ValueError, 'TP=1'):
                ep_compute._static_split_provider(module, tp, cp, ep_mesh)

    @patch.object(ep_compute, 'dist')
    def test_provider_requires_ordered_group_identity(self, distributed):
        """Same size or same members in another order do not establish the receive contract."""
        module, cp_mesh, ep_mesh = self._fixture()
        for cp_ranks in ([0, 1, 2, 4], [3, 2, 1, 0]):
            distributed.get_process_group_ranks.side_effect = [[0, 1, 2, 3], cp_ranks]
            with self.subTest(cp_ranks=cp_ranks), self.assertRaisesRegex(ValueError, 'identical ordered'):
                ep_compute._static_split_provider(module, None, cp_mesh, ep_mesh)
        distributed.get_process_group_ranks.side_effect = [[4, 5, 6, 7], [4, 5, 6, 7]]
        distributed.get_rank.return_value = 2
        provider = ep_compute._static_split_provider(module, None, cp_mesh, ep_mesh)
        self.assertEqual(provider(9), ([6, 4, 4, 4], [4, 4, 4, 4]))

    @patch.object(recipes, 'bind_local_expert_forward')
    @patch.object(recipes, 'ep_routed_forward')
    def test_recipe_passes_provider_without_changing_default_contract(self, routed, bind):
        """The generic recipe forwards the opt-in callback but adds no default keyword."""
        module = SimpleNamespace()
        mesh = MagicMock()
        hidden = torch.ones(1, 2, 3)
        router = MagicMock()
        provider = MagicMock()
        for split_fn in (None, provider):
            compute = recipes.build_ep_compute(
                module, mesh, router_fn=router, archetype_key='test', expected_attrs=[],
                combine=lambda _module, _hidden, result: result, split_fn=split_fn)
            self.assertIs(compute(module, hidden), routed.return_value)
            options = {} if split_fn is None else {'split_fn': provider}
            routed.assert_called_with(module, hidden, router_fn=router, ep_group=mesh.get_group('ep'), **options)
        self.assertEqual(bind.call_count, 2)

    def test_recipe_rejects_non_callable_provider(self):
        """Invalid public callback configuration fails before binding any experts."""
        with self.assertRaisesRegex(ValueError, 'callable'):
            recipes.build_ep_compute(None, None, router_fn=None, archetype_key='test',
                                     expected_attrs=[], combine=None, split_fn=3)

    @patch.object(experts, 'dist')
    def test_dispatch_matches_dynamic_values_and_gradients(self, distributed):
        """Skipping metadata exchange preserves token order, weights and autograd paths."""
        for real_tokens in (0, 1, 7):
            for padding in (0, 8):
                with self.subTest(tokens=real_tokens, padding=padding):
                    count = real_tokens + padding
                    indices = _routes(real_tokens, 8, 2, padding)
                    hidden = torch.arange(count * 3, dtype=torch.float64).reshape(1, count, 3).requires_grad_()
                    weights = torch.ones(count, 2, dtype=torch.float64, requires_grad=True)
                    provider = partial(ep_compute._fixed_route_splits, expert_count=8, top_k=2,
                                       padding=padding, ep_size=4, ep_rank=1)
                    expected_receive = provider(count)[1]
                    distributed.all_to_all_single.side_effect = (
                        lambda output, _input, **_kwargs: output.copy_(torch.tensor(expected_receive)))
                    arguments = {"local_expert_count": 2, "global_expert_count": 8,
                                 "ep_size": 4, "ep_group": object()}
                    reference = experts._prepare_ep_dispatch(hidden, indices, weights, **arguments)
                    distributed.reset_mock()
                    with patch.object(torch, 'bincount', side_effect=AssertionError('Unexpected device count')):
                        actual = experts._prepare_ep_dispatch(hidden, indices, weights, split_fn=provider, **arguments)
                    distributed.all_to_all_single.assert_not_called()
                    for field in ('source_indices', 'expert_weights', 'dispatch_order', 'states', 'expert_indices'):
                        torch.testing.assert_close(getattr(actual, field), getattr(reference, field), rtol=0, atol=0)
                    self.assertEqual(actual.send_counts, reference.send_counts)
                    self.assertEqual(actual.receive_counts, reference.receive_counts)
                    gradients = []
                    for dispatch in (reference, actual):
                        result = experts._aggregate_ep_outputs(
                            dispatch.states.square(), dispatch.expert_weights, dispatch.source_indices,
                            dispatch.dispatch_order, tuple(hidden.shape))
                        gradients.append(torch.autograd.grad(result.sum(), (hidden, weights), retain_graph=True))
                    for expected, value in zip(*gradients):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_dispatch_rejects_invalid_callback_counts(self):
        """Reject malformed metadata before invoking data collectives."""
        invalid = [([1], [1]), ([2, -1], [0, 1]), ([0, 0], [0, 0]), ([1.0, 0], [0, 1])]
        for send, receive in invalid:
            with self.subTest(send=send, receive=receive), self.assertRaisesRegex(ValueError, 'EP counts'):
                experts._prepare_ep_dispatch(
                    torch.ones(1, 1, 3), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1),
                    local_expert_count=2, global_expert_count=4, ep_size=2, ep_group=None,
                    split_fn=lambda _count: (send, receive))


    @patch.object(experts, 'dist')
    @patch.object(experts, 'ep_all_to_all')
    def test_routed_forward_threads_provider_through_data_pipeline(self, all_to_all, distributed):
        """The callback reaches dispatch while data collectives and weighted combine remain active."""
        group = MagicMock()
        group.size.return_value = 1
        distributed.get_rank.return_value = 0
        distributed.all_to_all_single.side_effect = lambda output, value, **_kwargs: output.copy_(value)
        all_to_all.side_effect = lambda values, *_args: values
        local_experts = MagicMock(side_effect=lambda values, _indices: values.square())
        local_experts.local_expert_count = 4
        module = SimpleNamespace(experts=local_experts)
        hidden = torch.arange(15, dtype=torch.float64).reshape(1, 5, 3).requires_grad_()
        weights = torch.rand(5, 2, dtype=torch.float64, requires_grad=True)
        indices = _routes(5, 4, 2, 0)

        def router(_module: SimpleNamespace, _hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Use the same routed slots and differentiable weights in both paths."""
            return indices, weights

        reference = experts.ep_routed_forward(module, hidden, router_fn=router, ep_group=group)
        distributed.reset_mock()
        all_to_all.reset_mock()
        provider = MagicMock(return_value=([10], [10]))
        actual = experts.ep_routed_forward(module, hidden, router_fn=router, ep_group=group, split_fn=provider)
        provider.assert_called_once_with(5)
        distributed.all_to_all_single.assert_not_called()
        self.assertEqual(all_to_all.call_count, 3)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        expected_gradients = torch.autograd.grad(reference.sum(), (hidden, weights), retain_graph=True)
        actual_gradients = torch.autograd.grad(actual.sum(), (hidden, weights))
        for value, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)