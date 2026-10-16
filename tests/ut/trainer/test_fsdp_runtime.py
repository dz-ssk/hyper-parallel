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
"""Unit tests for Trainer-side FSDP runtime diagnostics."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import yaml

from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.trainer.config.resolver import resolve_component
from hyper_parallel.trainer.runtime.fsdp import FSDPRuntimeDiagnostics, configure_fsdp_gradient_sync
from tests.common.mark_utils import arg_mark


def _make_state() -> SimpleNamespace:
    """Build an empty duck-typed Torch FSDP state."""
    return SimpleNamespace(param_group=None, hsdp_params=[])


class _FSDPUnit(torch.nn.Module):
    """CPU module carrying only the scheduler fields consumed by diagnostics."""

    def __init__(self, features: int = 4) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(features, features)
        self.hsdp_scheduler = SimpleNamespace(
            scheduler_ctx=SimpleNamespace(root_bp_state=False),
            hsdp_state=_make_state(),
            forward_prefetch_cells=[],
            backward_prefetch_cells=[],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the test projection."""
        return self.projection(inputs)


class _TwoUnitModel(_FSDPUnit):
    """Root plus two nested FSDP units with an optional second branch."""

    def __init__(self, run_second: bool = True) -> None:
        super().__init__()
        del self.projection
        self.first = _FSDPUnit()
        self.second = _FSDPUnit()
        self.run_second = run_second

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run child units in the fixed first-to-second execution order."""
        outputs = self.first(inputs)
        if self.run_second:
            outputs = self.second(outputs)
        return outputs


class TestFSDPRuntimeDiagnostics(unittest.TestCase):
    """Verify order and optimizer-boundary checks without distributed setup."""

    def setUp(self) -> None:
        """Create the CPU test model and attach diagnostics."""
        self.model = _TwoUnitModel()
        self.model.first.hsdp_scheduler.forward_prefetch_cells = [self.model.second]
        self.model.second.hsdp_scheduler.backward_prefetch_cells = [self.model.first]
        self.diagnostics = FSDPRuntimeDiagnostics([self.model])

    def tearDown(self) -> None:
        """Remove hooks so each test owns an isolated observer."""
        self.diagnostics.close()

    def _run_micro_step(self) -> None:
        """Run one forward/backward micro-step on CPU."""
        self.diagnostics.begin_micro_step(global_step=3, micro_step=0)
        inputs = torch.randn(2, 4, requires_grad=True)
        self.model(inputs).sum().backward()

    def test_matching_forward_backward_order_passes(self) -> None:
        """Accept prefetch edges that follow both observed execution phases."""
        self._run_micro_step()

        self.assertIsNone(self.diagnostics.validate_before_optimizer_step())

    def test_forward_prefetch_to_earlier_unit_fails(self) -> None:
        """Reject a construction-order edge that points backward at runtime."""
        self.model.first.hsdp_scheduler.forward_prefetch_cells = []
        self.model.second.hsdp_scheduler.forward_prefetch_cells = [self.model.first]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "HP-FSDP-008"):
            self.diagnostics.validate_before_optimizer_step()

    def test_prefetched_conditional_target_that_did_not_execute_fails(self) -> None:
        """Reject a prefetch target skipped by a conditional execution branch."""
        self.model.run_second = False
        self.model.first.hsdp_scheduler.forward_prefetch_cells = [self.model.second]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "target did not execute"):
            self.diagnostics.validate_before_optimizer_step()

    def test_non_fused_pending_all_gather_fails(self) -> None:
        """Reject an unconsumed per-parameter all-gather before optimizer.step."""
        hsdp_param = SimpleNamespace(
            _param_fqn="first.projection.weight",
            allgather_comm_ctx=SimpleNamespace(
                allgather_handle=object(),
                allgather_output=None,
            ),
        )
        self.model.first.hsdp_scheduler.hsdp_state.hsdp_params = [hsdp_param]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "HP-FSDP-009"):
            self.diagnostics.validate_before_optimizer_step()

    def test_non_fused_completed_but_unconsumed_output_fails(self) -> None:
        """Reject a non-fused output retained after its async handle completed."""
        hsdp_param = SimpleNamespace(
            _param_fqn="first.projection.weight",
            allgather_comm_ctx=SimpleNamespace(
                allgather_handle=None,
                allgather_output=object(),
            ),
        )
        self.model.first.hsdp_scheduler.hsdp_state.hsdp_params = [hsdp_param]
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "output=not-consumed"):
            self.diagnostics.validate_before_optimizer_step()

    def test_fused_completed_but_unconsumed_all_gather_fails(self) -> None:
        """Reject a fused result even when its async handle is already complete."""
        result = SimpleNamespace(handle=None)
        bucket = SimpleNamespace(all_gather_result=result)
        self.model.first.hsdp_scheduler.hsdp_state.param_group = SimpleNamespace(
            all_gather_buckets=[bucket]
        )
        self._run_micro_step()

        with self.assertRaisesRegex(RuntimeError, "completed-not-consumed"):
            self.diagnostics.validate_before_optimizer_step()

    def test_child_parameter_gradient_without_child_forward_fails(self) -> None:
        """Detect direct child-weight access that bypasses the FSDP unit call."""
        parameter = self.model.first.projection.weight
        parameter.main_grad = torch.ones_like(parameter)
        self.model.first.hsdp_scheduler.hsdp_state.hsdp_params = [
            SimpleNamespace(
                _param_fqn="first.projection.weight",
                sharded_param=parameter,
            )
        ]
        self.diagnostics.begin_micro_step(global_step=3, micro_step=0)
        inputs = torch.randn(2, 4, requires_grad=True)
        self.model.second(inputs).sum().backward()

        with self.assertRaisesRegex(RuntimeError, "HP-FSDP-007"):
            self.diagnostics.validate_before_optimizer_step()


class _SyncUnit(HSDPModule, torch.nn.Module):
    """Use real recursive HSDP APIs with a CPU-only scheduler stand-in."""

    def __init__(self) -> None:
        """Attach a recording scheduler without creating device resources."""
        torch.nn.Module.__init__(self)
        self.hsdp_scheduler = Mock()
        self.hsdp_scheduler.scheduler_ctx = SimpleNamespace(is_last_backward=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return inputs; these tests exercise runtime controls, not kernels."""
        return inputs


def _sync_tree() -> _SyncUnit:
    root = _SyncUnit()
    root.layer = _SyncUnit()
    root.layer.experts = _SyncUnit()
    root.layer.experts.projection = _SyncUnit()
    root.layer.dense = _SyncUnit()
    return root


def _sync_enabled(module: _SyncUnit) -> bool:
    return module.hsdp_scheduler.set_requires_grad_sync.call_args.args[0]


class TestDelayedGradientSync(unittest.TestCase):
    """Verify selected-subtree policy using the actual recursive HSDP setters."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_config_defaults_and_validation(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Default selectors are independent; malformed selectors fail early.
        Expectation: Defaults are isolated; non-list or invalid items raise ValueError.
        """
        first, second = FSDP2Config(), FSDP2Config()
        first.delayed_sync_patterns.append("*.experts")
        self.assertEqual(second.delayed_sync_patterns, [])
        for invalid in (None, "*.experts", ("*.experts",), [""], [1], [None]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "delayed_sync_patterns"):
                FSDP2Config(delayed_sync_patterns=invalid)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_yaml_subtree_resolves_typed_config(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: The trainer resolver accepts the new YAML list in FSDP2Config.
        Expectation: The selector list and global sync setting survive typed resolution.
        """
        raw = yaml.safe_load("dp_shard_size: 64\nrequires_grad_sync: true\ndelayed_sync_patterns: ['*.experts']")
        config = resolve_component(raw, annotation=FSDP2Config, path="fsdp_config")
        self.assertIsInstance(config, FSDP2Config)
        self.assertEqual(config.delayed_sync_patterns, ["*.experts"])
        self.assertEqual(config.dp_shard_size, 64)
        self.assertTrue(config.requires_grad_sync)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_experts_only_across_two_optimizer_steps(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Experts delay two micros, synchronize the third, then repeat next step.
        Expectation: Only experts delay; final sync is restored in both optimizer steps.
        """
        root = _sync_tree()
        config = FSDP2Config(dp_shard_size=64, delayed_sync_patterns=["*.experts"])
        for _ in range(2):
            for micro in range(3):
                configure_fsdp_gradient_sync([root], config, 1, micro, 3)
                self.assertTrue(_sync_enabled(root))
                self.assertTrue(_sync_enabled(root.layer.dense))
                self.assertEqual(_sync_enabled(root.layer.experts), micro == 2)
                self.assertEqual(_sync_enabled(root.layer.experts.projection), micro == 2)
                self.assertEqual(root.hsdp_scheduler.scheduler_ctx.is_last_backward, micro == 2)
        root.layer.dense.hsdp_scheduler.set_requires_all_reduce.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_empty_or_legacy_config_preserves_global_policy(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: An omitted field retains both global eager and delayed synchronization.
        Expectation: All units retain the configured global policy and final sync.
        """
        for requires_sync in (False, True):
            configs = [
                FSDP2Config(dp_shard_size=2, requires_grad_sync=requires_sync),
                SimpleNamespace(dp_shard_size=2, requires_grad_sync=requires_sync),
            ]
            for config in configs:
                with self.subTest(config=config):
                    root = _sync_tree()
                    for micro in range(3):
                        configure_fsdp_gradient_sync([root], config, 1, micro, 3)
                        for module in (root, root.layer.dense, root.layer.experts):
                            self.assertEqual(_sync_enabled(module), requires_sync or micro == 2)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_global_delay_and_selected_delay_compose(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Patterns do not re-enable non-final synchronization under global no-sync.
        Expectation: Neither selected nor unselected units sync before the final micro.
        """
        root = _sync_tree()
        config = FSDP2Config(dp_shard_size=2, requires_grad_sync=False, delayed_sync_patterns=["*.experts"])
        for micro in range(3):
            configure_fsdp_gradient_sync([root], config, 1, micro, 3)
            self.assertEqual(_sync_enabled(root.layer.dense), micro == 2)
            self.assertEqual(_sync_enabled(root.layer.experts), micro == 2)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_hsdp_all_reduce_policy_is_preserved(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Replica all-reduce remains delayed for all units until the last micro.
        Expectation: AllReduce occurs only on the final micro; dense gradient sync stays enabled.
        """
        root = _sync_tree()
        config = FSDP2Config(dp_shard_size=4, delayed_sync_patterns=["*.experts"])
        for micro in range(3):
            configure_fsdp_gradient_sync([root], config, 2, micro, 3)
            for module in (root, root.layer.dense, root.layer.experts):
                args = module.hsdp_scheduler.set_requires_all_reduce.call_args.args
                self.assertEqual(args, (micro == 2,))
            self.assertTrue(_sync_enabled(root.layer.dense))
            self.assertEqual(_sync_enabled(root.layer.experts), micro == 2)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_pattern_can_match_only_second_model_part(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Pattern coverage is checked across all roots, including non-first parts.
        Expectation: Only the selected second-part subtree delays and then synchronizes.
        """
        first, second = _sync_tree(), _sync_tree()
        second.layer.special_experts = _SyncUnit()
        config = FSDP2Config(dp_shard_size=2, delayed_sync_patterns=["*.special_experts"])
        configure_fsdp_gradient_sync([first, second], config, 1, 0, 3)
        self.assertTrue(_sync_enabled(first.layer.experts))
        self.assertTrue(_sync_enabled(second.layer.experts))
        self.assertFalse(_sync_enabled(second.layer.special_experts))
        configure_fsdp_gradient_sync([first, second], config, 1, 2, 3)
        self.assertTrue(_sync_enabled(second.layer.special_experts))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_missing_pattern_fails_before_any_state_change(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: One valid selector must not hide another unmatched or case-wrong one.
        Expectation: An unmatched pattern raises without invoking any sync setter.
        """
        for pattern in ("*.missing", "*.Experts"):
            with self.subTest(pattern=pattern):
                root = _sync_tree()
                config = FSDP2Config(dp_shard_size=2, delayed_sync_patterns=["*.experts", pattern])
                with self.assertRaisesRegex(ValueError, "matched no HSDP modules"):
                    configure_fsdp_gradient_sync([root], config, 1, 0, 3)
                for _, module in root.named_modules():
                    module.hsdp_scheduler.set_requires_grad_sync.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_non_hsdp_name_does_not_count_as_match(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Only an independently wrapped HSDP subtree can be selected.
        Expectation: Matching an ordinary module raises without changing state.
        """
        root = _sync_tree()
        root.layer.unsharded = torch.nn.Identity()
        config = FSDP2Config(dp_shard_size=2, delayed_sync_patterns=["*.unsharded"])
        with self.assertRaisesRegex(ValueError, "matched no HSDP modules"):
            configure_fsdp_gradient_sync([root], config, 1, 0, 3)
        root.hsdp_scheduler.set_requires_grad_sync.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_overlapping_patterns_and_recursive_subtrees(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Parent selectors cover descendants; duplicate/overlapping globs are harmless.
        Expectation: Descendants delay together and all restore sync on the last micro.
        """
        root = _sync_tree()
        config = FSDP2Config(dp_shard_size=2, delayed_sync_patterns=["layer", "*.experts", "*.experts"])
        configure_fsdp_gradient_sync([root], config, 1, 0, 3)
        self.assertTrue(_sync_enabled(root))
        self.assertFalse(_sync_enabled(root.layer.dense))
        self.assertFalse(_sync_enabled(root.layer.experts.projection))
        configure_fsdp_gradient_sync([root], config, 1, 2, 3)
        self.assertTrue(_sync_enabled(root.layer.dense))
        self.assertTrue(_sync_enabled(root.layer.experts.projection))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_no_accumulation_or_sharding_remains_noop(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Existing no-op topology and single-micro behavior stay unchanged.
        Expectation: No synchronization setter is called for either inactive case.
        """
        for shard_size, num_micros in ((1, 3), (2, 1)):
            with self.subTest(shard_size=shard_size, num_micros=num_micros):
                root = _sync_tree()
                config = FSDP2Config(dp_shard_size=shard_size, delayed_sync_patterns=["*.experts"])
                configure_fsdp_gradient_sync([root], config, 1, 0, num_micros)
                root.hsdp_scheduler.set_requires_grad_sync.assert_not_called()
                root.layer.experts.hsdp_scheduler.set_requires_grad_sync.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_removing_selector_restores_global_policy(self) -> None:
        """Validate module-selective gradient synchronization.

        Feature: Trainer delayed-sync subtree selection.
        Description: Each invocation restores root policy before applying current overrides.
        Expectation: Removing the selector restores the previously overridden subtree.
        """
        root = _sync_tree()
        config = FSDP2Config(dp_shard_size=2, delayed_sync_patterns=["*.experts"])
        configure_fsdp_gradient_sync([root], config, 1, 0, 3)
        self.assertFalse(_sync_enabled(root.layer.experts))
        config.delayed_sync_patterns.clear()
        configure_fsdp_gradient_sync([root], config, 1, 1, 3)
        self.assertTrue(_sync_enabled(root.layer.experts))


if __name__ == "__main__":
    unittest.main()