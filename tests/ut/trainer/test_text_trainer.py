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
"""Unit tests for the text Trainer step lifecycle."""

import unittest
import weakref
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import torch

from hyper_parallel.trainer.text_trainer import TextTrainer
from tests.common.mark_utils import arg_mark


def _prepared_text_batch(valid_targets: int, seq_len: int = 8) -> tuple[dict, dict]:
    """Build one prepared text micro-batch holding ``valid_targets`` supervised positions."""
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    labels[0, :valid_targets] = 1
    model_inputs = {"input_ids": torch.ones(1, seq_len, dtype=torch.long), "labels": labels}
    loss_inputs = {"labels": labels, "shift_labels": labels, "loss_mask": (labels >= 0).long()}
    return model_inputs, loss_inputs


def _train_step_base(
        get_batch: Callable[[Any], tuple[dict, dict]],
        forward_backward_step: Callable[[dict, dict], tuple[torch.Tensor, dict]],
        num_micro_batches: int,
) -> SimpleNamespace:
    """Build the BaseTrainer surface that one text optimizer step uses."""
    return SimpleNamespace(
        num_micro_batches=num_micro_batches,
        state=SimpleNamespace(global_step=0),
        model_integration=SimpleNamespace(begin_step=lambda step: None),
        get_batch=get_batch,
        forward_backward_step=forward_backward_step,
        on_step_begin=lambda: None,
        on_micro_step_begin=lambda micro_batch: None,
        model_reshard=lambda micro_step, num_micro_steps: None,
        configure_fsdp_gradient_sync=lambda micro_step, num_micro_steps: None,
        begin_fsdp_runtime_diagnostics=lambda micro_step: None,
        prepare_optimizer_step=lambda: 0.0,
        step_optimizers_and_schedulers=lambda: None,
        _end_model_integration_step=lambda metrics: None,
        on_step_end=lambda **kwargs: None,
    )


class TestTextTrainer(unittest.TestCase):
    """Verify text training dispatches prepared batches and whole-step token totals."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_forward_backward_dispatches_metric_hook_before_forward(self) -> None:
        """Send detached metric inputs before executing the model forward.

        Feature: Text Trainer micro-step callbacks.
        Description: Execute one forward-backward step on an already prepared batch.
        Expectation: Metric callback receives token metadata immediately before model execution.
        """
        events = []
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, -100]])
        model_inputs = {"input_ids": torch.ones(2, 4)}
        loss_inputs = {"labels": labels, "shift_labels": labels}

        def on_micro_step_begin(metric_inputs: dict[str, object]) -> None:
            """Record and validate the metric callback payload."""
            events.append("hook")
            token_count = int(metric_inputs["token_count"])
            self.assertEqual(token_count, 7, f"expected seven valid tokens, got={token_count}")

        def forward_backward_step(
                actual_model_inputs: dict[str, torch.Tensor],
                actual_loss_inputs: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Record model execution and validate forwarded batch identity."""
            events.append("forward")
            self.assertIs(
                actual_model_inputs,
                model_inputs,
                f"model input identity changed, got={actual_model_inputs!r}",
            )
            self.assertIs(
                actual_loss_inputs,
                loss_inputs,
                f"loss input identity changed, got={actual_loss_inputs!r}",
            )
            return torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)}

        base = SimpleNamespace(
            on_micro_step_begin=Mock(side_effect=on_micro_step_begin),
            forward_backward_step=Mock(side_effect=forward_backward_step),
        )
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = base

        loss, loss_dict = trainer.forward_backward_step(model_inputs, loss_inputs)

        self.assertEqual(events, ["hook", "forward"], f"expected metric hook before forward, got={events}")
        self.assertEqual(float(loss), 1.0, f"expected loss 1.0, got={float(loss)}")
        self.assertEqual(
            float(loss_dict["foundation_loss"]),
            1.0,
            f"expected foundation loss 1.0, got={float(loss_dict['foundation_loss'])}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_train_step_divides_every_micro_step_by_the_step_token_total(self) -> None:
        """Weight each micro-step by its valid targets over the optimizer-step total.

        Feature: Text Trainer gradient-accumulation token accounting.
        Description: Run one optimizer step over micro-batches holding 1, 2 and 5 valid targets.
        Expectation: Every micro-step keeps its own count and divides by the step total of 8,
            instead of by its own count times the number of micro-steps.
        """
        valid_targets = [1, 2, 5]
        observed = []

        def forward_backward_step(
                model_inputs: dict[str, torch.Tensor],
                loss_inputs: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Record the token counts that this micro-step's loss scaling reads."""
            del model_inputs, loss_inputs
            observed.append((
                int(trainer.base.current_token_counts["foundation_tokens"]),
                int(trainer.base.step_token_counts["foundation_tokens"]),
            ))
            return torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)}

        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = _train_step_base(next, forward_backward_step, num_micro_batches=len(valid_targets))

        trainer.train_step(iter([_prepared_text_batch(count) for count in valid_targets]))

        expected = [(count, sum(valid_targets)) for count in valid_targets]
        self.assertEqual(observed, expected, f"expected (micro-step, step) token counts={expected}, got={observed}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_train_step_releases_each_micro_batch_after_its_backward(self) -> None:
        """Keep each prefetched micro-batch referenced only until its backward completes.

        Feature: Text Trainer gradient-accumulation memory lifecycle.
        Description: Run one optimizer step over three micro-batches and track which prepared
            batches are still referenced before each forward and after the step.
        Expectation: Micro-step k sees all k earlier batches released, and no batch outlives the step.
        """
        produced = []
        released_before_forward = []

        def get_batch(data_iterator: Any) -> tuple[dict, dict]:
            """Fetch one prepared batch and watch its supervision tensor."""
            model_inputs, loss_inputs = next(data_iterator)
            produced.append(weakref.ref(loss_inputs["labels"]))
            return model_inputs, loss_inputs

        def forward_backward_step(
                model_inputs: dict[str, torch.Tensor],
                loss_inputs: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Count the earlier micro-batches that nothing references any more."""
            del model_inputs, loss_inputs
            released_before_forward.append(sum(ref() is None for ref in produced))
            return torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)}

        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = _train_step_base(get_batch, forward_backward_step, num_micro_batches=3)

        # A generator builds each batch on demand, so only the trainer can keep one alive.
        trainer.train_step(_prepared_text_batch(count) for count in (3, 4, 5))

        live_after_step = sum(ref() is not None for ref in produced)
        self.assertEqual(
            released_before_forward,
            [0, 1, 2],
            f"expected earlier micro-batches released before each forward, got={released_before_forward}",
        )
        self.assertEqual(live_after_step, 0, f"expected no micro-batch to outlive the step, got={live_after_step}")


if __name__ == "__main__":
    unittest.main()
