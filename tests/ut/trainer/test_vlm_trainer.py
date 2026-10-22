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
"""Unit tests for the VLM Trainer step lifecycle."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.common.mark_utils import arg_mark


def _prepared_omni_batch(valid_targets: int, seq_len: int = 8) -> tuple[dict, dict]:
    """Build one prepared Omni micro-batch holding ``valid_targets`` next-token targets."""
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    # Token counting drops the first label to align with next-token prediction.
    labels[0, 1:valid_targets + 1] = 1
    model_inputs = {"input_ids": torch.ones(1, seq_len, dtype=torch.long), "labels": labels}
    loss_inputs = {"labels": labels, "loss_mask": labels >= 0}
    return model_inputs, loss_inputs


class TestVLMTrainer(unittest.TestCase):
    """Verify VLM training divides every micro-step by whole-step token totals."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_train_step_divides_every_micro_step_by_the_step_token_total(self) -> None:
        """Weight each micro-step by its valid targets over the optimizer-step total.

        Feature: VLM Trainer gradient-accumulation token accounting.
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

        trainer = VLMTrainer.__new__(VLMTrainer)
        trainer.base = SimpleNamespace(
            num_micro_batches=len(valid_targets),
            state=SimpleNamespace(global_step=0),
            model_integration=SimpleNamespace(begin_step=lambda step: None),
            get_batch=next,
            forward_backward_step=forward_backward_step,
            on_step_begin=lambda micro_batches=None: None,
            model_reshard=lambda micro_step, num_micro_steps: None,
            configure_fsdp_gradient_sync=lambda micro_step, num_micro_steps: None,
            begin_fsdp_runtime_diagnostics=lambda micro_step: None,
            prepare_optimizer_step=lambda: 0.0,
            step_optimizers_and_schedulers=lambda: None,
            _end_model_integration_step=lambda metrics: None,
            on_step_end=lambda **kwargs: None,
        )

        trainer.train_step(iter([_prepared_omni_batch(count) for count in valid_targets]))

        expected = [(count, sum(valid_targets)) for count in valid_targets]
        self.assertEqual(observed, expected, f"expected (micro-step, step) token counts={expected}, got={observed}")


if __name__ == "__main__":
    unittest.main()
