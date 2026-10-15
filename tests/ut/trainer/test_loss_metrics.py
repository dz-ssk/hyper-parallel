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
"""CPU regressions for token-weighted loss values and backward scaling."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, call, patch

import torch

from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestMeanGlobalLoss(unittest.TestCase):
    """Exercise the real loss reducer while replacing only collectives."""

    def setUp(self) -> None:
        """Provide only the mesh metadata consumed by the loss reducer."""
        self.mesh = SimpleNamespace(
            dp_cp_mesh=None, sequence_parallel=False, device_mesh=None, dp_size=1, cp_size=1,
        )

    @patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **_: value)
    def test_single_microbatch_has_exact_identity_gradient(self, _reduce) -> None:
        """One rank and one microbatch must not perturb an incoming gradient."""
        for count in (1, 164, 330, 3817, 4096):
            with self.subTest(tokens=count):
                loss = torch.tensor(2.5, requires_grad=True)
                counts = {"foundation_tokens": torch.tensor(count)}
                result = mean_global_loss(loss, counts, counts, self.mesh)["foundation_loss"]
                result.backward()
                actual = loss.grad.item()
                self.assertEqual(actual, 1.0, f"tokens={count}: gradient={actual}, expected={1.0}")
                torch.testing.assert_close(result, loss.detach(), rtol=0, atol=0)

    @patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **_: value)
    def test_unequal_microbatches_keep_token_weighting(self, _reduce) -> None:
        """Microbatches with three and seven tokens contribute 30% and 70%."""
        losses = [torch.tensor(2.0, requires_grad=True), torch.tensor(4.0, requires_grad=True)]
        total = {"foundation_tokens": torch.tensor(10)}
        outputs = [
            mean_global_loss(loss, {"foundation_tokens": torch.tensor(count)}, total, self.mesh)
            ["foundation_loss"]
            for loss, count in zip(losses, (3, 7))
        ]
        sum(outputs).backward()
        torch.testing.assert_close(torch.stack(outputs).sum(), torch.tensor(3.4))
        torch.testing.assert_close(torch.stack([loss.grad for loss in losses]), torch.tensor([0.3, 0.7]))

    @patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=[20.0, 42.0])
    def test_dp_cp_average_is_compensated_only_in_backward(self, reduce) -> None:
        """Four-rank gradient averaging requires scale 4 * 3 / 20."""
        group = object()
        self.mesh.dp_cp_mesh = MagicMock()
        self.mesh.dp_cp_mesh.get_group.return_value = group
        self.mesh.dp_size = self.mesh.cp_size = 2
        loss = torch.tensor(2.0, requires_grad=True)
        result = mean_global_loss(
            loss, {"foundation_tokens": torch.tensor(3)}, {"foundation_tokens": torch.tensor(5)}, self.mesh,
        )["foundation_loss"]
        result.backward()
        torch.testing.assert_close(result, torch.tensor(2.1))
        torch.testing.assert_close(loss.grad, torch.tensor(0.6))
        expected = [call(5, op="sum", group=group), call(6.0, op="sum", group=group)]
        self.assertEqual(reduce.call_args_list, expected, f"calls={reduce.call_args_list}, expected={expected}")

    @patch("hyper_parallel.trainer.runtime.metrics.dist.get_world_size", return_value=2)
    @patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=[6.0, 24.0, 60.0])
    def test_sequence_parallel_keeps_its_separate_normalization(self, reduce, world_size) -> None:
        """TP sums current tokens, then divides the output and its gradient."""
        tp_group, dp_group = object(), object()
        self.mesh.sequence_parallel = True
        self.mesh.device_mesh = MagicMock(mesh_dim_names=("tp",))
        self.mesh.device_mesh.__getitem__.return_value.get_group.return_value = tp_group
        self.mesh.dp_cp_mesh = MagicMock()
        self.mesh.dp_cp_mesh.get_group.return_value = dp_group
        self.mesh.dp_size = self.mesh.cp_size = 2
        loss = torch.tensor(2.0, requires_grad=True)
        result = mean_global_loss(
            loss, {"foundation_tokens": torch.tensor(3)}, {"foundation_tokens": torch.tensor(12)}, self.mesh,
        )["foundation_loss"]
        result.backward()
        torch.testing.assert_close(result, torch.tensor(1.25))
        torch.testing.assert_close(loss.grad, torch.tensor(0.5))
        world_size.assert_called_once_with(group=tp_group)
        expected = [
            call(3, op="sum", group=tp_group), call(12, op="sum", group=dp_group),
            call(12.0, op="sum", group=dp_group),
        ]
        self.assertEqual(reduce.call_args_list, expected, f"calls={reduce.call_args_list}, expected={expected}")

    @patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **_: value)
    def test_named_losses_use_their_own_counts(self, _reduce) -> None:
        """Objectives with different valid-token masks retain independent weights."""
        losses = {name: torch.tensor(2.0, requires_grad=True) for name in ("foundation_loss", "image_decoder_loss")}
        current = {"foundation_tokens": torch.tensor(2), "image_decoder_tokens": torch.tensor(3)}
        total = {"foundation_tokens": torch.tensor(8), "image_decoder_tokens": torch.tensor(6)}
        outputs = mean_global_loss(losses, current, total, self.mesh)
        sum(outputs.values()).backward()
        for name, expected in (("foundation_loss", 0.25), ("image_decoder_loss", 0.5)):
            torch.testing.assert_close(losses[name].grad, torch.tensor(expected))
            torch.testing.assert_close(outputs[name], torch.tensor(2 * expected))

    @patch("hyper_parallel.trainer.runtime.metrics.all_reduce", return_value=0.0)
    def test_empty_objective_stays_finite(self, _reduce) -> None:
        """An all-ignored objective produces zero without dividing by zero."""
        counts = {"foundation_tokens": torch.tensor(0)}
        result = mean_global_loss(torch.tensor(0.0), counts, counts, self.mesh)["foundation_loss"]
        torch.testing.assert_close(result, torch.tensor(0.0), rtol=0, atol=0)

    @patch("hyper_parallel.trainer.runtime.metrics.all_reduce", return_value=0.0)
    def test_nonzero_loss_with_zero_tokens_is_rejected(self, _reduce) -> None:
        """Keep the existing invalid-count validation contract."""
        counts = {"foundation_tokens": torch.tensor(0)}
        with self.assertRaisesRegex(ValueError, "cur_loss is not 0"):
            mean_global_loss(torch.tensor(1.0), counts, counts, self.mesh)


if __name__ == "__main__":
    unittest.main()
