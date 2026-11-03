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
"""Loss objectives retain independent counts across accumulation microbatches."""

import unittest

import torch

from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestLossTokenAccumulation(unittest.TestCase):
    """Exercise the actual generic token counter with unequal/empty objectives."""

    def test_independent_objective_counts_sum_across_microbatches(self) -> None:
        """Later MTP depths have fewer valid targets, including an empty microbatch."""
        batches = [
            {"labels": torch.tensor([[1, 2, 3, 4]]),
             "mtp_depth1_labels": torch.tensor([[3, 4, -100, -100]]),
             "mtp_depth2_labels": torch.tensor([[4, -100, -100, -100]])},
            {"labels": torch.tensor([[1, 2, 3, -100]]),
             "mtp_depth1_labels": torch.tensor([[3, -100, -100, -100]]),
             "mtp_depth2_labels": torch.full((1, 4), -100)},
        ]
        counts = count_loss_token(batches)
        self.assertEqual(counts["foundation_tokens"].item(), 5)
        self.assertEqual(counts["mtp_depth1_tokens"].item(), 3)
        self.assertEqual(counts["mtp_depth2_tokens"].item(), 1)

    def test_single_batch_and_other_objectives_keep_their_contract(self) -> None:
        """The generic suffix rule applies to image targets as well as MTP."""
        batch = {"labels": torch.tensor([[1, 2, 3]]),
                 "shift_labels": torch.tensor([[2, 3, -100]]),
                 "loss_mask": torch.tensor([[1, 0, 1]]),
                 "image_labels": torch.tensor([[7, -100, 9]])}
        counts = count_loss_token(batch)
        self.assertEqual(counts["foundation_tokens"].item(), 1)
        self.assertEqual(counts["image_tokens"].item(), 2)
        accumulated = count_loss_token([batch, batch])
        self.assertEqual(accumulated["image_tokens"].item(), 4)
        self.assertEqual(accumulated["foundation_tokens"].item(), 2)
