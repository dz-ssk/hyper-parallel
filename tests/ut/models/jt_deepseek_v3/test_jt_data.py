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
"""Prepared Arrow supervision through the shared loader and JT runtime."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from datasets import Dataset, load_from_disk
import torch

from hyper_parallel.data.batching import FixedBatchDataLoader, TextParallelBatch
from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.online.mapping import MappingTransformDataset
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.data.text.text_transform import PreTokenizedSFTTransform
from hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime import JTSequenceRuntime
from tests.common.mark_utils import arg_mark


def _fixed_loader(source: Path, rank: int, workers: int, sampler_type: str) -> FixedBatchDataLoader:
    """Build the fixed loader with a fresh source and an independent DP cursor."""
    dataset = MappingTransformDataset(load_from_disk(source), PreTokenizedSFTTransform())
    sampler = build_dataset_batch_sampler(total_samples=len(dataset), micro_batch_size=1,
                                          global_batch_size=2, dp_rank=rank, dp_world_size=2,
                                          sampler_type=sampler_type, seed=19)
    return FixedBatchDataLoader(dataset, batch_sampler=sampler, collate_fn=TextPackingCollator(),
                                 sampler_type=sampler_type, num_workers=workers,
                                 prefetch_factor=2 if workers else None, seed=19)


class TestPreparedArrowDataset(unittest.TestCase):
    """Use a saved dataset through the public loader and JT model-input adapter."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_saved_arrow_to_jt_inputs(self) -> None:
        """Labels stay pre-shifted, optional masks are folded, and packed positions stay continuous."""
        records = [
            {"input_ids": [3, 4, 5, 6, 7, 8], "labels": [4, 5, -100, 7, 8, -100],
             "loss_mask": [1, 0, 0, 1, 1, 0], "cu_seqlens": [0, 3, 6]},
            {"input_ids": [9, 10, 11, 12, 13, 14], "labels": [10, -100, 12, 13, 14, -100],
             "loss_mask": [1, 0, 1, 1, 1, 0], "cu_seqlens": [0, 2, 6]},
        ]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "prepared"
            Dataset.from_list(records).save_to_disk(source)
            dataset = MappingTransformDataset(load_from_disk(source), PreTokenizedSFTTransform())
            iterator = iter(FixedBatchDataLoader(dataset, batch_size=1, drop_last=True, num_workers=0,
                                                 collate_fn=TextPackingCollator()))
            mesh = SimpleNamespace(cp_size=1, pp_size=1, tp_size=1, dp_size=1, dp_rank=0, device_mesh=None)
            batch = TextParallelBatch(mesh, torch.device("cpu"), None, {}, False,
                                      source_type="online", attention_mode="compressed", reset_position_ids=False,
                                      runtime_input_adapter=JTSequenceRuntime())
            for record in records:
                model, loss = batch(iterator)
                targets = torch.tensor([record["labels"]])
                targets[torch.tensor([record["loss_mask"]]) == 0] = -100
                self.assertEqual(model["actual_seq_len"], tuple(record["cu_seqlens"][1:]))
                positions = list(range(len(record["input_ids"])))
                torch.testing.assert_close(model["input_ids"], torch.tensor([record["input_ids"]]))
                torch.testing.assert_close(model["position_ids"], torch.tensor([positions]))
                torch.testing.assert_close(loss["shift_labels"], targets)
                torch.testing.assert_close(loss["loss_mask"], targets.ne(-100).long())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fixed_workers_preserve_dp_order_and_resume(self) -> None:
        """Prefetched records must not be skipped when each DP lane restores its cursor."""
        records = [{"input_ids": [index * 10 + value for value in range(1, 7)],
                    "labels": [2, -100, -100, 5, 6, -100], "cu_seqlens": [0, 3, 6]}
                   for index in range(12)]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "prepared"
            Dataset.from_list(records).save_to_disk(source)
            for sampler_type, epoch, workers in (("single", 0, 0), ("single", 0, 2), ("cyclic", 1, 2)):
                with self.subTest(sampler=sampler_type, epoch=epoch, workers=workers):
                    order = []
                    for rank in range(2):
                        baseline = _fixed_loader(source, rank, 0, sampler_type)
                        baseline.set_epoch(epoch)
                        expected = list(baseline)
                        loader = _fixed_loader(source, rank, workers, sampler_type)
                        loader.set_epoch(epoch)
                        iterator = iter(loader)
                        prefix = [next(iterator), next(iterator)]
                        state = copy.deepcopy(loader.state_dict())
                        suffix = list(iterator)
                        torch.testing.assert_close(prefix + suffix, expected, rtol=0, atol=0)
                        restored = _fixed_loader(source, rank, workers, sampler_type)
                        restored.load_state_dict(state)
                        restored.set_epoch(epoch)
                        torch.testing.assert_close(list(restored), suffix, rtol=0, atol=0)
                        order.append({batch["input_ids"][0, 0].item() for batch in expected})
                    self.assertFalse(order[0] & order[1])
                    self.assertEqual(order[0] | order[1], {row["input_ids"][0] for row in records})
