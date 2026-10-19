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
"""Complete optimizer-batch tails for fixed-sample Dataset loading."""

from collections.abc import Iterator
from types import SimpleNamespace
import unittest

from hyper_parallel.data.batching.build_dataloader import (
    FixedBatchDataLoader, TokenBatchLoader, build_dataloader,
)
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.trainer.config.target import Target
from tests.common.mark_utils import arg_mark


def _sampler(**overrides):
    options = {"total_samples": 100, "micro_batch_size": 1, "global_batch_size": 6,
               "dp_rank": 0, "dp_world_size": 2, "seed": 42}
    options.update(overrides)
    return build_dataset_batch_sampler(**options)


class _TrackedDataset:
    """Count real item reads and optionally reject reads of the unusable tail."""

    def __init__(self, size: int = 100, stop: int = 100) -> None:
        """Set the advertised length and first forbidden read index."""
        self.size = size
        self.stop = stop
        self.reads = []

    def __len__(self) -> int:
        """Return the advertised source length."""
        return self.size

    def __getitem__(self, index: int) -> int:
        """Record a valid source read or reject a dropped-tail read."""
        if index >= self.stop:
            raise AssertionError(f"Tail read: index={index}, stop={self.stop}")
        self.reads.append(index)
        return index


class _Stream:
    """Iterable-only source with no mapping-style random access."""

    def __iter__(self) -> Iterator[int]:
        """Yield a small deterministic source stream."""
        return iter(range(10))


class TestCompleteOptimizerBatches(unittest.TestCase):
    """Check order, resume, DP agreement and loader selection without accelerators."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_retained_order_and_length(self):
        """Feature: complete updates.
        Description: Compare all DP lanes across sequential and both cyclic orders.
        Expectation: Each retained sequence is exactly the old epoch prefix.
        """
        for mode, sharding in (("single", False), ("cyclic", False), ("cyclic", True)):
            for micro, dp, gbs, total in ((1, 2, 6, 100), (2, 4, 24, 101), (1, 4, 12, 96)):
                for rank in range(dp):
                    with self.subTest(mode=mode, sharding=sharding, micro=micro, rank=rank):
                        options = {"sampler_type": mode, "data_sharding": sharding,
                                   "micro_batch_size": micro, "dp_world_size": dp, "global_batch_size": gbs,
                                   "total_samples": total, "dp_rank": rank}
                        reference = _sampler(**options)
                        sampler = _sampler(**options, drop_last_global_batch=True)
                        stop = total // gbs * gbs
                        expected_length = stop // (micro * dp)
                        for epoch in (0, 1):
                            reference.set_epoch(epoch)
                            sampler.set_epoch(epoch)
                            self.assertEqual(len(sampler), expected_length)
                            expected = list(reference)[:expected_length]
                            self.assertEqual(list(sampler), expected)
                            self.assertEqual(sampler.consumed_samples, stop)
                            self.assertEqual(len(sampler), 0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_tail_is_not_loaded(self):
        """Feature: avoid unusable IO.
        Description: Exhaust a real stateful loader at GBS boundaries with DP slicing.
        Expectation: Exactly the retained samples are read and the next epoch restarts.
        """
        all_reads = []
        for rank in range(2):
            dataset = _TrackedDataset(stop=96)
            loader = FixedBatchDataLoader(dataset, batch_sampler=_sampler(dp_rank=rank, drop_last_global_batch=True),
                                         collate_fn=list, drop_last_global_batch=True)
            self.assertEqual(len(list(loader)), 48)
            self.assertEqual(dataset.reads, list(range(rank, 96, 2)))
            all_reads.extend(dataset.reads)
            loader.set_epoch(1)
            self.assertEqual(next(iter(loader)), [rank])
        self.assertEqual(sorted(all_reads), list(range(96)))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_prefetch_respects_stop(self):
        """Feature: worker prefetch.
        Description: Use a real worker that raises if it reads any dropped sample.
        Expectation: Prefetch never requests the unusable tail.
        """
        dataset = _TrackedDataset(stop=96)
        loader = FixedBatchDataLoader(dataset, batch_sampler=_sampler(drop_last_global_batch=True),
                                     collate_fn=list, drop_last_global_batch=True, num_workers=1, prefetch_factor=2)
        self.assertEqual(list(loader), [[index] for index in range(0, 96, 2)])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_resume_at_update_and_source_boundary(self):
        """Feature: resumable sampler.
        Description: Save inside or after an accumulation window for every cyclic policy.
        Expectation: Restored sampling equals the exact unconsumed reference suffix.
        """
        for mode, sharding in (("single", False), ("cyclic", False), ("cyclic", True)):
            for consumed_micro in (1, 3, 48):
                with self.subTest(mode=mode, sharding=sharding, consumed_micro=consumed_micro):
                    options = {"sampler_type": mode, "data_sharding": sharding, "drop_last_global_batch": True}
                    sampler = _sampler(**options)
                    sampler.enable_source_batch_resume()
                    iterator = iter(sampler)
                    for _ in range(consumed_micro):
                        next(iterator)
                    state = sampler.state_dict()
                    restored = _sampler(**options)
                    restored.enable_source_batch_resume()
                    restored.load_state_dict(state)
                    restored.set_epoch(0)
                    self.assertEqual(len(restored), 48 - consumed_micro)
                    self.assertEqual(list(restored), list(iterator))
                    restored.set_epoch(1)
                    self.assertEqual(len(restored), 48)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_fixed_loader_checkpoint(self):
        """Feature: stateful loader checkpoint.
        Description: Resume after one whole optimizer update and at end of epoch.
        Expectation: No duplicate retained batches and no dropped-tail reads.
        """
        for consumed_micro in (3, 48):
            with self.subTest(consumed_micro=consumed_micro):
                loader = FixedBatchDataLoader(_TrackedDataset(stop=96), collate_fn=list,
                                             batch_sampler=_sampler(drop_last_global_batch=True),
                                             drop_last_global_batch=True)
                iterator = iter(loader)
                for _ in range(consumed_micro):
                    next(iterator)
                state = loader.state_dict()
                restored = FixedBatchDataLoader(_TrackedDataset(stop=96), collate_fn=list,
                                               batch_sampler=_sampler(drop_last_global_batch=True),
                                               drop_last_global_batch=True)
                restored.load_state_dict(state)
                self.assertEqual(list(restored), list(iterator))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_checkpoint_policy_and_legacy_default(self):
        """Feature: checkpoint compatibility.
        Description: Load old default states and reject a changed tail policy.
        Expectation: Default states stay unchanged; policy changes fail explicitly.
        """
        reference = _sampler()
        legacy = reference.state_dict()
        self.assertNotIn("drop_last_global_batch", legacy)
        _sampler().load_state_dict(legacy)
        enabled = _sampler(drop_last_global_batch=True)
        with self.assertRaisesRegex(ValueError, "unchanged drop_last_global_batch"):
            enabled.load_state_dict(legacy)
        with self.assertRaisesRegex(ValueError, "unchanged drop_last_global_batch"):
            reference.load_state_dict(enabled.state_dict())
        self.assertEqual(len(list(_sampler())), 50)
        partial = _sampler(total_samples=5, micro_batch_size=2, global_batch_size=4, drop_last=False)
        self.assertEqual(list(partial), [[0, 1], [4]])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_small_dataset_and_mapping(self):
        """Feature: boundary datasets.
        Description: Use fewer samples than one micro/update, and a reversed index map.
        Expectation: Small sources are not read; mapping is applied only to kept indices.
        """
        for mode in ("single", "cyclic"):
            for total in (1, 2, 5):
                sampler = _sampler(total_samples=total, sampler_type=mode, drop_last_global_batch=True)
                self.assertEqual(len(sampler), 0)
                self.assertEqual(list(sampler), [])
        sampler = _sampler(index_mapping=list(reversed(range(100))), drop_last_global_batch=True)
        self.assertEqual(list(sampler), [[99 - index] for index in range(0, 96, 2)])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_builder_train_only(self):
        """Feature: Trainer target integration.
        Description: Build three real fixed-loader splits with one shared configuration.
        Expectation: Only training drops the incomplete optimizer batch.
        """
        for enabled in (False, True):
            target = Target(FixedBatchDataLoader, target_path="test.FixedBatchDataLoader",
                            drop_last=True, drop_last_global_batch=enabled)
            datasets = tuple(_TrackedDataset(size=10, stop=10) for _ in range(3))
            loaders, samplers = build_dataloader(
                target, datasets=datasets, collate_fn=list,
                mesh_context=SimpleNamespace(dp_size=2, dp_rank=0),
                training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=6, seed=42),
            )
            self.assertEqual([len(list(loader)) for loader in loaders], [3 if enabled else 5, 5, 5])
            self.assertEqual([sampler.drop_last_global_batch for sampler in samplers], [enabled, False, False])
            self.assertEqual([loader.drop_last_global_batch for loader in loaders], [enabled, False, False])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_invalid_policy_and_unsupported_loaders(self):
        """Feature: configuration validation.
        Description: Reject invalid flags, non-dropping batches, dynamic and iterable sources.
        Expectation: An explicit error occurs before reading any source items.
        """
        for flag in (1, "true", None):
            with self.assertRaisesRegex(ValueError, "drop_last_global_batch"):
                _sampler(drop_last_global_batch=flag)
        with self.assertRaisesRegex(ValueError, "drop_last_global_batch"):
            _sampler(drop_last=False, drop_last_global_batch=True)
        with self.assertRaisesRegex(ValueError, "matching batch_sampler"):
            FixedBatchDataLoader(_TrackedDataset(), drop_last_global_batch=True)
        for loader_class, dataset, drop_last in ((TokenBatchLoader, _TrackedDataset(), True),
                                                 (FixedBatchDataLoader, _Stream(), True),
                                                 (FixedBatchDataLoader, _TrackedDataset(), False)):
            target = Target(loader_class, target_path="test.loader", drop_last=drop_last, drop_last_global_batch=True)
            with self.assertRaisesRegex(ValueError, "drop_last_global_batch"):
                build_dataloader(target, datasets=(dataset, None, None), collate_fn=list,
                                 mesh_context=SimpleNamespace(dp_size=2, dp_rank=0),
                                 training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=6, seed=42))