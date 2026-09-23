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
"""Pretokenized SFT archives with explicit, already-shifted labels and masks."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzSFTDataset(Dataset):
    """Load an offline token archive without tokenization, packing or label shifting."""

    def __init__(self, data_path: str | Path) -> None:
        """Validate and retain a pretokenized archive on CPU.

        Args:
            data_path: NPZ archive with tokens, labels and mask.
        """
        with np.load(data_path, allow_pickle=False) as archive:
            self.arrays = {name: archive[name].copy() for name in ("input_ids", "labels", "loss_mask")}
        shapes = [value.shape for value in self.arrays.values()]
        if any(len(shape) != 2 or shape != shapes[0] for shape in shapes):
            raise ValueError("input_ids, pre-shifted labels and loss_mask require matching [samples, tokens] shapes")
        if not shapes[0][0] or not shapes[0][1]:
            raise ValueError("SFT archive must not be empty")
        if any(not np.issubdtype(self.arrays[key].dtype, np.integer) for key in ("input_ids", "labels")):
            raise ValueError("input_ids and labels must be integers")
        mask = self.arrays["loss_mask"]
        if not np.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("loss_mask must be finite and nonnegative")

    def __len__(self) -> int:
        """Return the number of complete sequences."""
        return self.arrays["input_ids"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return an independent sample with explicit token/mask dtypes.

        Args:
            index: Sample index.
        """
        return {key: torch.from_numpy(value[index].copy()).to(
            torch.float32 if key == "loss_mask" else torch.int64) for key, value in self.arrays.items()}


class PreShiftedSFTBatch:
    """Keep precomputed labels/masks intact for a CP1/PP1 Trainer recipe.

    FixedBatchDataLoader's DP sampler selects identical sample IDs within each
    TP group. Each rank reads the same immutable archive; this adapter performs
    no resampling or sequence slicing. Packed/CP/PP inputs require another adapter.
    """

    def __init__(self, mesh_context: Any, device: Any) -> None:
        """Validate the supported data-layout contract.

        Args:
            mesh_context: Resolved parallel mesh.
            device: Destination device.
        """
        if mesh_context.cp_size != 1 or mesh_context.pp_size != 1:
            raise ValueError("PreShiftedSFTBatch requires CP1 and PP1")
        self.device = device

    def __call__(self, data_iterator: Any) -> tuple[dict, dict]:
        """Move one collated sample without modifying labels or masks.

        Args:
            data_iterator: Iterator over configured batches.
        """
        batch = {key: value.to(self.device) for key, value in next(data_iterator).items()}
        return {"input_ids": batch["input_ids"]}, {
            "labels": batch["labels"], "shift_labels": batch["labels"], "loss_mask": batch["loss_mask"],
        }
