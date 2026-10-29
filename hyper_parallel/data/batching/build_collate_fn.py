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
"""Build final batch collators for Indexed and Online sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import default_collate

from hyper_parallel.data.constants import IGNORE_INDEX, OMNI_CONCAT_FIELDS


def _get_sequence_parallel_size(mesh_context: Any | None) -> int:
    """Return the unified sequence-sharding degree used for batch alignment."""
    if mesh_context is None:
        return 1

    cp_size = int(getattr(mesh_context, "cp_size", 1))
    tp_size = int(getattr(mesh_context, "tp_size", 1))
    sequence_parallel = bool(getattr(mesh_context, "sequence_parallel", False))
    if cp_size <= 0 or tp_size <= 0:
        raise ValueError("mesh_context tp_size and cp_size must be positive")

    return cp_size * (tp_size if sequence_parallel else 1)


@dataclass
class TextPackingCollator:
    """Collate unpadded text records, preserving internal document boundaries.

    Args:
        sequence_parallel_size: Pad each row to this positive alignment.
        pack: Concatenate records into one row, or pad one record per row.
        pad_token_id: Token written only to newly created row tails.
    """

    sequence_parallel_size: int = 1
    pack: bool = True
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        """Validate layout settings before constructing batches."""
        if (isinstance(self.sequence_parallel_size, bool)
                or not isinstance(self.sequence_parallel_size, int) or self.sequence_parallel_size <= 0):
            raise ValueError("sequence_parallel_size must be a positive integer")
        if not isinstance(self.pack, bool):
            raise ValueError("pack must be a boolean")
        if isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int) or self.pad_token_id < 0:
            raise ValueError("pad_token_id must be a nonnegative integer")

    @staticmethod
    def _boundaries(sample: Mapping[str, Any]) -> torch.Tensor:
        try:
            tokens, labels = sample["input_ids"], sample["labels"]
        except KeyError as error:
            raise ValueError("Text records require input_ids and labels") from error
        if (not isinstance(tokens, torch.Tensor) or not isinstance(labels, torch.Tensor)
                or tokens.ndim != 1 or not tokens.numel() or tokens.shape != labels.shape
                or tokens.dtype != torch.int64 or labels.dtype != torch.int64):
            raise ValueError("Text records require matching nonempty 1D int64 tokens and labels")
        boundaries = torch.as_tensor(sample.get("cu_seq_lens", [0, tokens.numel()]), device=tokens.device)
        if (boundaries.dtype not in (torch.int32, torch.int64) or boundaries.ndim != 1
                or boundaries.numel() < 2 or boundaries[0] != 0 or boundaries[-1] != tokens.numel()
                or torch.any(boundaries[1:] <= boundaries[:-1])):
            raise ValueError("Sample boundaries must strictly increase from zero to token count")
        return boundaries.to(torch.int64)

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Return [B,S] tokens/labels, global boundaries and real per-row lengths.

        Inputs must be unpadded. Prompt targets may be -100, so real lengths
        cannot be inferred from labels or from the value of a token ID.
        """
        if not model_samples:
            raise ValueError("model_samples must contain at least one text sample")
        sample_boundaries = [self._boundaries(sample) for sample in model_samples]
        sample_lengths = [sample["input_ids"].numel() for sample in model_samples]
        valid_lengths = [sum(sample_lengths)] if self.pack else sample_lengths
        row_length = max(valid_lengths)
        row_length += (-row_length) % self.sequence_parallel_size
        if row_length * len(valid_lengths) > torch.iinfo(torch.int32).max:
            raise ValueError("Flattened batch boundaries must fit int32")
        device = model_samples[0]["input_ids"].device
        shape = (len(valid_lengths), row_length)
        batch = {
            "input_ids": torch.full(shape, self.pad_token_id, dtype=torch.int64, device=device),
            "labels": torch.full(shape, IGNORE_INDEX, dtype=torch.int64, device=device),
            "valid_lengths": torch.tensor(valid_lengths, dtype=torch.int64, device=device),
        }
        ends = []
        row_offset = 0
        for index, (sample, boundaries, length) in enumerate(zip(model_samples, sample_boundaries, sample_lengths)):
            row = 0 if self.pack else index
            start = row_offset if self.pack else 0
            for field in ("input_ids", "labels"):
                batch[field][row, start:start + length] = sample[field]
            ends.append(boundaries[1:] + row * row_length + start)
            row_offset += length
            if not self.pack and length < row_length:
                ends.append(boundaries.new_tensor([(row + 1) * row_length]))
        if self.pack and valid_lengths[0] < row_length:
            ends.append(sample_boundaries[0].new_tensor([row_length]))
        batch["cu_seq_lens"] = torch.cat((sample_boundaries[0].new_zeros(1), *ends)).to(torch.int32)
        return batch


class OmniCollator:
    """Batch already prepared Omni samples without performing sequence packing."""

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Stack token fields and concatenate variable-length modality fields."""
        if not model_samples:
            raise ValueError("model_samples must contain at least one Omni sample")

        sample_fields = set()
        for model_sample in model_samples:
            sample_fields.update(model_sample)

        batch = {}
        for field in sample_fields:
            values = []
            for model_sample in model_samples:
                if field in model_sample:
                    values.append(model_sample[field])

            if field in OMNI_CONCAT_FIELDS:
                batch[field] = torch.cat(values, dim=0)
            else:
                batch[field] = default_collate(values)

        return batch


def build_default_collate_fn() -> Callable[[list[Any]], Any]:
    """Build ordinary PyTorch collation for already prepared samples.

    Returns:
        PyTorch default collation.
    """
    collate_fn = default_collate

    return collate_fn


# Indexed samples are already prepared and use the same ordinary collation.
build_indexed_collate_fn = build_default_collate_fn


def build_omni_collate_fn() -> OmniCollator:
    """Build ordinary collation for encoded or finally packed Omni samples."""
    collate_fn = OmniCollator()
    return collate_fn


def build_online_text_collate_fn(mesh_context: Any | None = None) -> TextPackingCollator:
    """Build Online text collation shared by fixed N and dynamic K batching.

    Packing concatenates ``input_ids`` and ``labels`` and emits ``cu_seq_lens``.
    Tail padding is derived from the runtime topology rather than user config.
    The unified alignment size is ``cp_size * tp_size`` when TP sequence
    parallelism is enabled, otherwise it is ``cp_size``. Real sample boundaries
    remain unchanged and one synthetic final boundary covers the physical
    padding segment.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.

    Returns:
        A collator producing one forward-backward batch.
    """
    sequence_parallel_size = _get_sequence_parallel_size(mesh_context)
    collate_fn = TextPackingCollator(sequence_parallel_size=sequence_parallel_size)

    return collate_fn


def build_pretokenized_collate_fn(
    mesh_context: Any | None = None, *, pack: bool = True, pad_token_id: int = 0,
) -> TextPackingCollator:
    """Build prepared text collation for packed or ordinary [B,S] batches.

    Args:
        mesh_context: Runtime topology determining CP/TP-SP alignment.
        pack: Concatenate records when true; otherwise keep one record per row.
        pad_token_id: Explicit row-tail padding token, independent of EOS/EOD.

    Returns:
        Collator retaining every internal document boundary and real row length.
    """
    return TextPackingCollator(_get_sequence_parallel_size(mesh_context), pack, pad_token_id)
