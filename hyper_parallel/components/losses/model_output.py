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
"""Loss module that reads the loss produced by a model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
# pylint: disable-next=forbidden-backend-import
import torch

from hyper_parallel.data.constants import IGNORE_INDEX


class ModelOutputLoss(torch.nn.Module):
    """Return the loss field from a Transformers-style model output."""

    def forward(  # pylint: disable=unused-argument
        self,
        *,
        model_output: Any,
        labels: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read the model-computed loss.

        Args:
            model_output: Model output exposing a ``loss`` attribute.
            labels: Labels associated with the output. This default loss keeps
                the argument only to share the trainer-facing call signature
                with replaceable loss modules.

        Returns:
            The loss tensor or named loss mapping from ``model_output.loss``.
        """
        local_loss = model_output.loss
        if labels is None or not isinstance(local_loss, torch.Tensor):
            return local_loss

        # Causal LM loss shifts labels by one position. A CP-local slice may
        # therefore contain no trainable target even when other CP ranks do.
        has_valid_labels = labels[..., 1:].ne(IGNORE_INDEX).any()
        local_loss = torch.where(has_valid_labels, local_loss, torch.zeros_like(local_loss))

        return local_loss


class ModelComputedLoss(ModelOutputLoss):
    """Adapt models that already compute a complete objective inside forward.

    Unlike the default causal-LM adapter, this class does not infer shifted
    targets or zero a combined objective: auxiliary losses may remain valid
    when all supervised tokens are masked.
    """

    def __init__(self, input_mapping: Mapping[str, str] | None = None,
                 loss_group_attribute: str | None = None) -> None:
        """Configure model input names and optional vocabulary-parallel binding.

        Args:
            input_mapping: Model argument to public batch field mapping; None
                passes existing model inputs through unchanged.
            loss_group_attribute: Optional model attribute receiving the TP
                group when loss parallelism is enabled. No binding by default.
        """
        super().__init__()
        self.input_mapping = None if input_mapping is None else dict(input_mapping)
        self.loss_group_attribute = loss_group_attribute
        self.loss_group = None

    def bind_model(self, model: torch.nn.Module, distributed_setup: Any = None) -> None:
        """Bind an explicitly requested vocabulary group without default-group fallback.

        Args:
            model: Constructed model owning the vocabulary objective.
            distributed_setup: Trainer mesh and loss parallelism settings.
        """
        if self.loss_group_attribute is None:
            return
        mesh = getattr(distributed_setup, "mesh_context", None)
        self.loss_group = None
        if mesh is not None and getattr(mesh, "tp_size", 1) > 1 and getattr(mesh, "loss_parallel", False):
            self.loss_group = mesh.device_mesh["tp"].get_group()
        setattr(model, self.loss_group_attribute, self.loss_group)

    def prepare_model_inputs(self, model_inputs: dict, loss_inputs: dict) -> dict:
        """Map public batch fields into the model signature without shifting or masking.

        Args:
            model_inputs: Forward fields from the public batch runtime.
            loss_inputs: Supervision and token-accounting fields from that runtime.
        """
        if self.input_mapping is None:
            return dict(model_inputs)
        fields = {**model_inputs, **loss_inputs}
        missing = set(self.input_mapping.values()) - fields.keys()
        if missing:
            raise ValueError(f"Model loss input mapping is missing batch fields: {sorted(missing)}")
        return {target: fields[source] for target, source in self.input_mapping.items()}

    def forward(self, *, model_output: Any, labels: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Use the model's complete objective without imposing causal label semantics.

        Args:
            model_output: Output containing the already combined objective.
            labels: Unused; the model already applied its own target semantics.
        """
        del labels
        return super().forward(model_output=model_output, labels=None)


__all__ = ["ModelOutputLoss", "ModelComputedLoss"]
