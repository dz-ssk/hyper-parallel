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
"""Build the JT model with native initialization or converted ``model.npz`` weights."""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_npu
from transformers import PreTrainedModel

from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.models.build_options import FSDP2Config, get_device_id
from hyper_parallel.models._transformers.model_builder import (
    _build_replacement_context,
    _initialize_model_weights,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config,
    JTDeepseekV3ForCausalLM,
    JTDeepseekV3MoE,
)
from hyper_parallel.models.replacement import _apply_module_replacement_actions
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import configure_context_parallel


def _load_reference_state(model: PreTrainedModel, arrays: dict[str, np.ndarray]) -> None:
    """Load converted weights that already use the model's parameter names and layouts.

    Args:
        model: Materialized JT model.
        arrays: Weights read from ``model.npz``, keyed by state-dict name.

    Raises:
        ValueError: If names, shapes or dtypes differ, a loaded value differs, or the
            embedding and LM head are tied.
    """
    expected = model.state_dict()
    if set(expected) != set(arrays):
        raise ValueError(
            f"State coverage mismatch: missing={set(expected) - set(arrays)}, "
            f"unexpected={set(arrays) - set(expected)}",
        )
    for name, value in arrays.items():
        if tuple(expected[name].shape) != value.shape:
            raise ValueError(f"Shape mismatch for {name}: {expected[name].shape} != {value.shape}")
        if torch.from_numpy(value).dtype != expected[name].dtype:
            raise ValueError(
                f"Reference dtype mismatch for {name}: "
                f"{torch.from_numpy(value).dtype} != {expected[name].dtype}",
            )
    model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in arrays.items()},
        strict=True,
    )
    for name, value in model.state_dict().items():
        if value.detach().numpy().tobytes() != arrays[name].tobytes():
            raise ValueError(f"Loaded tensor differs: {name}")
    if model.model.embed_tokens.weight is model.lm_head.weight:
        raise ValueError("JT embedding and LM head must not be tied")


def _bind_statistics_groups(model: PreTrainedModel, mesh: Any) -> None:
    """Bind the process groups over which the model reduces its training statistics.

    Args:
        model: Sharded JT model.
        mesh: Runtime mesh context of the training job.
    """
    dp_cp_mesh = mesh.dp_cp_mesh
    replica_group = None if dp_cp_mesh is None or dp_cp_mesh.size() == 1 else dp_cp_mesh.get_group()
    # QK clipping must use one maximum on every rank that holds the same attention heads.
    model.qk_clip_group = replica_group
    model.loss_sequence_parallel_size = mesh.tp_size if mesh.sequence_parallel else 1
    model.loss_tp_mesh = mesh.device_mesh["tp"] if mesh.loss_parallel else None
    # Bias updates sum unique token counts.
    model.expert_load_group = replica_group
    sequence_group = mesh.device_mesh["tp"].get_group() if mesh.sequence_parallel and mesh.tp_size > 1 else None
    # Every TP rank routes the same gathered tokens. Without loss parallelism FSDP already divides gradients by
    # these TP loss replicas, so averaging the auxiliary loss over TP as well would halve its gradient.
    auxiliary_group = mesh.device_mesh["tp"].get_group() if mesh.tp_size > 1 and mesh.loss_parallel else None
    for module in model.modules():
        if isinstance(module, JTDeepseekV3MoE):
            module.sequence_partition_group = sequence_group
            module.auxiliary_loss_group = auxiliary_group


def build_jt_model(*, config: dict[str, Any], reference_weights: str | Path | None = None,
                    distributed_setup: Any, **infrastructure_options: Any) -> PreTrainedModel:
    """Build the JT model with optional converted ``model.npz`` weights.

    Args:
        config: JT model configuration fields.
        reference_weights: Directory containing ``model.npz``. If omitted or None,
            initialize from ``config`` using the model's native initialization
            and the caller's random seed.
        distributed_setup: Trainer distributed setup.
        **infrastructure_options: Options forwarded to ``apply_model_infrastructure``.

    Returns:
        The sharded JT model.

    Raises:
        ValueError: If the weights or explicit context-parallel plan do not match the model.
    """
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    config = JTDeepseekV3Config(**config)
    mesh = distributed_setup.mesh_context
    # Source-layout FSDP owns parameters and gradient synchronization even at DP1.
    framework_setup = replace(
        distributed_setup, module_replacements=(),
        strategy_config=distributed_setup.strategy_config or FSDP2Config(),
    )
    planner, fsdp = instantiate_infrastructure(distributed_setup=framework_setup)
    with torch.device("meta"):
        model = JTDeepseekV3ForCausalLM(config)
        model, _ = _apply_module_replacement_actions(
            model,
            getattr(distributed_setup, "module_replacements", None),
            weights_mapping=get_model_conversion_mapping(model),
            context=_build_replacement_context(distributed_setup, None),
        )
    model.to_empty(device="cpu")
    if reference_weights is None:
        _initialize_model_weights(model)
    else:
        with np.load(Path(reference_weights) / "model.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        _load_reference_state(model, arrays)

    device = torch.device(mesh.device_mesh.device_type, get_device_id())
    model.to(device)
    model = apply_model_infrastructure(
        model,
        mesh=mesh,
        sharding_planner=planner,
        fsdp2_manager=fsdp,
        distributed_setup=framework_setup,
        device=device,
        is_meta_device=False,
        is_hf_model=True,
        **infrastructure_options,
    )
    _bind_statistics_groups(model, mesh)
    configure_context_parallel(model, mesh)
    return model
