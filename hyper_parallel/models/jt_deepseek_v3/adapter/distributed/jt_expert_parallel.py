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
"""Bind the model-owned router and expert computation to Hyper EP."""

# This integration uses the Torch/HF runtime.
# pylint: disable=forbidden-backend-import
from functools import partial
from typing import Any, Callable

import torch

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.expert_parallel.experts import EPDispatchPolicy
from hyper_parallel.distributed.recipe_spec import local_compute


@local_compute
def jt_deepseek_v3_ep_compute(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any) -> Callable:
    """Bind EP execution without installing or changing model semantics."""
    del mesh, tp_mesh, cp_mesh
    if ep_mesh is None:
        raise ValueError("DeepSeek V3.2 JT requires an EP mesh")
    module.ep_group = ep_mesh.get_group("ep")
    module.ep_world = ep_mesh["ep"].size()
    executor = build_ep_compute(
        module, ep_mesh, router_fn=type(module).route, archetype_key="jt_deepseek_v3_hf",
        expected_attrs=["gate", "experts", "shared_experts", "reference_config"],
        combine=module.combine_routed, use_grouped_gemm=True,
        dispatch_policy=EPDispatchPolicy(torch.bfloat16, torch.float32, True),
        aggregate_fn=module.aggregate_experts)
    module.ep_compute = partial(executor, module)
    return type(module).forward
