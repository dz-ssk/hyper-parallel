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
"""JT DeepSeek-V3 expert-parallel compute on the framework EP skeleton.

The JT MoE router and routed-branch combine run through the public
``build_ep_compute``; recipes bind it to the routed MoE layers with ``when: ep``
``plan_overrides`` entries.
"""

from functools import partial
from typing import Any, Callable, Optional

import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute


def _fixed_route_splits(token_count: int, *, expert_count: int, top_k: int,
                        padding: int, ep_size: int, ep_rank: int) -> tuple[list[int], list[int]]:
    """Count cyclic real routes and the pad prefix without reading device tensors."""
    if expert_count <= 0 or ep_size <= 0:
        raise ValueError("Fixed routing requires positive expert and EP counts")
    if token_count < padding or padding not in (0, expert_count):
        raise ValueError("Fixed routing requires a valid pad prefix and token count")
    if expert_count % ep_size or not 0 <= ep_rank < ep_size or not 1 <= top_k <= expert_count:
        raise ValueError("Fixed routing requires divisible experts, valid EP rank and top-k")
    local_experts = expert_count // ep_size
    quotient, remainder = divmod((token_count - padding) * top_k, expert_count)
    base = quotient + (top_k if padding else 0)
    send = [base * local_experts + min(max(remainder - rank * local_experts, 0), local_experts)
            for rank in range(ep_size)]
    return send, [send[ep_rank]] * ep_size


def _static_split_provider(module: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any) -> Optional[Callable]:
    """Require equal CP shards across exactly the same ordered EP group."""
    if not module.config.expert_static_splits:
        return None
    # Fixed routing is provided by the separate JT benchmark routing feature.
    if not getattr(module.config, "fix_router", False) or not callable(getattr(module, "fixed_routing", None)):
        raise ValueError("expert_static_splits requires JT fix_router support (PR #161) and fix_router=true")
    if (tp_mesh is not None and tp_mesh.size() != 1) or cp_mesh is None:
        raise ValueError("expert_static_splits requires fix_router, TP=1 and a CP mesh")
    ep_group = ep_mesh["ep"].get_group()
    cp_group = cp_mesh.get_group()
    if dist.get_process_group_ranks(ep_group) != dist.get_process_group_ranks(cp_group):
        raise ValueError("expert_static_splits requires identical ordered CP and EP groups")
    provider = partial(_fixed_route_splits, expert_count=module.config.n_routed_experts,
                       top_k=module.config.num_experts_per_tok, padding=module.padding,
                       ep_size=ep_mesh["ep"].size(), ep_rank=dist.get_rank(ep_group))
    provider(module.padding)
    return provider


@local_compute
def jt_deepseek_v3_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                                 ep_mesh: Any) -> Callable:
    """Bind expert-parallel execution of a JT MoE layer to ``ep_mesh``.

    Raises:
        ValueError: If no expert-parallel mesh is given.
    """
    del mesh
    if ep_mesh is None:
        raise ValueError("JT requires an EP mesh")
    executor = build_ep_compute(
        module,
        ep_mesh,
        router_fn=type(module).route,
        archetype_key="jt_deepseek_v3_hf",
        expected_attrs=["gate", "experts", "shared_experts", "config"],
        combine=module.combine_routed,
        use_grouped_gemm=True,
        split_fn=_static_split_provider(module, tp_mesh, cp_mesh, ep_mesh),
    )
    module.ep_compute = partial(executor, module)
    return type(module).forward


__all__ = ["jt_deepseek_v3_ep_compute_fn"]