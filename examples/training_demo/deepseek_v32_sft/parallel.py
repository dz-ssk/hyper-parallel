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
"""Explicit reference parameter placement through Hyper's sharding planner."""

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, TP


def plan_overrides(mtp_layers: int) -> dict[str, ModuleShardingSpec]:
    """Describe MLA head shards and replicated MTP/shared-expert projections.

    These declarations do not install an EP compute factory. A successful plan
    is a structural check; EP execution and numerical validation are separate.
    """
    attention = {
        "linear_qkv.weight": {TP: Replicate()},
        "q_b_proj.weight": {TP: Shard(0)},
        "kv_b_proj.weight": {TP: Shard(0)},
        "o_proj.weight": {TP: Shard(1)},
    }
    overrides = {
        "*.self_attn": ModuleShardingSpec(params=attention, region_dispatch=False),
        "*.self_attn.*_layernorm": ModuleShardingSpec(
            in_src={"hidden_states": {TP: Replicate()}}, in_dst={"hidden_states": {TP: Replicate()}},
            out_src={TP: Replicate()}, out_dst={TP: Replicate()},
        ),
        "*.mlp.shared_experts": ModuleShardingSpec(
            params={f"{name}.weight": {TP: Replicate()} for name in ("gate_proj", "up_proj", "down_proj")},
            in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}},
            out_src={TP: Shard(1)}, out_dst={TP: Shard(1)},
        ),
    }
    # Exact names insert new boundaries; wildcard overrides only merge existing ones.
    for index in range(mtp_layers):
        overrides[f"mtp.layers.{index}.eh_proj"] = ModuleShardingSpec(
            params={"weight": {TP: Replicate()}},
            in_src={"input": {TP: Shard(1)}}, in_dst={"input": {TP: Shard(1)}},
            out_src={TP: Shard(1)}, out_dst={TP: Shard(1)},
        )
    return overrides
