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
"""MTP training parity workers for real TP/SP, CP, EP and FSDP layouts."""

from copy import deepcopy
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3ForCausalLM

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.expert_parallel.recipes import deepseekv3_ep_compute_fn
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.core.pipeline_parallel.scheduler import ScheduleGPipe
from hyper_parallel.distributed.mtp_pipeline import build_mtp_pipeline_stage
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.components.modules.mtp_training import attach_mtp
from hyper_parallel.trainer.config import Target
from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from tests.common.mla_cp_utils import replace_and_load_mla


def _model() -> torch.nn.Module:
    config = DeepseekV3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=8, q_lora_rank=32, kv_lora_rank=32,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, first_k_dense_replace=1,
        n_routed_experts=8, n_shared_experts=1, moe_intermediate_size=32,
        num_experts_per_tok=2, n_group=1, topk_group=1, max_position_embeddings=256,
        attention_dropout=0.0, attention_bias=False, tie_word_embeddings=False,
        use_cache=False, rope_interleave=True, mtp_num_layers=2,
        mtp_loss_factor=0.3, mtp_loss_normalization="sequence",
    )
    config._attn_implementation = "sdpa"
    model = attach_mtp(DeepseekV3ForCausalLM(config), 2, mtp_boolean_mask=True).to(dtype=torch.bfloat16)
    return replace_and_load_mla(model, match=("*.self_attn",))[0]


def _case(mesh: MeshContext, device: torch.device, recompute: bool) -> None:
    torch.manual_seed(314)
    parallel = _model().to(device)
    reference = deepcopy(parallel)
    placement = {"tp": Shard(1) if mesh.sequence_parallel else Replicate(),
                 "cp": Shard(1), "ep": Replicate()}
    overrides = {
        "*.self_attn": ModuleShardingSpec(inner_target="self", region_dispatch=False,
            inner_wrapper=Target(mla_cp_wrapper,
                target_path="hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel.mla_cp_wrapper",
                strategy="expanded_ulysses", overlap=False)),
        "mtp.layers.*.eh_proj": ModuleShardingSpec(
            params={"weight": {"tp": Replicate(), "cp": Replicate(), "ep": Replicate()}},
            in_src={"input": placement}, in_dst={"input": placement},
            out_src={"output": placement}, out_dst={"output": placement}),
    }
    if mesh.ep_size > 1:
        for name, module in parallel.named_modules():
            if isinstance(module, DeepseekV3MoE):
                overrides[name] = ModuleShardingSpec(
                    region_dispatch=False, local_compute_fn=Target(deepseekv3_ep_compute_fn,
                        target_path="hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn"))
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        parallel, mesh.device_mesh, tp_size=mesh.tp_size, cp_size=mesh.cp_size,
        ep_size=mesh.ep_size, sequence_parallel=mesh.sequence_parallel)
    stacked = {}
    for prefix, spec in plan.modules.items():
        for target, sources in getattr(spec, "_ep_stack", {}).items():
            stacked[f"{prefix}.{target}"] = [f"{prefix}.{source}" for source in sources]
    parallel, source_info = apply_sharding_plan(parallel, plan, mesh)
    parallel.mtp_cp_size = mesh.cp_size
    parallel.mtp_cp_group = mesh.cp_mesh.get_group()
    if recompute:
        _apply_activation_checkpointing(parallel, "full")
    manager = FSDP2Manager(FSDP2Config(
        dp_shard_size=mesh.dp_shard_size, edp_shard_size=mesh.edp_shard_size,
        reshard_after_forward=True, mix_precision=FSDP2MixedPrecisionConfig(
            param_dtype="bfloat16", reduce_dtype="float32", cast_forward_inputs=False)), mesh)
    manager.parallelize(parallel, source_info)
    sequence = 32
    cp_rank, dp_rank = mesh.cp_mesh.get_local_rank(), mesh.device_mesh["dp"].get_local_rank()
    begin, end = cp_rank * sequence // mesh.cp_size, (cp_rank + 1) * sequence // mesh.cp_size
    positions = torch.arange(sequence, device=device)[None]
    allowed = torch.ones(1, 1, sequence, sequence, dtype=torch.bool, device=device).tril()
    for microbatch in range(2):
        torch.manual_seed(204 + microbatch)
        tokens = torch.randint(0, 128, (mesh.dp_size, sequence)).to(device)
        expected = reference(tokens, labels=tokens, attention_mask=allowed, position_ids=positions)
        (sum(expected.loss.values()) / 2).backward()
        local = tokens[dp_rank:dp_rank + 1, begin:end].contiguous()
        with SkipDTensorDispatch():
            actual = parallel(local, labels=local, attention_mask=allowed,
                              position_ids=positions[:, begin:end].contiguous())
            counts = {"foundation_tokens": local.new_tensor(local.numel() - int(end == sequence)),
                      "mtp_depth1_tokens": local.new_tensor(local.numel()),
                      "mtp_depth2_tokens": local.new_tensor(local.numel())}
            losses = mean_global_loss(actual.loss, counts, {key: value * 2 for key, value in counts.items()}, mesh)
            sum(losses.values()).backward()
        if dist.get_rank() == 0:
            print(json.dumps({"micro": microbatch, "actual_loss": {k: v.item() for k, v in losses.items()},
                              "reference_loss": {k: (v / 2).item() for k, v in expected.loss.items()},
                              "logits_max_error": (actual.logits - expected.logits[
                                  dp_rank:dp_rank + 1, begin:end]).abs().max().item()}), flush=True)
        for name, loss in losses.items():
            torch.testing.assert_close(loss, expected.loss[name] / 2, atol=3e-3, rtol=0.02,
                                       msg=lambda message: f"global loss {name}: {message}")
    gradients, expected_gradients = [], []
    for name, parameter in parallel.named_parameters():
        actual = parameter.grad.full_tensor()
        if name in stacked:
            expected = torch.stack([reference.get_parameter(source).grad for source in stacked[name]])
        else:
            expected = reference.get_parameter(name).grad
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=0.06, msg=name)
        gradients.append(actual.detach().cpu().double().flatten())
        expected_gradients.append(expected.detach().cpu().double().flatten())
    actual, expected = torch.cat(gradients), torch.cat(expected_gradients)
    relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    assert relative < 0.025, f"MTP gradient relative L2={relative.item()}"
    parallel.reset_iter_state()
    if dist.get_rank() == 0:
        print(json.dumps({"mtp_layers": 2, "tp": mesh.tp_size, "cp": mesh.cp_size,
                          "ep": mesh.ep_size, "sp": mesh.sequence_parallel,
                          "fsdp_shards": mesh.dp_shard_size,
                          "recompute": recompute, "gradient_relative_l2": relative.item()}), flush=True)


def test_mtp_parallel_training() -> None:
    """Compare complete trunk/MTP losses and gradients over two microbatches."""
    # Optional NPU runtime is only needed inside a launched worker.
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        ep = int(os.environ.get("MTP_TEST_EP", "2"))
        for shards, sp, recompute in ((4, False, False), (4, True, True), (2, True, True)):
            mesh = MeshContext(tp_size=2, cp_size=2, dp_size=2, dp_shard_size=shards,
                               dp_replicate_size=4 // shards, ep_size=ep,
                               edp_shard_size=shards, sequence_parallel=sp)
            mesh.build_meshs("npu", dist.get_world_size())
            _case(mesh, device, recompute)
    finally:
        dist.destroy_process_group()


def test_mtp_pipeline_training() -> None:
    """Exercise actual PP sends, backwards and shared-embedding gradient sync."""
    # Optional NPU runtime is only needed inside a launched worker.
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(314)
        model = _model().to(device)
        reference = deepcopy(model)
        stage = build_mtp_pipeline_stage(model, stage_index=dist.get_rank(), stage_count=2,
                                         device=device, loss_scale=0.5)
        schedule = ScheduleGPipe([stage], 2, kwargs_batch_dim={"input_ids": 0, "labels": 0})
        torch.manual_seed(204)
        tokens = torch.randint(0, 128, (2, 32)).to(device)
        expected = sum(reference(tokens, labels=tokens).loss.values())
        expected.backward()
        losses = schedule.run(tokens, input_ids=tokens, labels=tokens)
        if stage.is_last_stage:
            torch.testing.assert_close(sum(losses), expected, atol=3e-3, rtol=0.02)
        gradients, expected_gradients = [], []
        for name, parameter in model.named_parameters():
            original = reference.get_parameter(name)
            actual = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            expected_grad = torch.zeros_like(original) if original.grad is None else original.grad
            torch.testing.assert_close(actual, expected_grad, atol=4e-3, rtol=0.06, msg=name)
            gradients.append(actual.detach().cpu().double().flatten())
            expected_gradients.append(expected_grad.detach().cpu().double().flatten())
        actual, expected_grad = torch.cat(gradients), torch.cat(expected_gradients)
        relative = (actual - expected_grad).norm() / expected_grad.norm().clamp_min(1e-12)
        assert relative < 0.025, f"PP gradient relative L2={relative.item()}"
        print(json.dumps({"pp": 2, "rank": dist.get_rank(), "mtp_layers": 2,
                          "gradient_relative_l2": relative.item()}), flush=True)
    finally:
        dist.destroy_process_group()
