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
"""Exercise YAML -> real model -> optimizer and Trainer MTP metrics on CPU."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType
import yaml
from transformers import LlamaConfig, Qwen2Config
from transformers import LlamaForCausalLM

from hyper_parallel.components.losses import ModelOutputLoss, compute_mtp_loss
from hyper_parallel.distributed.activation_checkpoint import (
    _apply_activation_checkpointing, _find_transformer_block_modules,
)
from hyper_parallel.distributed.mtp import shift_mtp_tokens
from hyper_parallel.distributed.mtp_pipeline import partition_mtp_model, MTPPipelineModule
from hyper_parallel.data.batching.mtp import MTPTextParallelBatch, prepare_mtp_inputs
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.components.modules.mtp_training import attach_mtp, _decoder_inputs, _targets
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.callbacks.tqdm_callback import TqdmCallback
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


class TestMTPTraining(unittest.TestCase):
    """Test the production builder, actual HF decoders, loss gradients and logging."""

    def setUp(self) -> None:
        """Use tiny dimensions and CPU without mocking model construction."""
        torch.manual_seed(17)
        checkpoint_device = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        self.addCleanup(DefaultDeviceType.set_device_type, checkpoint_device)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        cpu = patch("hyper_parallel.models._transformers.auto_model._current_device",
                    return_value=torch.device("cpu"))
        cpu.start()
        self.addCleanup(cpu.stop)
        recipe = Path(__file__).resolve().parents[3] / "examples/mtp.yaml"
        self.raw = yaml.safe_load(recipe.read_text())
        self.ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]])

    def build(self, depth: int | None = 1, factor: float = 0.3) -> torch.nn.Module:
        """Use the same configuration resolver and target invoked by Trainer."""
        raw = copy.deepcopy(self.raw)
        raw["model"]["mtp_num_layers"] = depth
        raw["model"]["mtp_loss_factor"] = factor
        return resolve_config(raw).model.build(model_init_dtype="float32")

    def test_trainer_build_registers_mtp_before_optimizer(self) -> None:
        """Execute Trainer's real build path, not a model-specific constructor."""
        trainer = SimpleNamespace(config=resolve_config(copy.deepcopy(self.raw)),
                                  distributed_setup=None, global_rank=0, mesh=MeshContext())
        BaseTrainer._build_model(trainer)
        self.assertEqual(len(trainer.model.mtp.layers), 1)
        optimizer = torch.optim.AdamW(trainer.model.parameters(), foreach=False, fused=False)
        registered = {id(param) for group in optimizer.param_groups for param in group["params"]}
        self.assertTrue(all(id(param) in registered for param in trainer.model.mtp.parameters()))
        losses = trainer.model(self.ids, labels=self.ids, use_cache=False).loss
        sum(losses.values()).backward()
        self.assertGreater(trainer.model.mtp.layers[0].eh_proj.weight.grad.norm().item(), 0)

    def test_yaml_depth_optimizer_and_checkpoint(self) -> None:
        """One/two configured depths register, receive gradients, update and reload."""
        for depth in (1, 2):
            with self.subTest(depth=depth):
                model = self.build(depth)
                self.assertEqual(len(model.mtp.layers), depth)
                self.assertEqual(len(model.model.layers), 2)
                norms = (model.mtp.layers[0].enorm, model.mtp.layers[0].hnorm, model.mtp.layers[0].post_norm)
                for norm in norms:
                    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
                before = model.mtp.layers[0].eh_proj.weight.detach().clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    losses = model(self.ids, labels=self.ids, use_cache=False).loss
                    sum(losses.values()).backward()
                    for layer in model.mtp.layers:
                        self.assertGreater(layer.eh_proj.weight.grad.norm().item(), 0)
                    self.assertGreater(model.model.embed_tokens.weight.grad.norm().item(), 0)
                    optimizer.step()
                self.assertFalse(torch.equal(before, model.mtp.layers[0].eh_proj.weight))
                state = io.BytesIO()
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, state)
                state.seek(0)
                saved = torch.load(state, weights_only=True)
                resumed = self.build(depth)
                resumed.load_state_dict(saved["model"], strict=True)
                other_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001, foreach=False, fused=False)
                other_optimizer.load_state_dict(saved["optimizer"])
                for current, opt in ((model, optimizer), (resumed, other_optimizer)):
                    opt.zero_grad(set_to_none=True)
                    sum(current(self.ids, labels=self.ids, use_cache=False).loss.values()).backward()
                    opt.step()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)

    def test_disabled_is_native_hf(self) -> None:
        """Omitted/zero depth leaves the original architecture and loss unchanged."""
        for depth in (None, 0):
            model = self.build(depth)
            self.assertIs(type(model), LlamaForCausalLM)
            self.assertFalse(any(name.startswith("mtp.") for name, _ in model.named_parameters()))
            self.assertTrue(torch.isfinite(model(self.ids, labels=self.ids).loss))

    def test_shared_builder_supports_another_model_and_preserves_trunk(self) -> None:
        """The same public entry works for Llama and Qwen2 without family-specific MTP classes."""
        for config_type in (LlamaConfig, Qwen2Config):
            config = config_type(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                 max_position_embeddings=32, use_cache=False)
            torch.manual_seed(51)
            native = HyperAutoModelForCausalLM.from_config(config, attn_implementation="eager")
            torch.manual_seed(51)
            candidate = HyperAutoModelForCausalLM.from_config(
                config, mtp_num_layers=2, attn_implementation="eager")
            self.assertIs(type(native), type(candidate))
            for name, value in native.state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
            expected = native(self.ids, labels=self.ids)
            actual = candidate(self.ids, labels=self.ids)
            torch.testing.assert_close(expected.logits, actual.logits)
            torch.testing.assert_close(expected.loss, actual.loss["foundation_loss/lm_loss"])
            sum(actual.loss.values()).backward()
            for layer in candidate.mtp.layers:
                self.assertGreater(layer.eh_proj.weight.grad.norm().item(), 0)

    def test_meta_materialization_restores_norms_and_rotary_state(self) -> None:
        """Exercise the actual empty-storage initialization used before distributed training."""
        config = LlamaConfig(**{k: v for k, v in self.raw["model"]["config"].items() if k != "_target_"})
        with torch.device("meta"):
            model = attach_mtp(LlamaForCausalLM(config), 2)
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        rebuild_materialized_state(model, MaterializationContext(reason="random_init", device=torch.device("cpu")))
        for layer in model.mtp.layers:
            for norm in (layer.enorm, layer.hnorm, layer.post_norm):
                torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
        self.assertTrue(torch.isfinite(sum(model(self.ids, labels=self.ids).loss.values())))

    def test_factor_scales_only_mtp_gradients(self) -> None:
        """The YAML coefficient affects MTP and its gradient exactly once."""
        model, scaled = self.build(factor=0.2), self.build(factor=0.6)
        scaled.load_state_dict(model.state_dict())
        first = model(self.ids, labels=self.ids, use_cache=False).loss
        second = scaled(self.ids, labels=self.ids, use_cache=False).loss
        torch.testing.assert_close(first["foundation_loss/lm_loss"], second["foundation_loss/lm_loss"])
        torch.testing.assert_close(first["mtp_depth1_loss"] * 3, second["mtp_depth1_loss"])
        sum(first.values()).backward()
        sum(second.values()).backward()
        torch.testing.assert_close(model.mtp.layers[0].eh_proj.weight.grad * 3,
                                   scaled.mtp.layers[0].eh_proj.weight.grad)
        zero = self.build(factor=0)
        self.assertEqual(zero(self.ids, labels=self.ids, use_cache=False).loss["mtp_depth1_loss"].item(), 0)

    def test_labels_shift_and_empty_targets(self) -> None:
        """Labels shift once, masked labels are safe, and ignored targets give finite zero gradients."""
        model = self.build()
        shifted = torch.nn.functional.pad(self.ids[:, 1:], (0, 1), value=-100)
        ordinary = model(self.ids, labels=self.ids, use_cache=False)
        aligned = model(self.ids, shift_labels=shifted, use_cache=False)
        for name in ordinary.loss:
            torch.testing.assert_close(ordinary.loss[name], aligned.loss[name])
        masked = model(self.ids, labels=self.ids, loss_mask=torch.zeros_like(self.ids), use_cache=False)
        loss = sum(masked.loss.values())
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.mtp.layers[0].eh_proj.weight.grad).all())
        self.assertTrue(torch.equal(self.ids, torch.tensor([[1,3,5,7,9,11,13,15],[2,4,6,8,10,12,14,16]])))

    def test_trainer_and_console_report_both_losses_without_double_count(self) -> None:
        """Use real postforward, metric callback and console formatting with only collectives mocked."""
        model = self.build()
        output = model(self.ids, labels=self.ids, use_cache=False)
        trainer = SimpleNamespace(model=model, loss_fn=ModelOutputLoss(),
                                  mesh=SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1),
                                  optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]), lr_scheduler=None)
        count = {"foundation_tokens": torch.tensor(self.ids.shape[0] * (self.ids.shape[1] - 1)),
                 "mtp_depth1_tokens": torch.tensor(self.ids.numel())}
        with patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **_: value):
            loss, metrics = BaseTrainer.postforward(trainer, output, self.ids,
                                                    current_token_counts=count, step_token_counts=count)
        torch.testing.assert_close(loss, sum(output.loss.values()))
        with patch("hyper_parallel.trainer.callbacks.environ_meter_callback.get_world_size_safe", return_value=1), \
             patch("hyper_parallel.trainer.callbacks.environ_meter_callback.get_device_type", return_value="cpu"):
            meter = EnvironMeterCallback(trainer)
            state = TrainerState(global_step=1, epoch=0)
            meter.on_step_begin(state)
            meter.on_step_end(state, loss.item(), {key: value.item() for key, value in metrics.items()}, 1.0)
        postfix = TqdmCallback(trainer)._postfix()
        self.assertIn("foundation_loss/lm_loss", postfix)
        self.assertIn("mtp_depth1_loss", postfix)
        self.assertIn("loss", postfix)
        torch.testing.assert_close(loss, sum(output.loss.values()))

    def test_padding_excludes_lm_and_mtp_targets(self) -> None:
        """Left/right padded tokens contribute neither LM nor shifted MTP losses."""
        model = self.build(2)
        padding = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]])
        targets = _targets(model, self.ids, self.ids, None, None, padding)
        expected = torch.tensor([[-100, -100, 7, 9, 11, -100, -100, -100],
                                 [4, 6, 8, 10, -100, -100, -100, -100]])
        torch.testing.assert_close(targets, expected)
        prepared = prepare_mtp_inputs(self.ids, torch.arange(8)[None], targets, 2,
                                      attention_mask=padding)
        self.assertTrue(all((target[~padding.bool()] == -100).all() for target in prepared["mtp_targets"]))
        losses = model(self.ids, labels=self.ids, attention_mask=padding).loss
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))

    def test_validation_on_model_construction(self) -> None:
        """Reject invalid configuration and unsupported layouts instead of silently ignoring MTP."""
        config = LlamaConfig(**{k: v for k, v in self.raw["model"]["config"].items() if k != "_target_"})
        for options in ({"mtp_num_layers": -1}, {"mtp_num_layers": True}, {"mtp_loss_factor": float("nan")},
                        {"mtp_loss_factor": -0.1}, {"mtp_loss_normalization": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                HyperAutoModelForCausalLM.from_config(config, **({"mtp_num_layers": 1} | options))

    def test_real_text_batches_and_accumulated_loss_match_full_batch(self) -> None:
        """The data runtime and Trainer normalize each MTP depth over the whole GA window."""
        self.raw["model"]["mtp_loss_normalization"] = "valid_tokens"
        model = self.build(2)
        batcher = MTPTextParallelBatch(
            mtp_num_layers=2, mtp_loss_normalization="valid_tokens",
            mesh_context=MeshContext(), device=torch.device("cpu"), tokenizer=None,
            data_config={"labels_are_shifted": False}, pp_shared_data=False,
            source_type="online", attention_mode="dense")
        labels = self.ids.clone()
        labels[0, -2:] = -100
        labels[1, -1:] = -100
        batches = [batcher(iter([{"input_ids": self.ids[index:index + 1],
                                  "labels": labels[index:index + 1],
                                  "cu_seq_lens": torch.tensor([0, 8])}])) for index in range(2)]
        trainer = SimpleNamespace(model=model, loss_fn=ModelOutputLoss(), mesh=MeshContext())
        step_counts = count_loss_token([loss_inputs for _, loss_inputs in batches])
        actual = []
        with patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **_: value):
            for inputs, loss_inputs in batches:
                output = model(**inputs)
                loss, metrics = BaseTrainer.postforward(
                    trainer, output, loss_inputs["labels"],
                    current_token_counts=count_loss_token(loss_inputs), step_token_counts=step_counts)
                self.assertIn("mtp_loss", metrics)
                actual.append(loss)
        expected = sum(model(self.ids, labels=labels).loss.values())
        torch.testing.assert_close(sum(actual), expected)
        parameter = model.mtp.layers[0].eh_proj.weight
        first = torch.autograd.grad(sum(actual), parameter)[0]
        second = torch.autograd.grad(expected, parameter)[0]
        torch.testing.assert_close(first, second)

    def test_reload_constructor_and_inference(self) -> None:
        """Persist MTP configuration and keep inference on the ordinary LM head."""
        model = self.build(2)
        restored_config = LlamaConfig.from_dict(model.config.to_dict())
        restored = HyperAutoModelForCausalLM.from_config(restored_config)
        restored.load_state_dict(model.state_dict())
        self.assertEqual(len(restored.mtp.layers), 2)
        with torch.no_grad():
            output = restored(self.ids, use_cache=False)
        self.assertIsNone(output.loss)
        self.assertEqual(output.logits.shape, (2, 8, 32))
        disabled = HyperAutoModelForCausalLM.from_config(restored_config, mtp_num_layers=0)
        self.assertFalse(hasattr(disabled, "mtp"))
        self.assertEqual(disabled.config.mtp_num_layers, 0)

    def test_fixed_length_matches_truncated_objective(self) -> None:
        """Padding keeps valid logits/gradients equal to the original full-row loop."""
        model = self.build(2)
        hidden = model.model.embed_tokens(self.ids)
        positions = torch.arange(self.ids.shape[1])[None]
        kwargs = _decoder_inputs(model, hidden, positions, None)
        for layer in model.model.layers:
            hidden = layer(hidden, **kwargs)
        targets = torch.nn.functional.pad(self.ids[:, 1:], (0, 1), value=-100)
        reference = model.mtp(hidden, self.ids, embedding=model.model.embed_tokens, head=model.lm_head,
                              decoder_kwargs=[_decoder_inputs(model, hidden[:, :-k], positions[:, k:], None)
                                              for k in (1, 2)])
        reference_loss = compute_mtp_loss(reference.logits, targets)
        actual = model(self.ids, shift_labels=targets).loss
        actual_loss = actual["mtp_depth1_loss"] + actual["mtp_depth2_loss"]
        torch.testing.assert_close(actual_loss, reference_loss)
        parameter = model.mtp.layers[0].eh_proj.weight
        first = torch.autograd.grad(reference_loss, parameter, retain_graph=True)[0]
        second = torch.autograd.grad(actual_loss, parameter)[0]
        torch.testing.assert_close(first, second)

    def test_recompute_covers_trunk_and_all_mtp_depths(self) -> None:
        """Generic layer discovery and checkpoint wrappers preserve all parameter gradients."""
        plain, recompute = self.build(2), self.build(2)
        recompute.load_state_dict(plain.state_dict())
        blocks, _ = _find_transformer_block_modules(recompute)
        self.assertEqual({block.fqn for block in blocks},
                         {"model.layers.0", "model.layers.1", "mtp.layers.0", "mtp.layers.1"})
        _apply_activation_checkpointing(recompute, "full")
        losses = [sum(model(self.ids, labels=self.ids).loss.values()) for model in (plain, recompute)]
        torch.testing.assert_close(*losses)
        for loss in losses:
            loss.backward()
        for (name, first), (_, second) in zip(plain.named_parameters(), recompute.named_parameters()):
            if first.grad is None:
                self.assertIn(".mlp.experts.", name)
                self.assertIsNone(second.grad, name)
                continue
            torch.testing.assert_close(first.grad, second.grad, msg=name)

    def test_pipeline_partition_outputs_gradients_and_shared_embedding(self) -> None:
        """Two stages match monolithic training after summing the shared embedding gradients."""
        reference, first, last = self.build(2), self.build(2), self.build(2)
        first.load_state_dict(reference.state_dict())
        last.load_state_dict(reference.state_dict())
        first_shared = partition_mtp_model(first, 0, 2)
        last_shared = partition_mtp_model(last, 1, 2)
        self.assertEqual(first_shared[0].shared_stage, [0, 1])
        self.assertEqual(last_shared[0].shared_stage, [0, 1])
        self.assertNotIn("model.layers.0.mlp.gate_proj.weight", last.state_dict())
        self.assertIsNone(first.mtp)
        self.assertIsNone(first.lm_head)
        expected = sum(reference(self.ids, labels=self.ids).loss.values())
        first_runtime, last_runtime = MTPPipelineModule(first), MTPPipelineModule(last)
        hidden = first_runtime(self.ids, labels=self.ids)
        actual = last_runtime(hidden, input_ids=self.ids, labels=self.ids)
        self.assertIn("mtp_loss", last_runtime.last_loss_metrics)
        torch.testing.assert_close(actual, expected)
        expected.backward()
        actual.backward()
        shared_grad = first_shared[0].parameter.grad + last_shared[0].parameter.grad
        torch.testing.assert_close(shared_grad, reference.model.embed_tokens.weight.grad)
        baseline = dict(reference.named_parameters())
        for stage in (first, last):
            for name, parameter in stage.named_parameters():
                if name != "model.embed_tokens.weight":
                    torch.testing.assert_close(parameter.grad, baseline[name].grad, msg=name)

    def test_cp_shift_matches_global_sequence_with_multiple_rows(self) -> None:
        """Every shard, depth > local length, tail and row boundary agree with a full-row oracle."""
        tokens = torch.arange(24).reshape(2, 12)
        group = object()
        for offset in (0, 1, 2, 4, 5, 9, 12, 13):
            expected = torch.full_like(tokens, -100)
            if offset < 12:
                expected[:, :12 - offset] = tokens[:, offset:]
            for rank in range(3):
                local = tokens[:, rank * 4:(rank + 1) * 4]
                def gather(outputs: list[torch.Tensor], _input: torch.Tensor, group: object) -> None:
                    """Copy the selected prefixes as an explicit CP collective oracle."""
                    self.assertIsNotNone(group)
                    for index, output in enumerate(outputs):
                        output.copy_(tokens[:, index * 4:index * 4 + min(offset, 4)])
                with patch("hyper_parallel.distributed.mtp.dist.get_world_size", return_value=3), \
                     patch("hyper_parallel.distributed.mtp.dist.get_rank", return_value=rank), \
                     patch("hyper_parallel.distributed.mtp.dist.all_gather", side_effect=gather):
                    actual = shift_mtp_tokens(local, offset, group=group, pad_value=-100)
                torch.testing.assert_close(actual, expected[:, rank * 4:(rank + 1) * 4])

    def test_packed_mtp_targets_do_not_cross_document_boundaries(self) -> None:
        """MTP targets crossing the anchor document are excluded at every depth."""
        ids = self.ids[:, :8]
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        documents = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        mask = (documents[:, None] == documents[None, :]).tril()[None, None]
        targets = torch.nn.functional.pad(ids[:, 1:], (0, 1), value=-100)
        targets[:, 3] = -100
        prepared = prepare_mtp_inputs(ids, positions, targets, 2, attention_mask=mask)
        torch.testing.assert_close(prepared["mtp_targets"][0][:, 2:4], torch.full((2, 2), -100))
        torch.testing.assert_close(prepared["mtp_targets"][1][:, 1:4], torch.full((2, 3), -100))
