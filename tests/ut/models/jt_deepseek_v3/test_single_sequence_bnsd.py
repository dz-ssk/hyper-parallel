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
"""CPU contract tests for the optional single-document BNSD attention path."""
# pylint: disable=forbidden-backend-import,protected-access

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as modeling
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import _observed_attention
from tests.common.mark_utils import arg_mark


def _qkv(batch=1, query_length=5, key_length=5):
    generator = torch.Generator().manual_seed(42)
    return tuple(
        torch.randn(batch, length, 2, dim, dtype=torch.float64, generator=generator)
        .transpose(1, 2).requires_grad_()
        for length, dim in ((query_length, 4), (key_length, 4), (key_length, 3))
    )


def _observer(enabled=True):
    return SimpleNamespace(fa_single_sequence_bnsd=enabled, is_causal=True, max_logits_val=None)


def _recording_kernel(query, _key, value, _heads, _layout, **_options):
    """Return layout-correct placeholders; argument assertions test fallback behavior."""
    output = query.new_zeros((*query.shape[:-1], value.shape[-1]))
    maxima = query.new_ones((*query.shape[:-1], 8))
    return output, maxima


def _single_document_reference(query, key, value, _heads, layout, **options):
    """Evaluate one full causal document on CPU, including layout-specific maxima."""
    if layout == "TND":
        query, key, value = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value))
    scores = (query @ key.transpose(-1, -2)) * options["scale"]
    disallowed = torch.ones(scores.shape[-2:], dtype=torch.bool).triu(1)
    scores = scores.masked_fill(disallowed, float("-inf"))
    output = scores.softmax(-1) @ value
    maxima = scores.amax(-1, keepdim=True).expand(*scores.shape[:-1], 8)
    if layout == "TND":
        output, maxima = (tensor.squeeze(0).transpose(0, 1) for tensor in (output, maxima))
    return output, maxima


def _config(enabled):
    return modeling.JTDeepseekV3Config(
        fa_single_sequence_bnsd=enabled, hidden_size=16, num_attention_heads=2,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
    )


class TestSingleSequenceBNSD(unittest.TestCase):
    """Validate layout selection, autograd, QK clipping statistics and CP propagation."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_config_and_attention_replacement(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: Config round-trip and both attention module types retain the flag.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        self.assertFalse(modeling.JTDeepseekV3Config().fa_single_sequence_bnsd)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = modeling.JTDeepseekV3Config.from_dict(_config(enabled).to_dict())
                attention = modeling.JTDeepseekV3Attention(config, 0)
                replacement = modeling.JTDeepseekV3MLAAttention(module=attention)
                self.assertIs(attention.fa_single_sequence_bnsd, enabled)
                self.assertIs(replacement.fa_single_sequence_bnsd, enabled)
        for invalid in ("true", 1, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "must be a bool"):
                modeling.JTDeepseekV3Config(fa_single_sequence_bnsd=invalid)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_layout_output_gradients_and_statistics(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: Switching layout preserves the single-document mathematical reference.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        qkv = _qkv()
        upstream = torch.linspace(-1, 1, 30, dtype=torch.float64).reshape(1, 5, 2, 3)
        outputs, gradients, maxima = [], [], []
        for enabled in (False, True):
            observer = _observer(enabled)
            with patch.object(
                modeling.torch_npu, "npu_fusion_attention", side_effect=_single_document_reference
            ) as kernel:
                output, weights = modeling.observed_fusion_attention(
                    observer, *qkv, None, actual_seq_len=[5], scaling=0.37
                )
            self.assertIsNone(weights)
            self.assertEqual(output.shape, (1, 5, 2, 3))
            args, options = kernel.call_args
            self.assertEqual(args[4], "BNSD" if enabled else "TND")
            self.assertEqual(options["sparse_mode"], 3)
            self.assertEqual(options["scale"], 0.37)
            self.assertEqual(options["keep_prob"], 1.0)
            self.assertEqual(options["actual_seq_qlen"], None if enabled else [5])
            self.assertEqual(options["actual_seq_kvlen"], None if enabled else [5])
            self.assertEqual(options["softmax_layout"], "" if enabled else "TND")
            self.assertEqual(options["atten_mask"].shape, (2048, 2048))
            if enabled:
                for actual, original in zip(args[:3], qkv):
                    self.assertTrue(actual.is_contiguous())
                    torch.testing.assert_close(actual, original)
            outputs.append(output)
            gradients.append(torch.autograd.grad(output, qkv, upstream))
            maxima.append(observer.max_logits_val)
        torch.testing.assert_close(outputs[0], outputs[1], atol=1e-12, rtol=1e-12)
        for original, changed in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(original, changed, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(maxima[0], maxima[1], atol=1e-12, rtol=1e-12)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_maxima_accumulate_across_layout_switch(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: Changing layout retains previous per-head QK clipping maxima.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        qkv, observer = _qkv(), _observer(False)
        with patch.object(modeling.torch_npu, "npu_fusion_attention", side_effect=_single_document_reference):
            modeling.observed_fusion_attention(observer, *qkv, None, actual_seq_len=[5])
            first = observer.max_logits_val.clone()
            observer.max_logits_val[0] = 100
            observer.fa_single_sequence_bnsd = True
            modeling.observed_fusion_attention(observer, *qkv, None, actual_seq_len=[5])
        expected = first.clone()
        expected[0] = 100
        torch.testing.assert_close(observer.max_logits_val, expected)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_sequence_length_aliases(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: All supported one-document cumulative-length forms select BNSD.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        aliases = [
            {"actual_seq_len": (5,)},
            {"actual_seq_qlen": [5], "actual_seq_kvlen": [5]},
            {"cu_seq_lens_q": [0, 5], "cu_seq_lens_k": [0, 5]},
            {"packed_seq_params": SimpleNamespace(cu_seqlens_q=[0, 5], cu_seqlens_kv=[0, 5])},
        ]
        for kwargs in aliases:
            with self.subTest(kwargs=kwargs), \
                    patch.object(modeling.torch_npu, "npu_fusion_attention", side_effect=_recording_kernel) as kernel:
                modeling.observed_fusion_attention(_observer(), *_qkv(), None, **kwargs)
                self.assertEqual(kernel.call_args.args[4], "BNSD")
                self.assertIsNone(kernel.call_args.kwargs["actual_seq_qlen"])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_ineligible_inputs_preserve_tnd_and_boundaries(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: Packed boundaries and kernel options remain unchanged outside the eligible case.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        cases = [
            (_qkv(), None, {"actual_seq_len": [2, 5]}),
            (_qkv(2), None, {"actual_seq_len": [10]}),
            (_qkv(query_length=3), None, {"actual_seq_qlen": [3], "actual_seq_kvlen": [5]}),
            (_qkv(), torch.ones(5, 5, dtype=torch.bool), {"actual_seq_len": [5]}),
            (_qkv(), None, {"actual_seq_len": [5], "dropout": 0.1}),
            (_qkv(), None, {"actual_seq_len": [5], "sliding_window": 2}),
            (_qkv(), None, {"actual_seq_len": [5], "is_causal": False}),
        ]
        for qkv, mask, kwargs in cases:
            with self.subTest(kwargs=kwargs):
                calls = []
                for enabled in (False, True):
                    with patch.object(
                        modeling.torch_npu, "npu_fusion_attention", side_effect=_recording_kernel
                    ) as kernel:
                        modeling.observed_fusion_attention(_observer(enabled), *qkv, mask, **kwargs)
                    calls.append(kernel.call_args)
                for call in calls:
                    self.assertEqual(call.args[4], "TND")
                for original, changed in zip(calls[0].args[:3], calls[1].args[:3]):
                    torch.testing.assert_close(original, changed)
                for name in calls[0].kwargs:
                    left, right = calls[0].kwargs[name], calls[1].kwargs[name]
                    if isinstance(left, torch.Tensor):
                        torch.testing.assert_close(left, right)
                    else:
                        self.assertEqual(left, right)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_unpacked_path_is_unchanged(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: Inputs without packed metadata continue to use the original BNSD path.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                    patch.object(modeling.torch_npu, "npu_fusion_attention", side_effect=_recording_kernel) as kernel:
                modeling.observed_fusion_attention(_observer(enabled), *_qkv(), None)
                self.assertEqual(kernel.call_args.args[4], "BNSD")
                self.assertEqual(kernel.call_args.kwargs["sparse_mode"], 0)
                self.assertEqual(kernel.call_args.kwargs["atten_mask"].shape, (5, 5))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_invalid_sequence_end_is_rejected(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: Incomplete document boundaries fail existing validation before any kernel.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        with patch.object(modeling.torch_npu, "npu_fusion_attention") as kernel:
            with self.assertRaisesRegex(ValueError, "final query sequence length"):
                modeling.observed_fusion_attention(_observer(), *_qkv(), None, actual_seq_len=[4])
        kernel.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_prepared_sparse_mode_is_preserved(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: Exercise prepared modes 2, 3 and a non-causal mode.
        Expectation: Only causal modes switch layout; the prepared mode is retained.
        """
        prepare = modeling._prepare_attention_inputs
        for mode in (2, 3, 4):
            def _with_mode(*args, **kwargs):
                prepared = prepare(*args, **kwargs)
                return (*prepared[:-1], mode)

            with self.subTest(mode=mode), \
                    patch.object(modeling, "_prepare_attention_inputs", side_effect=_with_mode), \
                    patch.object(modeling.torch_npu, "npu_fusion_attention", side_effect=_recording_kernel) as kernel:
                modeling.observed_fusion_attention(_observer(), *_qkv(), None, actual_seq_len=[5])
            self.assertEqual(kernel.call_args.args[4], "BNSD" if mode in (2, 3) else "TND")
            self.assertEqual(kernel.call_args.kwargs["sparse_mode"], mode)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_cp_observer_propagates_flag_and_head_coordinates(self) -> None:
        """Validate single-document BNSD layout selection.

        Feature: Optional JT single-sequence BNSD attention.
        Description: CP forwards the option and writes maxima only into its assigned head range.
        Expectation: Layout, metadata, output and state obey the asserted contract.
        """
        qkv = _qkv()
        runtime = SimpleNamespace(compute_heads=2, local_heads=4, head_range=(1, 3))
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                module = SimpleNamespace(
                    config=_config(enabled), attention_interface=modeling.observed_fusion_attention,
                    scaling=0.37, max_logits_val=torch.zeros(4, dtype=torch.float64)
                )
                with patch.object(
                    modeling.torch_npu, "npu_fusion_attention", side_effect=_single_document_reference
                ) as kernel:
                    output = _observed_attention(module, runtime, *qkv, (5,))
                self.assertEqual(kernel.call_args.args[4], "BNSD" if enabled else "TND")
                self.assertEqual(output.shape, (1, 5, 2, 3))
                observer = _observer(enabled)
                with patch.object(modeling.torch_npu, "npu_fusion_attention", side_effect=_single_document_reference):
                    modeling.observed_fusion_attention(observer, *qkv, None, scaling=0.37, actual_seq_len=[5])
                torch.testing.assert_close(module.max_logits_val[1:3], observer.max_logits_val)
                torch.testing.assert_close(module.max_logits_val[[0, 3]], torch.zeros(2, dtype=torch.float64))