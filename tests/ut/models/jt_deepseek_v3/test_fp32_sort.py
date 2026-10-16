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
"""JT model configuration and adapter wiring for optional FP32 expert keys."""

import unittest
from unittest.mock import Mock, patch

from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.ep_compute import jt_deepseek_v3_ep_compute_fn
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config, JTDeepseekV3MoE
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


class TestJTExpertSort(unittest.TestCase):
    """Keep the option disabled unless explicitly configured."""

    def test_config_defaults_roundtrip_and_invalid_values(self) -> None:
        """A saved model configuration retains its explicit choice and rejects non-bools."""
        self.assertFalse(JTDeepseekV3Config().expert_sort_fp32)
        for enabled in (False, True):
            config = JTDeepseekV3Config(expert_sort_fp32=enabled)
            self.assertEqual(JTDeepseekV3Config.from_dict(config.to_dict()).expert_sort_fp32, enabled)
        for invalid in ("true", 1, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "expert_sort_fp32"):
                JTDeepseekV3Config(expert_sort_fp32=invalid)

    def test_adapter_forwards_model_option(self) -> None:
        """The JT EP adapter passes the model option into the shared recipe."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = small_config()
                config.expert_sort_fp32 = enabled
                module = JTDeepseekV3MoE(config)
                with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed.ep_compute.build_ep_compute",
                           return_value=Mock()) as factory:
                    jt_deepseek_v3_ep_compute_fn(module=module, mesh=None, tp_mesh=None,
                                                cp_mesh=None, ep_mesh=object())
                self.assertEqual(factory.call_args.kwargs["sort_in_fp32"], enabled)