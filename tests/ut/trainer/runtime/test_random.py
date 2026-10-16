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
"""Unit tests for enable_full_determinism's Ascend NPU environment handling."""

import os
import unittest
from unittest.mock import patch

from hyper_parallel.trainer.runtime import random as random_runtime
from tests.common.mark_utils import arg_mark

MODULE = "hyper_parallel.trainer.runtime.random"


class TestEnableFullDeterminism(unittest.TestCase):
    """Verify CLOSE_MATMUL_K_SHIFT is left to the process environment, not HP."""

    def setUp(self) -> None:
        """Isolate each test from any CLOSE_MATMUL_K_SHIFT set outside this process."""
        self._saved = os.environ.pop("CLOSE_MATMUL_K_SHIFT", None)

    def tearDown(self) -> None:
        """Restore whatever CLOSE_MATMUL_K_SHIFT state existed before the test."""
        os.environ.pop("CLOSE_MATMUL_K_SHIFT", None)
        if self._saved is not None:
            os.environ["CLOSE_MATMUL_K_SHIFT"] = self._saved

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_npu_path_does_not_set_k_shift_when_absent(self) -> None:
        """Leave CLOSE_MATMUL_K_SHIFT unset on NPU when the caller never set it."""
        with patch(f"{MODULE}.IS_NPU_AVAILABLE", True), patch(f"{MODULE}.torch"):
            random_runtime.enable_full_determinism(42)

        self.assertNotIn("CLOSE_MATMUL_K_SHIFT", os.environ)
        # The other NPU-specific determinism knob this function owns is untouched.
        self.assertEqual(os.environ["NCCL_DETERMINISTIC"], "true")

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_npu_path_preserves_a_caller_set_value(self) -> None:
        """Never clobber a value the caller (or CANN's own default) already set."""
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "0"

        with patch(f"{MODULE}.IS_NPU_AVAILABLE", True), patch(f"{MODULE}.torch"):
            random_runtime.enable_full_determinism(42)

        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "0")

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_non_npu_path_is_unaffected(self) -> None:
        """Take no position on CLOSE_MATMUL_K_SHIFT at all off the NPU path."""
        with patch(f"{MODULE}.IS_NPU_AVAILABLE", False), patch(f"{MODULE}.torch"):
            random_runtime.enable_full_determinism(42)

        self.assertNotIn("CLOSE_MATMUL_K_SHIFT", os.environ)
        self.assertEqual(os.environ["NCCL_DETERMINISTIC"], "1")
