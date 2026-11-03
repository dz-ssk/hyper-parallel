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
"""Thin NPU launchers for MTP distributed training parity."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_mtp_tp_cp_ep_fsdp_hsdp() -> None:
    """MTP losses and gradients match the unsharded model with SP and recompute."""
    torchrun_case(str(Path(__file__).with_name("_test_mtp_parallel.py")),
                  "test_mtp_parallel_training", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_mtp_pp() -> None:
    """Two-stage shared-embedding PP matches the monolithic MTP gradient."""
    torchrun_case(str(Path(__file__).with_name("_test_mtp_parallel.py")),
                  "test_mtp_pipeline_training", num_proc=2)
