# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
Test the loading helpers for autoguidance: truncating the action head DiT to build a
smaller guide model, and checking that a guide normalizes actions like the main model.
"""

import json

from gr00t.model.gr00t_n1d7.setup import (
    drop_truncated_dit_block_keys,
    truncated_diffusion_model_cfg,
)
from gr00t.policy.gr00t_policy import check_autoguidance_processor_compatibility
import pytest


BASE_DIT_CFG = {"num_layers": 32, "num_attention_heads": 32, "attention_head_dim": 48}


class TestTruncatedDiT:
    def test_truncates_num_layers_only(self):
        cfg = truncated_diffusion_model_cfg(BASE_DIT_CFG, 8)
        assert cfg == {**BASE_DIT_CFG, "num_layers": 8}
        assert BASE_DIT_CFG["num_layers"] == 32  # input not mutated

    @pytest.mark.parametrize("num_layers", [0, -4, 36])
    def test_rejects_out_of_range(self, num_layers):
        with pytest.raises(ValueError, match="must be in"):
            truncated_diffusion_model_cfg(BASE_DIT_CFG, num_layers)

    @pytest.mark.parametrize("num_layers", [2, 6, 10])
    def test_rejects_non_multiple_of_four(self, num_layers):
        with pytest.raises(ValueError, match="multiple of 4"):
            truncated_diffusion_model_cfg(BASE_DIT_CFG, num_layers)

    def test_drops_only_discarded_blocks(self):
        keys = [
            "action_head.model.transformer_blocks.7.attn1.to_q.weight",
            "action_head.model.transformer_blocks.8.attn1.to_q.weight",
            "action_head.model.transformer_blocks.31.ff.net.0.proj.bias",
            "action_head.vl_self_attention.transformer_blocks.9.attn1.to_q.weight",
            "backbone.model.unexpected.weight",
        ]
        assert drop_truncated_dit_block_keys(keys, 8) == [
            "action_head.model.transformer_blocks.7.attn1.to_q.weight",
            "action_head.vl_self_attention.transformer_blocks.9.attn1.to_q.weight",
            "backbone.model.unexpected.weight",
        ]


def _write_processor(directory, *, q99=1.0, use_percentiles=True, embodiment_index=5):
    directory.mkdir(parents=True)
    (directory / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gr00tN1d7Processor",
                "processor_kwargs": {
                    "max_action_dim": 132,
                    "max_action_horizon": 40,
                    "use_percentiles": use_percentiles,
                    "use_relative_action": True,
                },
            }
        )
    )
    (directory / "statistics.json").write_text(
        json.dumps({"libero_sim": {"action": {"x": {"q01": [-1.0], "q99": [q99]}}}})
    )
    (directory / "embodiment_id.json").write_text(json.dumps({"libero_sim": embodiment_index}))
    return directory


class TestProcessorCompatibility:
    def test_matching_processors_pass(self, tmp_path):
        main = _write_processor(tmp_path / "main")
        guide = _write_processor(tmp_path / "guide", q99=1.0 + 1e-9)
        check_autoguidance_processor_compatibility(main, guide, "libero_sim")

    @pytest.mark.parametrize(
        "guide_kwargs, message",
        [
            ({"q99": 1.5}, "statistics.json"),
            ({"use_percentiles": False}, "use_percentiles"),
            ({"embodiment_index": 6}, "embodiment_id.json"),
        ],
    )
    def test_mismatch_raises(self, tmp_path, guide_kwargs, message):
        main = _write_processor(tmp_path / "main")
        guide = _write_processor(tmp_path / "guide", **guide_kwargs)
        with pytest.raises(ValueError, match=message):
            check_autoguidance_processor_compatibility(main, guide, "libero_sim")

    def test_missing_embodiment_raises(self, tmp_path):
        main = _write_processor(tmp_path / "main")
        guide = _write_processor(tmp_path / "guide")
        with pytest.raises(ValueError, match="differs or is missing"):
            check_autoguidance_processor_compatibility(main, guide, "oxe_droid")
