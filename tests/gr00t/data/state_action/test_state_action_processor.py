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
Test StateActionProcessor: normalization, denormalization, sin/cos encoding.

Uses the same modality configs and statistics from the test fixtures directory
(libero_sim embodiment).
"""

import json
from pathlib import Path

from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"


def _load_fixture_configs():
    """Load modality configs and statistics from test fixtures."""
    with open(FIXTURE_DIR / "processor_config.json") as f:
        proc_config = json.load(f)
    modality_configs = proc_config["processor_kwargs"]["modality_configs"]

    with open(FIXTURE_DIR / "statistics.json") as f:
        statistics = json.load(f)
    return modality_configs, statistics


@pytest.fixture
def fixture_data():
    modality_configs, statistics = _load_fixture_configs()
    state_keys = modality_configs[EMBODIMENT]["state"]["modality_keys"]
    action_keys = modality_configs[EMBODIMENT]["action"]["modality_keys"]
    proc = StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        use_percentiles=False,
        clip_outliers=True,
    )
    return proc, state_keys, action_keys, statistics


@pytest.fixture
def processor(fixture_data):
    return fixture_data[0]


@pytest.fixture
def modality_keys(fixture_data):
    return fixture_data[1], fixture_data[2]


@pytest.fixture
def statistics(fixture_data):
    return fixture_data[3]


def _get_dim_from_stats(statistics, embodiment, modality, key):
    """Get the dimension of a key from its statistics (length of min array)."""
    return len(statistics[embodiment][modality][key]["min"])


def _random_state(state_keys, statistics):
    """Create random state dict with dimensions matching statistics."""
    return {
        k: np.random.randn(1, _get_dim_from_stats(statistics, EMBODIMENT, "state", k)).astype(
            np.float32
        )
        for k in state_keys
    }


def _random_action(action_keys, statistics, horizon=16):
    """Create random action dict with dimensions matching statistics."""
    return {
        k: np.random.randn(
            horizon, _get_dim_from_stats(statistics, EMBODIMENT, "action", k)
        ).astype(np.float32)
        for k in action_keys
    }


class TestStateNormalization:
    """Test state normalization and denormalization roundtrip."""

    def test_apply_state_returns_all_keys(self, processor, modality_keys, statistics):
        state_keys, _ = modality_keys
        raw = _random_state(state_keys, statistics)
        result = processor.apply_state(raw, EMBODIMENT)
        assert set(result.keys()) == set(state_keys)

    def test_normalized_state_within_bounds(self, processor, modality_keys, statistics):
        state_keys, _ = modality_keys
        raw = _random_state(state_keys, statistics)
        result = processor.apply_state(raw, EMBODIMENT)
        for key, val in result.items():
            assert val.min() >= -1.0, f"{key}: value {val.min()} < -1"
            assert val.max() <= 1.0, f"{key}: value {val.max()} > 1"

    def test_state_roundtrip(self, processor, modality_keys, statistics):
        """Normalize then denormalize should recover original values."""
        state_keys, _ = modality_keys
        raw = _random_state(state_keys, statistics)
        # Keep values within normalization range to avoid clipping
        for k in raw:
            stats = statistics[EMBODIMENT]["state"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw[k] = low + (high - low) * np.random.rand(*raw[k].shape).astype(np.float32)
        normalized = processor.apply_state(raw, EMBODIMENT)
        recovered = processor.unapply_state(normalized, EMBODIMENT)
        for key in state_keys:
            np.testing.assert_allclose(
                recovered[key], raw[key], atol=1e-4, err_msg=f"Roundtrip failed for {key}"
            )

    def test_missing_key_raises(self, processor):
        with pytest.raises(KeyError, match="not found in state dict"):
            processor.apply_state({"nonexistent": np.zeros(1)}, EMBODIMENT)


class TestActionNormalization:
    """Test action normalization and denormalization."""

    def test_apply_action_returns_all_keys(self, processor, modality_keys, statistics):
        _, action_keys = modality_keys
        raw = _random_action(action_keys, statistics)
        result = processor.apply_action(raw, EMBODIMENT)
        assert set(result.keys()) == set(action_keys)

    def test_normalized_action_clipped(self, processor, modality_keys, statistics):
        _, action_keys = modality_keys
        raw = _random_action(action_keys, statistics)
        result = processor.apply_action(raw, EMBODIMENT)
        for key, val in result.items():
            assert val.min() >= -1.0, f"{key}: action value {val.min()} < -1"
            assert val.max() <= 1.0, f"{key}: action value {val.max()} > 1"

    def test_action_roundtrip(self, processor, modality_keys, statistics):
        _, action_keys = modality_keys
        # Use values within the normalization range to avoid clipping
        raw = _random_action(action_keys, statistics)
        # Scale down to stay within range for perfect roundtrip
        for k in raw:
            stats = statistics[EMBODIMENT]["action"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw[k] = low + (high - low) * np.random.rand(*raw[k].shape).astype(np.float32)
        normalized = processor.apply_action(raw, EMBODIMENT)
        recovered = processor.unapply_action(normalized, EMBODIMENT)
        for key in action_keys:
            np.testing.assert_allclose(
                recovered[key], raw[key], atol=1e-4, err_msg=f"Action roundtrip failed for {key}"
            )


class TestApplyConvenience:
    """Test the combined apply/unapply convenience methods."""

    def test_apply_returns_tuple(self, processor, modality_keys, statistics):
        state_keys, action_keys = modality_keys
        raw_state = _random_state(state_keys, statistics)
        raw_action = _random_action(action_keys, statistics)
        result = processor.apply(raw_state, raw_action, EMBODIMENT)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_full_roundtrip(self, processor, modality_keys, statistics):
        state_keys, action_keys = modality_keys
        raw_state = _random_state(state_keys, statistics)
        raw_action = _random_action(action_keys, statistics)
        # Keep values within normalization range for exact roundtrip
        for k in raw_state:
            stats = statistics[EMBODIMENT]["state"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw_state[k] = low + (high - low) * np.random.rand(*raw_state[k].shape).astype(
                np.float32
            )
        for k in raw_action:
            stats = statistics[EMBODIMENT]["action"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw_action[k] = low + (high - low) * np.random.rand(*raw_action[k].shape).astype(
                np.float32
            )
        norm_state, norm_action = processor.apply(raw_state, raw_action, EMBODIMENT)
        rec_state, rec_action = processor.unapply(norm_state, norm_action, EMBODIMENT)
        for key in state_keys:
            np.testing.assert_allclose(rec_state[key], raw_state[key], atol=1e-4)
        for key in action_keys:
            np.testing.assert_allclose(rec_action[key], raw_action[key], atol=1e-4)


def _self_consistent_absolute_setup():
    """Rebuild the libero_sim fixture's processor with state statistics
    overridden to match action statistics per key - apply_reversed needs
    state[state_key] and action[key] to be dimensionally compatible (they
    represent the same physical channel at different times), which the
    plain libero_sim fixture's independently-chosen per-key state/action
    dims don't otherwise guarantee (apply_action never needs that for an
    ABSOLUTE-rep key, since it never combines the two; apply_reversed does).
    Returns (processor, action_keys, fixed_statistics)."""
    modality_configs, statistics = _load_fixture_configs()
    fixed_statistics = json.loads(json.dumps(statistics))
    action_keys = modality_configs[EMBODIMENT]["action"]["modality_keys"]
    for key in action_keys:
        fixed_statistics[EMBODIMENT]["state"][key] = fixed_statistics[EMBODIMENT]["action"][key]
    proc = StateActionProcessor(
        modality_configs=modality_configs,
        statistics=fixed_statistics,
        clip_outliers=False,
    )
    return proc, action_keys, fixed_statistics


def _random_within_range(keys, statistics, horizon):
    """Random values kept strictly within each key's [min, max] (mirrors
    TestActionNormalization.test_action_roundtrip's approach). Necessary
    here (unlike the unconstrained _random_action/_random_state helpers)
    because unnormalize_values_minmax clips its input to the normalized
    [-1, 1] range independent of clip_outliers, so an unconstrained value
    would silently lose precision on the way back out - a pre-existing,
    unrelated property of denormalization, not of apply_reversed."""
    result = {}
    for k in keys:
        stats = statistics[EMBODIMENT]["action"][k]
        low, high = np.array(stats["min"]), np.array(stats["max"])
        result[k] = (low + (high - low) * np.random.rand(horizon, len(low))).astype(np.float32)
    return result


class TestApplyReversedAbsolute:
    """Test apply_reversed for ABSOLUTE-rep actions (the libero_sim fixture -
    all keys are ABSOLUTE/NON_EEF), i.e. the temporal-reversal-only path."""

    def test_apply_reversed_returns_tuple(self):
        proc, action_keys, statistics = _self_consistent_absolute_setup()
        raw_state = _random_within_range(action_keys, statistics, horizon=1)
        raw_action = _random_within_range(action_keys, statistics, horizon=4)
        result = proc.apply_reversed(raw_state, raw_action, EMBODIMENT)
        assert isinstance(result, tuple) and len(result) == 2
        norm_state, norm_action = result
        assert set(norm_state.keys()) == set(action_keys)
        assert set(norm_action.keys()) == set(action_keys)

    def test_matches_manual_temporal_reversal(self):
        """Denormalized apply_reversed output must match a hand-computed
        temporal reversal: backward[-1] == state_t, new_state == forward[-1]."""
        proc, action_keys, statistics = _self_consistent_absolute_setup()
        raw_state = _random_within_range(action_keys, statistics, horizon=1)
        raw_action = _random_within_range(action_keys, statistics, horizon=4)

        norm_state, norm_action = proc.apply_reversed(raw_state, raw_action, EMBODIMENT)
        raw_new_state = proc.unapply_state(norm_state, EMBODIMENT)
        raw_backward_action = proc.unapply_action(norm_action, EMBODIMENT)

        for key in action_keys:
            current_value = raw_state[key][-1]  # (D,)
            forward_array = raw_action[key]  # (H, D)
            full = np.concatenate([current_value[None, :], forward_array], axis=0)
            expected_backward = full[::-1][1:]
            expected_new_reference = full[::-1][0]

            np.testing.assert_allclose(
                raw_backward_action[key], expected_backward, atol=1e-3, rtol=1e-3
            )
            # new state was tiled across the state-history dimension
            np.testing.assert_allclose(
                raw_new_state[key][-1], expected_new_reference, atol=1e-3, rtol=1e-3
            )
            # Round-trip: last element of the backward chunk recovers state_t.
            np.testing.assert_allclose(
                raw_backward_action[key][-1], current_value, atol=1e-3, rtol=1e-3
            )


# ---------------------------------------------------------------------------
# Custom RELATIVE-rep (EEF) fixture for apply_reversed's SE(3)-composition path
# ---------------------------------------------------------------------------

RELATIVE_EMBODIMENT = "test_relative_eef"


def _random_eef_pose() -> EndEffectorPose:
    return EndEffectorPose(
        translation=np.random.randn(3),
        rotation=Rotation.random().as_quat(),
        rotation_type="quat",
        rotation_order="xyzw",
    )


def _wide_stats(dim: int) -> dict:
    """Statistics with a wide enough range that clip_outliers=False plus
    random composed poses never hits a normalization edge case."""
    return {"min": [-100.0] * dim, "max": [100.0] * dim, "mean": [0.0] * dim, "std": [50.0] * dim}


def _relative_eef_processor(action_rep: ActionRepresentation = ActionRepresentation.RELATIVE):
    modality_configs = {
        RELATIVE_EMBODIMENT: {
            "state": ModalityConfig(delta_indices=[0], modality_keys=["eef", "gripper"]),
            "action": ModalityConfig(
                delta_indices=list(range(4)),
                modality_keys=["eef", "gripper"],
                action_configs=[
                    ActionConfig(
                        rep=action_rep,
                        type=ActionType.EEF,
                        format=ActionFormat.XYZ_ROTVEC,
                        state_key="eef",
                    ),
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                        state_key="gripper",
                    ),
                ],
            ),
        }
    }
    statistics = {
        RELATIVE_EMBODIMENT: {
            "state": {"eef": _wide_stats(6), "gripper": _wide_stats(1)},
            "action": {"eef": _wide_stats(6), "gripper": _wide_stats(1)},
            "relative_action": {"eef": _wide_stats(6)},
        }
    }
    return StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        use_relative_action=True,
        clip_outliers=False,
    )


class TestApplyReversedRelative:
    """Test apply_reversed for RELATIVE-rep (EEF) actions - the SE(3)
    composition path via ActionChunk.to_reversed_chunking."""

    def test_matches_manual_composition(self):
        np.random.seed(2)
        proc = _relative_eef_processor()
        horizon = 4

        state_pose = _random_eef_pose()
        gripper_t = np.array([0.3])
        state = {
            "eef": state_pose.xyz_rotvec[None, :],  # (1, 6)
            "gripper": gripper_t[None, :],  # (1, 1)
        }

        absolute_poses = [_random_eef_pose() for _ in range(horizon)]
        action_eef = EndEffectorActionChunk(absolute_poses).to(ActionFormat.XYZ_ROTVEC)  # (H, 6)
        gripper_forward = np.random.uniform(-1, 1, size=(horizon, 1))
        action = {"eef": action_eef, "gripper": gripper_forward}

        norm_state, norm_action = proc.apply_reversed(state, action, RELATIVE_EMBODIMENT)
        raw_new_state = proc.unapply_state(norm_state, RELATIVE_EMBODIMENT)
        raw_backward_action = proc.unapply_action(
            norm_action, RELATIVE_EMBODIMENT, state=raw_new_state
        )

        # Round-trip: the last timestep of the decoded backward EEF chunk
        # (composed forward from the new reference state_{t+H}) must recover
        # the original state_t.
        np.testing.assert_allclose(
            raw_backward_action["eef"][-1], state["eef"][0], atol=1e-3, rtol=1e-3
        )
        # Gripper (ABSOLUTE rep): plain temporal reversal.
        np.testing.assert_allclose(
            raw_backward_action["gripper"][-1], gripper_t, atol=1e-3, rtol=1e-3
        )
        np.testing.assert_allclose(
            raw_new_state["gripper"][-1], gripper_forward[-1], atol=1e-3, rtol=1e-3
        )

    def test_delta_representation_raises(self):
        proc = _relative_eef_processor(action_rep=ActionRepresentation.DELTA)
        state = {"eef": np.zeros((1, 6)), "gripper": np.zeros((1, 1))}
        action = {"eef": np.zeros((4, 6)), "gripper": np.zeros((4, 1))}
        with pytest.raises(NotImplementedError, match="DELTA"):
            proc.apply_reversed(state, action, RELATIVE_EMBODIMENT)


class TestSinCosEncoding:
    """Test sin/cos state encoding mode."""

    def test_sincos_doubles_dimension(self, modality_keys, statistics):
        state_keys, _ = modality_keys
        modality_configs, statistics_local = _load_fixture_configs()
        modality_configs[EMBODIMENT]["state"]["sin_cos_embedding_keys"] = state_keys
        proc = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics_local,
            apply_sincos_state_encoding=True,
        )
        raw = _random_state(state_keys, statistics)
        result = proc.apply_state(raw, EMBODIMENT)
        for key in state_keys:
            assert result[key].shape[-1] == raw[key].shape[-1] * 2, (
                f"{key}: sin/cos should double dimension"
            )


class TestStatistics:
    """Test statistics management."""

    def test_set_statistics(self):
        modality_configs, statistics = _load_fixture_configs()
        proc = StateActionProcessor(modality_configs=modality_configs)
        # No statistics yet — normalization params empty
        assert len(proc.norm_params) == 0
        proc.set_statistics(statistics)
        assert EMBODIMENT in proc.norm_params

    def test_set_statistics_no_override(self, caplog):
        modality_configs, statistics = _load_fixture_configs()
        proc = StateActionProcessor(modality_configs=modality_configs, statistics=statistics)
        original_min = proc.norm_params[EMBODIMENT]["state"]["x"]["min"].copy()
        # Modify statistics
        modified = json.loads(json.dumps(statistics))
        modified[EMBODIMENT]["state"]["x"]["min"] = [999.0]
        with caplog.at_level("WARNING"):
            proc.set_statistics(modified, override=False)
        # Should NOT have changed
        np.testing.assert_array_equal(
            proc.norm_params[EMBODIMENT]["state"]["x"]["min"], original_min
        )
        # Discarded merge must surface as a WARNING (not a print) so callers
        # running with override_pretraining_statistics=False on a mixture
        # dataset see it in their training logs instead of in stdout.
        assert any(
            r.levelname == "WARNING" and "DISCARDED" in r.message and EMBODIMENT in r.message
            for r in caplog.records
        ), f"expected a WARNING about discarded statistics; got: {caplog.records}"

    def test_set_statistics_override(self, caplog):
        modality_configs, statistics = _load_fixture_configs()
        proc = StateActionProcessor(modality_configs=modality_configs, statistics=statistics)
        modified = json.loads(json.dumps(statistics))
        modified[EMBODIMENT]["state"]["x"]["min"] = [999.0]
        with caplog.at_level("WARNING"):
            proc.set_statistics(modified, override=True)
        assert proc.norm_params[EMBODIMENT]["state"]["x"]["min"][0] == pytest.approx(999.0)
        # No DISCARDED warning when the caller explicitly asked for override.
        assert not any("DISCARDED" in r.message for r in caplog.records)


class TestTrainEvalMode:
    """Test train/eval mode switching."""

    def test_default_is_train(self, processor):
        assert processor.training is True

    def test_eval(self, processor):
        processor.eval()
        assert processor.training is False

    def test_train(self, processor):
        processor.eval()
        processor.train()
        assert processor.training is True
