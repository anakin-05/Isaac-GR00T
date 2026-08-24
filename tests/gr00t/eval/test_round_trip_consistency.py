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
Tests for gr00t.eval.round_trip_consistency: the C_i diagnostic computation.

Uses the real demo_data/droid_sample dataset (small, CPU-only, no GPU or
model checkpoint needed) so extract_step_data / parse_observation_gr00t are
exercised for real, with a lightweight stub policy standing in for
Gr00tPolicy (no model). Video is dropped from the modality config: neither
stub below needs vision, and it sidesteps having to match this dataset's
video column naming.
"""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType
from gr00t.eval.round_trip_consistency import compute_round_trip_consistency
import numpy as np
import pytest


DATASET_PATH = "demo_data/droid_sample"
EMBODIMENT_TAG = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
TRAJ_ID = 1  # traj 0 has an empty language instruction (see examples/DROID/README.md)


def _modality_configs_no_video() -> dict:
    # LeRobotEpisodeLoader wants a flat {modality: ModalityConfig} dict (not
    # nested by embodiment tag - that nesting is StateActionProcessor's contract).
    configs = deepcopy(MODALITY_CONFIGS["oxe_droid_relative_eef_relative_joint"])
    configs["video"].modality_keys = []
    return configs


@pytest.fixture(scope="module")
def loader():
    return LeRobotEpisodeLoader(
        dataset_path=DATASET_PATH,
        modality_configs=_modality_configs_no_video(),
    )


def _key_to_state_key(loader) -> dict:
    action_keys = loader.modality_configs["action"].modality_keys
    action_configs = loader.modality_configs["action"].action_configs
    return {k: (ac.state_key if ac.state_key else k) for k, ac in zip(action_keys, action_configs)}


def _invert_chunk(action_config, state_t: np.ndarray, forward_chunk: np.ndarray) -> np.ndarray:
    """Independent oracle: the mathematically-exact inverse of `forward_chunk`
    (raw absolute values, shape (H, D)), starting from `state_t` (D,), in the
    SAME space compute_round_trip_consistency expects a decoded backward
    prediction to be in. Deliberately re-derives the math here (does not call
    state_action_processor) so this is a genuine check of the
    compose/decode pipeline, not two identical code paths agreeing.
    """
    fmt = action_config.format
    if action_config.rep == ActionRepresentation.RELATIVE and action_config.type == ActionType.EEF:
        absolute_chunk = EndEffectorActionChunk.from_array(forward_chunk, fmt)
        reference_frame = EndEffectorPose.from_action_format(state_t, fmt)
        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference_frame)
        reversed_chunk, new_reference = relative_chunk.to_reversed_chunking(reference_frame)
        return reversed_chunk.to_absolute_chunking(reference_frame=new_reference).to(fmt)
    elif (
        action_config.rep == ActionRepresentation.RELATIVE
        and action_config.type == ActionType.NON_EEF
    ):
        absolute_chunk = JointActionChunk([JointPose(row) for row in forward_chunk])
        reference_frame = JointPose(state_t)
        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference_frame)
        reversed_chunk, new_reference = relative_chunk.to_reversed_chunking(reference_frame)
        return reversed_chunk.to_absolute_chunking(reference_frame=new_reference).to(fmt)
    else:
        # ABSOLUTE rep: plain temporal reversal.
        full = np.concatenate([state_t[None, :], forward_chunk], axis=0)
        return full[::-1][1:]


class TestComputeRoundTripConsistencyShapes:
    """Shape/finiteness smoke test with a constant-output stub (no exact-
    recovery claim - just that the plumbing runs end-to-end)."""

    class _ConstantStubPolicy:
        def __init__(self, action_keys, dims):
            self.action_keys = action_keys
            self.dims = dims

        def get_action(self, observation, options=None):
            assert options is not None and options.get("direction") == 1
            return (
                {key: np.zeros((4, self.dims[key])) for key in self.action_keys},
                {},
            )

    def test_shapes_and_finiteness(self, loader):
        action_keys = loader.modality_configs["action"].modality_keys
        traj = loader[TRAJ_ID]
        dims = {key: np.asarray(traj[f"action.{key}"].iloc[0]).reshape(-1).shape[0] for key in action_keys}
        policy = self._ConstantStubPolicy(action_keys, dims)

        results = compute_round_trip_consistency(
            policy, loader, TRAJ_ID, EMBODIMENT_TAG, depths=[1], steps=120
        )

        assert 1 in results
        result = results[1]
        assert result.depth == 1
        assert len(result.start_steps) == len(result.c_i)
        assert result.c_i.ndim == 1
        assert len(result.c_i) > 0
        assert np.all(np.isfinite(result.c_i))
        for key, arr in result.per_key_c_i.items():
            assert key in action_keys
            assert np.all(np.isfinite(arr))

    def test_use_predicted_forward_not_implemented(self, loader):
        with pytest.raises(NotImplementedError):
            compute_round_trip_consistency(
                None, loader, TRAJ_ID, EMBODIMENT_TAG, use_predicted_forward=True
            )


class TestComputeRoundTripConsistencyExactRecovery:
    """The mathematically-exact-inverse stub must drive C_i to ~0 (machine
    precision) - validates the composition/decoding pipeline itself,
    independent of any trained model (see the integration plan)."""

    class _ExactInverseStubPolicy:
        """No model: looks up which trajectory row an observation's state
        corresponds to (exact match against the trajectory - the state was
        read directly from it, with no lossy transform in between) and
        returns the precomputed exact inverse for that row."""

        def __init__(self, traj, probe_key: str, expected: dict[int, dict]):
            self.traj = traj
            self.probe_key = probe_key
            self.expected = expected

        def _find_step(self, observation) -> int:
            target = np.asarray(observation["state"][self.probe_key][0, -1])
            col = self.traj[f"state.{self.probe_key}"]
            for i in range(len(col)):
                if np.array_equal(np.asarray(col.iloc[i]).reshape(-1), target.reshape(-1)):
                    return i
            raise AssertionError(
                f"round_trip_consistency queried a state not present in the trajectory "
                f"(probe_key={self.probe_key})"
            )

        def get_action(self, observation, options=None):
            assert options is not None and options.get("direction") == 1
            future_step = self._find_step(observation)
            return self.expected[future_step], {}

    def test_exact_recovery(self, loader):
        depth = 1
        action_keys = loader.modality_configs["action"].modality_keys
        action_configs = loader.modality_configs["action"].action_configs
        key_to_state_key = _key_to_state_key(loader)
        action_horizon = int(loader.modality_configs["action"].delta_indices[-1]) + 1
        stride = depth * action_horizon

        traj = loader[TRAJ_ID]
        steps = 120
        actual_steps = min(steps, len(traj))

        # Mirror compute_round_trip_consistency's own loop exactly to build
        # the (future_step -> exact inverse) lookup the stub will answer from.
        expected: dict[int, dict] = {}
        for t in range(0, actual_steps - stride, action_horizon):
            future_step = t + stride
            if future_step >= len(traj):
                continue
            backward = {}
            for key, ac in zip(action_keys, action_configs):
                state_key = key_to_state_key[key]
                state_t = np.asarray(traj[f"state.{state_key}"].iloc[t]).reshape(-1).astype(float)
                forward_chunk = np.stack(
                    [
                        np.asarray(traj[f"action.{key}"].iloc[t + j]).reshape(-1).astype(float)
                        for j in range(stride)
                    ]
                )
                backward[key] = _invert_chunk(ac, state_t, forward_chunk)
            expected[future_step] = backward

        assert expected, "trajectory too short for this test's depth/steps"

        probe_key = key_to_state_key[action_keys[0]]
        policy = self._ExactInverseStubPolicy(traj, probe_key, expected)

        results = compute_round_trip_consistency(
            policy, loader, TRAJ_ID, EMBODIMENT_TAG, depths=[depth], steps=steps
        )

        assert depth in results
        c_i = results[depth].c_i
        assert len(c_i) == len(expected)
        assert np.all(np.isfinite(c_i))
        np.testing.assert_allclose(c_i, 0.0, atol=1e-4)
