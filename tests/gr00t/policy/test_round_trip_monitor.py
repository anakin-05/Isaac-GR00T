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
Tests for gr00t.policy.round_trip_monitor.RoundTripMonitor: interval/depth bookkeeping,
ring-buffer reset, JSONL log shape, and failure isolation. Mocked BasePolicy, CPU-safe, no
sim/GPU - mirrors the mocked-policy pattern in tests/gr00t/eval/test_round_trip_consistency.py.

The stub policy below uses Gr00tSimPolicyWrapper's flat observation/action key convention
(``state.<key>``, ``action.<key>``), since that's what every closed-loop benchmark actually
sends in practice (LIBERO/RoboCasa/RoboCasa-GR1/RoboLab all use --use-sim-policy-wrapper).
"""

from __future__ import annotations

import json
from typing import Any

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.policy.policy import BasePolicy
from gr00t.policy.round_trip_monitor import RoundTripMonitor
import numpy as np
import pytest


def _make_modality_config(action_dims: dict[str, int]) -> dict[str, ModalityConfig]:
    keys = list(action_dims.keys())
    return {
        "state": ModalityConfig(delta_indices=[0], modality_keys=keys),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=keys,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in keys
            ],
        ),
    }


class _StubPolicy(BasePolicy):
    """A minimal BasePolicy: forward calls return zeros; backward calls
    (direction=1) return a caller-supplied reconstruction of the current
    single-env observation's state, so tests can control exactly what
    RoundTripMonitor compares against cached history."""

    def __init__(
        self,
        action_dims: dict[str, int],
        state_dims: dict[str, int] | None = None,
        reconstruct=None,
    ):
        super().__init__(strict=False)
        self.action_dims = action_dims
        self.state_dims = state_dims or action_dims
        self.reconstruct = reconstruct  # fn(single_obs) -> {key: np.ndarray (D,)} or None
        self.forward_calls = 0
        self.backward_calls = 0
        self.reset_calls = 0

    def check_observation(self, observation: dict[str, Any]) -> None:
        pass

    def check_action(self, action: dict[str, Any]) -> None:
        pass

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return _make_modality_config(self.action_dims)

    def _get_action(self, observation, options=None):
        batch = next(v for k, v in observation.items() if k.startswith("state.")).shape[0]
        if options and options.get("direction") == 1:
            self.backward_calls += 1
            assert batch == 1, "backward queries are always single-env"
            recon = self.reconstruct(observation) if self.reconstruct else {}
            action = {}
            for key, dim in self.action_dims.items():
                val = recon.get(key, np.zeros(dim, dtype=np.float32))
                action[f"action.{key}"] = np.asarray(val, dtype=np.float32).reshape(1, 1, dim)
            return action, {}
        self.forward_calls += 1
        action = {
            f"action.{key}": np.zeros((batch, 1, dim), dtype=np.float32)
            for key, dim in self.action_dims.items()
        }
        return action, {}

    def reset(self, options=None):
        self.reset_calls += 1
        return {}


def _obs(values: dict[str, float | list[float]], batch: int = 1) -> dict[str, Any]:
    """Build a flat batched observation: state.<key> = (batch, 1, D)."""
    result = {}
    for key, val in values.items():
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        result[f"state.{key}"] = np.tile(arr, (batch, 1, 1))
    return result


class TestIntervalAndDepthBookkeeping:
    def test_check_fires_at_interval_and_logs_expected_rows(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(action_dims={"x": 1, "gripper": 1}, state_dims={"x": 1, "gripper": 2})
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=2,
            round_trip_depths=[1, 2],
            round_trip_log_path=str(log_path),
        )

        for i in range(5):
            monitor.get_action(_obs({"x": float(i), "gripper": [0.0, 0.0]}))

        # Checks fire at call_index 1 (depth=1 only has a valid target) and 3
        # (both depth=1 and depth=2 have valid targets): 1 + 2 = 3 rows.
        assert policy.forward_calls == 5
        assert policy.backward_calls == 2  # one check call at index 1, one at index 3
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(rows) == 3
        for row in rows:
            assert set(row.keys()) == {"env_slot", "call_index", "depth", "c_i", "per_key"}
            assert row["env_slot"] == 0
            # "gripper" has a dimension-mismatched state/action pair (2 vs 1) -
            # excluded from per_key, same as state_action_processor.apply_reversed.
            assert set(row["per_key"].keys()) == {"x"}
            assert np.isfinite(row["c_i"])

    def test_multiple_env_slots_tracked_independently(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(action_dims={"x": 1})
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=2,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        for i in range(4):
            monitor.get_action(_obs({"x": float(i)}, batch=3))

        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        # 3 env slots x checks at call_index 1 and 3 = 6 rows.
        assert len(rows) == 6
        assert {row["env_slot"] for row in rows} == {0, 1, 2}

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            RoundTripMonitor(_StubPolicy(action_dims={"x": 1}), round_trip_check_interval=0)


class TestExactRecoveryDepth1:
    def test_perfect_depth1_reconstruction_gives_zero_c_i(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(
            action_dims={"x": 1},
            # A depth-1-only oracle: reconstruct "the state one call ago" as
            # (current state - 1), matching how _obs below advances x by 1 per call.
            reconstruct=lambda obs: {"x": obs["state.x"][0, -1] - 1.0},
        )
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=1,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        for i in range(4):
            monitor.get_action(_obs({"x": float(i)}))

        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        # No valid depth=1 target at call_index 0 (nothing one call before it) -
        # 3 rows from call_index 1, 2, 3.
        assert len(rows) == 3
        for row in rows:
            assert row["c_i"] == pytest.approx(0.0, abs=1e-6)


class TestResetClearsHistory:
    def test_reset_clears_ring_buffer_and_delegates(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(action_dims={"x": 1})
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=1,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        monitor.get_action(_obs({"x": 1.0}))
        monitor.get_action(_obs({"x": 2.0}))
        assert monitor._slots  # history populated

        monitor.reset()
        assert policy.reset_calls == 1
        assert monitor._slots == {}

        # Immediately after reset, the first call has no history yet - no new
        # backward query (still just the 2 from before reset).
        monitor.get_action(_obs({"x": 3.0}))
        assert (
            policy.backward_calls == 1
        )  # only the one from before reset (call_index 1, pre-reset)


class TestFailureIsolation:
    def test_round_trip_failure_does_not_break_get_action(self, monkeypatch):
        policy = _StubPolicy(action_dims={"x": 1})
        monitor = RoundTripMonitor(policy, round_trip_check_interval=1, round_trip_depths=[1])

        def _boom(self, observation):
            raise RuntimeError("simulated diagnostic failure")

        monkeypatch.setattr(RoundTripMonitor, "_maybe_check_round_trip", _boom)

        action, info = monitor.get_action(_obs({"x": 1.0}))
        assert "action.x" in action
        assert policy.forward_calls == 1
