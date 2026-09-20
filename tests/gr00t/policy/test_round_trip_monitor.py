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
        latent_reconstruct=None,
        hidden_size: int = 3,
    ):
        super().__init__(strict=False)
        self.action_dims = action_dims
        self.state_dims = state_dims or action_dims
        self.reconstruct = reconstruct  # fn(single_obs) -> {key: np.ndarray (D,)} or None
        # fn(single_obs) -> np.ndarray (hidden_size,) or None. When set, forward
        # calls also report a latent (state.x, the only key these tests ever
        # give it, tiled to hidden_size) so a test can control the *backward*
        # reconstruction independently and check latent_c_i exactly - same
        # opt-in-by-callback pattern as `reconstruct` above, just for
        # info["latent_pred"] instead of the decoded action.
        self.latent_reconstruct = latent_reconstruct
        self.hidden_size = hidden_size
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
            info = {}
            if options.get("return_latent") and self.latent_reconstruct is not None:
                latent = np.asarray(self.latent_reconstruct(observation), dtype=np.float32)
                info["latent_pred"] = latent.reshape(1, self.hidden_size)
            return action, info
        self.forward_calls += 1
        action = {
            f"action.{key}": np.zeros((batch, 1, dim), dtype=np.float32)
            for key, dim in self.action_dims.items()
        }
        info = {}
        if options and options.get("return_latent") and self.latent_reconstruct is not None:
            x = next(v for k, v in observation.items() if k.startswith("state."))[:, -1, 0]
            info["latent_pred"] = np.tile(x[:, None], (1, self.hidden_size)).astype(np.float32)
        return action, info

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
            assert set(row.keys()) == {
                "env_slot",
                "call_index",
                "depth",
                "action_c_i",
                "latent_c_i",
                "episode_id",
                "task",
                "per_key",
            }
            assert row["env_slot"] == 0
            # No round_trip_episode_ids/round_trip_task supplied in this test -
            # both fields log as null, same as a client that's never heard of them.
            assert row["episode_id"] is None
            assert row["task"] is None
            # "gripper" has a dimension-mismatched state/action pair (2 vs 1) -
            # excluded from per_key, same as state_action_processor.apply_reversed.
            assert set(row["per_key"].keys()) == {"x"}
            assert np.isfinite(row["action_c_i"])
            # This stub never sets latent_reconstruct - no latent to compare.
            assert row["latent_c_i"] is None

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
            assert row["action_c_i"] == pytest.approx(0.0, abs=1e-6)


class TestLatentCi:
    """latent_c_i - see round_trip_monitor.py's "Two C_i metrics" section.
    latent_reconstruct mirrors reconstruct's role above, but for
    info["latent_pred"] instead of the decoded action."""

    def test_perfect_depth1_latent_reconstruction_gives_zero_latent_c_i(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(
            action_dims={"x": 1},
            reconstruct=lambda obs: {"x": obs["state.x"][0, -1] - 1.0},
            # Forward calls cache tile(x, hidden_size) as this call's latent (see
            # _StubPolicy._get_action); reconstructing "x one call ago" here,
            # tiled the same way, should exactly match it at depth=1.
            latent_reconstruct=lambda obs: np.full(3, obs["state.x"][0, -1] - 1.0),
            hidden_size=3,
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
        assert len(rows) == 3
        for row in rows:
            assert row["action_c_i"] == pytest.approx(0.0, abs=1e-6)
            assert row["latent_c_i"] == pytest.approx(0.0, abs=1e-6)

    def test_off_by_one_latent_reconstruction_gives_nonzero_latent_c_i(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(
            action_dims={"x": 1},
            reconstruct=lambda obs: {"x": obs["state.x"][0, -1] - 1.0},
            # Deliberately wrong by a constant offset of 2 in latent space only -
            # action_c_i (still exact) and latent_c_i (now 4.0 = 2**2 per
            # component) must be independent numbers, not the same value twice.
            latent_reconstruct=lambda obs: np.full(3, obs["state.x"][0, -1] - 1.0 + 2.0),
            hidden_size=3,
        )
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=1,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        for i in range(2):
            monitor.get_action(_obs({"x": float(i)}))

        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["action_c_i"] == pytest.approx(0.0, abs=1e-6)
        assert rows[0]["latent_c_i"] == pytest.approx(4.0, abs=1e-6)

    def test_latent_c_i_is_none_when_policy_does_not_support_return_latent(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(
            action_dims={"x": 1}, reconstruct=lambda obs: {"x": obs["state.x"][0, -1] - 1.0}
        )
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=1,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        for i in range(2):
            monitor.get_action(_obs({"x": float(i)}))

        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["action_c_i"] == pytest.approx(0.0, abs=1e-6)
        assert rows[0]["latent_c_i"] is None


class TestEpisodeIdAndTask:
    """options["round_trip_episode_ids"] / options["round_trip_task"] - see
    round_trip_monitor.py's "Per-episode and per-task identity" section."""

    def test_task_and_episode_id_logged_when_supplied(self, tmp_path):
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(action_dims={"x": 1})
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=1,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        options = {"round_trip_task": "libero_sim/some_task", "round_trip_episode_ids": ["0:0"]}
        monitor.get_action(_obs({"x": 1.0}), options=options)
        monitor.get_action(_obs({"x": 2.0}), options=options)

        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["task"] == "libero_sim/some_task"
        assert rows[0]["episode_id"] == "0:0"

    def test_round_trip_keys_stripped_before_forwarding(self):
        """The wrapped policy must never see round_trip_task/round_trip_episode_ids
        - those are this monitor's own bookkeeping, not the model's options contract."""
        seen_options = []

        class _RecordingPolicy(_StubPolicy):
            def _get_action(self, observation, options=None):
                seen_options.append(options)
                return super()._get_action(observation, options)

        policy = _RecordingPolicy(action_dims={"x": 1})
        monitor = RoundTripMonitor(policy, round_trip_check_interval=1, round_trip_depths=[1])
        monitor.get_action(
            _obs({"x": 1.0}),
            options={
                "round_trip_task": "t",
                "round_trip_episode_ids": ["0:0"],
                "some_other_key": 1,
            },
        )
        # return_latent is injected by RoundTripMonitor itself on every forwarded
        # call (see _get_action) - not something the caller asked for.
        assert seen_options == [{"some_other_key": 1, "return_latent": True}]

    def test_episode_id_change_clears_stale_history(self, tmp_path):
        """A new episode starting in the same env slot (gymnasium's inline
        autoreset) must not let a round-trip window span the boundary."""
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(action_dims={"x": 1})
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=1,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        opts_ep0 = {"round_trip_episode_ids": ["0:0"]}
        opts_ep1 = {"round_trip_episode_ids": ["0:1"]}

        # Two calls in episode 0 build up history...
        monitor.get_action(_obs({"x": 1.0}), options=opts_ep0)
        monitor.get_action(_obs({"x": 2.0}), options=opts_ep0)
        assert len(log_path.read_text().splitlines()) == 1  # one valid depth=1 check so far

        # ...then episode 1 starts in the SAME slot with no explicit reset() call.
        # Without episode-id tracking this would spuriously round-trip against
        # episode 0's last state; with it, history is cleared and no row logs.
        monitor.get_action(_obs({"x": 100.0}), options=opts_ep1)
        rows_after_boundary = log_path.read_text().splitlines()
        assert len(rows_after_boundary) == 1  # unchanged: no valid target in the new episode yet

    def test_episode_ids_optional_no_behavior_change_when_absent(self, tmp_path):
        """A client that never supplies round_trip_episode_ids (e.g. RoboLab) sees
        identical behavior to before this feature existed."""
        log_path = tmp_path / "rtc.jsonl"
        policy = _StubPolicy(action_dims={"x": 1})
        monitor = RoundTripMonitor(
            policy,
            round_trip_check_interval=1,
            round_trip_depths=[1],
            round_trip_log_path=str(log_path),
        )
        for i in range(3):
            monitor.get_action(_obs({"x": float(i)}))  # no options at all
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(rows) == 2
        assert all(r["episode_id"] is None and r["task"] is None for r in rows)


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
