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
Tests for Mechanism 2 - the client-side round-trip-consistency hook in
_collect_rollout_episodes (see SESSION_HANDOFF.md's "Mechanism 2" design
section, and that function's own docstring). Mocked env/policy, CPU-safe, no
sim/GPU dependency - mirrors the fake-env/fake-policy pattern already used in
test_rollout_round_trip_options.py, plus a modality config in the style of
test_round_trip_monitor.py's _StubPolicy.

The fake env's state.x increments by exactly 1 every step (reset to 0 when an
episode ends), and the fake policy's backward direction reconstructs "the
state one call ago" as (current x - 1) - exactly right for depth=1 (C_i = 0)
and off by exactly 1 for depth=2 (C_i = 1.0), giving every assertion below an
exact expected value instead of an approximate one.
"""

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.eval.rollout_policy import _collect_rollout_episodes
from gr00t.policy.policy import BasePolicy
import numpy as np
import pytest


def _modality_config() -> dict[str, ModalityConfig]:
    return {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            ],
        ),
    }


class _FakePolicy(BasePolicy):
    """Forward calls return zeros (the fake env ignores actions entirely).
    Backward calls (direction=1) reconstruct every env's "one call ago" state
    as (current x - 1) in a *single batched* call - records the batch size
    seen on each backward call so tests can confirm Mechanism 2 never issues
    one call per env slot the way Mechanism 1 does."""

    def __init__(self):
        super().__init__(strict=False)
        self.forward_calls = 0
        self.backward_batch_sizes: list[int] = []

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def get_modality_config(self):
        return _modality_config()

    def _get_action(self, observation, options=None):
        batch = observation["state.x"].shape[0]
        if options and options.get("direction") == 1:
            self.backward_batch_sizes.append(batch)
            cur_x = observation["state.x"][:, -1, 0]
            recon_x = (cur_x - 1.0).reshape(batch, 1, 1).astype(np.float32)
            return {"action.x": recon_x}, {}
        self.forward_calls += 1
        return {"action.x": np.zeros((batch, 1, 1), dtype=np.float32)}, {}

    def reset(self, options=None):
        return {}


class _FakeEnv:
    """n_envs sub-envs in lockstep; state.x increments by 1 every step and
    resets to 0 (gymnasium's inline-autoreset convention) whenever
    ``episode_length`` steps have elapsed."""

    def __init__(self, n_envs: int, episode_length: int):
        self.n_envs = n_envs
        self.episode_length = episode_length
        self.step_count = 0
        self.x = np.zeros(n_envs, dtype=np.float32)

    def reset(self, seed=None):
        self.step_count = 0
        self.x = np.zeros(self.n_envs, dtype=np.float32)
        return self._obs(), {}

    def _obs(self):
        return {"state.x": self.x.reshape(self.n_envs, 1, 1).copy()}

    def step(self, actions):
        self.step_count += 1
        self.x = self.x + 1.0
        done = (self.step_count % self.episode_length) == 0
        terminations = np.full(self.n_envs, done)
        truncations = np.zeros(self.n_envs, dtype=bool)
        if done:
            # Inline autoreset: next_obs is the FRESH episode's first
            # observation, not one more step of the episode that just ended -
            # matches the real gymnasium convention _macro_step_env_steps'
            # docstring describes, and what the terminating-step's env.step()
            # actually hands back in production.
            self.x = np.zeros(self.n_envs, dtype=np.float32)
        next_obs = self._obs()
        rewards = np.zeros(self.n_envs)
        env_infos = {"n_env_steps": np.ones(self.n_envs, dtype=int)}
        return next_obs, rewards, terminations, truncations, env_infos


class TestDisabledByDefault:
    def test_no_backward_calls_or_new_keys_when_disabled(self):
        policy = _FakePolicy()
        env = _FakeEnv(n_envs=2, episode_length=4)

        _, _, _, episode_infos = _collect_rollout_episodes(
            env, policy, n_episodes=2, n_envs=2, seed=None
        )

        assert policy.backward_batch_sizes == []
        assert not any(k.startswith("round_trip_c_i_depth") for k in episode_infos)


class TestBatchedBackwardCall:
    def test_one_batched_call_covers_every_env_slot(self):
        """The whole point of Mechanism 2 over Mechanism 1: one backward call
        per check, not one per env slot."""
        policy = _FakePolicy()
        env = _FakeEnv(n_envs=4, episode_length=100)

        _collect_rollout_episodes(
            env,
            policy,
            n_episodes=4,
            n_envs=4,
            seed=None,
            enable_round_trip_check=True,
            round_trip_check_interval=2,
            round_trip_depths=(1,),
        )

        assert len(policy.backward_batch_sizes) > 0
        assert all(batch == 4 for batch in policy.backward_batch_sizes)


class TestExactValuesAndEpisodeBoundary:
    def test_per_episode_mean_ci_and_no_cross_episode_contamination(self):
        """One env, two 4-step episodes back to back (episode_length=4,
        check_interval=2, depths=(1, 2)). Depth=1 reconstructs exactly (C_i=0)
        in both episodes; depth=2 is off by exactly 1 (C_i=1.0) whenever a
        valid target exists. Both episodes landing on the same exact values
        proves the second episode's checks never spuriously used episode 1's
        cached state after the autoreset boundary - if they had, the huge x
        discontinuity (episode 1 ends at x=4, episode 2 starts at x=0) would
        show up as a wildly different, inexact C_i instead of these clean
        values.
        """
        policy = _FakePolicy()
        env = _FakeEnv(n_envs=1, episode_length=4)

        successes, lengths, rewards, episode_infos = _collect_rollout_episodes(
            env,
            policy,
            n_episodes=2,
            n_envs=1,
            seed=None,
            enable_round_trip_check=True,
            round_trip_check_interval=2,
            round_trip_depths=(1, 2),
        )

        assert len(successes) == 2
        assert episode_infos["round_trip_c_i_depth1"] == pytest.approx([0.0, 0.0], abs=1e-6)
        assert episode_infos["round_trip_c_i_depth2"] == pytest.approx([1.0, 1.0], abs=1e-6)

    def test_episode_shorter_than_check_interval_records_nan(self):
        """An episode that ends before a single check ever fires must still
        append one entry per depth (index-aligned with episode_successes) -
        NaN, not a missing/short list."""
        policy = _FakePolicy()
        env = _FakeEnv(n_envs=1, episode_length=1)

        _, _, _, episode_infos = _collect_rollout_episodes(
            env,
            policy,
            n_episodes=3,
            n_envs=1,
            seed=None,
            enable_round_trip_check=True,
            round_trip_check_interval=8,
            round_trip_depths=(1,),
        )

        assert policy.backward_batch_sizes == []
        assert len(episode_infos["round_trip_c_i_depth1"]) == 3
        assert all(np.isnan(v) for v in episode_infos["round_trip_c_i_depth1"])
