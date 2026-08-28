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
Regression test for _collect_rollout_episodes' round-trip options bookkeeping:
every policy.get_action call must carry options["round_trip_task"] (constant for
the whole rollout) and options["round_trip_episode_ids"] (one id per env-batch
slot, bumped exactly when that slot's episode ends) - see
gr00t/policy/round_trip_monitor.py's "Per-episode and per-task identity" section
for why RoundTripMonitor needs this. A fake 2-env, 2-episode env/policy drives
the real loop directly; no gymnasium/sim dependency.
"""

from gr00t.eval.rollout_policy import _collect_rollout_episodes
from gr00t.policy.policy import BasePolicy
import numpy as np


class _FakePolicy(BasePolicy):
    def __init__(self):
        super().__init__(strict=False)
        self.seen_options: list[dict] = []

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def _get_action(self, observation, options=None):
        # _collect_rollout_episodes builds a fresh options dict per call
        # (never mutates one in place), so storing the reference directly is
        # safe - it will never change out from under this list afterward.
        self.seen_options.append(options)
        return {"action.x": np.zeros((2, 1, 1), dtype=np.float32)}, {}

    def reset(self, options=None):
        return {}


class _FakeEnv:
    """2 sub-envs; env 0's episode ends on the first step, env 1's on the
    second - enough to exercise one episode-id bump per slot independently."""

    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None):
        return {"state.x": np.zeros((2, 1, 1), dtype=np.float32)}, {}

    def step(self, actions):
        self.step_count += 1
        next_obs = {"state.x": np.zeros((2, 1, 1), dtype=np.float32)}
        rewards = np.array([0.0, 0.0])
        if self.step_count == 1:
            terminations = np.array([True, False])
        else:
            terminations = np.array([False, True])
        truncations = np.array([False, False])
        env_infos = {"n_env_steps": np.array([1, 1])}
        return next_obs, rewards, terminations, truncations, env_infos


def test_round_trip_episode_ids_bump_independently_per_slot():
    policy = _FakePolicy()
    env = _FakeEnv()

    _collect_rollout_episodes(env, policy, n_episodes=2, n_envs=2, seed=None, env_name="my_task")

    assert len(policy.seen_options) == 2
    # Before any step: both slots at episode 0.
    assert policy.seen_options[0] == {
        "round_trip_task": "my_task",
        "round_trip_episode_ids": ["0:0", "1:0"],
    }
    # After step 1 (env 0's episode ended): only slot 0 bumped.
    assert policy.seen_options[1] == {
        "round_trip_task": "my_task",
        "round_trip_episode_ids": ["0:1", "1:0"],
    }


def test_round_trip_task_defaults_to_empty_string():
    """env_name is optional (defaults to "") - callers that don't pass it (or
    that predate this feature) still get a well-formed, if unlabeled, options dict."""
    policy = _FakePolicy()
    env = _FakeEnv()

    _collect_rollout_episodes(env, policy, n_episodes=1, n_envs=2, seed=None)

    assert policy.seen_options[0]["round_trip_task"] == ""


def test_episode_infos_records_the_id_of_the_episode_that_just_ended():
    """episode_infos["round_trip_episode_id"] must be index-aligned with
    episode_successes/episode_lengths/episode_rewards (same completion-order
    convention as task_progress/q_score above it), so a round-trip JSONL log's
    per-episode C_i can be joined back to whether that episode actually
    succeeded - the whole point of adding this field."""
    policy = _FakePolicy()
    env = _FakeEnv()

    successes, lengths, rewards, episode_infos = _collect_rollout_episodes(
        env, policy, n_episodes=2, n_envs=2, seed=None, env_name="my_task"
    )

    assert len(successes) == len(lengths) == len(rewards) == 2
    # Env 0's episode ends first (step 1), env 1's second (step 2) - the id
    # recorded is the one that JUST ended (counter value before the bump).
    assert episode_infos["round_trip_episode_id"] == ["0:0", "1:0"]
