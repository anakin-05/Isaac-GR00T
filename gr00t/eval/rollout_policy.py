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

from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
import logging
from pathlib import Path
import sys
import time
from typing import Any
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.deployment.modes import InferenceMode
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
from gr00t.policy.round_trip_monitor import (
    _last_timestep,
    _modality_dict,
    action_key_to_state_key_map,
    reconstruct_last_timesteps_batched,
)
from gr00t.utils.determinism import seed_everything
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import tyro


logger = logging.getLogger(__name__)


ROBOCASA_PANDA_RECORD_VIDEO_KEYS = (
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
)

ROBOCASA365_PANDA_RECORD_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)

ROBOCASA_RECORD_VIDEO_KEYS_BY_PREFIX = {
    "robocasa_panda_omron": ROBOCASA_PANDA_RECORD_VIDEO_KEYS,
    "robocasa365_panda_omron": ROBOCASA365_PANDA_RECORD_VIDEO_KEYS,
}

# Canonical per-episode step budget for the default (LIBERO) backend. Kept in one
# place so the CLI, video, and multi-step defaults cannot drift.
DEFAULT_MAX_EPISODE_STEPS = 720


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    fps: int = 20
    codec: str = "h264"
    overlay_text: bool = True
    record_video_keys: tuple[str, ...] | None = None


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        contract: policy-resolved :class:`PolicyHorizonSpec` carrying
            ``n_action_steps`` and the video / state delta-indices.
        max_episode_steps: Maximum number of steps per episode.
        terminate_on_success: End the episode once the task is reported solved.
    """

    contract: PolicyHorizonSpec
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        multistep: Configuration for multi-step processing (required; carries
            the policy-resolved horizon contract).
        video: Configuration for video recording.
    """

    multistep: MultiStepConfig
    video: VideoConfig = field(default_factory=VideoConfig)


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_robocasa_env_fn(
    env_name: str,
    robocasa_split: str = "",
):
    def env_fn():
        kwargs = {}
        if env_name.startswith("robocasa365_panda_omron/"):
            import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401

            if robocasa_split:
                kwargs["split"] = robocasa_split
        else:
            import robocasa  # noqa: F401
            import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        return gym.make(env_name, enable_render=True, **kwargs)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int, robocasa_split: str = ""):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)
    env_prefix = env_name.split("/")[0]

    if env_prefix in ("robocasa_panda_omron", "robocasa365_panda_omron", "gr1_unified"):
        env_fn = get_robocasa_env_fn(env_name, robocasa_split=robocasa_split)

    elif env_embodiment in (EmbodimentTag.SIMPLER_ENV_GOOGLE, EmbodimentTag.SIMPLER_ENV_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    wrapper_configs: WrapperConfigs,
    robocasa_split: str = "",
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs, robocasa_split=robocasa_split)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper

        record_video_keys = wrapper_configs.video.record_video_keys
        env_prefix = env_name.split("/")[0]
        if record_video_keys is None:
            record_video_keys = ROBOCASA_RECORD_VIDEO_KEYS_BY_PREFIX.get(env_prefix)

        env = VideoRecordingWrapper(
            env,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            overlay_text=wrapper_configs.video.overlay_text,
            record_video_keys=record_video_keys,
        )

    env = MultiStepWrapper(
        env,
        contract=wrapper_configs.multistep.contract,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """AsyncVectorEnv that tolerates variable-shaped info arrays across envs.

    Gymnasium's default _add_info pre-allocates a numpy array based on the
    first env's value shape and then assigns subsequent envs into it.  When
    envs return differently-shaped values (e.g. variable-length contact arrays)
    the assignment raises ValueError.  We catch that and fall back to a plain
    Python list for that key so the rest of the step can proceed normally.
    """

    def _add_info(self, infos, info, env_num):
        for k, v in info.items():
            if k not in infos:
                infos[k] = [None] * self.num_envs
                infos[f"_{k}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[k], np.ndarray):
                try:
                    infos[k][env_num] = v
                except (ValueError, TypeError):
                    lst = list(infos[k])
                    lst[env_num] = v
                    infos[k] = lst
            else:
                infos[k][env_num] = v
            infos[f"_{k}"][env_num] = True
        return infos


def _macro_step_env_steps(env_infos: dict, env_idx: int) -> int:
    """Inner env-steps advanced by one macro-step for ``env_idx``.

    ``MultiStepWrapper`` records the number of inner ``super().step()`` calls in
    ``info["n_env_steps"]``. The vector env relocates that info depending on the
    gymnasium version:

    * gymnasium 0.29.1 (sim venvs) uses inline autoreset: a terminating step's
      info is moved into ``final_info`` while the top-level info describes the
      freshly reset env, which has **no** ``n_env_steps``.
    * gymnasium >=1.0 keeps the terminating step's info at the top level.

    ``final_info`` is therefore consulted first; otherwise the terminal
    macro-step is silently counted as 0 env-steps and ``episode_length``
    collapses to 0, tripping the zero-length-episode invariant downstream.
    """
    final_info = env_infos.get("final_info")
    if final_info is not None and final_info[env_idx] is not None:
        step_info = final_info[env_idx]
        if "n_env_steps" in step_info:
            return int(step_info["n_env_steps"])
    if "n_env_steps" in env_infos:
        # The key can be present with a None entry for this env; count 0 rather
        # than calling int(None).
        n_env_steps = env_infos["n_env_steps"][env_idx]
        if n_env_steps is not None:
            return int(n_env_steps)
    return 0


def _update_round_trip_state(
    *,
    policy: BasePolicy,
    observations: dict[str, Any],
    call_index: int,
    n_envs: int,
    history: list[deque[tuple[int, dict[str, np.ndarray]]]],
    pending: list[dict[int, list[float]]],
    max_depth: int,
    check_interval: int,
    depths: tuple[int, ...],
    action_keys: list[str],
    key_to_state_key: dict[str, str],
) -> None:
    """Mechanism 2's per-macro-step hook (see ``_collect_rollout_episodes``'s
    docstring): cache this call's true state for every env slot, then every
    ``check_interval`` calls issue *one* batched backward-direction query and
    accumulate each env's C_i (per requested depth) into ``pending`` for that
    env's in-progress episode. Mirrors ``RoundTripMonitor._check_slot``'s math
    (Mechanism 1) but with a single batched call instead of one per env slot,
    since this loop's observation is already naturally batched.
    """
    state = _modality_dict(observations, "state")
    for slot in range(n_envs):
        true_state = {key: _last_timestep(np.asarray(arr[slot])) for key, arr in state.items()}
        history[slot].append((call_index, true_state))
        while len(history[slot]) > max_depth + 1:
            history[slot].popleft()

    if (call_index + 1) % check_interval != 0:
        return

    histories_by_index = [dict(h) for h in history]
    has_target = any(
        (call_index - depth) in histories_by_index[slot]
        for slot in range(n_envs)
        for depth in depths
    )
    if not has_target:
        # Nothing would be logged from the (expensive) backward query yet -
        # e.g. still early in the rollout - so skip issuing it at all.
        return

    backward_action, _ = policy.get_action(
        observations, options={"direction": 1, "drop_vision": True}
    )
    backward_state = _modality_dict(backward_action, "action")
    reconstructed = {
        key: reconstruct_last_timesteps_batched(backward_state[key])
        for key in action_keys
        if key in backward_state
    }

    for slot in range(n_envs):
        for depth in depths:
            cached = histories_by_index[slot].get(call_index - depth)
            if cached is None:
                continue
            per_key: dict[str, float] = {}
            for key in action_keys:
                state_key = key_to_state_key[key]
                if state_key not in cached or key not in reconstructed:
                    continue
                true_val = cached[state_key]
                est_val = reconstructed[key][slot]
                if true_val.shape != est_val.shape:
                    # Same dimension-mismatch case state_action_processor.apply_reversed
                    # already skips (e.g. LIBERO's "gripper") - no comparable anchor.
                    continue
                per_key[key] = float(np.mean((true_val - est_val) ** 2))
            if per_key:
                pending[slot][depth].append(float(np.mean(list(per_key.values()))))


def _collect_rollout_episodes(
    env,
    policy: BasePolicy,
    n_episodes: int,
    n_envs: int,
    seed: int | None,
    env_name: str = "",
    enable_round_trip_check: bool = False,
    round_trip_check_interval: int = 8,
    round_trip_depths: tuple[int, ...] = (1, 2, 4),
):
    """Run the rollout loop and return ``(successes, lengths, rewards, infos)``.

    Owns the progress bar and the initial / inter-episode env resets; the
    caller owns env teardown so ``env.close()`` still runs if this raises.

    ``env_name`` is optional and purely additive: it's forwarded to the policy
    as ``options["round_trip_task"]``/``options["round_trip_episode_ids"]`` for
    RoundTripMonitor's benefit (see gr00t/policy/round_trip_monitor.py's
    "Per-episode and per-task identity" section) - a plain policy that doesn't
    look for those keys is unaffected.

    ``enable_round_trip_check`` turns on Mechanism 2 (see
    SESSION_HANDOFF.md's "Mechanism 2" design section): a client-side
    round-trip-consistency diagnostic, complementary to the server-side
    ``RoundTripMonitor`` (Mechanism 1). Every ``round_trip_check_interval``
    macro-steps it issues one additional *batched* backward-direction
    ``policy.get_action`` call (one call for the whole vectorized batch, not
    one per env slot - this loop already has a naturally batched observation),
    computes each env's C_i against its own cached true state from ``depth``
    calls ago, and appends the *mean* of whatever C_i values were computed
    during that episode into ``episode_infos[f"round_trip_c_i_depth{depth}"]``
    when that env's episode ends (``float("nan")`` if none were - e.g. an
    episode shorter than one check interval) - the same per-episode
    aggregation sink ``"task_progress"``/``"q_score"`` already use, so results
    surface in whatever summary a benchmark already produces, with no new
    reporting plumbing. Because this function owns episode boundaries
    directly (unlike Mechanism 1, which must infer them from the
    ``round_trip_episode_ids`` this same function supplies via ``options``),
    a round-trip window can never span an episode boundary without needing
    that extra id-based bookkeeping: each env slot's own history ring buffer
    is simply cleared the moment that slot's episode ends. Only meaningful
    for a checkpoint trained with ``--enable-direction-conditioning``; adds a
    real, non-optional compute cost identical to Mechanism 1's
    (``round_trip_check_interval`` should be kept equal to the caller's
    action-execution horizon, same reason and same caveat as Mechanism 1's).
    """
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    current_rewards = [0.0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)
    # One counter per env-batch slot, incremented exactly when that slot's
    # episode ends (same event as the episode_lengths/rewards/successes
    # bookkeeping below) - gives RoundTripMonitor a way to detect gymnasium's
    # inline autoreset starting a new episode in the same slot mid-rollout,
    # which positional slot index alone can't reveal. A fresh options dict is
    # built per get_action call (below) rather than mutating one shared
    # instance in place, so nothing downstream can observe a call's episode
    # ids changing retroactively after the fact.
    round_trip_episode_counters = [0] * n_envs

    def _round_trip_options() -> dict:
        return {
            "round_trip_task": env_name,
            "round_trip_episode_ids": [
                f"{i}:{c}" for i, c in enumerate(round_trip_episode_counters)
            ],
        }

    # Mechanism 2 bookkeeping (see docstring above) - every piece here is a
    # no-op when enable_round_trip_check is False.
    round_trip_max_depth = max(round_trip_depths) if round_trip_depths else 0
    round_trip_history: list[deque[tuple[int, dict[str, np.ndarray]]]] = [
        deque() for _ in range(n_envs)
    ]
    round_trip_pending: list[dict[int, list[float]]] = [
        {depth: [] for depth in round_trip_depths} for _ in range(n_envs)
    ]
    round_trip_action_keys: list[str] = []
    round_trip_key_to_state_key: dict[str, str] = {}
    if enable_round_trip_check:
        round_trip_action_keys, round_trip_key_to_state_key = action_key_to_state_key_map(policy)

    # Initial reset; if a seed is provided, give each sub-env a distinct but
    # deterministic seed so that parallel workers don't all start from the
    # same initial state while still being run-to-run reproducible.
    if seed is not None:
        reset_seeds = [int(seed) + i for i in range(n_envs)]
        observations, _ = env.reset(seed=reset_seeds)
    else:
        observations, _ = env.reset()
    policy.reset()

    call_index = 0
    pbar = tqdm(total=n_episodes, desc="Episodes")
    try:
        while completed_episodes < n_episodes:
            actions, _ = policy.get_action(observations, options=_round_trip_options())

            if enable_round_trip_check:
                try:
                    _update_round_trip_state(
                        policy=policy,
                        observations=observations,
                        call_index=call_index,
                        n_envs=n_envs,
                        history=round_trip_history,
                        pending=round_trip_pending,
                        max_depth=round_trip_max_depth,
                        check_interval=round_trip_check_interval,
                        depths=round_trip_depths,
                        action_keys=round_trip_action_keys,
                        key_to_state_key=round_trip_key_to_state_key,
                    )
                except Exception:
                    # The diagnostic must never take down a real rollout.
                    logger.exception("Mechanism 2 round-trip check failed; continuing without it.")

            next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
            # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
            # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
            for env_idx in range(n_envs):
                if "success" in env_infos:
                    env_success = env_infos["success"][env_idx]
                    if isinstance(env_success, list):
                        env_success = np.any(env_success)
                    elif isinstance(env_success, np.ndarray):
                        env_success = np.any(env_success)
                    elif isinstance(env_success, bool):
                        env_success = env_success
                    elif isinstance(env_success, int):
                        env_success = bool(env_success)
                    else:
                        raise ValueError(f"Unknown success dtype: {type(env_success)}")
                    current_successes[env_idx] |= bool(env_success)
                else:
                    current_successes[env_idx] = False

                if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                    env_success = env_infos["final_info"][env_idx]["success"]
                    if isinstance(env_success, list):
                        env_success = any(env_success)
                    elif isinstance(env_success, np.ndarray):
                        env_success = np.any(env_success)
                    elif isinstance(env_success, bool):
                        env_success = env_success
                    elif isinstance(env_success, int):
                        env_success = bool(env_success)
                    else:
                        raise ValueError(f"Unknown success dtype: {type(env_success)}")
                    current_successes[env_idx] |= bool(env_success)
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += _macro_step_env_steps(env_infos, env_idx)

                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    if "final_info" in env_infos:
                        current_successes[env_idx] |= any(
                            env_infos["final_info"][env_idx]["success"]
                        )
                    if "task_progress" in env_infos:
                        episode_infos["task_progress"].append(
                            env_infos["task_progress"][env_idx][-1]
                        )
                    if "q_score" in env_infos:
                        episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                    if "valid" in env_infos:
                        episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                    # Accumulate per-episode results. Both lists are captured
                    # BEFORE the per-env trackers are reset to 0 below — without
                    # this ordering downstream consumers silently see
                    # episode_length=0 / episode_reward=0.0.
                    episode_lengths.append(current_lengths[env_idx])
                    episode_rewards.append(float(current_rewards[env_idx]))
                    episode_successes.append(current_successes[env_idx])
                    # Same completion-order convention as the three lists above -
                    # index-aligned with episode_successes - so a round-trip log's
                    # "episode_id" column (see round_trip_monitor.py) can be joined
                    # back to this episode's actual outcome. Captured BEFORE the
                    # counter bumps below, i.e. this is the id of the episode that
                    # JUST ended, not the one about to start.
                    episode_infos["round_trip_episode_id"].append(
                        f"{env_idx}:{round_trip_episode_counters[env_idx]}"
                    )
                    if enable_round_trip_check:
                        # Mechanism 2: flush this episode's accumulated C_i values
                        # (mean per depth, NaN if the episode ended before any
                        # check fired) and clear this slot's history so the next
                        # episode in this slot starts with a clean ring buffer -
                        # no window can span the boundary that way.
                        for depth in round_trip_depths:
                            values = round_trip_pending[env_idx][depth]
                            episode_infos[f"round_trip_c_i_depth{depth}"].append(
                                float(np.mean(values)) if values else float("nan")
                            )
                            round_trip_pending[env_idx][depth] = []
                        round_trip_history[env_idx].clear()
                    # Reset trackers for this environment.
                    current_successes[env_idx] = False
                    # only update completed_episodes if valid
                    if "valid" in episode_infos:
                        if episode_infos["valid"][-1]:
                            completed_episodes += 1
                            pbar.update(1)
                    else:
                        # envs don't return valid
                        completed_episodes += 1
                        pbar.update(1)
                    # Reset with `0.0` to match the `[0.0] * n_envs` init and the
                    # `float(...)` cast above; otherwise the per-env entry's
                    # static type silently flips int <-> float across iterations.
                    current_rewards[env_idx] = 0.0
                    current_lengths[env_idx] = 0
                    # A new episode starts in this slot next step (gymnasium's
                    # inline autoreset) - bump its id so RoundTripMonitor can
                    # tell, even though env_idx itself doesn't change.
                    round_trip_episode_counters[env_idx] += 1
            observations = next_obs
            call_index += 1

        # Best-effort: a failed final reset must not skip the caller's env.close().
        try:
            env.reset()
        except Exception as reset_err:
            print(f"Final env.reset() before close failed; closing env anyway: {reset_err}")
    finally:
        pbar.close()

    return episode_successes, episode_lengths, episode_rewards, episode_infos


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    seed: int | None = None,
    robocasa_split: str = "",
    enable_round_trip_check: bool = False,
    round_trip_check_interval: int = 8,
    round_trip_depths: tuple[int, ...] = (1, 2, 4),
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        seed: If set, forwards per-env seeds (``seed+i``) to the first
            ``env.reset`` so each sub-env is reproducible. Should be paired
            with :func:`gr00t.utils.determinism.seed_everything` upstream to
            also constrain policy-side RNGs.
        enable_round_trip_check: Mechanism 2 - see
            ``_collect_rollout_episodes``'s docstring for the full design.
        round_trip_check_interval: Only used when ``enable_round_trip_check``.
        round_trip_depths: Only used when ``enable_round_trip_check``.
    Returns:
        ``(env_name, episode_successes, episode_infos)``. ``episode_lengths``
        in ``episode_infos`` is in **env-steps** (inner ``super().step()``
        calls), the same unit as ``MultiStepWrapper.max_episode_steps``.
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
            robocasa_split=robocasa_split,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Reap the vector env (and the ffmpeg / async-worker children it owns) on
    # every exit path; a leaked child blocks the next eval shard's ports/GPUs.
    try:
        episode_successes, episode_lengths, episode_rewards, episode_infos = (
            _collect_rollout_episodes(
                env,
                policy,
                n_episodes,
                n_envs,
                seed,
                env_name=env_name,
                enable_round_trip_check=enable_round_trip_check,
                round_trip_check_interval=round_trip_check_interval,
                round_trip_depths=round_trip_depths,
            )
        )
    finally:
        # Don't let a teardown error mask an in-flight rollout exception.
        try:
            env.close()
        except Exception as close_err:
            print(f"env.close() during teardown failed: {close_err}")
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    # Every captured episode ran >= 1 env-step, so episode_length >= 1.
    assert all(length >= 1 for length in episode_lengths), (
        f"Internal invariant violated: rollout produced zero-length episode(s) "
        f"in {episode_lengths!r}."
    )

    # Surface the per-episode length and reward that were tracked locally so
    # downstream metrics (SimplerEnv / LIBERO / Robocasa / Wholebody) can
    # read them off episode_infos instead of silently falling back to 0.
    # Planted BEFORE the "valid" filter so they get filtered in lockstep
    # with the other episode_infos fields.
    episode_infos["episode_lengths"] = episode_lengths
    episode_infos["episode_rewards"] = episode_rewards

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    trt_engine_path: str = "",
    trt_mode: InferenceMode = InferenceMode.n17_full_pipeline,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        gr00t_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=0,
        )
        if trt_engine_path:
            deploy_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment")
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from trt_model_forward import close_tensorrt_engines, setup_tensorrt_engines

            try:
                setup_tensorrt_engines(gr00t_policy, trt_engine_path, mode=trt_mode)
            except BaseException:
                # Free engines attached before the failing one so a partial
                # setup doesn't leak GPU memory.
                close_tensorrt_engines(gr00t_policy)
                raise
        policy = Gr00tSimPolicyWrapper(gr00t_policy)
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    trt_engine_path: str = "",
    trt_mode: InferenceMode = InferenceMode.n17_full_pipeline,
    seed: int | None = None,
    robocasa_split: str = "",
    enable_round_trip_check: bool = False,
    round_trip_check_interval: int = 8,
    round_trip_depths: tuple[int, ...] = (1, 2, 4),
):
    # seed_everything resolves `None` via the GR00T_EVAL_SEED env var and is a
    # no-op when that is also unset, so the historical non-deterministic
    # behavior is preserved by default. Returns the effective seed (or None)
    # which we forward to env.reset below.
    seed = seed_everything(seed)

    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if video_dir is None:
        if model_path:
            parts = model_path.split("/")
            model_slug = parts[-3] if len(parts) >= 3 else parts[-1]
            video_dir = f"/tmp/sim_eval_videos_{model_slug}_ac{n_action_steps}_{uuid.uuid4()}"
        else:
            video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    policy = create_gr00t_sim_policy(
        model_path,
        embodiment_tag,
        policy_client_host,
        policy_client_port,
        trt_engine_path=trt_engine_path,
        trt_mode=trt_mode,
    )

    # Release TRT engines explicitly on every exit path: the sim-eval entrypoint
    # hard-exits via os._exit, which skips Engine.__del__ and would leak GPU memory.
    if trt_engine_path:
        deploy_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment")
        if deploy_dir not in sys.path:
            sys.path.insert(0, deploy_dir)
        from trt_model_forward import closing_tensorrt_engines

        engine_cleanup = closing_tensorrt_engines(policy)
    else:
        engine_cleanup = nullcontext()

    with engine_cleanup:
        # Resolve the horizon contract from the policy *before* building the
        # wrapper config. The video / state delta-indices are sourced from the
        # policy's modality config (no policy-independent defaults), and
        # ``n_action_steps`` is the receding-horizon execution length validated
        # against the policy's action horizon. A mismatch now raises here at
        # construction instead of surfacing as an IndexError / cryptic
        # check_observation assert deep inside the rollout loop.
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=n_action_steps)

        wrapper_configs = WrapperConfigs(
            multistep=MultiStepConfig(
                contract=contract,
                max_episode_steps=max_episode_steps,
                terminate_on_success=True,
            ),
            video=VideoConfig(
                video_dir=video_dir,
                max_episode_steps=max_episode_steps,
            ),
        )

        results = run_rollout_gymnasium_policy(
            env_name=env_name,
            policy=policy,
            wrapper_configs=wrapper_configs,
            n_episodes=n_episodes,
            n_envs=n_envs,
            seed=seed,
            robocasa_split=robocasa_split,
            enable_round_trip_check=enable_round_trip_check,
            round_trip_check_interval=round_trip_check_interval,
            round_trip_depths=round_trip_depths,
        )
        print("Video saved to: ", wrapper_configs.video.video_dir)
        return results


@dataclass
class RolloutConfig:
    """Configuration for rollout policy evaluation."""

    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    """Maximum number of steps per episode."""

    n_episodes: int = 50
    """Number of episodes to run."""

    model_path: str = ""
    """Path to model checkpoint."""

    policy_client_host: str = ""
    """Host for policy client."""

    policy_client_port: int | None = None
    """Port for policy client."""

    env_name: str = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    """Environment name."""

    n_envs: int = 8
    """Number of parallel environments."""

    n_action_steps: int = 8
    """Number of action steps."""

    video_dir: str | None = None
    """Directory to save videos. If None, uses /tmp/sim_eval_videos_<env>_<uuid>."""

    trt_engine_path: str = ""
    """Path to TRT engine directory. If set, uses TRT inference instead of PyTorch."""

    trt_mode: InferenceMode = InferenceMode.n17_full_pipeline
    """TRT mode: which engine subset to swap in (e.g. 'n17_full_pipeline' for all
    engines, 'vit_llm_only', 'action_head', or 'dit_only')."""

    seed: int | None = None
    """Optional seed for deterministic evaluation. When set, seeds Python /
    NumPy / torch / cuda RNGs, enables cuDNN determinism, and forwards
    per-env seeds to the sim envs. If left as ``None``, falls back to the
    ``GR00T_EVAL_SEED`` env var; if that is also unset, keeps the historical
    non-deterministic behavior."""

    robocasa_split: str = ""
    """Optional RoboCasa/RoboCasa365 split forwarded to the simulator."""

    enable_round_trip_check: bool = False
    """Mechanism 2: client-side round-trip-consistency diagnostic (see
    _collect_rollout_episodes's docstring in this file). Ties C_i directly to
    each episode's outcome via episode_infos, unlike the server-side
    RoundTripMonitor (--enable-round-trip-monitor on run_gr00t_server.py),
    which logs to a separate JSONL sidecar that must be cross-referenced by
    episode id/task afterward. Only meaningful for a checkpoint trained with
    --enable-direction-conditioning; only reaches this script's own client
    loop (LIBERO/RoboCasa/RoboCasa-GR1) - RoboLab is an external client this
    repo doesn't control, so it needs Mechanism 1 instead."""

    round_trip_check_interval: int = 8
    """Run Mechanism 2's diagnostic every this many macro-steps. Keep equal to
    n_action_steps above, matching the same convention Mechanism 1 requires
    and for the same reason: this loop has no way to notice a mismatch on its
    own. Only used when enable_round_trip_check is set."""

    round_trip_depths: tuple[int, ...] = (1, 2, 4)
    """Rollout depths (in macro-steps) Mechanism 2 evaluates. Only used when
    enable_round_trip_check is set."""


if __name__ == "__main__":
    args = tyro.cli(RolloutConfig)

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy-client-host and --policy-client-port, and set --model-path ""\n'
        '  - To use model path: set --model-path, and set --policy-client-host "" (and leave --policy-client-port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        trt_engine_path=args.trt_engine_path,
        trt_mode=args.trt_mode,
        seed=args.seed,
        robocasa_split=args.robocasa_split,
        enable_round_trip_check=args.enable_round_trip_check,
        round_trip_check_interval=args.round_trip_check_interval,
        round_trip_depths=args.round_trip_depths,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
