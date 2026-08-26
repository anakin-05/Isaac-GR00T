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
Round-trip consistency (C_i) diagnostic - the round-trip-consistency signal
(arXiv:2608.00675) adapted to GR00T's action-chunk-space bidirectional dynamics.

Mechanism (see the integration plan for the full derivation):

  1. At rollout depth i (in units of action-chunk executions, i.e.
     i * action_horizon raw timesteps), read the recorded state at t and at
     t + i*action_horizon directly from the trajectory. Offline eval has the
     full trajectory in memory, so no forward composition is needed for this
     leg (this also sidesteps the action-vs-observation-stream ambiguity
     noted in the plan's caveats - we read real recorded values, not a
     composition of them).
  2. Query the model's BACKWARD direction, conditioned on the state (and,
     optionally, the real video frame) at t + i*action_horizon:
     policy.get_action(future_obs, options={"direction": 1, "drop_vision": ...}).
  3. Reuse Gr00tPolicy's existing decode/compose machinery (relative ->
     absolute via Gr00tN1d7Processor.decode_action, invoked internally by
     policy.get_action) to turn the predicted backward action chunk into
     absolute poses. By construction (see ActionChunk.to_reversed_chunking
     and state_action_processor._reverse_absolute_action), the LAST timestep
     of that composed chunk is the model's round-trip estimate of the state
     at t - no extra composition code is needed in this module.
  4. C_i = MSE(true_state_t, estimated_state_t).

Costs exactly one extra policy.get_action call per (start step, depth) pair -
"one extra rollout", matching the source paper's framing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy import BasePolicy
import numpy as np
import pandas as pd


def _build_observation(
    traj: pd.DataFrame,
    step_count: int,
    modality_configs_no_action: dict,
    full_modality_configs: dict,
    embodiment_tag: EmbodimentTag,
) -> dict:
    """Build a policy-ready observation at `step_count`. Mirrors the observation
    construction in open_loop_eval.evaluate_single_trajectory exactly (same
    action-removed config for extraction, full config for parsing)."""
    data_point = extract_step_data(traj, step_count, modality_configs_no_action, embodiment_tag)
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v  # (T, D)
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
    for language_key in full_modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text
    return parse_observation_gr00t(obs, full_modality_configs)


@dataclass
class RoundTripConsistencyResult:
    depth: int
    start_steps: list[int]
    c_i: np.ndarray  # (N,) aggregate (mean over action keys) MSE per start step
    per_key_c_i: dict[str, np.ndarray]  # action key -> (N,) MSE for that key


def compute_round_trip_consistency(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    depths: list[int] | None = None,
    steps: int = 200,
    use_predicted_forward: bool = False,
    use_real_future_vision: bool = False,
) -> dict[int, RoundTripConsistencyResult]:
    """
    Compute the round-trip consistency signal C_i at each requested depth.

    Args:
        policy: A policy whose underlying model has enable_direction_conditioning
            (and Gr00tPolicy._get_action forwarding `options` - see the
            integration plan's gr00t_policy.py fix).
        loader: Episode loader with the full trajectory (state/action/video/
            language) available in memory.
        traj_id: Which trajectory to evaluate.
        embodiment_tag: Embodiment tag for this trajectory.
        depths: Rollout depths to evaluate, in units of action-chunk
            executions (i.e. depth `i` means i * action_horizon raw
            timesteps). Defaults to [1, 2, 4].
        steps: Maximum number of raw timesteps to consider (capped by
            trajectory length), same convention as open_loop_eval.
        use_predicted_forward: If True, anchor the backward query on the
            model's own chained forward predictions rather than the recorded
            trajectory. NOT YET IMPLEMENTED - raises NotImplementedError; a
            documented follow-up. The default (False) still exercises a real,
            measurement-free self-consistency check of the model's inverse-
            dynamics understanding, anchored on the recorded trajectory.
        use_real_future_vision: If True, condition the backward query on the
            real video frame at the future step. Defaults to False to match
            the model's actual training distribution: backward examples are
            trained with vision dropped by default
            (vision_dropout_prob_backward=1.0 - see the plan's caveats), so
            True exercises an out-of-training-distribution conditioning path.

    Returns:
        Dict mapping depth -> RoundTripConsistencyResult. A depth with no
        valid (start_step, future_step) window inside the trajectory is
        omitted (with a warning), not present with empty arrays.
    """
    if depths is None:
        depths = [1, 2, 4]
    if use_predicted_forward:
        raise NotImplementedError(
            "use_predicted_forward=True (chaining the model's own forward "
            "predictions rather than the recorded trajectory) is not yet "
            "implemented - a documented follow-up. Use the default (False)."
        )

    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)

    action_keys = loader.modality_configs["action"].modality_keys
    action_configs = loader.modality_configs["action"].action_configs
    if action_configs is not None:
        key_to_state_key = {
            key: (ac.state_key if ac.state_key else key)
            for key, ac in zip(action_keys, action_configs)
        }
    else:
        key_to_state_key = {key: key for key in action_keys}

    action_horizon = int(loader.modality_configs["action"].delta_indices[-1]) + 1

    modality_configs_no_action = deepcopy(loader.modality_configs)
    modality_configs_no_action.pop("action")

    results: dict[int, RoundTripConsistencyResult] = {}

    for depth in depths:
        stride = depth * action_horizon
        if actual_steps - stride <= 0:
            logging.warning(
                "Trajectory %d too short for round-trip depth %d (needs > %d "
                "steps, has %d) - skipping.",
                traj_id,
                depth,
                stride,
                actual_steps,
            )
            continue

        per_key_errs: dict[str, list[float]] = {key: [] for key in action_keys}
        agg_errs: list[float] = []
        valid_start_steps: list[int] = []

        for t in range(0, actual_steps - stride, action_horizon):
            future_step = t + stride
            if future_step >= traj_length:
                continue

            future_obs = _build_observation(
                traj,
                future_step,
                modality_configs_no_action,
                loader.modality_configs,
                embodiment_tag,
            )
            backward_action, _ = policy.get_action(
                future_obs,
                options={"direction": 1, "drop_vision": not use_real_future_vision},
            )

            errs = []
            for key in action_keys:
                if key not in backward_action:
                    continue
                state_key = key_to_state_key[key]
                col = f"state.{state_key}"
                if col not in traj.columns:
                    continue
                true_val = np.asarray(traj[col].iloc[t]).astype(np.float64).reshape(-1)
                # Last timestep of the decoded backward chunk = round-trip
                # estimate of the state at t (see ActionChunk.to_reversed_chunking
                # / _reverse_absolute_action docstrings for why the last slot is
                # the right one). backward_action[key] may be (H, D) - a bare
                # chunk, the convention this module's own unit tests' stub
                # policies use - or (B, H, D) - the real Gr00tPolicy.get_action
                # contract (see Gr00tPolicy._get_action / check_action), which
                # always has B=1 here since `future_obs` is a single-window
                # observation. Collapsing every leading dim before indexing the
                # last row handles both without guessing which one applies.
                raw_est = np.asarray(backward_action[key])
                est_val = raw_est.reshape(-1, raw_est.shape[-1])[-1].astype(np.float64)
                if est_val.shape != true_val.shape:
                    # Action/state dims can legitimately differ (e.g. a rot6d
                    # action vs. a differently-parameterized state column) -
                    # skip this key's comparison rather than guess a broadcast.
                    continue
                err = float(np.mean((true_val - est_val) ** 2))
                per_key_errs[key].append(err)
                errs.append(err)

            if errs:
                agg_errs.append(float(np.mean(errs)))
                valid_start_steps.append(t)

        if not agg_errs:
            logging.warning("No valid round-trip windows for traj %d at depth %d.", traj_id, depth)
            continue

        results[depth] = RoundTripConsistencyResult(
            depth=depth,
            start_steps=valid_start_steps,
            c_i=np.array(agg_errs),
            per_key_c_i={k: np.array(v) for k, v in per_key_errs.items() if v},
        )

    return results


def plot_round_trip_consistency(
    results: dict[int, RoundTripConsistencyResult],
    traj_id: int,
    save_path: str,
) -> None:
    """Plot C_i (mean +/- std over start steps) vs. rollout depth, analogous to
    the source repo's Fig. 2/4 (C_i vs. depth)."""
    from matplotlib import pyplot as plt

    if not results:
        logging.warning("No round-trip consistency results to plot for traj %d.", traj_id)
        return

    depths = sorted(results.keys())
    means = [results[d].c_i.mean() for d in depths]
    stds = [results[d].c_i.std() for d in depths]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(depths, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Rollout depth i (action-chunk executions)")
    ax.set_ylabel(r"Round-trip consistency $\mathcal{C}_i$ (state MSE)")
    ax.set_title(f"Trajectory {traj_id} — round-trip consistency vs. depth")
    ax.set_xticks(depths)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
