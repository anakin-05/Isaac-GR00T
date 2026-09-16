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
Server-side round-trip-consistency diagnostic - Mechanism 1 of the closed-loop
evaluation extension (see the integration plan's "Extension: Closed-Loop Evaluation"
section, and gr00t/eval/round_trip_consistency.py for the offline/open-loop equivalent
this mirrors).

``RoundTripMonitor`` wraps any ``BasePolicy`` transparently. Every
``round_trip_check_interval`` ``get_action`` calls, for each env-batch slot, it issues one
additional internal query of the wrapped policy's backward direction
(``options={"direction": 1, "drop_vision": True}``) anchored on the *current* observation,
decodes the reconstructed past state, and diffs it against a cached true state from
``depth`` calls ago for each requested depth. Results are appended to a JSONL sidecar log -
independent of any one benchmark's own result format, the same pattern RoboLab itself uses
for ``episode_results.jsonl``.

This reaches every closed-loop client (LIBERO, RoboCasa, RoboCasa-GR1-Tabletop, RoboLab)
with zero client-side changes, because ``options`` already forwards transparently end-to-end
through ``PolicyServer.get_action`` -> ``Gr00tPolicy`` / ``Gr00tSimPolicyWrapper`` (verified
directly in gr00t_policy.py - no server/RPC changes needed). Enable it by wrapping the
constructed policy before handing it to ``PolicyServer`` (see
``gr00t/eval/run_gr00t_server.py``'s ``--enable-round-trip-monitor`` flag).

Per-episode and per-task identity (both optional, both additive)
------------------------------------------------------------------
By default this monitor identifies an env purely by its *positional* index in the batched
observation (``range(batch_size)``). That's stable for one ``gr00t/eval/rollout_policy.py``
run against a fixed ``n_envs`` (confirmed by reading ``_collect_rollout_episodes`` - the
per-env-index loop never resizes), but it can't tell when gymnasium's inline autoreset
silently starts a *new episode* in the same slot mid-rollout, nor which task a given row
belongs to when many tasks run sequentially against one long-lived server (as in a
multi-task benchmark). A caller that wants either can pass, alongside the usual
``direction``/``drop_vision`` keys:

- ``options["round_trip_episode_ids"]``: a list of length ``batch_size``, one opaque
  per-slot identifier that changes exactly when that slot starts a new episode (e.g.
  ``f"{env_idx}:{episode_counter}"``). When a slot's incoming id differs from the one
  cached from its previous call, this monitor clears that slot's history immediately -
  a round-trip window can then never span an undetected episode boundary, closing the
  gap the per-slot-``reset()``-granularity caveat below describes. ``rollout_policy.py``
  supplies this (see its own ``round_trip_episode_ids`` bookkeeping).
- ``options["round_trip_task"]``: a constant string for the whole rollout (e.g. the
  ``env_name``), logged verbatim into every JSONL row's ``"task"`` field, so rows from a
  multi-task benchmark against one server can be grouped without guessing from timing.

Both are extracted and stripped before ``options`` is forwarded to the wrapped policy (the
model layer has no use for them), and both default to ``None`` when absent - a client that
never sets them (RoboLab, or any caller written before this existed) sees no behavior
change: same positional-slot tracking, ``"episode_id"``/``"task"`` simply log as ``null``.

Caveats (see the plan for the full discussion):
- **Per-slot ``reset()`` granularity is unconfirmed and, in the one client actually checked
  (``gr00t/eval/rollout_policy.py``), not exercised at all mid-rollout**: that script calls
  ``policy.reset()`` exactly once, with no arguments, before the rollout loop begins - it
  does not detect or forward individual sub-env auto-resets within a vectorized batch. This
  monitor therefore clears every slot's history on *any* ``reset()`` call rather than guess
  which slot(s) it refers to. Without ``round_trip_episode_ids`` (above), a round-trip
  window that happens to span an undetected episode boundary will compare states from two
  different episodes; depth `i` windows are short relative to typical episode lengths, so
  this is a rare, not systemic, contamination even then.
- **One extra ``get_action`` call per env slot every ``round_trip_check_interval`` calls is a
  real, non-optional compute cost** (e.g. +12.5% at interval=8) - budget for it.
- **``drop_vision=True`` is intentional, not a shortcut**: the model was only ever trained
  with backward examples vision-dropped by default (``vision_dropout_prob_backward=1.0``),
  so querying with real vision here would be out-of-training-distribution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .policy import BasePolicy, PolicyWrapper


logger = logging.getLogger(__name__)


def _slice_leaf(value: Any, slot: int) -> Any:
    """Slice one env-batch-slot out of a single observation/action leaf value,
    keeping a length-1 batch dimension."""
    if isinstance(value, np.ndarray):
        return value[slot : slot + 1]
    if isinstance(value, (list, tuple)):
        return type(value)([value[slot]])
    return value


def _slice_observation(observation: dict[str, Any], slot: int) -> dict[str, Any]:
    """Build a single-env observation (batch size 1) from a batched one, preserving
    whichever key convention (nested dict-of-dicts or flat dotted keys) was used."""
    sliced: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            sliced[key] = {k: _slice_leaf(v, slot) for k, v in value.items()}
        else:
            sliced[key] = _slice_leaf(value, slot)
    return sliced


def _modality_dict(observation: dict[str, Any], modality: str) -> dict[str, Any]:
    """{key: value} for a top-level modality, supporting both Gr00tPolicy's nested
    format (``observation[modality][key]``) and Gr00tSimPolicyWrapper's flat format
    (``observation[f"{modality}.{key}"]``) - this monitor works with either, since it
    wraps whichever policy layer is outermost before ``PolicyServer``."""
    nested = observation.get(modality)
    if isinstance(nested, dict):
        return nested
    prefix = f"{modality}."
    return {k[len(prefix) :]: v for k, v in observation.items() if k.startswith(prefix)}


def _batch_size(observation: dict[str, Any]) -> int:
    for value in _modality_dict(observation, "state").values():
        if isinstance(value, np.ndarray):
            return value.shape[0]
    raise ValueError(
        "RoundTripMonitor could not determine batch size: no array-valued 'state' key "
        "found in the observation."
    )


def _last_timestep(value: np.ndarray) -> np.ndarray:
    """(..., T, D) -> (..., D): the most recent timestep."""
    return np.asarray(value)[..., -1, :]


def _reconstructed_last_step(value: Any) -> np.ndarray:
    """Collapse every leading dim of a decoded backward action chunk - (H, D) or the
    real Gr00tPolicy (B, H, D) contract, B=1 here - and return its last row, which by
    construction (see ActionChunk.to_reversed_chunking / state_action_processor.
    _reverse_absolute_action) is the round-trip estimate of the state the chunk
    started from. Mirrors round_trip_consistency.py's own fix for this exact shape
    ambiguity between this module's convention and the real policy's."""
    arr = np.asarray(value)
    return arr.reshape(-1, arr.shape[-1])[-1]


def reconstruct_last_timesteps_batched(value: Any) -> np.ndarray:
    """(B, H, D) -> (B, D): the last-timestep round-trip estimate for every batch
    row at once. Companion to ``_reconstructed_last_step`` for Mechanism 2
    (``gr00t/eval/rollout_policy.py``'s client-side hook), which issues one
    backward query for the *whole* vectorized batch instead of one per env slot
    (this monitor's approach), so it needs each row's answer kept apart rather
    than collapsed to a single row. Always the real Gr00tPolicy ``(B, H, D)``
    contract here - Mechanism 2 only ever calls through a real policy or
    ``PolicyClient`` with batch size ``n_envs``, never the ``(H, D)`` mocked-stub
    shape ``_reconstructed_last_step`` also has to tolerate for historical
    reasons (see its own docstring)."""
    arr = np.asarray(value)
    return arr[:, -1, :]


def action_key_to_state_key_map(policy: BasePolicy) -> tuple[list[str], dict[str, str]]:
    """``(action_keys, {action_key: state_key})`` from a policy's modality config.

    Shared by ``RoundTripMonitor`` (Mechanism 1) and Mechanism 2
    (``gr00t/eval/rollout_policy.py``) so both map an action key back to the
    state key it round-trips against identically - a checkpoint's
    ``ActionConfig.state_key`` override (when an action key doesn't share its
    name with the state key it corresponds to) must be honored the same way by
    either mechanism, or their C_i numbers would not be comparable.
    """
    action_modality = policy.get_modality_config()["action"]
    action_keys = action_modality.modality_keys
    action_configs = action_modality.action_configs
    if action_configs is not None:
        return action_keys, {
            k: (ac.state_key if ac.state_key else k) for k, ac in zip(action_keys, action_configs)
        }
    return action_keys, {k: k for k in action_keys}


def _extract_round_trip_options(
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str] | None, str | None]:
    """Split an incoming options dict into (options to forward downstream,
    per-slot episode ids, task label) - see module docstring's "Per-episode
    and per-task identity" section. Both extras are optional and additive:
    a client that never sets them (e.g. RoboLab) sees no behavior change."""
    if options is None:
        return None, None, None
    episode_ids = options.get("round_trip_episode_ids")
    task = options.get("round_trip_task")
    if episode_ids is None and task is None:
        return options, None, None
    forwarded = {
        k: v for k, v in options.items() if k not in ("round_trip_episode_ids", "round_trip_task")
    }
    return (forwarded or None), episode_ids, task


@dataclass
class _SlotState:
    call_index: int = 0
    episode_id: str | None = None
    history: deque[tuple[int, dict[str, np.ndarray]]] = field(default_factory=deque)


class RoundTripMonitor(PolicyWrapper):
    """Server-side round-trip-consistency diagnostic - see module docstring."""

    def __init__(
        self,
        policy: BasePolicy,
        *,
        round_trip_check_interval: int = 8,
        round_trip_depths: list[int] | None = None,
        round_trip_log_path: str | None = None,
        strict: bool = True,
    ):
        super().__init__(policy, strict=strict)
        if round_trip_check_interval <= 0:
            raise ValueError(
                f"round_trip_check_interval must be positive; got {round_trip_check_interval}"
            )
        self.round_trip_check_interval = round_trip_check_interval
        self.round_trip_depths = sorted(round_trip_depths or [1, 2, 4])
        if round_trip_log_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            round_trip_log_path = f"/tmp/round_trip_consistency/{timestamp}.jsonl"
        self.round_trip_log_path = Path(round_trip_log_path)
        self.round_trip_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_depth = max(self.round_trip_depths)
        self._slots: dict[int, _SlotState] = {}

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        # Per-slot granularity is unconfirmed (see module docstring) - clear every
        # slot's history on any reset() call rather than guess which slot(s) it means.
        self._slots.clear()
        return self.policy.reset(options)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        forwarded_options, episode_ids, task = _extract_round_trip_options(options)
        action, info = self.policy.get_action(observation, forwarded_options)

        try:
            self._maybe_check_round_trip(observation, episode_ids, task)
        except Exception:
            # The diagnostic must never take down real closed-loop control.
            logger.exception("RoundTripMonitor: round-trip check failed; continuing without it.")

        return action, info

    def _maybe_check_round_trip(
        self, observation: dict[str, Any], episode_ids: list[str] | None, task: str | None
    ) -> None:
        state = _modality_dict(observation, "state")
        batch_size = _batch_size(observation)
        action_keys, key_to_state_key = action_key_to_state_key_map(self.policy)

        for slot in range(batch_size):
            slot_state = self._slots.setdefault(slot, _SlotState())
            new_episode_id = episode_ids[slot] if episode_ids is not None else None
            if (
                new_episode_id is not None
                and slot_state.episode_id is not None
                and new_episode_id != slot_state.episode_id
            ):
                # A caller-supplied episode id changed under this slot without an
                # explicit reset() call - e.g. gymnasium's inline autoreset starting
                # a new episode in the same env-batch position mid-rollout. Clear
                # this slot's history so no round-trip window spans the boundary;
                # see the module docstring's "Per-episode and per-task identity"
                # section for why positional slot index alone can't catch this.
                slot_state.history.clear()
            if new_episode_id is not None:
                slot_state.episode_id = new_episode_id

            call_index = slot_state.call_index
            true_state = {key: _last_timestep(np.asarray(arr[slot])) for key, arr in state.items()}
            slot_state.history.append((call_index, true_state))
            while len(slot_state.history) > self._max_depth + 1:
                slot_state.history.popleft()

            if (call_index + 1) % self.round_trip_check_interval == 0:
                history_by_index = dict(slot_state.history)
                has_target = any(
                    (call_index - depth) in history_by_index for depth in self.round_trip_depths
                )
                # Skip the (expensive) backward query entirely if no requested depth
                # has a cached target yet (e.g. early in a rollout) - nothing would
                # be logged from it anyway.
                if has_target:
                    self._check_slot(
                        slot,
                        call_index,
                        observation,
                        action_keys,
                        key_to_state_key,
                        history_by_index,
                        slot_state.episode_id,
                        task,
                    )

            slot_state.call_index = call_index + 1

    def _check_slot(
        self,
        slot: int,
        call_index: int,
        observation: dict[str, Any],
        action_keys: list[str],
        key_to_state_key: dict[str, str],
        history_by_index: dict[int, dict[str, np.ndarray]],
        episode_id: str | None,
        task: str | None,
    ) -> None:
        single_obs = _slice_observation(observation, slot)
        backward_action, _ = self.policy.get_action(
            single_obs, options={"direction": 1, "drop_vision": True}
        )
        backward_state = _modality_dict(backward_action, "action")

        rows = []
        for depth in self.round_trip_depths:
            target_index = call_index - depth
            cached = history_by_index.get(target_index)
            if cached is None:
                continue
            per_key: dict[str, float] = {}
            for key in action_keys:
                state_key = key_to_state_key[key]
                if state_key not in cached or key not in backward_state:
                    continue
                true_val = cached[state_key]
                est_val = _reconstructed_last_step(backward_state[key])
                if true_val.shape != est_val.shape:
                    # Same dimension-mismatch case state_action_processor.apply_reversed
                    # already skips (e.g. LIBERO's "gripper") - no comparable anchor.
                    continue
                per_key[key] = float(np.mean((true_val - est_val) ** 2))
            if not per_key:
                continue
            rows.append(
                {
                    "env_slot": slot,
                    "call_index": call_index,
                    "depth": depth,
                    "c_i": float(np.mean(list(per_key.values()))),
                    "episode_id": episode_id,
                    "task": task,
                    "per_key": per_key,
                }
            )

        if rows:
            with open(self.round_trip_log_path, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
