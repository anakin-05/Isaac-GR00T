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

Caveats (see the plan for the full discussion):
- **Per-slot ``reset()`` granularity is unconfirmed and, in the one client actually checked
  (``gr00t/eval/rollout_policy.py``), not exercised at all mid-rollout**: that script calls
  ``policy.reset()`` exactly once, with no arguments, before the rollout loop begins - it
  does not detect or forward individual sub-env auto-resets within a vectorized batch. This
  monitor therefore clears every slot's history on *any* ``reset()`` call rather than guess
  which slot(s) it refers to; a round-trip window that happens to span an undetected episode
  boundary will compare states from two different episodes. Depth `i` windows are short
  relative to typical episode lengths, so this is a rare, not systemic, contamination.
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


@dataclass
class _SlotState:
    call_index: int = 0
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
        action, info = self.policy.get_action(observation, options)

        try:
            self._maybe_check_round_trip(observation)
        except Exception:
            # The diagnostic must never take down real closed-loop control.
            logger.exception("RoundTripMonitor: round-trip check failed; continuing without it.")

        return action, info

    def _maybe_check_round_trip(self, observation: dict[str, Any]) -> None:
        state = _modality_dict(observation, "state")
        batch_size = _batch_size(observation)
        action_modality = self.policy.get_modality_config()["action"]
        action_keys = action_modality.modality_keys
        action_configs = action_modality.action_configs
        key_to_state_key = (
            {k: (ac.state_key if ac.state_key else k) for k, ac in zip(action_keys, action_configs)}
            if action_configs is not None
            else {k: k for k in action_keys}
        )

        for slot in range(batch_size):
            slot_state = self._slots.setdefault(slot, _SlotState())
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
                    "per_key": per_key,
                }
            )

        if rows:
            with open(self.round_trip_log_path, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
