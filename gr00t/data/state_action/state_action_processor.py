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
Unified processor for robot state and action data.

Handles:
- State normalization (min/max, mean/std, sin/cos encoding)
- Action normalization
- Absolute <-> Relative action representation conversion
- Action processing with state dependency
"""

from copy import deepcopy
import logging

from gr00t.configs.data.embodiment_configs import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.utils import (
    apply_sin_cos_encoding,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
import numpy as np


logger = logging.getLogger(__name__)

# (embodiment_tag, key) pairs already warned about via `_warn_once_dim_mismatch`, so the
# per-datapoint `apply_reversed` hot path doesn't spam identical warnings every example.
_WARNED_DIM_MISMATCH_KEYS: set[tuple[str, str]] = set()


def _warn_once_dim_mismatch(
    embodiment_tag: str, key: str, state_key: str, state_dim: int, action_dim: int
) -> None:
    cache_key = (embodiment_tag, key)
    if cache_key in _WARNED_DIM_MISMATCH_KEYS:
        return
    _WARNED_DIM_MISMATCH_KEYS.add(cache_key)
    logger.warning(
        f"apply_reversed: skipping ABSOLUTE-rep reversal for action key '{key}' in "
        f"embodiment '{embodiment_tag}' - its reference state key '{state_key}' has dim "
        f"{state_dim} but the action has dim {action_dim}. These represent different "
        f"physical quantities that happen to share a name (e.g. a multi-joint gripper "
        f"state vs. a single-channel gripper command), so no dimension-matched anchor "
        f"exists for a temporal reversal. This key's action/state are left at their "
        f"forward-direction values in the backward example instead of crashing."
    )


def _reverse_absolute_action(
    current_value: np.ndarray, forward_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Temporally reverse an ABSOLUTE-representation action chunk (e.g. gripper).

    ABSOLUTE-rep values have no SE(3)/vector "inverse" - the backward target is
    simply the same recorded values walked in reverse chronological order. This
    has no physical "undo" meaning (you cannot un-grasp an object by reversing a
    gripper command); it is a regression-consistency construction, not a claim
    about dynamics reversibility.

    Args:
        current_value: (D,) value at t (the state this chunk starts from).
        forward_array: (H, D) recorded values at t+1..t+H.

    Returns:
        Tuple of (backward_array, new_reference_value) where backward_array is
        (H, D) = values at t+H-1..t (chronologically reversed, dropping the
        endpoint which becomes the new reference), and new_reference_value is
        (D,) = the value at t+H (this chunk's own endpoint).
    """
    full = np.concatenate([current_value[None, :], forward_array], axis=0)  # (H+1, D)
    reversed_full = full[::-1]
    return reversed_full[1:], reversed_full[0]


class StateActionProcessor:
    """
    Unified processor for robot state and action data.

    Handles:
    - State normalization (min/max, mean/std, sin/cos encoding)
    - Action normalization
    - Absolute <-> Relative action representation conversion
    - Action processing with state dependency
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        """
        Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
                where modality in ["state", "action"]
                Example: {"gr1": {"state": ModalityConfig(...), "action": ModalityConfig(...)}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
                where modality in ["state", "action", "relative_action"]
                and stat_type in ["min", "max", "mean", "std", "q01", "q99"]
                Example: {"gr1": {"state": {"left_arm": {"min": [...], "max": [...], ...}}}}
            use_percentiles: Whether to use percentiles (q01/q99) instead of min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Global flag to enable sin/cos encoding for states
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        # Format: norm_params[embodiment_tag][modality][joint_group][stat_type]
        # where stat_type in ["min", "max", "mean", "std", "dim"]

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """
        Set dataset statistics for normalization.

        Args:
            statistics: Nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
        """
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                # Surfaced as a warning (not print) because callers running with
                # override_pretraining_statistics=False on a mixture dataset will
                # otherwise silently keep the pre-existing pretraining stats and
                # discard newly-merged per-dataset stats — training proceeds with
                # the wrong mean/std and only an easy-to-miss stdout line records
                # the drop.
                logger.warning(
                    "Statistics for embodiment %r already present; new stats "
                    "DISCARDED (override=False). If the new data differs from "
                    "the existing distribution this will cause silent "
                    "normalization mismatch — pass override=True (or "
                    "override_pretraining_statistics=True at the dataset level) "
                    "to use the merged stats instead.",
                    key,
                )
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics for all embodiments and modalities."""
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    # Compute range, ensuring it's not zero
                    range_vals = max_vals - min_vals
                    range_vals = np.maximum(range_vals, 1e-8)

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[0]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            # Override absolute action stats with relative stats where specified
            if "action" in self.modality_configs[embodiment_tag]:
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs):
                        if (
                            action_config.rep == ActionRepresentation.RELATIVE
                            and self.use_relative_action
                        ):
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                                self.statistics[embodiment_tag]["relative_action"][key]
                            )
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1")

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)
        """
        normalized_values = {}
        state = deepcopy(state)  # Avoid modifying input

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding (doubles dimension)
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])

            # Strategy 2: Mean/std normalization
            elif (
                hasattr(
                    self.modality_configs[embodiment_tag]["state"],
                    "mean_std_embedding_keys",
                )
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Min/max normalization to [-1, 1]
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Reverse state processing (denormalization).

        Args:
            state: Dict mapping joint_group -> processed state values
            embodiment_tag: Embodiment identifier

        Returns:
            Dict mapping joint_group -> raw state values

        Raises:
            ValueError: If attempting to reverse sin/cos encoding (not reversible)
        """
        unnormalized_values = {}

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Sin/cos encoding is not reversible
            if sin_cos_keys and joint_group in sin_cos_keys:
                raise ValueError(
                    f"Cannot unapply sin/cos encoding for joint group '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. This transformation is not reversible."
                )

            # Reverse mean/std normalization
            elif (
                hasattr(
                    self.modality_configs[embodiment_tag]["state"],
                    "mean_std_embedding_keys",
                )
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized = unnormalize_values_meanstd(state[joint_group], params)
                unnormalized_values[joint_group] = unnormalized

            # Reverse min/max normalization
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized_values[joint_group] = unnormalize_values_minmax(
                    state[joint_group], params
                )

        return unnormalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Apply action processing (absolute->relative conversion, normalization).

        Processing order:
        1. Convert absolute actions to relative (if configured)
        2. Normalize actions

        Args:
            action: Dict mapping joint_group -> raw action values
                Shape per group: (T, D) where T is action horizon, D is action dimension
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) where last timestep is used as reference

        Returns:
            Dict mapping joint_group -> processed action values
                Shape per group: (T, D)

        Raises:
            ValueError: If state is None but required for relative action conversion
        """
        action = deepcopy(action)  # Avoid modifying input

        # Step 1: Convert absolute actions to relative (if needed)
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative action processing of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    # Use last state as reference frame
                    reference_state = state[state_key][-1]

                    # Convert absolute to relative
                    action[key] = self._convert_to_relative_action(
                        action=action[key],
                        reference_state=reference_state,
                        action_type=action_config.type,
                        action_format=action_config.format,
                    )

        # Step 2: Normalize actions
        return self._normalize_action_dict(action, embodiment_tag)

    def _normalize_action_dict(
        self, action: dict[str, np.ndarray], embodiment_tag: str
    ) -> dict[str, np.ndarray]:
        """
        Normalize an already-representation-resolved action dict (i.e. the
        conversion to relative/absolute representation, if any, has already
        happened). Shared by `apply_action` (Step 2) and `apply_reversed`.
        """
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        normalized_values = {}
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                normalized = normalize_values_meanstd(action[joint_group], params)
            else:
                normalized = normalize_values_minmax(action[joint_group], params)

            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Reverse action processing (denormalization, relative->absolute conversion).

        Processing order:
        1. Denormalize actions
        2. Convert relative actions to absolute (if configured)

        Args:
            action: Dict mapping joint_group -> processed action values
                Shape per group: (T, D) or (B, T, D) for batched
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) or (B, T_state, D) for batched

        Returns:
            Dict mapping joint_group -> raw absolute action values
                Shape per group: (T, D) or (B, T, D) for batched

        Raises:
            ValueError: If state is None but required for relative->absolute conversion
        """
        # Step 1: Unnormalize actions
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else:
                unnormalized = unnormalize_values_minmax(group_values, params)

            unnormalized_values[joint_group] = unnormalized

        # Step 2: Convert relative actions to absolute (if needed)
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    relative_action = unnormalized_values[key]

                    # Handle batched and unbatched cases
                    is_batched = relative_action.ndim == 3
                    if not is_batched:
                        assert relative_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]
                        relative_action = relative_action[None, :]
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]

                    # Convert batched relative actions to absolute
                    absolute_actions = []
                    for s, a in zip(reference_state, relative_action):
                        # Use last timestep of state as reference
                        absolute_action = self._convert_to_absolute_action(
                            action=a,
                            reference_state=s[-1],
                            action_type=action_config.type,
                            action_format=action_config.format,
                        )
                        absolute_actions.append(absolute_action)

                    if is_batched:
                        unnormalized_values[key] = np.stack(absolute_actions, axis=0)
                    else:
                        unnormalized_values[key] = absolute_actions[0]

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Apply both state and action processing together.

        Convenience method that processes state and action in one call,
        automatically passing raw state to action processor for relative conversion.

        Args:
            state: Dict mapping joint_group -> raw state values
            action: Dict mapping joint_group -> raw action values
            embodiment_tag: Embodiment identifier

        Returns:
            Tuple of (processed_state, processed_action)
        """
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def apply_reversed(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Construct a BACKWARD (round-trip consistency, direction=1) training example
        from the same (state, action) pair already loaded for the forward direction.

        Rather than sampling new data, this derives a "return trip": the
        conditioning state becomes this chunk's own composed endpoint
        (state_{t+H}), and the target action becomes the chronologically
        reversed ("undo") chunk that composes back onto the original state_t.
        This requires no new I/O - only the same (state, action) already
        extracted for the forward example.

        Args:
            state: Dict mapping joint_group -> raw state values, same contract as `apply`.
            action: Dict mapping joint_group -> raw action values, same contract as `apply`.
            embodiment_tag: Embodiment identifier.

        Returns:
            Tuple of (processed_state, processed_action) for the backward example, in
            the same normalized representation `apply()` returns for the forward direction.

        Raises:
            ValueError: If action_configs is not set for this embodiment (needed to
                determine each action key's representation/type for inversion), or a
                referenced state key is missing.
            NotImplementedError: If any action key uses ActionRepresentation.DELTA
                (chained per-step deltas), which this method does not support.
        """
        action = deepcopy(action)
        # Default: carry forward the original state unchanged for any joint_group with
        # no corresponding action key to derive an updated value from (a low-risk
        # approximation - see plan caveats).
        raw_new_state = deepcopy(state)

        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is None:
            raise ValueError(
                f"apply_reversed requires action_configs to be set for embodiment "
                f"'{embodiment_tag}' (needed to determine each action key's "
                f"representation/type for inversion)."
            )

        for key, action_config in zip(modality_keys, action_configs):
            state_key = action_config.state_key if action_config.state_key else key
            if state_key not in state:
                raise KeyError(
                    f"Reference state key '{state_key}' not found in state dict "
                    f"for embodiment '{embodiment_tag}'"
                )
            reference_state = state[state_key][-1]

            if action_config.rep == ActionRepresentation.DELTA:
                raise NotImplementedError(
                    f"apply_reversed does not support ActionRepresentation.DELTA "
                    f"(chained per-step deltas) for action key '{key}' in embodiment "
                    f"'{embodiment_tag}'. Only RELATIVE and ABSOLUTE are supported."
                )
            elif action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                reversed_action, new_reference = self._reverse_relative_action(
                    action=action[key],
                    reference_state=reference_state,
                    action_type=action_config.type,
                    action_format=action_config.format,
                )
            else:
                # ABSOLUTE rep, or RELATIVE rep with use_relative_action disabled -
                # mirrors apply_action's Step 1 gate, which likewise skips conversion
                # (and normalizes the raw values directly) in that case.
                if reference_state.shape[-1] != action[key].shape[-1]:
                    # E.g. LIBERO_PANDA's "gripper": a 2-dim multi-finger joint state and
                    # a 1-dim actuation command that happen to share a name but are
                    # different physical quantities - there's no dimension-matched anchor
                    # to build a temporal reversal from. Leave this key's forward-direction
                    # action/state untouched rather than crash (see plan Caveat: gripper
                    # reversal is already a regression-consistency approximation, not a
                    # physical-undo claim, so skipping it here is a further instance of
                    # the same documented limitation, not a new one).
                    _warn_once_dim_mismatch(
                        embodiment_tag,
                        key,
                        state_key,
                        reference_state.shape[-1],
                        action[key].shape[-1],
                    )
                    continue
                reversed_action, new_reference = _reverse_absolute_action(
                    current_value=reference_state, forward_array=action[key]
                )

            action[key] = reversed_action
            raw_new_state[state_key] = np.tile(
                new_reference[None, :], (state[state_key].shape[0], 1)
            )

        processed_state = self.apply_state(raw_new_state, embodiment_tag)
        processed_action = self._normalize_action_dict(action, embodiment_tag)
        return processed_state, processed_action

    def unapply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Reverse both state and action processing together.

        Args:
            state: Dict mapping joint_group -> processed state values
            action: Dict mapping joint_group -> processed action values
            embodiment_tag: Embodiment identifier
            raw_state: Optional dict of raw states for relative->absolute conversion
                If None, will use unapplied state (but won't work for sin/cos encoded states)

        Returns:
            Tuple of (raw_state, raw_action)
        """
        # Unapply state first
        try:
            unapplied_state = self.unapply_state(state, embodiment_tag)
        except ValueError as e:
            if "sin/cos encoding" in str(e) and raw_state is None:
                raise ValueError(
                    "Cannot unapply sin/cos encoded state. Please provide raw_state parameter."
                ) from e
            raise

        # Use provided raw_state if available, otherwise use unapplied state
        state_for_action = raw_state if raw_state is not None else unapplied_state

        # Unapply action
        unapplied_action = self.unapply_action(action, embodiment_tag, state=state_for_action)

        return unapplied_state, unapplied_action

    def get_state_dim(self, embodiment_tag: str, include_sincos_expansion: bool = False) -> int:
        """
        Get total state dimension after processing.

        Args:
            embodiment_tag: Embodiment identifier
            include_sincos_expansion: If True, accounts for sin/cos encoding doubling dimensions

        Returns:
            Total state dimension across all joint groups
        """
        total_dim = 0
        state_config = self.modality_configs[embodiment_tag]["state"]

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = set()
        if self.apply_sincos_state_encoding and hasattr(state_config, "sin_cos_embedding_keys"):
            sin_cos_keys = set(state_config.sin_cos_embedding_keys)

        for joint_group in state_config.modality_keys:
            base_dim = self.norm_params[embodiment_tag]["state"][joint_group]["dim"].item()

            # Sin/cos encoding doubles the dimension
            if include_sincos_expansion and joint_group in sin_cos_keys:
                total_dim += base_dim * 2
            else:
                total_dim += base_dim

        return total_dim

    def get_action_dim(self, embodiment_tag: str) -> int:
        """
        Get total action dimension.

        Args:
            embodiment_tag: Embodiment identifier

        Returns:
            Total action dimension across all joint groups
        """
        total_dim = 0
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            total_dim += self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
        return total_dim

    def _convert_to_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert absolute action to relative action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        if action_type == ActionType.EEF:
            action_chunking = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            action_chunking = JointActionChunk([JointPose(m) for m in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        relative_action_chunking = action_chunking.relative_chunking(
            reference_frame=reference_frame
        )
        return relative_action_chunking.to(action_format)

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert relative action to absolute action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert reference_state.shape[0] == action.shape[1], (
            f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"
        )

        if action_type == ActionType.EEF:
            rel_action = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            rel_action = JointActionChunk([JointPose(pose) for pose in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        abs_action = rel_action.to_absolute_chunking(reference_frame=reference_frame)
        return abs_action.to(action_format)

    def _reverse_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given an absolute action chunk and the reference state it starts from,
        derive the chronologically-reversed ("undo") relative action chunk plus
        the new reference state it composes from.

        Args:
            action: Absolute action chunk, shape (T, D) (same contract as
                `_convert_to_relative_action`'s `action` argument).
            reference_state: Raw reference state, shape (D,) (the state this
                chunk starts from, i.e. state_t).
            action_type: EEF or NON_EEF - determines Pose/ActionChunk subclass.
            action_format: The array format of both `action` and `reference_state`.

        Returns:
            Tuple of (reversed_relative_action, new_reference_state):
                - reversed_relative_action: (T, D) relative chunk, relative to
                  new_reference_state, that composes back onto new_reference_state
                  to recover reference_state at its last step.
                - new_reference_state: (D,) this chunk's own composed absolute
                  endpoint (state_{t+H}), same raw format as reference_state.
        """
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        if action_type == ActionType.EEF:
            chunk_cls = EndEffectorActionChunk
            absolute_chunking = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            chunk_cls = JointActionChunk
            absolute_chunking = JointActionChunk([JointPose(m) for m in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        relative_chunking = absolute_chunking.relative_chunking(reference_frame=reference_frame)
        reversed_chunking, new_reference_frame = relative_chunking.to_reversed_chunking(
            reference_frame
        )

        new_reference_state = chunk_cls([new_reference_frame]).to(action_format)[0]
        return reversed_chunking.to(action_format), new_reference_state

    def __str__(self) -> str:
        return f"StateActionProcessor(modality_configs={self.modality_configs}, statistics={self.statistics}, use_percentiles={self.use_percentiles}, clip_outliers={self.clip_outliers}, apply_sincos_state_encoding={self.apply_sincos_state_encoding}, use_relative_action={self.use_relative_action})"
