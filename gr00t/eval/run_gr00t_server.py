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

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    # Round-trip consistency monitor configs (see gr00t/policy/round_trip_monitor.py)
    enable_round_trip_monitor: bool = False
    """If True, wrap the policy in RoundTripMonitor - a server-side round-trip-consistency
    diagnostic that periodically queries the model's backward direction and logs C_i to a
    JSONL sidecar. Reaches any closed-loop client transparently (no client-side changes
    needed). Only meaningful for a checkpoint trained with --enable-direction-conditioning."""

    round_trip_check_interval: int = 8
    """Run the diagnostic every this many get_action calls (matches the --n-action-steps /
    --open-loop-horizon convention used by LIBERO/RoboCasa/RoboCasa-GR1/RoboLab examples).
    Only used when --enable-round-trip-monitor is set."""

    round_trip_depths: tuple[int, ...] = (1, 2, 4)
    """Rollout depths (in units of get_action calls) to evaluate. Only used when
    --enable-round-trip-monitor is set."""

    round_trip_log_path: str | None = None
    """JSONL sidecar log path. Defaults to /tmp/round_trip_consistency/<timestamp>.jsonl.
    Only used when --enable-round-trip-monitor is set."""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    # RoundTripMonitor wraps whatever is outermost so far, transparent to every
    # closed-loop client's wire protocol (see gr00t/policy/round_trip_monitor.py).
    if config.enable_round_trip_monitor:
        from gr00t.policy.round_trip_monitor import RoundTripMonitor

        policy = RoundTripMonitor(
            policy,
            round_trip_check_interval=config.round_trip_check_interval,
            round_trip_depths=list(config.round_trip_depths),
            round_trip_log_path=config.round_trip_log_path,
        )
        print(
            f"  Round-trip monitor: enabled (interval={config.round_trip_check_interval}, "
            f"depths={list(config.round_trip_depths)}, log={policy.round_trip_log_path})"
        )

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
