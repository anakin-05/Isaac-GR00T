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
Record LIBERO rollout episodes to disk, so failed rollouts of a policy can be turned into a
failure dataset (see ``examples/LIBERO/build_failure_dataset.py``).

Enabled by environment variables (inherited by the vectorized env workers):

``GR00T_LIBERO_RECORD_DIR``
    Output directory. Unset = no recording.
``GR00T_LIBERO_RECORD_MODE``
    ``failure`` (default) records only failed episodes, ``all`` records every episode.
``GR00T_LIBERO_RECORD_MIN_STEPS``
    An episode only counts as finished (and is recorded) if it ran at least this many env steps
    or succeeded. Set it to the rollout's max episode steps so that an episode cut short when the
    evaluation ends is not mistaken for a failure. Default 0.

Each recorded episode is a directory ``<dir>/<task>/<id>/`` with ``image.mp4``,
``wrist_image.mp4``, ``state.npy`` ([T, 8]), ``action.npy`` ([T, 7], the action passed to the env
in the dataset's convention) and ``meta.json``.
"""

import json
import os
from pathlib import Path
import uuid

import imageio.v2 as imageio
import numpy as np


FPS = 20


class EpisodeRecorder:
    def __init__(self, task_name: str, task_description: str):
        self.out_dir = Path(os.environ["GR00T_LIBERO_RECORD_DIR"])
        self.mode = os.environ.get("GR00T_LIBERO_RECORD_MODE", "failure")
        if self.mode not in ("failure", "all"):
            raise ValueError(f"GR00T_LIBERO_RECORD_MODE must be failure or all, got {self.mode!r}")
        self.min_steps = int(os.environ.get("GR00T_LIBERO_RECORD_MIN_STEPS", "0"))
        self.task_name = task_name
        self.task_description = task_description
        self._clear()

    @staticmethod
    def enabled() -> bool:
        return bool(os.environ.get("GR00T_LIBERO_RECORD_DIR"))

    def _clear(self) -> None:
        self.images: list[np.ndarray] = []
        self.wrist_images: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.success = False
        self._pending_obs = None

    @staticmethod
    def _state_vector(obs: dict) -> np.ndarray:
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        return np.concatenate(
            [np.asarray(obs[f"state.{k}"], dtype=np.float32).reshape(-1) for k in keys]
            + [np.asarray(obs["state.gripper"], dtype=np.float32).reshape(-1)]
        )

    def reset(self, obs: dict) -> None:
        """Call after the env reset, with the first observation. Flushes the previous episode."""
        self.flush()
        self._clear()
        self._pending_obs = obs

    def step(self, action: np.ndarray, next_obs: dict, success: bool) -> None:
        """Record (observation, action) of the step just taken and keep the next observation."""
        if self._pending_obs is None:
            return
        obs = self._pending_obs
        self.images.append(np.ascontiguousarray(obs["video.image"]))
        self.wrist_images.append(np.ascontiguousarray(obs["video.wrist_image"]))
        self.states.append(self._state_vector(obs))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.success |= bool(success)
        self._pending_obs = next_obs

    def flush(self) -> None:
        n = len(self.actions)
        if n == 0:
            return
        finished = self.success or n >= self.min_steps
        if not finished or (self.mode == "failure" and self.success):
            return
        episode_dir = self.out_dir / self.task_name / f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        for name, frames in (("image", self.images), ("wrist_image", self.wrist_images)):
            imageio.mimwrite(
                episode_dir / f"{name}.mp4",
                frames,
                fps=FPS,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                pixelformat="yuv420p",
            )
        np.save(episode_dir / "state.npy", np.stack(self.states))
        np.save(episode_dir / "action.npy", np.stack(self.actions))
        (episode_dir / "meta.json").write_text(
            json.dumps(
                {
                    "task_name": self.task_name,
                    "task_description": self.task_description,
                    "success": self.success,
                    "length": n,
                }
            )
        )

    def close(self) -> None:
        self.flush()
        self._clear()
