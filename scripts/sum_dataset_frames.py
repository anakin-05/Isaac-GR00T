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

"""Sum `total_frames` across every task in a LeRobot-format dataset directory.

Each task is a subdirectory containing a `meta/info.json` file with a
`total_frames` field. This script walks the dataset root, reads every
task's `info.json`, and reports the per-task and overall frame counts.

Usage:
    python scripts/sum_dataset_frames.py \
        examples/robocasa/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
"""

import argparse
import json
from pathlib import Path
import sys


def find_task_info_files(dataset_root: Path) -> list[Path]:
    """Return the `meta/info.json` path for every task under `dataset_root`."""
    return sorted(dataset_root.glob("*/meta/info.json"))


def sum_total_frames(dataset_root: Path) -> tuple[int, dict[str, int]]:
    """Sum `total_frames` from each task's `info.json` under `dataset_root`.

    Returns the overall total and a mapping of task name -> total_frames.
    """
    info_files = find_task_info_files(dataset_root)
    if not info_files:
        raise FileNotFoundError(f"No task meta/info.json files found under {dataset_root}")

    per_task_frames: dict[str, int] = {}
    for info_file in info_files:
        task_name = info_file.parent.parent.name
        with open(info_file) as f:
            info = json.load(f)
        if "total_frames" not in info:
            print(f"warning: {info_file} has no 'total_frames' field, skipping", file=sys.stderr)
            continue
        per_task_frames[task_name] = info["total_frames"]

    return sum(per_task_frames.values()), per_task_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset root containing one subdirectory per task",
    )
    args = parser.parse_args()

    total, per_task_frames = sum_total_frames(args.dataset_root)

    name_width = max(len(name) for name in per_task_frames)
    for task_name, frames in per_task_frames.items():
        print(f"{task_name:<{name_width}}  {frames:>12,}")
    print("-" * (name_width + 14))
    print(f"{'TOTAL':<{name_width}}  {total:>12,}")
    print(f"\n{len(per_task_frames)} tasks, {total:,} total frames")


if __name__ == "__main__":
    main()
