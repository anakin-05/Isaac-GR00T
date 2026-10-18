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

"""Build a LeRobot v2.1 failure dataset from recorded LIBERO rollouts.

Rollouts are recorded by running ``examples/LIBERO/eval_libero_10.sh`` with
``RECORD_FAILURES_DIR=<dir>`` (see ``gr00t/eval/sim/LIBERO/episode_recorder.py``). This script
packs them into a dataset in the same format as the success (demonstration) dataset, which it
uses as a template:

* ``meta/info.json``, ``tasks.jsonl`` and ``modality.json`` come from the template.
* ``meta/stats.json`` and ``meta/relative_stats.json`` are copied from the template too, so a
  failure head trained on this dataset normalizes states and actions exactly like the main model
  (autoguidance / AFIL mix velocities of both models and require identical normalization).

Usage:
    uv run python examples/LIBERO/build_failure_dataset.py \
        --recordings-dir examples/LIBERO/logs/failures/<run> \
        --template-dataset examples/LIBERO/libero_10_no_noops_1.0.0_lerobot \
        --output-dir examples/LIBERO/libero_10_failures_lerobot
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


VIDEO_KEYS = {"image": "observation.images.image", "wrist_image": "observation.images.wrist_image"}


@dataclass
class Args:
    recordings_dir: str
    """Directory the failed rollouts were recorded to (``<task>/<episode>/`` inside)."""

    template_dataset: str
    """Success dataset whose format, tasks and statistics are reused."""

    output_dir: str
    """Where to write the failure dataset."""

    min_length: int = 16
    """Skip episodes shorter than this many steps."""

    include_successful: bool = False
    """Also keep recorded episodes marked successful (default: failures only)."""

    overwrite: bool = False


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def main(args: Args) -> None:
    template = Path(args.template_dataset)
    out = Path(args.output_dir)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)

    tasks = [json.loads(line) for line in (template / "meta/tasks.jsonl").read_text().splitlines()]
    task_index = {_norm(t["task"]): t["task_index"] for t in tasks}
    info = json.loads((template / "meta/info.json").read_text())
    schema = pq.read_schema(next((template / "data").glob("chunk-*/episode_*.parquet")))
    chunk_size = info["chunks_size"]
    fps = info["fps"]

    episodes = []
    for meta_path in sorted(Path(args.recordings_dir).glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["success"] and not args.include_successful:
            continue
        if meta["length"] < args.min_length:
            continue
        key = _norm(meta["task_description"])
        if key not in task_index:
            raise ValueError(
                f"Task {meta['task_description']!r} is not in {template}/meta/tasks.jsonl"
            )
        episodes.append((meta_path.parent, meta, task_index[key]))
    if not episodes:
        raise ValueError(f"No matching episodes found in {args.recordings_dir}")

    (out / "meta").mkdir(parents=True)
    episode_records = []
    total_frames = 0
    for ep_idx, (ep_dir, meta, t_idx) in enumerate(episodes):
        chunk = ep_idx // chunk_size
        state = np.load(ep_dir / "state.npy").astype(np.float32)
        action = np.load(ep_dir / "action.npy").astype(np.float32)
        n = len(action)
        assert len(state) == n, (ep_dir, len(state), n)
        table = pa.table(
            {
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(state.reshape(-1)), state.shape[1]
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(action.reshape(-1)), action.shape[1]
                ),
                "timestamp": pa.array(np.arange(n, dtype=np.float32) / fps),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep_idx, dtype=np.int64)),
                "index": pa.array(np.arange(total_frames, total_frames + n, dtype=np.int64)),
                "task_index": pa.array(np.full(n, t_idx, dtype=np.int64)),
            }
        ).cast(schema)
        data_path = out / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)
        for short, video_key in VIDEO_KEYS.items():
            video_path = out / info["video_path"].format(
                episode_chunk=chunk, video_key=video_key, episode_index=ep_idx
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ep_dir / f"{short}.mp4", video_path)
        episode_records.append(
            {"episode_index": ep_idx, "tasks": [meta["task_description"]], "length": n}
        )
        total_frames += n

    with open(out / "meta/episodes.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in episode_records)
    for name in ("tasks.jsonl", "modality.json", "stats.json", "relative_stats.json"):
        shutil.copy(template / "meta" / name, out / "meta" / name)
    info.update(
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_videos=len(episodes) * len(VIDEO_KEYS),
        total_chunks=(len(episodes) - 1) // chunk_size + 1,
        splits={"train": f"0:{len(episodes)}"},
    )
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))

    per_task: dict[str, int] = {}
    for _, meta, _ in episodes:
        per_task[meta["task_description"]] = per_task.get(meta["task_description"], 0) + 1
    print(f"Wrote {len(episodes)} episodes / {total_frames} frames to {out}")
    for task, count in sorted(per_task.items()):
        print(f"  {count:4d}  {task}")


if __name__ == "__main__":
    main(tyro.cli(Args))
