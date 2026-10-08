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
Tests for VideoRecordingWrapper's raw_video mode, its per-episode C_i JSON
sidecar, and gr00t/eval/sim/round_trip_viewer.py, which turns those sidecars
into HTML viewers. ffmpeg writes are stubbed out; CPU-safe.
"""

import json
from pathlib import Path
import re

from gr00t.eval.sim import round_trip_viewer
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper
import gymnasium as gym
import numpy as np
import pytest


class _DummyLanguageEnv(gym.Env):
    def reset(self, **kwargs):
        return self._obs(), {}

    def _obs(self):
        return {
            "video.image": np.full((64, 96, 3), 40, dtype=np.uint8),
            "annotation.task": "do the thing",
        }

    def step(self, action):
        return self._obs(), 0.0, False, False, {"success": False}


def _make_wrapper(**kwargs):
    kwargs.setdefault("round_trip_depths", (1, 2))
    wrapper = VideoRecordingWrapper(
        _DummyLanguageEnv(), video_dir=None, steps_per_render=1, max_episode_steps=50, **kwargs
    )
    wrapper.reset()
    wrapper.file_path = Path("/tmp/unused_for_this_test.mp4")
    return wrapper


@pytest.fixture
def written(monkeypatch):
    frames = []
    monkeypatch.setattr(
        VideoRecordingWrapper, "_write_video_frame", lambda self, frame: frames.append(frame)
    )
    return frames


class TestRawVideo:
    def test_raw_frames_are_the_camera_footage(self, written):
        wrapper = _make_wrapper(raw_video=True, round_trip_live_plot=True, overlay_text=True)
        wrapper.set_round_trip_overlay([{1: 0.5, 2: 0.7}], [{1: 0.1, 2: 0.2}])
        wrapper.step(None)

        assert wrapper.overlay_text is False
        assert wrapper.round_trip_live_plot is False
        assert written[0].shape == (64, 96, 3)
        assert np.all(written[0] == 40)

    def test_raw_video_still_tracks_ci(self, written):
        wrapper = _make_wrapper(raw_video=True)
        wrapper.step(None)
        wrapper.set_round_trip_overlay([{1: 0.5}], [{1: 0.1}])
        wrapper.step(None)

        assert len(wrapper._round_trip_series) == len(written) == 2


class TestSidecar:
    def test_one_entry_per_recorded_frame_with_nulls_until_known(self, written, tmp_path):
        wrapper = _make_wrapper(raw_video=True)
        wrapper.step(None)  # nothing known yet
        wrapper.set_round_trip_overlay([{1: 0.5}], [{1: 0.1}])
        wrapper.step(None)  # depth 1 known, depth 2 not yet
        wrapper.set_round_trip_overlay([{1: 0.6, 2: float("nan")}], [{1: 0.2, 2: 0.3}])
        wrapper.step(None)

        video = tmp_path / "ep_s1.mp4"
        wrapper._save_round_trip_data(video.with_suffix(".json"), video, success=True)
        data = json.loads(video.with_suffix(".json").read_text())

        assert data["format"] == round_trip_viewer.SIDECAR_FORMAT
        assert data["video"] == "ep_s1.mp4"
        assert data["success"] is True
        assert data["depths"] == [1, 2]
        assert data["frame_steps"] == [s for s, _, _ in wrapper._round_trip_series]
        assert len(data["frame_steps"]) == len(written)
        assert data["action_c_i"] == {"1": [None, 0.5, 0.6], "2": [None, None, None]}
        assert data["latent_c_i"] == {"1": [None, 0.1, 0.2], "2": [None, None, 0.3]}


def _write_episode(folder: Path, name: str, success: bool, values=(0.5, 0.6)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.mp4").write_bytes(b"")
    payload = {
        "format": round_trip_viewer.SIDECAR_FORMAT,
        "video": f"{name}.mp4",
        "fps": 20,
        "steps_per_render": 2,
        "max_episode_steps": 720,
        "success": success,
        "depths": [1],
        "frame_steps": [2, 4],
        "action_c_i": {"1": list(values)},
        "latent_c_i": {"1": [0.1, 0.2]},
    }
    (folder / f"{name}.json").write_text(json.dumps(payload))


def _embedded_data(page: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="viewer-data">(.*?)</script>', page, re.S
    )
    assert match is not None
    return json.loads(match.group(1))


class TestViewer:
    def test_task_viewer_embeds_every_sidecar(self, tmp_path):
        _write_episode(tmp_path, "b_s0", success=False)
        _write_episode(tmp_path, "a_s1", success=True)
        (tmp_path / "unrelated.json").write_text('{"hello": 1}')

        out = round_trip_viewer.build_task_viewer(tmp_path, title="my task")

        assert out == tmp_path / "index.html"
        page = out.read_text()
        data = _embedded_data(page)
        assert data["title"] == "my task"
        assert [e["video"] for e in data["episodes"]] == ["a_s1.mp4", "b_s0.mp4"]
        assert "All tasks" not in page

    def test_sidecar_without_its_video_is_skipped(self, tmp_path):
        _write_episode(tmp_path, "kept_s1", success=True)
        _write_episode(tmp_path, "gone_s1", success=True)
        (tmp_path / "gone_s1.mp4").unlink()

        data = _embedded_data(round_trip_viewer.build_task_viewer(tmp_path).read_text())

        assert [e["video"] for e in data["episodes"]] == ["kept_s1.mp4"]

    def test_no_sidecars_writes_nothing(self, tmp_path):
        assert round_trip_viewer.build_task_viewer(tmp_path) is None
        assert not (tmp_path / "index.html").exists()

    def test_script_close_tag_in_data_cannot_break_out(self, tmp_path):
        _write_episode(tmp_path, "x_s1", success=True)
        out = round_trip_viewer.build_task_viewer(tmp_path, title="</script><b>")

        page = out.read_text()
        assert _embedded_data(page)["title"] == "</script><b>"
        # Only "</" can terminate a script element early; markup inside the
        # JSON block is otherwise inert text.
        block = page.split('id="viewer-data">', 1)[1].split("</script>", 1)[0]
        assert "</" not in block
        # The visible <title>/<h1> copies are HTML-escaped.
        assert "<title>C_i viewer: &lt;/script&gt;&lt;b&gt;</title>" in page

    def test_root_index_summarizes_tasks(self, tmp_path):
        _write_episode(tmp_path / "task_a", "e1_s1", success=True, values=(1.0, 2.0))
        _write_episode(tmp_path / "task_a", "e2_s0", success=False, values=(3.0, 5.0))
        _write_episode(tmp_path / "task_b", "e1_s1", success=True)
        (tmp_path / "empty_task").mkdir()

        out = round_trip_viewer.build_root_index(tmp_path, title="run")

        rows = {r["task"]: r for r in _embedded_data(out.read_text())["tasks"]}
        assert set(rows) == {"task_a", "task_b"}
        assert rows["task_a"]["n"] == 2 and rows["task_a"]["n_success"] == 1
        assert rows["task_a"]["mean_ci_succ"] == pytest.approx(1.5)
        assert rows["task_a"]["mean_ci_fail"] == pytest.approx(4.0)
        assert rows["task_a"]["href"] == "task_a/index.html"
        assert "All tasks" in (tmp_path / "task_a" / "index.html").read_text()
