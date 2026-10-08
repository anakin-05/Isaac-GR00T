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
Tests for VideoRecordingWrapper's round_trip_live_plot feature: a live
C_i-vs-step panel appended to the right of every recorded frame when
round_trip_depths is set. ffmpeg writes are stubbed out (same pattern as
test_video_round_trip_overlay.py), so these are CPU-safe and only exercise the
frame composition itself.
"""

from pathlib import Path

from gr00t.eval.sim.wrapper import video_recording_wrapper
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper
import gymnasium as gym
import numpy as np
import pytest


class _DummyLanguageEnv(gym.Env):
    def reset(self, **kwargs):
        return self._obs(), {}

    def _obs(self):
        return {
            "video.image": np.full((128, 128, 3), 40, dtype=np.uint8),
            "annotation.task": "do the thing",
        }

    def step(self, action):
        return self._obs(), 0.0, False, False, {"success": False}


def _make_wrapper(round_trip_depths=(1, 2, 4), round_trip_live_plot=True, max_episode_steps=64):
    wrapper = VideoRecordingWrapper(
        _DummyLanguageEnv(),
        video_dir=None,
        steps_per_render=1,
        max_episode_steps=max_episode_steps,
        round_trip_depths=round_trip_depths,
        round_trip_live_plot=round_trip_live_plot,
    )
    wrapper.reset()
    # step() only records when file_path is set; video_dir=None never sets one.
    wrapper.file_path = Path("/tmp/unused_for_this_test.mp4")
    return wrapper


@pytest.fixture
def written(monkeypatch):
    frames = []
    monkeypatch.setattr(
        VideoRecordingWrapper, "_write_video_frame", lambda self, frame: frames.append(frame)
    )
    return frames


class TestPanelComposition:
    def test_panel_widens_frame_and_keeps_height(self, written):
        with_plot = _make_wrapper(round_trip_live_plot=True)
        with_plot.step(None)
        without_plot = _make_wrapper(round_trip_live_plot=False)
        without_plot.step(None)

        plotted, plain = written
        assert plotted.shape[0] == plain.shape[0]
        panel_width = plotted.shape[1] - plain.shape[1]
        assert panel_width > 0
        # H.264 / yuv420p needs even dimensions.
        assert panel_width % 2 == 0
        assert plotted.dtype == np.uint8

    def test_live_plot_ignored_without_round_trip_depths(self, written):
        wrapper = _make_wrapper(round_trip_depths=None, round_trip_live_plot=True)
        wrapper.step(None)

        assert wrapper.round_trip_live_plot is False
        assert written[0].shape[1] == 128

    def test_frame_shape_constant_as_ci_arrives(self, written):
        wrapper = _make_wrapper()
        wrapper.step(None)  # no C_i yet
        wrapper.set_round_trip_overlay([{1: 0.5, 2: 0.9, 4: 1.7}], [{1: 1e-3, 2: 2e-3, 4: 5e-3}])
        wrapper.step(None)
        wrapper.set_round_trip_overlay([{1: 12.0, 2: 30.0, 4: 80.0}], [{1: 4.0, 2: 9.0, 4: 20.0}])
        wrapper.step(None)

        assert len({frame.shape for frame in written}) == 1


class TestRenderThrottling:
    def test_only_rerenders_on_new_values_or_periodically(self, written, monkeypatch):
        calls = []
        original = VideoRecordingWrapper._render_live_plot

        def counting(self):
            calls.append(self.step_count)
            original(self)

        monkeypatch.setattr(VideoRecordingWrapper, "_render_live_plot", counting)
        wrapper = _make_wrapper(max_episode_steps=200)
        wrapper.set_round_trip_overlay([{1: 0.5}], [{1: 0.1}])

        n_frames = 40
        for _ in range(n_frames):
            wrapper.step(None)

        every = video_recording_wrapper._LIVE_PLOT_REDRAW_EVERY
        # First frame renders, then one re-render per (every + 1) frames.
        assert len(calls) == -(-n_frames // (every + 1))

        calls.clear()
        wrapper.set_round_trip_overlay([{1: 0.6}], [{1: 0.2}])
        wrapper.step(None)
        assert len(calls) == 1  # a new value forces an immediate re-render


class TestCursor:
    def test_cursor_moves_between_frames(self, written):
        wrapper = _make_wrapper(max_episode_steps=32)
        wrapper.set_round_trip_overlay([{1: 0.5}], [{1: 0.1}])
        wrapper.step(None)
        for _ in range(10):
            wrapper.step(None)

        first, last = written[0], written[-1]
        panel_start = 128
        # Same cached render (no new C_i, within the redraw window) except for
        # the cursor line, so the two panels must differ somewhere.
        assert not np.array_equal(first[:, panel_start:], last[:, panel_start:])


class TestLifecycle:
    def test_reset_drops_live_plot_state(self, written):
        wrapper = _make_wrapper()
        wrapper.step(None)
        assert wrapper._live_plot is not None

        wrapper.reset()
        assert wrapper._live_plot is None

    def test_too_many_depths_rejected(self):
        n = len(video_recording_wrapper._LIVE_PLOT_DEPTH_COLORS) + 1
        with pytest.raises(ValueError, match="at most"):
            _make_wrapper(round_trip_depths=tuple(range(1, n + 1)))

    def test_too_many_depths_fine_when_live_plot_disabled(self):
        n = len(video_recording_wrapper._LIVE_PLOT_DEPTH_COLORS) + 1
        _make_wrapper(round_trip_depths=tuple(range(1, n + 1)), round_trip_live_plot=False)
