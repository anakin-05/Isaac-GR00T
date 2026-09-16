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
Tests for VideoRecordingWrapper's round_trip_depths overlay/plot feature (see
gr00t/eval/rollout_policy.py's Mechanism 2, which pushes the latest known C_i
onto the wrapper every macro-step via
``env.call("set_round_trip_overlay", ...)``). Mocked ffmpeg (writes are
stubbed out, matching the pattern already used in
test_video_recording_wrapper.py), CPU-safe, no real video/plot I/O required
for most assertions.

TestCallReachesTheCorrectSlotVideoWrapper is the one class here that goes
through a real gym.vector.SyncVectorEnv + MultiStepWrapper(VideoRecordingWrapper(...))
chain instead of poking wrapper state directly - the others test the
wrapper's own caption/plot logic in isolation, which is correct but was
never the part that broke. See that class's docstring for why.
"""

from pathlib import Path

from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class _DummyLanguageEnv(gym.Env):
    """Minimal env satisfying VideoRecordingWrapper.step()'s requirements: one
    video.* frame key, one annotation.*/language.* caption key, and a
    "success" field in step()'s info."""

    def reset(self, **kwargs):
        return self._obs(), {}

    def _obs(self):
        return {
            "video.image": np.zeros((8, 8, 3), dtype=np.uint8),
            "annotation.task": "do the thing",
        }

    def step(self, action):
        return self._obs(), 0.0, False, False, {"success": False}


def _make_wrapper(
    round_trip_depths, video_dir=None, max_episode_steps=720
) -> VideoRecordingWrapper:
    wrapper = VideoRecordingWrapper(
        _DummyLanguageEnv(),
        video_dir=video_dir,
        steps_per_render=1,
        max_episode_steps=max_episode_steps,
        round_trip_depths=round_trip_depths,
    )
    wrapper.reset()
    if video_dir is None:
        # step() only writes a frame when file_path is set; give it a
        # placeholder path since video_dir=None means reset() never set one.
        wrapper.file_path = Path("/tmp/unused_for_this_test.mp4")
    return wrapper


class TestOverlayDisabledIsUnchanged:
    def test_round_trip_depths_none_never_touches_overlay_state(self, monkeypatch):
        monkeypatch.setattr(VideoRecordingWrapper, "_write_video_frame", lambda self, frame: None)
        wrapper = _make_wrapper(round_trip_depths=None)

        wrapper.step(None)

        assert wrapper._round_trip_series == []


class TestSecondLineReservedUpfront:
    def test_caption_taller_than_baseline_before_any_ci_arrives(self, monkeypatch):
        monkeypatch.setattr(VideoRecordingWrapper, "_write_video_frame", lambda self, frame: None)

        baseline = _make_wrapper(round_trip_depths=None)
        baseline.step(None)

        with_ci = _make_wrapper(round_trip_depths=(1, 2))
        with_ci.step(None)  # no C_i set yet - still shows "(pending)"

        assert with_ci.caption_height > baseline.caption_height


class TestFrameShapeStaysConstant:
    def test_frame_height_unchanged_when_ci_arrives_mid_episode(self, monkeypatch):
        written_shapes = []
        monkeypatch.setattr(
            VideoRecordingWrapper,
            "_write_video_frame",
            lambda self, frame: written_shapes.append(frame.shape),
        )
        wrapper = _make_wrapper(round_trip_depths=(1, 2))

        wrapper.step(None)  # frame 1: no C_i yet ("(pending)")
        wrapper._round_trip_overlay = {1: 1.234567, 2: 2.345678}
        wrapper.step(None)  # frame 2: C_i now present
        wrapper._round_trip_overlay = {1: 1.2, 2: 2.3}
        wrapper.step(None)  # frame 3: different C_i values

        assert len(written_shapes) == 3
        # H.264 rejects a mid-stream shape change - every frame this wrapper
        # ever hands to ffmpeg must be identical, regardless of whether/what
        # C_i text was drawn into the reserved second line.
        assert len(set(written_shapes)) == 1


class TestSeriesAccumulation:
    def test_series_records_one_entry_per_written_frame(self, monkeypatch):
        monkeypatch.setattr(VideoRecordingWrapper, "_write_video_frame", lambda self, frame: None)
        wrapper = _make_wrapper(round_trip_depths=(1, 2))

        wrapper.step(None)
        wrapper._round_trip_overlay = {1: 1.5}
        wrapper.step(None)
        wrapper._round_trip_overlay = {1: 1.6, 2: 1.7}
        wrapper.step(None)

        assert [values for _, values in wrapper._round_trip_series] == [
            {},
            {1: 1.5},
            {1: 1.6, 2: 1.7},
        ]

    def test_series_resets_on_new_episode(self, monkeypatch):
        monkeypatch.setattr(VideoRecordingWrapper, "_write_video_frame", lambda self, frame: None)
        wrapper = _make_wrapper(round_trip_depths=(1,), max_episode_steps=1)

        wrapper._round_trip_overlay = {1: 1.5}
        wrapper.step(None)
        assert wrapper._round_trip_series != []

        wrapper.reset()

        assert wrapper._round_trip_series == []
        assert wrapper._round_trip_overlay is None


class TestPlotExport:
    def test_plot_saved_alongside_a_kept_video(self, tmp_path, monkeypatch):
        def _fake_write_frame(self, frame):
            # Bypass the real ffmpeg subprocess entirely - just touch the
            # file once, matching what a real encoder would eventually
            # produce, so reset()'s exists()-gated rename/plot logic runs.
            if not self.file_path.exists():
                self.file_path.touch()

        monkeypatch.setattr(VideoRecordingWrapper, "_write_video_frame", _fake_write_frame)

        wrapper = _make_wrapper(round_trip_depths=(1, 2), video_dir=tmp_path, max_episode_steps=2)
        wrapper._round_trip_overlay = {1: 1.5, 2: 1.6}
        wrapper.step(None)
        wrapper._round_trip_overlay = {1: 1.55, 2: 1.62}
        wrapper.step(None)  # step_count now >= max_episode_steps -> kept on next reset()

        wrapper.reset()

        mp4_files = list(tmp_path.glob("*.mp4"))
        png_files = list(tmp_path.glob("*.png"))
        assert len(mp4_files) == 1
        assert len(png_files) == 1
        assert png_files[0].stat().st_size > 0
        assert png_files[0].stem == mp4_files[0].stem

    def test_no_plot_saved_when_video_discarded(self, tmp_path, monkeypatch):
        def _fake_write_frame(self, frame):
            if not self.file_path.exists():
                self.file_path.touch()

        monkeypatch.setattr(VideoRecordingWrapper, "_write_video_frame", _fake_write_frame)

        # max_episode_steps very high, episode never "finishes" or succeeds -
        # should_keep_video is False, matching an interrupted/unfinished run.
        wrapper = _make_wrapper(round_trip_depths=(1,), video_dir=tmp_path, max_episode_steps=1000)
        wrapper._round_trip_overlay = {1: 1.5}
        wrapper.step(None)

        wrapper.reset()

        assert list(tmp_path.glob("*.mp4")) == []
        assert list(tmp_path.glob("*.png")) == []


class _FakeContract:
    """Just enough of PolicyHorizonSpec's surface for MultiStepWrapper.__init__."""

    n_action_steps = 2
    video_delta_indices_array = np.array([-1, 0])
    state_delta_indices_array = None


class _DummyStepEnv(_DummyLanguageEnv):
    """_DummyLanguageEnv plus real action/observation spaces, since
    MultiStepWrapper.__init__ (unlike VideoRecordingWrapper) actually reads
    them (repeated_space(env.action_space, ...), convert_observation_space
    (self.observation_space, ...))."""

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    observation_space = spaces.Dict(
        {
            "video.image": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
            "annotation.task": spaces.Text(max_length=64),
        }
    )


class TestCallReachesTheCorrectSlotVideoWrapper:
    """Regression test for a bug where the rollout loop pushed the latest C_i
    via ``env.set_attr("_round_trip_overlay", ...)``, which appeared to work
    (get_attr read back whatever set_attr had just written) but the video
    caption/plot stayed stuck on "(pending)" forever in production.

    Root cause: MultiStepWrapper sits directly in a VectorEnv's ``self.envs``
    (it's the outermost wrapper), with VideoRecordingWrapper nested one level
    in via ``self.env``. gymnasium 0.29.1 (what the LIBERO client's own venv
    actually pins) implements ``VectorEnv.set_attr`` as a *shallow*
    ``setattr(self.envs[i], name, value)`` that lands on MultiStepWrapper and
    never reaches VideoRecordingWrapper - it silently creates an unrelated
    attribute on the wrong object. gymnasium>=1.0's set_attr instead calls
    ``env.set_wrapper_attr(name, value)``, which walks the chain correctly.
    Because this repo's top-level .venv (which runs `pytest`) has
    gymnasium>=1.0, an isolated repro built and run in *this* venv cannot
    reproduce the bug - it only manifests under the other venv's older
    gymnasium. That asymmetry is what let the original bug slip past manual
    isolated testing for a long time.

    env.call(name, *args) does not have this asymmetry: both versions
    resolve the name via delegation that walks the wrapper chain
    (``get_wrapper_attr`` on >=1.0; plain ``getattr`` triggering
    ``Wrapper.__getattr__`` delegation on 0.29.1, which is why the target
    method below is deliberately not underscore-prefixed - 0.29.1's
    ``__getattr__`` refuses to delegate names starting with "_"). This test
    exercises the exact call the rollout loop makes,
    ``env.call("set_round_trip_overlay", values_per_slot)``, through a real
    two-level wrapper chain with n_envs=2, confirming each slot's value lands
    on the VideoRecordingWrapper instance matching its own env_idx - not
    just that *some* attribute got set somewhere.
    """

    def _make_env_fn(self, env_idx):
        def _init():
            inner = VideoRecordingWrapper(
                _DummyStepEnv(),
                video_dir=None,
                steps_per_render=1,
                max_episode_steps=1000,
                round_trip_depths=(1, 2),
                env_idx=env_idx,
            )
            return MultiStepWrapper(inner, contract=_FakeContract())

        return _init

    def test_call_reaches_the_correct_slot(self):
        vec_env = gym.vector.SyncVectorEnv([self._make_env_fn(0), self._make_env_fn(1)])
        try:
            values_per_slot = [{1: 0.1, 2: 0.2}, {1: 0.9, 2: 0.8}]

            vec_env.call("set_round_trip_overlay", values_per_slot)

            video_wrappers = [sub_env.env for sub_env in vec_env.envs]
            assert video_wrappers[0]._round_trip_overlay == {1: 0.1, 2: 0.2}
            assert video_wrappers[1]._round_trip_overlay == {1: 0.9, 2: 0.8}
        finally:
            vec_env.close()
