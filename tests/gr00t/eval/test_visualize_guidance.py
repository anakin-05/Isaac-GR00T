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

"""CPU tests for the geometry and metric helpers of gr00t/eval/visualize_guidance.py."""

from gr00t.eval.visualize_guidance import DensityRegion, Projection, chunk_metrics
import numpy as np


def _data(n=800, dim=12, seed=0):
    rng = np.random.RandomState(seed)
    latent = rng.randn(n, 2) * np.array([3.0, 1.0])
    return (latent @ rng.randn(2, dim) + 0.01 * rng.randn(n, dim)).astype(np.float32)


def test_projection_captures_the_dominant_directions():
    ref = _data()
    proj = Projection(ref)
    z = proj.zscore(ref)
    explained = (proj(ref) ** 2).sum() / (z**2).sum()
    assert explained > 0.95  # rank-2 data


def test_density_region_levels_and_membership():
    pts = np.random.RandomState(1).randn(2000, 2)
    region = DensityRegion(pts)
    assert region.level(0.5) > region.level(0.9)  # smaller mass -> higher density threshold
    inside = region.inside(pts, 0.9).mean()
    assert 0.85 <= inside <= 0.95
    assert not region.inside(np.array([[8.0, 8.0]]), 0.9)[0]


def test_chunk_metrics_zero_for_demo_and_main():
    ref = _data()
    proj = Projection(ref)
    demo = ref[0]
    samples = np.repeat(demo[None], 5, axis=0)
    m = chunk_metrics(samples, demo, samples, ref, proj)
    assert m["dist_to_demo"] < 1e-6 and m["shift_vs_main"] < 1e-6 and m["spread"] < 1e-6
    assert m["dist_to_manifold"] < 1e-5
    far = chunk_metrics(samples + 5.0, demo, samples, ref, proj)
    assert far["dist_to_demo"] > 1 and far["dist_to_manifold"] > 1 and far["shift_vs_main"] > 1
