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
Tests for gr00t.eval.calibration - pytest wrapper around the module's own
synthetic self-test (see its __main__ block), plus the tiny-dataset
graceful-degradation path directly relevant to a small demo-scale run.
"""

from gr00t.eval.calibration import (
    calibration_metrics,
    check_monotone,
    fit_heteroscedastic_gaussian,
    risk_coverage,
    select_degrees,
    spearman_fixed_depth,
    spearman_within,
    to_log,
)
import numpy as np
import pytest


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(0)
    n, D = 60, 10
    # log E = 1.2 log C + 0.3 + heteroscedastic noise
    logC = np.sort(rng.normal(-4, 1, (n, D)), axis=1)
    logE = 1.2 * logC + 0.3 + rng.normal(0, 0.15, (n, D)) * (1 + 0.2 * (logC - logC.min()))
    C, E = np.exp(logC), np.exp(logE)
    return C, E


class TestSpearman:
    def test_within_trajectory_correlation_is_high(self, synthetic_data):
        C, E = synthetic_data
        sw = spearman_within(C, E)
        assert sw.mean() > 0.9

    def test_fixed_depth_correlation_is_high(self, synthetic_data):
        C, E = synthetic_data
        sf = spearman_fixed_depth(C, E)
        assert sf.min() > 0.5


class TestFitHeteroscedasticGaussian:
    def test_recovers_known_linear_relation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 2000)
        true_a, true_b = 2.0, -1.0
        y = true_a * x + true_b + rng.normal(0, 0.1, x.shape)
        mu_fn, sigma_fn = fit_heteroscedastic_gaussian(x, y, deg_mu=1, deg_sigma=0)
        probe = np.linspace(x.min(), x.max(), 20)
        # A degree-1 fit of a genuinely linear relation should track it closely.
        np.testing.assert_allclose(mu_fn(probe), true_a * probe + true_b, atol=0.15)
        assert np.all(sigma_fn(probe) > 0)

    def test_monotone_fit_on_synthetic_data(self, synthetic_data):
        C, E = synthetic_data
        x, y = to_log(C).ravel(), to_log(E).ravel()
        mu_fn, _ = fit_heteroscedastic_gaussian(x, y, deg_mu=1, deg_sigma=0)
        assert check_monotone(mu_fn, x)

    def test_clamps_outside_training_range(self):
        """mu_fn/sigma_fn must not extrapolate wildly - inputs are clamped to
        the training range."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mu_fn, _ = fit_heteroscedastic_gaussian(x, y, deg_mu=1, deg_sigma=0)
        far_outside = mu_fn(np.array([1000.0]))
        at_max = mu_fn(np.array([4.0]))
        np.testing.assert_allclose(far_outside, at_max, atol=1e-8)


class TestSelectDegrees:
    def test_select_degrees_on_synthetic_data(self, synthetic_data):
        C, E = synthetic_data
        x_tr, y_tr = to_log(C[:40]), to_log(E[:40])
        dm, ds = select_degrees(x_tr, y_tr, verbose=False)
        assert dm in (1, 2, 3, 4)
        assert ds in (0, 1, 2)

    def test_graceful_degradation_for_tiny_dataset(self, synthetic_data):
        """n_sims < 4 (e.g. a 3-episode demo run) must not crash CV - it
        falls back to (1, 0) with a warning, per the plan's execution phase."""
        C, E = synthetic_data
        dm, ds = select_degrees(C[:3], E[:3], verbose=False)
        assert (dm, ds) == (1, 0)

    def test_rejects_1d_input(self, synthetic_data):
        C, E = synthetic_data
        with pytest.raises(AssertionError):
            select_degrees(C.ravel(), E.ravel(), verbose=False)


class TestCalibrationMetrics:
    def test_well_calibrated_synthetic_gaussian(self):
        rng = np.random.default_rng(2)
        mu = rng.normal(0, 1, 5000)
        sd = np.full(5000, 0.5)
        y = mu + rng.normal(0, 0.5, 5000)
        res = calibration_metrics(y, mu, sd, tag="test", verbose=False)
        assert 0.5 < res["cov68"] < 0.85
        assert res["cov95"] > 0.9

    def test_a_tighter_fit_beats_a_flat_baseline(self, synthetic_data):
        C, E = synthetic_data
        x_tr, y_tr = to_log(C[:40]).ravel(), to_log(E[:40]).ravel()
        x_va, y_va = to_log(C[40:]).ravel(), to_log(E[40:]).ravel()

        mu_fn, sd_fn = fit_heteroscedastic_gaussian(x_tr, y_tr, deg_mu=1, deg_sigma=0)
        res_c = calibration_metrics(y_va, mu_fn(x_va), sd_fn(x_va), "mu(C)", verbose=False)
        res_g = calibration_metrics(
            y_va, np.full(y_va.size, y_tr.mean()), np.full(y_va.size, y_tr.std() + 1e-9),
            "global", verbose=False,
        )
        assert res_c["nll"] < res_g["nll"]


class TestRiskCoverage:
    def test_deferring_highest_score_reduces_risk(self, synthetic_data):
        C, E = synthetic_data
        cov, rel, aurc = risk_coverage(E.ravel(), C.ravel())
        # Deferring the worst-scored fraction should not increase mean risk
        # at any coverage level (C and E are positively correlated here).
        assert rel[np.argmin(np.abs(cov - 0.5))] < rel[-1] + 1e-6
        assert np.isfinite(aurc)
