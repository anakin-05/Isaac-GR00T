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
Calibration of the round-trip consistency signal C_i against the true
prediction error E_i - ported (not copied verbatim) from the
round-trip-consistency reference repo's
`LE_PDE_UQ/3_fno_calibration_V3.py`, itself ported from the paper's MHD
analysis notebook. Framework-agnostic (pure numpy/scipy).

At deployment only C_i (the round-trip consistency signal) is available, not
the true error E_i. This module fits, on held-out (train/val) data where both
C_i and E_i ARE known, a heteroscedastic-Gaussian envelope
mu(log C) +/- 2*sigma(log C) that predicts E_i from C_i alone - this is what
produces the shaded mu +/- 2sigma bands in the source paper's Fig. 2/4.

Dropped from the source module (PDE/video-decoder-specific, not applicable to
GR00T's per-trajectory state-space C_i): decode_predictions, true_frames,
frame_mse, accuracy_metrics, pixel_scale, lepdeuq_l2, toolbox_metrics,
hybrid_pixel_sigma, fit_hybrid, comparison_table, LEPDEUQ_REFERENCE, and the
sweep-array-specific extract_CE / depth_only_baseline / global_baseline /
pooled_and_fixed_depth_report / check_alignment helpers (those assume a
[n_sims, T] PDE-rollout sweep-array layout that does not match GR00T's
per-trajectory C_i arrays - see gr00t.eval.round_trip_consistency).

Run this file directly for a synthetic self-test (mirrors the source
module's __main__ self-test).
"""

from __future__ import annotations

from math import erf

import numpy as np
from scipy.optimize import minimize


EPS = 1e-12
LS_CLIP = 8.0  # |log sigma| bound inside the NLL


# ------------------------- Spearman (generic) -------------------------


def rankdata(a):
    a = np.asarray(a)
    sorter = a.argsort()
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.flatnonzero(obs), len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x, y):
    rx = rankdata(np.asarray(x))
    ry = rankdata(np.asarray(y))
    return np.corrcoef(rx, ry)[0, 1]


def spearman_within(C, E):
    """Per-trajectory Spearman of C vs E across depths. C, E: [n, D].
    Returns [n]; report mean +/- std."""
    C, E = np.asarray(C), np.asarray(E)
    return np.array([spearman(C[i], E[i]) for i in range(C.shape[0])])


def spearman_fixed_depth(C, E):
    """Across-trajectory Spearman at each depth. Returns [D]. Needs
    n_sims >= ~10 to be meaningful."""
    C, E = np.asarray(C), np.asarray(E)
    return np.array([spearman(C[:, k], E[:, k]) for k in range(C.shape[1])])


# --------------- heteroscedastic calibrator (ported, hardened) ---------------


def fit_heteroscedastic_gaussian(x, y, deg_mu=3, deg_sigma=2):
    """Joint ML fit of polynomial mu(x), log sigma(x); returns
    (mu_fn, sigma_fn) with inputs clamped to the training range (no
    extrapolation beyond what was fit)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, xs = x.mean(), x.std()
    xlo, xhi = x.min(), x.max()  # training range
    x0 = (x - xm) / xs
    Xmu = np.vander(x0, deg_mu + 1, increasing=True)
    Xsig = np.vander(x0, deg_sigma + 1, increasing=True)
    n_mu = Xmu.shape[1]

    def nll(p):
        a, b = p[:n_mu], p[n_mu:]
        mu = Xmu @ a
        ls = np.clip(Xsig @ b, -LS_CLIP, LS_CLIP)  # variance-collapse guard
        return np.sum(0.5 * (y - mu) ** 2 / np.exp(2 * ls) + ls)

    a0 = np.linalg.lstsq(Xmu, y, rcond=None)[0]
    b0 = np.zeros(Xsig.shape[1])
    b0[0] = np.log(np.std(y - Xmu @ a0) + 1e-9)
    res = minimize(nll, np.concatenate([a0, b0]), method="L-BFGS-B")
    if not res.success:
        print(f"  [warn] optimizer: {res.message} (deg {deg_mu}/{deg_sigma})")
    a, b = res.x[:n_mu], res.x[n_mu:]

    def _feats(xn, d):
        xn = np.clip(np.asarray(xn, float), xlo, xhi)  # extrapolation clamp
        return np.vander((xn - xm) / xs, d + 1, increasing=True)

    def mu_fn(xn):
        return _feats(xn, deg_mu) @ a

    def sigma_fn(xn):
        return np.exp(np.clip(_feats(xn, deg_sigma) @ b, -LS_CLIP, LS_CLIP))

    return mu_fn, sigma_fn


def _val_nll(mu_fn, sigma_fn, x, y):
    mu, sd = mu_fn(x), np.maximum(sigma_fn(x), 1e-9)
    return float(np.mean(0.5 * ((y - mu) / sd) ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)))


def select_degrees(
    C_tr,
    E_tr,
    deg_mu_grid=(1, 2, 3, 4),
    deg_sigma_grid=(0, 1, 2),
    k_folds=5,
    seed=0,
    verbose=True,
    monotone_only=False,
):
    """Trajectory-level K-fold CV on TRAINING data; one-SE choice.
    Inputs MUST be 2-D [n_sims, n_depths] in the fitting space (log). Folds
    are over trajectories, not individual (C, E) points, because points
    within a trajectory are correlated and pointwise CV would bless overfit
    degrees. Falls back to (1, 0) with a warning when there are too few
    trajectories for CV (e.g. a small demo dataset) - this is an expected,
    graceful degradation, not an error."""
    C_tr, E_tr = np.asarray(C_tr), np.asarray(E_tr)
    assert C_tr.ndim == 2 and E_tr.ndim == 2, (
        f"expected 2-D [n_sims, n_depths]; got C {C_tr.shape}, E {E_tr.shape}"
    )
    assert C_tr.shape == E_tr.shape, (C_tr.shape, E_tr.shape)
    n_sims, n_depths = C_tr.shape
    k_folds = min(k_folds, n_sims)
    if n_sims < 4:
        print(
            f"[select_degrees] only {n_sims} sims - skipping CV, "
            f"returning (deg_mu, deg_sigma) = (1, 0)"
        )
        return 1, 0
    if verbose:
        print(
            f"select_degrees: {n_sims} sims x {n_depths} depths, {k_folds}-fold CV over sims"
        )
    folds = np.random.default_rng(seed).permutation(n_sims) % k_folds
    rows = []
    for dm in deg_mu_grid:
        for ds in deg_sigma_grid:
            scores = []
            for f in range(k_folds):
                tr, va = folds != f, folds == f
                mu_fn, sd_fn = fit_heteroscedastic_gaussian(
                    C_tr[tr].ravel(), E_tr[tr].ravel(), dm, ds
                )
                scores.append(_val_nll(mu_fn, sd_fn, C_tr[va].ravel(), E_tr[va].ravel()))
            rows.append((dm, ds, float(np.mean(scores)), float(np.std(scores) / np.sqrt(k_folds))))
    if monotone_only:
        kept = []
        for dm, ds, m, se in rows:
            mu_fn, _ = fit_heteroscedastic_gaussian(C_tr.ravel(), E_tr.ravel(), dm, ds)
            if check_monotone(mu_fn, C_tr):
                kept.append((dm, ds, m, se))
            elif verbose:
                print(f"  [monotone_only] dropping deg ({dm},{ds}): non-monotone full-train fit")
        if kept:
            rows = kept
        else:
            print(
                "  [monotone_only] NO monotone candidate - keeping all "
                "(inspect the fit; the C-E relation may be absent)"
            )
    rows.sort(key=lambda r: r[2])
    best = rows[0]
    if verbose:
        print(f"{'deg_mu':>6} {'deg_sig':>7} {'CV NLL':>9} {'SE':>7}")
        for dm, ds, m, se in sorted(rows):
            mark = " <-- best" if (dm, ds) == best[:2] else ""
            print(f"{dm:6d} {ds:7d} {m:9.4f} {se:7.4f}{mark}")
    thresh = best[2] + best[3]  # one-SE rule: simplest within reach
    eligible = [r for r in rows if r[2] <= thresh]
    dm, ds, m, _ = min(eligible, key=lambda r: (r[0] + r[1], r[2]))
    if verbose:
        print(
            f"one-SE choice: deg_mu={dm}, deg_sigma={ds}  (CV NLL {m:.4f}; "
            f"best was {best[0]}/{best[1]} at {best[2]:.4f})"
        )
    return dm, ds


def check_monotone(mu_fn, x, tol=-1e-9, warn_tag=None):
    """Non-crashing monotonicity check; returns bool. Physically: larger C
    must not predict smaller E. A False on real, trained-model data means
    the fit (or the C-E relation) deserves inspection."""
    grid = np.linspace(np.min(x), np.max(x), 512)
    ok = bool(np.all(np.diff(mu_fn(grid)) >= tol))
    if not ok and warn_tag is not None:
        print(
            f"  [WARN] {warn_tag}: mu non-monotone on the data range - "
            f"lower deg_mu, use isotonic, or inspect the C-E scatter"
        )
    return ok


def assert_monotone(mu_fn, x, tol=-1e-9):
    """Physically: larger C must not predict smaller E."""
    assert check_monotone(mu_fn, x, tol), (
        "mu(C) non-monotone on the data range - lower deg_mu (or use isotonic)"
    )


# --------------------------- metrics (ported) ---------------------------


def _phi(z):  # standard normal CDF
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def calibration_metrics(y, mu, sd, tag="", verbose=True):
    """NLL | logRMSE | factor-of-X (50/68/95) | +/-1/2 sigma coverage |
    miscalibration area (PIT) | sharpness. All args in LOG space."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    sd = np.maximum(np.asarray(sd, float), 1e-9)
    z = (y - mu) / sd
    nll = float(np.mean(0.5 * z**2 + np.log(sd) + 0.5 * np.log(2 * np.pi)))
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    fac = np.exp(np.abs(y - mu))  # |resid| in log = log factor
    facs = tuple(float(np.quantile(fac, q)) for q in (0.5, 0.68, 0.95))
    cov1 = float(np.mean(np.abs(z) < 1))
    cov2 = float(np.mean(np.abs(z) < 1.96))
    u = np.sort(_phi(z))  # PIT; uniform if calibrated
    mca = float(np.mean(np.abs(u - (np.arange(len(u)) + 0.5) / len(u))))
    sharp = float(np.mean(sd))
    if verbose:
        print(
            f"  {tag:<12s} NLL {nll:7.3f} | logRMSE {rmse:5.3f} | "
            f"within x{facs[0]:.2f}/{facs[1]:.2f}/{facs[2]:.2f} (50/68/95%)"
            f" | cov68 {cov1 * 100:4.1f}% cov95 {cov2 * 100:4.1f}% | "
            f"miscal {mca:.3f} | sharp {sharp:.3f}"
        )
    return dict(
        nll=nll, rmse=rmse, x50=facs[0], x68=facs[1], x95=facs[2], cov68=cov1, cov95=cov2,
        mca=mca, sharp=sharp,
    )


def to_log(A):
    return np.log(np.asarray(A, float) + EPS)


# ------------------------- selective prediction -------------------------


def risk_coverage(E_flat, score_flat, grid=None):
    """Defer the (1 - c) fraction with the HIGHEST score; report the mean
    incurred error over the kept fraction, relative to full coverage.
    Returns (coverages, relative_risk, aurc)."""
    E = np.asarray(E_flat, float)
    s = np.asarray(score_flat, float)
    order = np.argsort(s)  # keep lowest-score first
    E_sorted = E[order]
    csum = np.cumsum(E_sorted)
    n = len(E)
    if grid is None:
        grid = np.arange(1, n + 1) / n
    ks = np.maximum(1, np.round(np.asarray(grid) * n).astype(int))
    risk = csum[ks - 1] / ks
    rel = risk / E.mean()
    # np.trapezoid was added in numpy 2.0; np.trapz (deprecated there, but
    # still present) is what this repo's pinned numpy actually has.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    aurc = float(trapz(rel, np.asarray(grid)))
    return np.asarray(grid), rel, aurc


# ------------------------------ self-test ------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, D = 60, 10
    # synthetic: log E = 1.2 log C + 0.3 + heteroscedastic noise
    logC = np.sort(rng.normal(-4, 1, (n, D)), axis=1)
    logE = 1.2 * logC + 0.3 + rng.normal(0, 0.15, (n, D)) * (1 + 0.2 * (logC - logC.min()))
    C, E = np.exp(logC), np.exp(logE)

    sw = spearman_within(C, E)
    sf = spearman_fixed_depth(C, E)
    print(
        f"spearman within {sw.mean():.3f} +/- {sw.std():.3f} | "
        f"fixed-depth {sf.min():.3f}..{sf.max():.3f}"
    )
    assert sw.mean() > 0.9

    x_tr, y_tr = to_log(C[:40]), to_log(E[:40])
    dm, ds = select_degrees(x_tr, y_tr)
    mu_fn, sd_fn = fit_heteroscedastic_gaussian(x_tr.ravel(), y_tr.ravel(), dm, ds)
    assert_monotone(mu_fn, x_tr)

    x_va, y_va = to_log(C[40:]), to_log(E[40:])
    res_c = calibration_metrics(y_va.ravel(), mu_fn(x_va.ravel()), sd_fn(x_va.ravel()), "mu(C)")
    res_g = calibration_metrics(
        y_va.ravel(), np.full(y_va.size, y_tr.mean()), np.full(y_va.size, y_tr.std() + 1e-9),
        "global",
    )
    assert res_c["nll"] < res_g["nll"]
    assert 0.5 < res_c["cov68"] < 0.85

    cov, rel, aurc = risk_coverage(np.exp(y_va).ravel(), mu_fn(x_va.ravel()))
    print(
        f"risk-coverage: rel risk @80% cov = {rel[np.argmin(np.abs(cov - 0.8))]:.3f}, "
        f"AURC {aurc:.4f}"
    )
    assert rel[np.argmin(np.abs(cov - 0.8))] < 1.0

    # graceful degradation on a tiny (n_sims < 4) dataset, e.g. a 3-episode demo
    tiny_dm, tiny_ds = select_degrees(x_tr[:3], y_tr[:3])
    assert (tiny_dm, tiny_ds) == (1, 0)

    print("\nSELF-TEST PASSED")
