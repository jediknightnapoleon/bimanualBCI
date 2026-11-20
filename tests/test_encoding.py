"""Tests for bimanual.encoding on synthetic neurons with KNOWN tuning.

These are the main validation for the Fig 2 machinery. The estimators must
recover true magnitudes and correlations, and be unbiased (about 0) for
untuned channels, in exactly the cases where naive estimates are biased.
"""

import numpy as np
import pytest

from bimanual import encoding as E

N_CH = 96
BINS_PER_TRIAL = 20  # 300-700 ms window = 20 bins of 20 ms
NOISE = 3.0


def _simulate(B_true, n_trials, rng):
    """Synthetic data for a [1, dx, dy] design with one random target direction per trial.

    Returns X (samples x 3), Y (samples x N) and groups (trial IDs).
    """
    ang = rng.uniform(0, 2 * np.pi, n_trials)
    d = np.repeat(np.c_[np.cos(ang), np.sin(ang)], BINS_PER_TRIAL, axis=0)
    X = np.c_[np.ones(len(d)), d]
    Y = X @ B_true + rng.normal(0, NOISE, (len(d), B_true.shape[1]))
    groups = np.repeat(np.arange(n_trials), BINS_PER_TRIAL)
    return X, Y, groups


def test_group_folds_keep_trials_whole():
    groups = np.repeat(np.arange(23), 4)
    folds = E.group_folds(groups, 5)
    assert sum(m.sum() for m in folds) == groups.size
    for m in folds:
        # no trial is split across folds
        assert set(groups[m]).isdisjoint(set(groups[~m]))


def test_untuned_channels_have_unbiased_zero_magnitude():
    rng = np.random.default_rng(0)
    B_true = np.zeros((3, N_CH))
    naive, cv = [], []
    for rep in range(30):
        X, Y, g = _simulate(B_true, 40, rng)
        fit = E.cv_ols(X, Y, g, seed=rep)
        naive.append(np.sum(fit.full[1] ** 2))
        cv.append(E.cv_sq_norm(fit, 1))
    assert np.mean(naive) > 1.0  # naive squared norm is clearly biased upward
    # the cv estimate averages to ~0 (within 4 standard errors)
    assert abs(np.mean(cv)) < 4 * np.std(cv) / np.sqrt(len(cv))


def test_magnitude_recovered():
    rng = np.random.default_rng(1)
    B_true = np.zeros((3, N_CH))
    B_true[1] = rng.normal(0, 0.5, N_CH)  # x tuning
    B_true[2] = rng.normal(0, 0.25, N_CH)  # y tuning, half as strong
    est_x, est_y = [], []
    for rep in range(20):
        X, Y, g = _simulate(B_true, 40, rng)
        fit = E.cv_ols(X, Y, g, seed=rep, names=("b0", "x", "y"))
        est_x.append(E.cv_magnitude(fit, "x"))
        est_y.append(E.cv_magnitude(fit, "y"))
    assert np.mean(est_x) == pytest.approx(np.linalg.norm(B_true[1]), rel=0.1)
    assert np.mean(est_y) == pytest.approx(np.linalg.norm(B_true[2]), rel=0.15)
    ts = E.tuning_strength(fit, "x", "y")
    assert ts == pytest.approx(0.5 * (est_x[-1] + est_y[-1]))


def test_cv_correlation_between_models_is_not_attenuated():
    """The uni vs bi contexts have independent data and a true correlation of 0.5."""
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, N_CH)
    b = 0.5 * a + np.sqrt(1 - 0.25) * rng.normal(0, 1, N_CH)
    true_r = np.corrcoef(a, b)[0, 1]
    Ba = np.zeros((3, N_CH)); Ba[1] = 0.3 * a
    Bb = np.zeros((3, N_CH)); Bb[1] = 0.3 * b
    naive, cv = [], []
    for rep in range(20):
        fa = E.cv_ols(*_simulate(Ba, 40, rng), seed=rep)
        fb = E.cv_ols(*_simulate(Bb, 40, rng), seed=rep + 100)
        naive.append(np.corrcoef(fa.full[1], fb.full[1])[0, 1])
        cv.append(E.cv_correlation(fa, 1, fb, 1))
    assert np.mean(naive) < true_r - 0.1  # noise attenuates the naive correlation
    assert np.mean(cv) == pytest.approx(true_r, abs=0.08)


def test_cv_correlation_within_model_x_vs_y():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, N_CH)
    B = np.zeros((3, N_CH)); B[1] = 0.4 * a; B[2] = -0.4 * a  # x and y anti-correlated
    rs = [E.cv_correlation(E.cv_ols(*_simulate(B, 40, rng), seed=s), 1, None, 2) for s in range(10)]
    assert np.mean(rs) == pytest.approx(-1.0, abs=0.1)


def test_laterality_axis_and_removal():
    rng = np.random.default_rng(4)
    n_trials = 60
    c_lat = np.repeat(np.where(np.arange(n_trials) % 2 == 0, 1.0, -1.0), BINS_PER_TRIAL)
    lat_true = rng.normal(0, 1, N_CH)
    X = np.c_[np.ones_like(c_lat), c_lat]
    Y = np.outer(c_lat, lat_true) + rng.normal(0, 0.5, (c_lat.size, N_CH))
    u = E.laterality_axis(X, Y, lat_col=1)
    assert abs(u @ lat_true / np.linalg.norm(lat_true)) > 0.99
    Y_rm = E.remove_axis(Y, u)
    # Refitting Eq. 4 on cleaned data gives b_lat ~ 0 (the paper's check)
    assert np.linalg.norm(E.ols(X, Y_rm)[1]) < 1e-10


def test_zero_regressor_column_is_harmless():
    """A uni-R-only fit with LH columns all zero gives LH coefficients of 0, not an error."""
    rng = np.random.default_rng(5)
    B = np.zeros((3, N_CH)); B[1] = 1.0
    X, Y, g = _simulate(B, 20, rng)
    X5 = np.c_[X, np.zeros((len(X), 2))]
    fit = E.cv_ols(X5, Y, g)
    assert np.allclose(fit.full[3:], 0)
