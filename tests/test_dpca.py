"""Tests for bimanual.dpca_analysis (marginalization and cross-validated variance)."""

import numpy as np
import pytest

from bimanual import dpca_analysis as D


def test_marginals_are_orthogonal_and_sum_to_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2, 4, 10))
    M = D.marginalize(X)
    Xc = X - X.mean(axis=(1, 2, 3), keepdims=True)
    np.testing.assert_allclose(sum(M.values()), Xc, atol=1e-12)
    names = list(M)
    for i in range(4):
        for j in range(i + 1, 4):
            assert abs(np.sum(M[names[i]] * M[names[j]])) < 1e-9
    assert sum(D.marginal_variances(X).values()) == pytest.approx(np.sum(Xc**2))


def _simulate(rng, lat_gain, dir_gain, noise, n_per=30, N=40, T=10):
    """Trials with known laterality / direction signals plus a shared time course."""
    lat_vec, time_vec = rng.normal(size=N), rng.normal(size=N)
    dir_vecs = rng.normal(size=(4, N))
    ramp = np.linspace(-1, 1, T)
    trials, lat, dirs = [], [], []
    for l in range(2):
        for d in range(4):
            mean = ((2 * l - 1) * lat_gain * lat_vec + dir_gain * dir_vecs[d])[None, :] + ramp[:, None] * time_vec
            trials.append(mean[None] + rng.normal(0, noise, (n_per, T, N)))
            lat += [l] * n_per; dirs += [d] * n_per
    return np.concatenate(trials), np.array(lat), np.array(dirs)


def test_cv_variance_removes_noise_bias():
    """No laterality/direction signal: naive variance is clearly > 0, the CV estimate averages ~0.

    A single dataset's CV estimate scatters around 0 (sd ~10 here), so the
    bias is checked as a mean over 12 simulated datasets.
    """
    rng = np.random.default_rng(1)
    naive, cv = [], []
    for _ in range(12):
        tr, lat, dirs = _simulate(rng, lat_gain=0.0, dir_gain=0.0, noise=3.0)
        naive.append(D.marginal_variances(D.condition_average(tr, lat, dirs))["laterality"])
        cv.append(D.cv_marginal_variances(tr, lat, dirs, n_splits=10)["laterality"])
    assert np.mean(naive) > 50  # noise alone creates spurious "laterality" variance
    assert abs(np.mean(cv)) < 4 * np.std(cv) / np.sqrt(len(cv))  # unbiased: mean within 4 SE of 0
    assert abs(np.mean(cv)) < 0.1 * np.mean(naive)


def test_cv_fractions_rank_known_signals():
    rng = np.random.default_rng(2)
    tr, lat, dirs = _simulate(rng, lat_gain=1.0, dir_gain=0.5, noise=3.0)
    f = D.fractions(D.cv_marginal_variances(tr, lat, dirs, n_splits=30))
    assert f["laterality"] > f["direction"] > f["interaction"]
    assert f["interaction"] < 0.05
