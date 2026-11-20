"""Tests for bimanual.decoders on synthetic data where the right answer is known."""

import numpy as np

from bimanual import decoders as Dc


def test_stratified_folds_balanced():
    tt = np.repeat([1, 2, 3], [50, 40, 10])
    f = Dc.stratified_trial_folds(tt, 5)
    for t in (1, 2, 3):
        counts = np.bincount(f[tt == t], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_pearson_per_dim():
    y = np.c_[np.arange(10.0), np.arange(10.0)]
    np.testing.assert_allclose(Dc.pearson_per_dim(y, np.c_[2 * y[:, 0] + 1, -y[:, 1]]), [1, -1])


def _gated_data(rng, n_trials=200, bins=10, N=30):
    """Two 'hands' with IDENTICAL velocity tuning but a laterality offset.

    A linear decoder can't tell which hand moved; a nonlinear one can gate
    its outputs on the laterality dimension.
    """
    W = rng.normal(size=(2, N))
    lat_vec = rng.normal(size=N) * 3
    X, Y, g, tt = [], [], [], []
    for t in range(n_trials):
        hand = t % 2
        v = rng.normal(size=(bins, 2))
        X.append(v @ W + (1 if hand == 0 else -1) * lat_vec + rng.normal(0, 0.3, (bins, N)))
        y = np.zeros((bins, 4)); y[:, 2 * hand : 2 * hand + 2] = v
        Y.append(y); g += [t] * bins; tt.append(hand)
    return np.concatenate(X), np.concatenate(Y), np.array(g), np.array(tt)


def test_ffn_beats_ridge_when_hands_share_tuning():
    rng = np.random.default_rng(0)
    X, Y, g, tt = _gated_data(rng)
    folds = Dc.stratified_trial_folds(tt, 5)
    r_ridge = Dc.pearson_per_dim(Y, Dc.cross_validate(X, Y, g, folds, "ridge", alpha=1.0)).mean()
    r_ffn = Dc.pearson_per_dim(Y, Dc.cross_validate(X, Y, g, folds, "ffn")).mean()
    assert r_ffn > r_ridge + 0.15
