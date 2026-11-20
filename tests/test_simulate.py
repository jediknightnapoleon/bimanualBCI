"""Tests for bimanual.simulate."""

import numpy as np
import pytest

from bimanual import simulate as S

MAGS = dict(Rx=100.0, Ry=80.0, Lx=60.0, Ly=50.0, lat=70.0)


def test_fsnr_recovers_known_gain_and_noise():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(20000, 2))
    yhat = y @ np.array([[0.5, 0.1], [0.0, 0.7]]).T + 0.3 + rng.normal(0, 0.2, y.shape)
    assert S.fsnr(y, yhat) == pytest.approx(0.5 * (0.5 + 0.7) / 0.2, rel=0.03)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 1.0])
def test_population_has_requested_correlation_and_magnitudes(rho):
    rng = np.random.default_rng(1)
    p = S.make_population(np.full(192, 20.0), MAGS, rho, lat_scale=2.0, rng=rng)
    assert np.corrcoef(p.bRx, p.bLx)[0, 1] == pytest.approx(rho, abs=1e-6)
    assert np.linalg.norm(p.bLy) == pytest.approx(50.0)
    assert np.linalg.norm(p.blat) == pytest.approx(140.0)


def test_simulate_trials_shapes_targets_and_clipping():
    rng = np.random.default_rng(2)
    p = S.make_population(np.full(50, 1.0), MAGS, 0.3, 1.0, rng)
    X, Y, g, hand = S.simulate_trials(p, n_per_hand=30, bins_per_trial=5, sigma=10.0, rng=rng)
    assert X.shape == (300, 50) and Y.shape == (300, 4) and len(hand) == 60
    assert X.min() >= 0  # negative rates clipped
    yR = Y[np.repeat(hand, 5) == 1]
    assert np.all(yR[:, 2:] == 0) and np.allclose(np.hypot(yR[:, 0], yR[:, 1]), 1)
    ang = np.degrees(np.arctan2(yR[:, 1], yR[:, 0])) % 45
    assert np.allclose(np.minimum(ang, 45 - ang), 0, atol=1e-9)  # 8 wedge directions only
