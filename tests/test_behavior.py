"""Tests for bimanual.behavior and bimanual.stats on hand-built trajectories."""

import numpy as np
import pytest

from bimanual import behavior as B
from bimanual.stats import bootstrap_ci

N = 700  # 14 s of 20-ms bins
GO = 50  # go cue at bin 50 (0-based)


def _still_trial():
    """Both cursors at the origin; RH target at (10, 0), LH target at (-10, 0)."""
    Cp = np.zeros((4, N))
    Tp = np.zeros((4, N))
    Tp[0, :] = 10.0
    Tp[2, :] = -10.0
    return Cp, Tp


def _move_into_target(Cp, rows, xy, from_bin):
    Cp[rows, from_bin:] = np.asarray(xy, float)[:, None]


def test_tta_counts_full_dwell():
    Cp, Tp = _still_trial()
    _move_into_target(Cp, B.HAND_ROWS["R"], (10, 0), GO + 40)  # RH in at +0.8 s
    _move_into_target(Cp, B.HAND_ROWS["L"], (-10, 0), GO + 60)  # LH in at +1.2 s
    # Both in from +1.2 s; the dwell completes at +1.7 s.
    assert B.time_to_acquisition(Cp, Tp, GO, radius_R=1) == pytest.approx(1.7)
    assert B.time_to_acquisition(Cp, Tp, GO, radius_R=1, include_dwell=False) == pytest.approx(1.2)


def test_tta_dwell_resets_when_leaving_target():
    Cp, Tp = _still_trial()
    _move_into_target(Cp, B.HAND_ROWS["R"], (10, 0), GO + 10)
    _move_into_target(Cp, B.HAND_ROWS["L"], (-10, 0), GO + 10)
    Cp[0, GO + 30] = 0.0  # RH leaves for one bin after 0.4 s of dwell
    # The dwell restarts at bin GO+31 and completes 25 bins later.
    assert B.time_to_acquisition(Cp, Tp, GO, radius_R=1) == pytest.approx((31 + 25) * 0.02)


def test_tta_timeout_is_failure():
    Cp, Tp = _still_trial()
    # Acquire only after 10 s -> failure
    _move_into_target(Cp, B.HAND_ROWS["R"], (10, 0), GO + 510)
    _move_into_target(Cp, B.HAND_ROWS["L"], (-10, 0), GO + 510)
    assert np.isnan(B.time_to_acquisition(Cp, Tp, GO, radius_R=1))


def test_tta_respects_end_bin():
    Cp, Tp = _still_trial()
    _move_into_target(Cp, B.HAND_ROWS["R"], (10, 0), GO)
    _move_into_target(Cp, B.HAND_ROWS["L"], (-10, 0), GO)
    assert np.isnan(B.time_to_acquisition(Cp, Tp, GO, radius_R=1, end_bin=GO + 10))


def test_angular_error_straight_and_tilted():
    Cp, Tp = _still_trial()
    rows = B.HAND_ROWS["R"]
    # RH moves straight toward its target (+x): error 0
    Cp[0, GO:] = np.arange(N - GO) * 0.1
    assert B.angular_error(Cp, Tp, GO, "R") == pytest.approx(0.0)
    # RH moves at 30 deg off the target direction: error 30
    t = np.arange(N - GO) * 0.1
    Cp[rows, GO:] = np.vstack([t * np.cos(np.radians(30)), t * np.sin(np.radians(30))])
    assert B.angular_error(Cp, Tp, GO, "R") == pytest.approx(30.0)


def test_angular_error_wraps_and_handles_no_movement():
    Cp, Tp = _still_trial()
    # LH target is at 180 deg; moving at -170 deg (= 190 deg) is a 10-deg error, not 350
    t = np.arange(N - GO) * 0.1
    a = np.radians(-170)
    Cp[2:4, GO:] = np.vstack([t * np.cos(a), t * np.sin(a)])
    assert B.angular_error(Cp, Tp, GO, "L") == pytest.approx(10.0)
    # The RH never moves -> NaN
    assert np.isnan(B.angular_error(Cp, Tp, GO, "R"))


def test_speed_profile():
    Cp = np.zeros((4, N))
    Cp[0, :] = np.arange(N) * 0.2  # 0.2 units per bin = 10 units/s
    sp = B.speed_profile(Cp, GO, "R", n_bins=5, pre_bins=2)
    np.testing.assert_allclose(sp, 10.0)
    assert np.isnan(B.speed_profile(Cp, N - 2, "R", n_bins=3)[-1])


def test_bootstrap_ci_contains_mean_and_drops_nan():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(5, 1, 200), np.nan]
    est, lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < est < hi
    assert 4.7 < est < 5.3


def test_anova_fvaf():
    from bimanual.stats import anova_fvaf

    rng = np.random.default_rng(0)
    g = np.repeat([1, 2, 3, 4], 50).astype(float)
    tuned = np.array([0, 5, 0, -5])[g.astype(int) - 1] + rng.normal(0, 1, g.size)
    F, p, fvaf = anova_fvaf(tuned, g)
    assert p < 1e-5 and 0.8 < fvaf < 0.98
    F, p, fvaf = anova_fvaf(rng.normal(0, 1, g.size), g)
    assert p > 1e-3 and fvaf < 0.1
    assert anova_fvaf(np.ones(200), g)[1] == 1.0  # silent electrode
