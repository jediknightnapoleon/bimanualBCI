"""Tests for bimanual.targets and bimanual.augment (temporal fracturing)."""

import numpy as np
import pytest

from bimanual import augment as A
from bimanual import targets as T

# ----------------------------------------------------------------------------- targets


def test_minjerk_and_saturated_profiles():
    mj = T.minjerk_speed(50)
    assert mj.max() == pytest.approx(1.0) and np.argmax(mj) in (24, 25)
    sat = T.saturated_speed(50, sat=0.5)
    assert sat.max() == pytest.approx(1.0)
    assert (sat == 1.0).sum() > 10  # has a plateau
    np.testing.assert_allclose(T.saturated_speed(50, sat=1.0), mj)  # sat=1 -> plain bell
    hold = T.saturated_speed(50, sat=0.5, hold_to_end=True)
    assert hold[-1] == 1.0 and hold[0] < 1.0


def test_build_targets_timing_and_idle_hand():
    tr = [
        T.TrialSpec(start_bin=0, go_bin=50, move_end_bin=150, end_bin=200, move_type=1,
                    dir_R=np.array([1.0, 0.0]), dir_L=np.array([0.0, 1.0]), quadrant_R="E"),
        T.TrialSpec(start_bin=200, go_bin=250, move_end_bin=350, end_bin=400, move_type=3,
                    dir_R=np.array([0.0, -1.0]), dir_L=np.array([-1.0, 0.0]),
                    quadrant_R="S", quadrant_L="W"),
    ]
    vel, ctx, strat = T.build_targets(400, tr)
    # 200-ms reaction time: nothing before go + 10 bins
    assert np.all(vel[:60] == 0) and np.all(ctx[:60] == T.CTX_NONE)
    assert ctx[60] == T.CTX_UNI_R and vel[60:150, 0].max() == pytest.approx(1.0)
    assert np.all(vel[:, 2:4][:200] == 0)  # uni-R trial: left hand idle
    assert strat[10] == "delay_uniR" and strat[100] == ("uniR", "E", None)
    assert strat[160] == "rest" and ctx[160] == T.CTX_NONE
    # bimanual trial: both hands move, in their own directions
    assert ctx[300] == T.CTX_BI
    assert vel[300, 1] < 0 and vel[300, 2] < 0 and strat[300] == ("bi", "S", "W")


# ----------------------------------------------------------------------------- warping


@pytest.mark.parametrize("factor", [0.5, 0.73, 1.0, 1.6, 2.0])
def test_time_warp_length_and_endpoints(factor):
    x = np.arange(20, dtype=float)[:, None] * np.ones((1, 3))
    y = A.time_warp(x, factor)
    assert y.shape == (round(20 * factor), 3)
    assert y[0, 0] == 0 and y[-1, 0] == 19
    assert np.all(np.diff(y[:, 0]) > 0)  # linear ramp stays a monotone ramp


def test_nearest_warp_keeps_label_set():
    lab = np.array([0] * 5 + [2] * 5)
    out = A.nearest_warp(lab, 1.7)
    assert set(out) == {0, 2} and len(out) == 17


# ----------------------------------------------------------------------------- sampler


def _toy_recording(n_trials=40, trial_len=100, n_ch=6, seed=0):
    """Toy data to check alignment: channel 0 = 10**context, channel 1 = trial ID.

    10**label (1, 10, 100, 1000) is used so that an interpolated blend of two
    labels (e.g. 50.5) can never be mistaken for a valid label value.
    """
    rng = np.random.default_rng(seed)
    n = n_trials * trial_len
    ctx = np.zeros(n, int)
    strat = np.full(n, "rest", object)
    for k in range(n_trials):
        a = k * trial_len
        mt = 1 + k % 3
        ctx[a + 40 : a + 80] = mt
        strat[a : a + 40] = f"delay{mt}"
        strat[a + 40 : a + 80] = f"move{mt}"
    X = rng.normal(size=(n, n_ch))
    X[:, 0] = 10.0**ctx
    X[:, 1] = np.repeat(np.arange(n_trials), trial_len)
    vel = np.repeat(ctx[:, None].astype(float), 4, axis=1)
    bounds = np.array([[k * trial_len, (k + 1) * trial_len] for k in range(n_trials)])
    return X, vel, ctx, strat, bounds


def test_snippets_shape_and_label_alignment():
    X, vel, ctx, strat, bounds = _toy_recording()
    s = A.SnippetSampler(X, vel, ctx, strat, bounds, seed=1).make_snippets(20)
    assert s.X.shape == (20, 500, 6) and s.vel.shape == (20, 500, 4) and s.context.shape == (20, 500)
    # Wherever interpolated channel 0 is exactly a label code (not a blend at a label edge),
    # the nearest-neighbor context label must match it.
    codes = 10.0 ** np.arange(4)
    exact = np.isclose(s.X[..., 0][..., None], codes, rtol=1e-6).any(-1)
    assert exact.mean() > 0.8
    np.testing.assert_array_equal(np.rint(np.log10(s.X[..., 0][exact])), s.context[exact])
    # velocity (= context here) is warped with exactly the same time base as the features
    np.testing.assert_allclose(10.0 ** s.vel[..., 0][exact], s.X[..., 0][exact], rtol=1e-5)


def test_val_pool_never_leaks_into_train_snippets():
    X, vel, ctx, strat, bounds = _toy_recording()
    train, val = A.split_segments_by_trial(bounds, val_frac=0.2, seed=3)
    val_ids = {a // 100 for a, _ in val}
    s = A.SnippetSampler(X, vel, ctx, strat, train, seed=2).make_snippets(30)
    # channel 1 = trial ID (interpolated only within a clip, so the values stay integers)
    used = set(np.unique(np.round(s.X[..., 1]).astype(int)))
    assert used.isdisjoint(val_ids)
    assert len(val) == 8 and len(train) == 32


def test_strata_are_balanced():
    X, vel, ctx, strat, bounds = _toy_recording()
    sampler = A.SnippetSampler(X, vel, ctx, strat, bounds, seed=4)
    picks = [sampler.strata[sampler.rng.integers(len(sampler.strata))] for _ in range(6000)]
    counts = np.array([picks.count(s) for s in sampler.strata])
    # "rest" fills 20% of the bins but is picked about 1/7 of the time, like every stratum
    assert len(sampler.strata) == 7
    assert counts.min() > 0.8 * counts.mean()


def test_raw_snippets_preserve_order():
    X, vel, ctx, strat, bounds = _toy_recording()
    r = A.raw_snippets(X, vel, ctx, bounds[:12])
    assert r.X.shape == (2, 500, 6)
    np.testing.assert_array_equal(r.context[0], ctx[:500])
