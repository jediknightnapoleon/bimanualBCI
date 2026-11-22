"""Tests for bimanual.sbp (Phase 11, TX vs SBP on T5) on synthetic data."""

import h5py
import numpy as np

from bimanual import sbp as S
from bimanual.decoders import ridge_fit_predict


def test_gram_ridge_matches_closed_form_on_subsets():
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((500, 10)), rng.standard_normal((500, 2))
    Xte = rng.standard_normal((50, 10))
    gr = S.GramRidge(X, Y)
    s = np.array([1, 4, 7])
    np.testing.assert_allclose(gr.predict(Xte, s, 3.0), ridge_fit_predict(X[:, s], Y, Xte[:, s], 3.0), atol=1e-10)


def test_target_class_radial8():
    ang = np.arange(8) * np.pi / 4
    tgt = 400 * np.c_[np.cos(ang), np.sin(ang)]
    np.testing.assert_array_equal(S.target_class(tgt), np.arange(8))


def test_mute_modes_and_mask_counts():
    m = S.random_masks(48, 0.25, 20, seed=1)
    assert m.shape == (20, 48) and np.all(m.sum(1) == 12)
    z = np.ones((3, 5, 48))
    dead = -np.arange(48.0)
    assert np.all(S.mute(z, m[0], "zero")[..., m[0]] == 0) and np.all(S.mute(z, m[0], "zero")[..., ~m[0]] == 1)
    np.testing.assert_array_equal(S.mute(z, m[0], "dead", dead_z=dead)[0, 0, m[0]], dead[m[0]])
    out = S.mute(z, m[0], "noisy", rng=np.random.default_rng(0))
    assert out[..., m[0]].std() > 0.5 and np.all(out[..., ~m[0]] == 1)
    assert np.array_equal(S.mute(z, np.zeros(48, bool), "zero"), z)


def test_greedy_recovers_planted_channels_and_skips_duplicate():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2000, 12))
    X[:, 5] = X[:, 3] + 0.01 * rng.standard_normal(2000)  # near-duplicate of channel 3
    Y = (X[:, [3]] + X[:, [8]]) + 0.3 * rng.standard_normal((2000, 1))
    tr, te = slice(0, 1500), slice(1500, None)
    gr = S.GramRidge(X[tr], Y[tr])
    score = lambda s: S.mean_r(Y[te], gr.predict(X[te], s, 1.0))
    chosen, scores = S.greedy_select(score, np.arange(12), 3)
    assert set(chosen[:2]) in ({3, 8}, {5, 8})
    assert scores[1] > scores[0] and scores[2] - scores[1] < 0.01


def test_own_channel_residual_removes_planted_linear_part():
    rng = np.random.default_rng(3)
    ztx = rng.standard_normal((60, 40, 4))
    noise = 0.5 * rng.standard_normal(ztx.shape)
    zsbp = 0.8 * ztx + 0.3 * S.shift_bins(ztx, 2) + noise
    train = np.arange(60) < 40
    resid, pred, coefs = S.own_channel_residual(ztx, zsbp, train)
    test = ~train
    # the residual on test trials is (nearly) the planted independent noise
    assert np.corrcoef(resid[test].ravel(), noise[test].ravel())[0, 1] > 0.95
    np.testing.assert_allclose(coefs[:, 2], 0.8, atol=0.05)  # shift 0
    np.testing.assert_allclose(coefs[:, 3], 0.3, atol=0.05)  # shift +2 (past); shifts are (-4,-2,0,2,4)


def test_shift_and_norm():
    x = np.arange(10.0)[None, :, None]
    assert S.shift_bins(x, 2)[0, :, 0].tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert S.shift_bins(x, -2)[0, :, 0].tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 9, 9]
    raw = np.exp(np.random.default_rng(4).standard_normal((5, 20, 3)))
    n = S.Norm.fit(raw, log=True)
    z = n(raw)
    np.testing.assert_allclose(z.mean((0, 1)), 0, atol=1e-9)
    np.testing.assert_allclose(n.z_of_raw(raw[0, 0]), z[0, 0])


def test_paired_bootstrap_ci_covers_difference():
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal(300), rng.standard_normal(300) + 0.5
    res = S.paired_bootstrap(lambda i: a[i].mean(), lambda i: b[i].mean(), 300, n_boot=500)
    assert res["ci"][0] < res["diff"] < res["ci"][1] and res["ci"][0] > 0.2


def test_load_rt_on_synthetic_file(tmp_path):
    rng = np.random.default_rng(6)
    n, b, c = 6, 15, 4
    p = tmp_path / "toy.h5"
    with h5py.File(p, "w") as f:
        tx = rng.random((n, b, c)); tx[2, 3, 1] = np.nan
        f["spike_data_all"] = tx
        f["sbp_data_all"] = 100 + rng.random((n, b, c))
        f["cursor_data_all"] = rng.random((n, b, 2))
        f["target_data_all"] = np.tile([[400.0, 0.0]], (n, 1))
        f["block_id_data_all"] = np.array([7, 7, 8, 8, 9, 9.0])
        f["delay_data_all"] = np.full(n, 1.0)
        f["no_go_bool_all"] = np.zeros(n)
        f["t_data"] = np.arange(b) * 20 - 100
        f["keep_chans_all"] = np.array([3, 50, 100, 150])
    d = S.load_rt(p)
    assert d.tx.shape == (5, b, c) and d.n_dropped == 1
    assert d.array.tolist() == ["A", "A", "B", "B"]
    assert d.block.tolist() == [7, 7, 8, 9, 9]
