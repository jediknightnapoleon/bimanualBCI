"""Tests for bimanual.proxy: intention targets on a hand-built closed-loop session, and the metrics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from bimanual import proxy as PX


def _session():
    """Two CL blocks, one trial each: a successful uni-R trial (block 1), a timed-out uni-L trial (block 2)."""
    n = 1400
    Cp = np.zeros((4, n)); Tp = np.zeros((4, n))
    block = np.r_[np.full(700, 1), np.full(700, 2)]
    # trial 1: R target at (+100, 0); successful, move 200 bins (last 25 = dwell)
    Tp[0, 100:400] = 100.0
    # trial 2: L target at (0, +100); timeout (499 bins)
    Tp[3, 750:1300] = 100.0
    trials = pd.DataFrame(dict(
        start=[50, 720], go=[100, 750], move_end=[300, 1249], move_type=[1, 2], move_bins=[200, 499],
        block=[1, 2], kind=["CL", "CL"], variant=["raw", "augmented"]))
    return SimpleNamespace(Cp=Cp, Tp=Tp, block=block, trials=trials)


def test_intention_target_windows():
    s = _session()
    pdat = PX.build(s)
    V = pdat.V
    # trial 1: zero before go + RT, unit +x from go+10 to move_end-25, zero in the dwell
    assert np.all(V[:110] == 0)
    np.testing.assert_allclose(V[110:275, 0], 1.0); assert np.all(V[110:275, 1:] == 0)
    assert np.all(V[275:300] == 0)
    # trial 2 (timeout): unit +y for the LEFT hand until move_end, idle right hand stays 0
    np.testing.assert_allclose(V[760:1249, 3], 1.0); assert np.all(V[760:1249, :2] == 0)
    assert pdat.segments == {1: (0, 700), 2: (700, 1400)} and pdat.variant == {1: "raw", 2: "augmented"}


def test_metrics_for_a_perfect_and_a_fading_decoder():
    s = _session()
    pdat = PX.build(s)
    perfect = PX.score(pdat.V.copy(), pdat)
    assert np.nanmin(perfect["all"]["r"]) > 0.999  # R y and L x are always 0 in this toy (r undefined)
    assert perfect["all"]["ang_err_deg"] < 1e-6 and perfect["all"]["jitter"] == 0.0
    # only the timed-out trial is still in pursuit at 3 s; a constant output sustains (= 1)
    assert perfect["augmented"]["n_sustain_trials"] == 1
    np.testing.assert_allclose(perfect["augmented"]["sustain"], 1.0)
    # a decoder whose output fades to 0 after 1 s gives sustain = 0
    fade = pdat.V.copy(); fade[750 + 50:1249] = 0
    assert PX.score(fade, pdat)["augmented"]["sustain"] == 0.0
    # a decoder that points 90 deg off gives a 90-deg angular error
    rot = np.zeros_like(pdat.V); rot[:, 1] = pdat.V[:, 0]; rot[:, 2] = -pdat.V[:, 3]
    np.testing.assert_allclose(PX.score(rot, pdat)["all"]["ang_err_deg"], 90.0)


def test_decode_blocks_starts_each_block_fresh():
    s = _session()
    pdat = PX.build(s)
    calls = []
    out = PX.decode_blocks(lambda x: (calls.append(len(x)), np.ones((len(x), 4)))[1], np.zeros((1400, 3)), pdat)
    assert calls == [700, 700] and out.shape == (1400, 4)
