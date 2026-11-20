"""Tests for bimanual.blocksets and conditions.trial_directions."""

import numpy as np

from bimanual import blocksets as BS
from bimanual import conditions as C


def test_set_counts_per_scheme():
    n = {s: sum(len(BS.block_sets(d, s)) for d in BS.FIG2_DAYS) for s in BS.SCHEMES}
    assert n == {"adjacent_all": 25, "all_pairs_mixed": 38, "adjacent_mixed": 17}


def test_adjacent_all_includes_uni_only_blocks():
    assert BS.block_sets(1881, "adjacent_all")[:3] == [(1, 4), (4, 6), (6, 7)]
    assert (7, 9) in BS.block_sets(1881, "adjacent_mixed")


def test_trial_directions():
    Tp = np.zeros((4, 2)); Cp = np.zeros((4, 2))
    Tp[:, 0] = [3, 4, -2, 0]
    d = C.trial_directions(Tp, Cp, [0, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(d[0], [0.6, 0.8, 0, 0])  # uni-R: LH zero
    np.testing.assert_allclose(d[1], [0, 0, -1, 0])  # uni-L: RH zero
    np.testing.assert_allclose(d[2], [0.6, 0.8, -1, 0])
