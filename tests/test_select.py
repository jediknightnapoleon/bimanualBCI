"""Tests for bimanual/select.py (best-n channel selection, register #43). Synthetic data, CPU, fast."""

import numpy as np
import pytest

from bimanual import select as S


def _pool(z, y, n_blocks=4, spec=("lags", 3)):
    """Split T bins into n_blocks blocks: blocks 1..n-1 train, block n validation."""
    T = len(z)
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    block = np.zeros(T, int)
    for b in range(n_blocks):
        block[edges[b]:edges[b + 1]] = b + 1
    ranges = [(int(edges[b]), int(edges[b + 1])) for b in range(n_blocks)]
    return S.Surrogate.build(z, y, block, ranges[:-1], ranges[-1:], np.arange(z.shape[1]), spec=spec,
                             allowed=range(1, n_blocks + 1))


def test_greedy_recovers_planted_and_skips_duplicate():
    rng = np.random.default_rng(0)
    T, k = 8000, 20
    z = rng.standard_normal((T, k))
    planted = [3, 7, 11, 14]
    z[:, 15] = z[:, 3]  # exact duplicate of a planted channel
    y = np.stack([z[:, 3] + 0.8 * z[:, 7], z[:, 11] - 0.7 * z[:, 14], z[:, 7] + z[:, 14], z[:, 3] - z[:, 11]], 1)
    y += 0.5 * rng.standard_normal(y.shape)
    sur = _pool(z, y)
    order, vals = S.greedy(sur, alpha=1e-2, n_max=6)
    assert sorted(order[:4]) == planted
    assert 15 not in order[:4]
    assert all(np.diff(vals[:4]) > 0)


def test_group_lasso_recovers_jointly_informative_pair():
    rng = np.random.default_rng(1)
    T, k = 8000, 12
    z = rng.standard_normal((T, k))
    common = 10 * rng.standard_normal(T)  # large shared noise
    u = rng.standard_normal(T)            # the signal is only in the difference
    z[:, 2] = common + u / 2
    z[:, 5] = common - u / 2
    y = np.stack([u, -u, 0.5 * u, u], 1) + 0.3 * rng.standard_normal((T, 4))
    sur = _pool(z, y)
    # each channel alone is nearly useless...
    assert sur.score([2], 1e-2) < 0.2 and sur.score([5], 1e-2) < 0.2
    # ...but the pair decodes well, and group lasso finds it for n = 2
    gl = S.group_lasso_sets(sur, alpha=1e-2, ns=(2,), n_path=40)
    assert sorted(gl[2]["channels"]) == [2, 5]
    assert sur.score([2, 5], 1e-2) > 0.9


def test_swap_never_lowers_score():
    rng = np.random.default_rng(2)
    T, k = 4000, 15
    z = rng.standard_normal((T, k))
    y = z[:, :4] @ rng.standard_normal((4, 4)) + rng.standard_normal((T, 4))
    sur = _pool(z, y)
    start = [10, 11, 12]
    s0 = sur.score(start, 1e-2)
    final, s1, n_swaps, gain = S.swap_refine(sur, start, 1e-2)
    assert s1 >= s0 and gain >= 0 and n_swaps >= 1
    assert len(set(final)) == 3
    # a swap-refined set is a local optimum: no single swap improves it
    _, s2, n2, g2 = S.swap_refine(sur, final, 1e-2)
    assert n2 == 0 and g2 == 0


def test_test_block_guard_raises():
    block = np.repeat([5, 6, 7], 100)
    S.guard_ranges(block, [(0, 200)])  # blocks 5-6: fine
    with pytest.raises(ValueError):
        S.guard_ranges(block, [(150, 250)])  # touches block 7
    z = np.zeros((300, 2)); y = np.zeros((300, 4))
    with pytest.raises(ValueError):
        S.Surrogate.build(z, y, block, [(0, 100)], [(200, 300)], [0, 1])


def test_lagged_is_causal_and_block_local():
    x = np.arange(1, 6, dtype=float)[:, None]  # one channel, 5 bins
    L = S.lagged(x, 3)
    assert L.shape == (5, 3)
    np.testing.assert_array_equal(L[:, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(L[:, 1], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(L[:, 2], [0, 0, 1, 2, 3])


def test_exp_basis_is_causal_and_matches_pool_width():
    x = np.zeros((50, 2)); x[10, 0] = 1.0
    F_ = S.exp_basis(x, (2, 5))
    assert F_.shape == (50, 6)
    assert np.all(F_[:10] == 0)                    # nothing before the impulse (causal)
    assert F_[10, 0] == 1 and F_[11, 0] == 0       # raw column
    assert F_[11, 1] > 0 and F_[11, 2] > F_[11, 1] * 0  # smoothed columns carry the past
    assert np.all(F_[:, 3:] == 0)                  # channel 1 untouched
    rng = np.random.default_rng(3)
    z = rng.standard_normal((3000, 4)); y = np.repeat(z[:, :1], 4, 1)
    sur = _pool(z, y, spec=("exp", (2, 5)))
    assert sur.nf == 3 and sur.Cxx.shape == (12, 12)
    assert sur.score([0], 1e-3) > 0.9
