"""R^2 (coefficient of determination) used for the R^2 rescoring of every decoding result."""

import numpy as np

from bimanual.decoders import pearson_per_dim, r2_per_dim


def test_r2_known_values():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((500, 3))
    assert np.allclose(r2_per_dim(y, y), 1.0)  # perfect prediction
    assert np.allclose(r2_per_dim(y, np.broadcast_to(y.mean(0), y.shape)), 0.0)  # the constant mean
    assert np.all(r2_per_dim(y, -y) < -2.9)  # worse than the mean: negative (exactly -3 here)


def test_r2_penalises_gain_and_offset_but_r_does_not():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((1000, 2))
    for yhat in (2 * y, 0.5 * y, y + 1.0):  # right shape, wrong gain or offset
        assert np.allclose(pearson_per_dim(y, yhat), 1.0)
        assert np.all(r2_per_dim(y, yhat) < 0.8)
    yc = y - y.mean(0)  # exact for zero-mean data: SSE = 0.25 * SST
    assert np.allclose(r2_per_dim(yc, 0.5 * yc), 0.75)


def test_r2_matches_sklearn_definition():
    rng = np.random.default_rng(2)
    y = rng.standard_normal((200, 4)) * [1, 2, 3, 4] + [0, 1, -1, 5]
    yhat = y + rng.standard_normal((200, 4))
    manual = [1 - np.sum((y[:, d] - yhat[:, d]) ** 2) / np.sum((y[:, d] - y[:, d].mean()) ** 2) for d in range(4)]
    assert np.allclose(r2_per_dim(y, yhat), manual)
