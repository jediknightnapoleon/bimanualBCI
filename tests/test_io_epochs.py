"""Tests for bimanual.io preprocessing and bimanual.epochs windowing (no data files needed)."""

import numpy as np
import pytest

from bimanual import epochs, io


def test_remove_block_means():
    X = np.r_[np.full((5, 2), 3.0), np.full((4, 2), 10.0)] + np.arange(9)[:, None]
    block = np.r_[np.zeros(5), np.ones(4)]
    out = io.remove_block_means(X, block)
    assert np.allclose(out[block == 0].mean(0), 0) and np.allclose(out[block == 1].mean(0), 0)


def test_zscore_by_block_constant_channel():
    X = np.c_[np.random.default_rng(0).normal(5, 2, 100), np.ones(100)]
    out = io.zscore_by_block(X, np.zeros(100))
    assert out[:, 0].std() == pytest.approx(1) and np.all(out[:, 1] == 0)


def test_extract_windows_inclusive_and_out_of_range():
    F = np.arange(100.0)[:, None]
    w = epochs.extract_windows(F, np.array([10, 95, 96]), (2, 4))
    assert w.shape == (3, 3, 1)
    np.testing.assert_array_equal(w[0, :, 0], [12, 13, 14])  # inclusive window
    np.testing.assert_array_equal(w[1, :, 0], [97, 98, 99])  # ends exactly at the last bin
    assert np.isnan(w[2]).all()  # 96 + 4 = 100 is past the end -> NaN


def test_feature_tensor_layout_and_skipped_slot():
    F = np.arange(200.0)[:, None] * np.array([[1.0, -1.0]])  # (T=200, N=2)
    codes = np.array([[1, 2], [1, 2], [2, 1], [1, 2]])
    events = np.array([10, 195, 50, 70])  # trial 1's window (195..204) runs out of range
    t = epochs.feature_tensor(F, codes, events, (0, 9))
    assert t.shape == (2, 2, 2, 10, 3)  # N x C1 x C2 x W x maxRep
    np.testing.assert_array_equal(t[0, 0, 1, :, 0], np.arange(10, 20))
    assert np.isnan(t[0, 0, 1, :, 1]).all()  # skipped trial keeps an empty slot
    np.testing.assert_array_equal(t[1, 0, 1, :, 2], -np.arange(70, 80))
    np.testing.assert_array_equal(t[0, 1, 0, :, 0], np.arange(50, 60))
