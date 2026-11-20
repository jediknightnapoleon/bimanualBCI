"""Tests for bimanual.cache (re-run safety helpers)."""

import numpy as np

from bimanual import cache


def test_should_skip(tmp_path):
    out = tmp_path / "a.json"
    assert not cache.should_skip([out])
    cache.atomic_write_json(out, {"x": np.float64(1.5), "arr": np.arange(3)})
    assert cache.should_skip([out])
    assert not cache.should_skip([out], force=True)


def test_jsonl_roundtrip_ignores_partial_line(tmp_path):
    p = tmp_path / "log.jsonl"
    cache.append_jsonl(p, {"trial": 0})
    cache.append_jsonl(p, {"trial": 1})
    with open(p, "a") as f:
        f.write('{"trial": 2')  # simulated interrupted write
    assert [r["trial"] for r in cache.read_jsonl(p)] == [0, 1]


def test_atomic_npz(tmp_path):
    p = tmp_path / "x.npz"
    cache.atomic_write_npz(p, a=np.ones(4))
    assert np.load(p)["a"].sum() == 4
