"""Tests for the file-based job queue (bimanual/jobs.py, Phase 10)."""

import json

from bimanual.cache import append_jsonl
from bimanual.jobs import Queue


def _jobs():
    return [dict(run_id="a", priority=2), dict(run_id="b", priority=1),
            dict(run_id="c", priority=3, requires=[["prereq.json", "sets.x"]])]


def test_claims_are_exclusive_and_priority_ordered(tmp_path):
    q = Queue(tmp_path)
    q.write_jobs(_jobs())
    j1, s1 = q.next_job(0)
    j2, s2 = q.next_job(1)
    assert (j1["run_id"], j2["run_id"]) == ("b", "a") and s1 == s2 == "ok"
    assert not q.claim("b", 1)  # already claimed by worker 0
    j3, s3 = q.next_job(2)  # only "c" left, and its prerequisite is missing
    assert j3 is None and s3 == "blocked"


def test_prerequisite_with_key_and_done_records(tmp_path, monkeypatch):
    import bimanual.jobs as J
    monkeypatch.setattr(J, "REPO", tmp_path)
    q = Queue(tmp_path / "q")
    q.write_jobs(_jobs())
    for rid in ("a", "b"):
        append_jsonl(q.trials_path(0), dict(run_id=rid))
    (tmp_path / "prereq.json").write_text(json.dumps({"sets": {}}))
    assert q.next_job(0) == (None, "blocked")  # file exists, key does not
    (tmp_path / "prereq.json").write_text(json.dumps({"sets": {"x": [1]}}))
    job, st = q.next_job(0)
    assert job["run_id"] == "c" and st == "ok"
    append_jsonl(q.trials_path(1), dict(run_id="c"))  # another worker's file counts too
    q.release("c")
    assert q.next_job(0) == (None, "empty")


def test_stale_claims_are_cleared(tmp_path):
    q = Queue(tmp_path)
    q.write_jobs(_jobs()[:2])
    q.next_job(0)  # claims "b", then the worker "crashes"
    assert q.claimed() == {"b"}
    assert q.clear_claims() == 1 and q.claimed() == set()
    assert q.next_job(1)[0]["run_id"] == "b"


def test_duplicate_run_ids_rejected(tmp_path):
    import pytest
    with pytest.raises(AssertionError):
        Queue(tmp_path).write_jobs([dict(run_id="a", priority=1), dict(run_id="a", priority=2)])
