"""File-based job queue so several GPU worker processes can share one list of training runs (Phase 10).

Why
---
One laptop GPU (RTX 3050, 4 GB) trains a 512-unit LSTM at well under full use, so running 2-3
training processes at once raises throughput. The queue keeps them from duplicating work and
lets a stopped run be resumed by simply starting the workers again.

Layout (all under one root directory, e.g. results/ext_feas/)
----------------------------------------------------------------
* ``jobs.json``           {"meta": {...}, "jobs": [job, ...]}; a job is a dict with at least
                          ``run_id`` and ``priority`` (lower runs first) and optional ``requires``:
                          a list of [path, dotted_key_or_null] that must exist before it can run
                          (e.g. ["results/ext_feas/selection.json", "sets.greedy_swap"]).
                          The file may be rewritten while workers run (e.g. to append jobs);
                          workers re-read it before every claim.
* ``trials_w{k}.jsonl``   one file per worker, one line per finished run (so no two processes
                          append to the same file). ``records()`` merges them.
* ``claims/<run_id>.lock`` created with O_EXCL by the worker that takes a job, removed when the
                          record is written. A crashed worker leaves its lock behind; the driver
                          clears all locks before it starts a new set of workers (``clear_claims``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from bimanual.cache import atomic_write_json, read_jsonl

# Paths in ``requires`` are relative to the repository root.
REPO = Path(__file__).resolve().parents[1]


class Queue:
    def __init__(self, root: os.PathLike | str):
        self.root = Path(root)
        self.claims = self.root / "claims"
        self.jobs_path = self.root / "jobs.json"

    # ------------------------------------------------------------------ job list
    def write_jobs(self, jobs: list[dict], meta: dict | None = None) -> None:
        ids = [j["run_id"] for j in jobs]
        assert len(ids) == len(set(ids)), "duplicate run_ids in the job list"
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.jobs_path, dict(meta=meta or {}, jobs=jobs))

    def read(self) -> dict:
        if not self.jobs_path.exists():
            return dict(meta={}, jobs=[])
        return json.loads(self.jobs_path.read_text())

    def jobs(self) -> list[dict]:
        return sorted(self.read()["jobs"], key=lambda j: (j["priority"], j.get("order", 0)))

    # ------------------------------------------------------------------ results
    def trials_path(self, worker: int) -> Path:
        return self.root / f"trials_w{worker}.jsonl"

    def records(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("trials_w*.jsonl")):
            out += read_jsonl(p)
        return out

    def done(self) -> set[str]:
        return {r["run_id"] for r in self.records()}

    # ------------------------------------------------------------------ prerequisites
    @staticmethod
    def ready(job: dict) -> bool:
        """True if every [path, key] in ``job['requires']`` exists (key = dotted path in the JSON)."""
        for path, key in job.get("requires", []):
            p = Path(path) if Path(path).is_absolute() else REPO / path
            if not p.exists():
                return False
            if key:
                try:
                    obj = json.loads(p.read_text())
                except (json.JSONDecodeError, OSError):  # being written; try again later
                    return False
                for k in key.split("."):
                    if not isinstance(obj, dict) or k not in obj:
                        return False
                    obj = obj[k]
        return True

    # ------------------------------------------------------------------ claims
    def claim(self, run_id: str, worker: int) -> bool:
        self.claims.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.claims / f"{run_id}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump(dict(worker=worker, pid=os.getpid(), t=time.time()), f)
        return True

    def release(self, run_id: str) -> None:
        try:
            (self.claims / f"{run_id}.lock").unlink()
        except FileNotFoundError:
            pass

    def claimed(self) -> set[str]:
        return {p.stem for p in self.claims.glob("*.lock")} if self.claims.exists() else set()

    def clear_claims(self) -> int:
        """Remove every lock (call only when no worker is running). Returns how many were removed."""
        n = 0
        for p in self.claims.glob("*.lock") if self.claims.exists() else []:
            p.unlink(); n += 1
        return n

    # ------------------------------------------------------------------ scheduling
    def next_job(self, worker: int) -> tuple[dict | None, str]:
        """Claim the highest-priority runnable job.

        Returns (job, "ok"), or (None, "blocked") if jobs remain but all wait on prerequisites or
        on other workers, or (None, "empty") if nothing is left.
        """
        done, claimed = self.done(), self.claimed()
        waiting = False
        for job in self.jobs():
            rid = job["run_id"]
            if rid in done:
                continue
            if rid in claimed or not self.ready(job):
                waiting = True
                continue
            if self.claim(rid, worker):
                if rid in self.done():  # finished between our two reads
                    self.release(rid)
                    continue
                return job, "ok"
            waiting = True
        return None, ("blocked" if waiting else "empty")

    def remaining(self) -> list[dict]:
        done = self.done()
        return [j for j in self.jobs() if j["run_id"] not in done]
