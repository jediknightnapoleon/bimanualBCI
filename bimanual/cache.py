"""Helpers that make scripts safe to re-run after an interruption.

Purpose
-------
If a session is cut off (for example by hitting a usage limit), scripts
should resume cheaply instead of recomputing everything. Three tools:

* ``should_skip``  -- skip a step whose outputs already exist (unless forced).
* ``atomic_write_*`` -- write to a temp file and rename it, so a crash never
  leaves a half-written result that later looks "done".
* ``append_jsonl`` / ``read_jsonl`` -- append-only logs, used for example by
  the RNN hyperparameter search so finished trials are never redone.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def should_skip(outputs: Iterable[os.PathLike | str], force: bool = False) -> bool:
    """Return True if every output file exists and ``force`` is False.

    Args:
        outputs: paths the step would produce.
        force: set True (e.g. via a ``--force`` CLI flag) to always recompute.
    """
    if force:
        return False
    return all(Path(p).exists() for p in outputs)


def _atomic_replace(path: Path, write_fn) -> None:
    """Write via ``write_fn(tmp_path)`` then atomically move it onto ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        write_fn(Path(tmp))
        os.replace(tmp, path)  # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_json(path: os.PathLike | str, obj: Any) -> None:
    """Save ``obj`` as pretty-printed JSON; numpy scalars/arrays are converted."""

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    _atomic_replace(
        Path(path),
        lambda tmp: tmp.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8"),
    )


def atomic_write_npz(path: os.PathLike | str, **arrays: np.ndarray) -> None:
    """Save named arrays to a compressed ``.npz`` file atomically."""

    def write(tmp: Path):
        # np.savez appends ".npz" if missing, so write through a file handle.
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)

    _atomic_replace(Path(path), write)


def append_jsonl(path: os.PathLike | str, record: dict) -> None:
    """Append one JSON record as a line (flushed immediately)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: os.PathLike | str) -> list[dict]:
    """Read all records from a JSONL file; a truncated last line is ignored."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # partial line from an interrupted write
    return records
