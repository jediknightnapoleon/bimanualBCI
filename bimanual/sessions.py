"""Session catalogue: file names, trial days, paper figures, and per-block labels.

Sources
-------
* Dates, session numbers, trial days and figure use: Supplementary Table 2
  of the paper, extracted from the supplementary PDF text.
* Block conditions: ``Data/readme.txt`` (Dryad), section "Experiment Block
  Conditions". It is transcribed by hand below; ``scripts/01_inventory.py``
  checks every listed block against the blocks actually present in the data.

Block labels
------------
Each block maps to a :class:`Block` with these fields:

* ``kind``:     "OL"  = open-loop instructed delay (cursors move on their own),
                "CLR" = closed-loop recalibration (assisted, lock mode),
                "CL"  = closed-loop evaluation.
* ``decoder``:  "RNN" or "linear" (closed-loop blocks only).
* ``gain``:     decoder output gain where the readme gives one.
* ``variant``:  experiment-specific condition, e.g. "augmented" / "raw" (Fig 4),
                "sequential" / "simultaneous" (Fig 5b).
* ``uni_only``: True when the block had only unimanual trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


@dataclass(frozen=True)
class Block:
    kind: str  # "OL" | "CLR" | "CL"
    decoder: str | None = None  # "RNN" | "linear"
    gain: float | None = None
    variant: str | None = None
    uni_only: bool = False


@dataclass
class Session:
    date: str  # MM_DD_YYYY as in the file name
    session_num: int
    trial_day: int
    figures: str
    blocks: dict[int, Block] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return DATA_DIR / f"t5_{self.date}.mat"


def _blocks(ids, **kw) -> dict[int, Block]:
    """The same label for many block numbers."""
    return {int(b): Block(**kw) for b in ids}


def _gains(ids, gains, **kw) -> dict[int, Block]:
    """Per-block gains (readme lists them in the same order as the blocks)."""
    assert len(ids) == len(gains)
    return {int(b): Block(gain=g, **kw) for b, g in zip(ids, gains)}


r = range  # shorthand; readme ranges such as "12 to 28" are inclusive -> r(12, 29)

SESSIONS: list[Session] = [
    Session("06_02_2021", 320, 1750, "Fig 1B-C; SFig 1B,C,E", _blocks(r(12, 29), kind="OL")),
    Session("06_04_2021", 321, 1752, "Fig 5A-C (pilot calibration)", _blocks(r(2, 7), kind="OL")),
    Session("06_23_2021", 324, 1771, "Fig 5A-C (pilot calibration)", _blocks(r(6, 10), kind="OL")),
    Session("06_28_2021", 325, 1776, "Fig 2B-D; Fig 5A; SFig 1D", {
        **_blocks(r(5, 10), kind="OL"),
        **_blocks(r(11, 15), kind="CLR", decoder="RNN"),
        **_blocks(r(15, 21), kind="CL", decoder="RNN"),
    }),
    Session("06_30_2021", 326, 1778, "Fig 2B-D; Fig 5A; SMovie 1", {
        **_blocks(r(9, 14), kind="OL"),
        **_blocks(r(15, 19), kind="CLR", decoder="RNN"),
        **_blocks(r(20, 28), kind="CL", decoder="RNN"),
    }),
    Session("07_12_2021", 329, 1790, "Fig 5A", {
        **_blocks(r(8, 13), kind="OL"),
        **_blocks(r(14, 18), kind="CLR", decoder="RNN"),
        **_blocks(r(18, 22), kind="CL", decoder="RNN"),
    }),
    Session("07_14_2021", 330, 1792, "Fig 2B-D; Fig 5A", {
        **_blocks([7, 8, 14, 15], kind="OL"),
        **_blocks([16, 17, 18, 20, 21], kind="CL", decoder="RNN"),
    }),
    Session("09_13_2021", 336, 1853, "Fig 5C", {
        **_blocks(r(1, 5), kind="OL", uni_only=True),
        6: Block("CLR", "linear", uni_only=True), 7: Block("CLR", "RNN", uni_only=True),
        8: Block("CLR", "linear", uni_only=True), 9: Block("CLR", "RNN", uni_only=True),
        **_gains([12, 14, 16, 18, 20, 21], [0.5, 0.4, 0.3, 0.8, 1, 1.4], kind="CL", decoder="linear"),
        **_gains([13, 15, 17, 19], [0.3, 0.4, 0.2, 0.5], kind="CL", decoder="RNN"),
    }),
    Session("09_15_2021", 337, 1855, "Fig 5C; SMovie 3", {
        **_blocks([1, 2, 3, 5], kind="OL", uni_only=True),
        6: Block("CLR", "linear", uni_only=True), 7: Block("CLR", "linear", uni_only=True),
        8: Block("CLR", "RNN", uni_only=True), 9: Block("CLR", "RNN", uni_only=True),
        **_gains([17, 19, 21, 23], [1, 0.8, 1.1, 0.9], kind="CL", decoder="linear"),
        **_gains([16, 18, 20, 22], [0.5, 0.7, 0.4, 0.6], kind="CL", decoder="RNN"),
        **_gains([25, 27, 29, 31], [1.1, 0.9, 0.8, 1.2], kind="CL", decoder="linear", uni_only=True),
        **_gains([24, 26, 28, 30, 32], [0.7, 0.5, 0.6, 0.4, 0.8], kind="CL", decoder="RNN", uni_only=True),
    }),
    Session("09_27_2021", 340, 1867, "Fig 4E", {
        **_blocks([*r(1, 5), *r(12, 18), *r(22, 28)], kind="OL", uni_only=True),
        **_blocks([7, 9, 19, 21, 29, 31], kind="CL", decoder="RNN", variant="augmented", uni_only=True),
        **_blocks([8, 10, 18, 20, 28, 30], kind="CL", decoder="RNN", variant="raw", uni_only=True),
    }),
    Session("09_29_2021", 341, 1869, "Fig 4D-E; SMovie 4", {
        **_blocks(r(1, 8), kind="OL", uni_only=True),
        **_blocks([9, 13, 15, 17, 19, 21], kind="CL", decoder="RNN", variant="augmented", uni_only=True),
        **_blocks([8, 12, 14, 16, 20, 22], kind="CL", decoder="RNN", variant="raw", uni_only=True),
    }),
    Session("10_11_2021", 344, 1881, "Fig 2B-D; Fig 5A-B", {
        **_blocks([1, 4, 6, 8], kind="OL", uni_only=True),
        **_blocks([7, 9, 19, 28], kind="OL"),
        **_blocks([22, 25, 30, 32], kind="CL", decoder="RNN", variant="sequential"),
        **_blocks([23, 29, 31, 33], kind="CL", decoder="RNN", variant="simultaneous"),
    }),
    Session("10_13_2021", 345, 1883, "Fig 2B-D; Fig 3A-C; Fig 5A-B; SFig 2A-C; SMovie 2", {
        **_blocks([2, 4, 6, 8], kind="OL", uni_only=True),
        **_blocks([5, 7, 9, 27], kind="OL"),
        **_blocks([22, 24, 26, 29, 32], kind="CL", decoder="RNN", variant="sequential"),
        **_blocks([20, 23, 25, 28, 31], kind="CL", decoder="RNN", variant="simultaneous"),
    }),
]

BY_DAY = {s.trial_day: s for s in SESSIONS}
BY_DATE = {s.date: s for s in SESSIONS}


def get(day_or_date: int | str) -> Session:
    """Look up a session by trial day (e.g. 1750) or date string ('06_02_2021')."""
    return BY_DAY[day_or_date] if isinstance(day_or_date, int) else BY_DATE[day_or_date]
