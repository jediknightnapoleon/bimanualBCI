"""Group open-loop blocks into "sets" for the Fig 2b-c tuning comparisons.

Background
----------
Methods: data from 5 days (1776, 1778, 1792, 1881, 1883) were "grouped in
pairs of blocks to ~40 repetitions per trial type", giving a sample size of
25. The exact pairing rule isn't stated, so three rules are implemented
(register #10):

* ``"adjacent_all"`` (MAIN, user decision 2026-11-21): consecutive pairs of
  ALL open-loop blocks of a day, in block order, including the
  "unimanual-only" OL blocks of days 1881/1883. This gives exactly
  4 + 4 + 3 + 7 + 7 = 25 pairs, matching the paper's n. Three of these pairs
  contain only unimanual-only blocks (about 2 bimanual trials), so bimanual
  tuning can't be estimated for them and they are reported as unusable.
* ``"all_pairs_mixed"`` (robustness): every unordered pair of the day's
  mixed OL blocks (all 3 trial types): 10 + 10 + 6 + 6 + 6 = 38 sets.
* ``"adjacent_mixed"`` (robustness): consecutive pairs of the mixed OL
  blocks only: 4 + 4 + 3 + 3 + 3 = 17 sets.
"""

from __future__ import annotations

from itertools import combinations

from bimanual import sessions

FIG2_DAYS = (1776, 1778, 1792, 1881, 1883)
SCHEMES = ("adjacent_all", "all_pairs_mixed", "adjacent_mixed")
MAIN_SCHEME = "adjacent_all"


def ol_blocks(day: int, mixed_only: bool) -> list[int]:
    """Sorted open-loop block numbers of a day (from the readme labels)."""
    s = sessions.get(day)
    return sorted(b for b, lab in s.blocks.items() if lab.kind == "OL" and not (mixed_only and lab.uni_only))


def block_sets(day: int, scheme: str) -> list[tuple[int, int]]:
    """Block pairs of one day under the given scheme (see the module docstring)."""
    if scheme == "adjacent_all":
        b = ol_blocks(day, mixed_only=False)
        return list(zip(b[:-1], b[1:]))
    if scheme == "adjacent_mixed":
        b = ol_blocks(day, mixed_only=True)
        return list(zip(b[:-1], b[1:]))
    if scheme == "all_pairs_mixed":
        return list(combinations(ol_blocks(day, mixed_only=True), 2))
    raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
