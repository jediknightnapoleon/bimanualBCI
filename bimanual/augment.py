"""Temporal fracturing: the data augmentation that lets the RNN generalize online.

Reproduces
----------
Fig 4c-d and Methods, "Data augmentation". The open-loop training data is
very stereotyped in time: delay, go, then a 1-2 s movement, every trial. An
RNN trained on it learns the timing and makes pulse-like movements online.
The paper breaks this timing structure as follows:

1. Cut CLIPS starting at random times from the training data.
2. Stretch or compress each clip in time by a factor drawn uniformly from
   [0.5, 2], using linear interpolation.
3. Concatenate clips in random order into 10-s synthetic SNIPPETS
   (T = 500 bins of 20 ms), about 2000 snippets in total.
4. Balance the clips across movement types x 4 direction quadrants and the
   "no movement" sub-types (uni-R delay, uni-L delay, bimanual delay, rest).
5. Split the training and validation pools BEFORE augmentation, so no
   validation data leaks into the training snippets.

Assumptions
-----------
* ASSUMPTION: clip length is uniform in [200, 800] ms (10-40 bins). The
  paper's wording is ambiguous ("0.2-0.8 x" vs "200-800 ms"); this is
  re-checked against the PDF in step 6.2.
* ASSUMPTION: a clip's stratum (used for balancing) is the label of its
  middle bin.
* ASSUMPTION: velocity-target AMPLITUDES are not rescaled when a clip is
  stretched. The targets encode intended direction and "go" strength, not
  physical speed.
* Continuous signals (features, velocities) are linearly interpolated.
  Discrete labels (context) use the nearest source bin.

Memory
------
2000 x 500 x 192 float32 is about 770 MB. ``SnippetSampler.sample`` can
therefore make snippets on the fly, per training batch, as well as building
a fixed set with ``make_snippets``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

SNIPPET_BINS = 500  # 10 s
CLIP_BINS = (10, 40)  # 200-800 ms (ASSUMPTION, see module docstring)
DILATION = (0.5, 2.0)


def time_warp(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample ``x`` (n_bins, ...) to round(n_bins * factor) bins by linear interpolation.

    factor > 1 stretches (slower), factor < 1 compresses (faster). The first
    and last samples are kept.
    """
    n = x.shape[0]
    m = max(1, int(round(n * factor)))
    if n == 1:
        return np.repeat(x, m, axis=0)
    pos = np.linspace(0, n - 1, m)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[lo] + w * x[hi]


def nearest_warp(labels: np.ndarray, factor: float) -> np.ndarray:
    """Like :func:`time_warp`, for discrete labels: take the nearest source bin."""
    n = labels.shape[0]
    m = max(1, int(round(n * factor)))
    pos = np.linspace(0, n - 1, m) if n > 1 else np.zeros(m)
    return labels[np.rint(pos).astype(int)]


def split_segments_by_trial(
    trial_bounds: np.ndarray, val_frac: float = 0.1, seed: int | None = 0
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split trials into training and validation pools BEFORE augmentation.

    Args:
        trial_bounds: (n_trials, 2) [start, stop) bins of each trial (0-based).
        val_frac: fraction of trials held out (the paper uses ~10%).

    Returns:
        (train_segments, val_segments): lists of [start, stop) bin ranges.
        Clips are drawn only from inside these ranges.
    """
    rng = np.random.default_rng(seed)
    n = len(trial_bounds)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = set(rng.choice(n, n_val, replace=False).tolist())
    train = [tuple(map(int, b)) for i, b in enumerate(trial_bounds) if i not in val_idx]
    val = [tuple(map(int, b)) for i, b in enumerate(trial_bounds) if i in val_idx]
    return train, val


@dataclass
class Snippets:
    """Augmented (or raw) training data. Shapes: (n_snippets, T, ...)."""

    X: np.ndarray  # (S, T, N) neural features
    vel: np.ndarray  # (S, T, 4) velocity targets
    context: np.ndarray  # (S, T) discrete context targets


class SnippetSampler:
    """Draw balanced, time-warped, shuffled 10-s snippets from a pool of segments.

    Args:
        X: (n_bins, N) neural features of the whole recording (z-scored).
        vel: (n_bins, 4) velocity targets (from targets.build_targets).
        context: (n_bins,) discrete context labels.
        stratum: (n_bins,) balancing labels (from targets.build_targets).
        segments: [start, stop) ranges the clips may come from (one pool,
            e.g. training trials only). A clip never crosses a segment edge.
        clip_bins, dilation, snippet_bins: augmentation parameters.
        seed: RNG seed.
    """

    def __init__(
        self,
        X,
        vel,
        context,
        stratum,
        segments,
        clip_bins=CLIP_BINS,
        dilation=DILATION,
        snippet_bins=SNIPPET_BINS,
        seed=0,
    ):
        self.X, self.vel, self.context = X, vel, context
        self.clip_bins, self.dilation, self.T = clip_bins, dilation, snippet_bins
        self.rng = np.random.default_rng(seed)
        # Index every bin by stratum, so a stratum can be picked uniformly first
        # and then a clip center within it (= balancing). A center can be any
        # bin of a segment: _clip() shifts the clip so it stays inside the
        # segment, so short labels near segment edges (e.g. a short rest
        # period) can still be sampled.
        self.segments = [(int(a), int(b)) for a, b in segments if b - a >= clip_bins[1]]
        if not self.segments:
            raise ValueError("no segment is long enough for the longest clip")
        by_stratum = defaultdict(list)
        for a, b in self.segments:
            for c in range(a, b):
                by_stratum[stratum[c]].append(c)
        self.strata = sorted(by_stratum, key=repr)
        self.centers = {s: np.array(by_stratum[s]) for s in self.strata}
        self._seg_of = {c: (a, b) for a, b in self.segments for c in range(a, b)}

    def _clip(self):
        """One balanced, warped clip: returns (X, vel, context), each warped."""
        s = self.strata[self.rng.integers(len(self.strata))]
        c = int(self.rng.choice(self.centers[s]))
        L = int(self.rng.integers(self.clip_bins[0], self.clip_bins[1] + 1))
        a, b = self._seg_of[c]
        start = int(np.clip(c - L // 2, a, b - L))  # keep the clip inside its segment
        sl = slice(start, start + L)
        f = float(self.rng.uniform(*self.dilation))
        return (
            time_warp(self.X[sl], f),
            time_warp(self.vel[sl], f),
            nearest_warp(self.context[sl], f),
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One 10-s snippet: clips in random order, concatenated, trimmed to T bins."""
        xs, vs, cs, n = [], [], [], 0
        while n < self.T:
            x, v, c = self._clip()
            xs.append(x), vs.append(v), cs.append(c)
            n += len(c)
        return (
            np.concatenate(xs)[: self.T].astype(np.float32),
            np.concatenate(vs)[: self.T].astype(np.float32),
            np.concatenate(cs)[: self.T],
        )

    def make_snippets(self, n: int) -> Snippets:
        """A fixed set of ``n`` snippets (the paper used ~2000)."""
        out = [self.sample() for _ in range(n)]
        return Snippets(*(np.stack(z) for z in zip(*out)))


def raw_snippets(X, vel, context, segments, snippet_bins=SNIPPET_BINS) -> Snippets:
    """NON-augmented baseline for Fig 4d: consecutive T-bin windows of the real data.

    The segments are concatenated in their original order, so the stereotyped
    timing is kept, and then cut into non-overlapping windows. A leftover
    shorter than T is dropped.
    """
    idx = np.concatenate([np.arange(a, b) for a, b in segments])
    n = len(idx) // snippet_bins
    idx = idx[: n * snippet_bins].reshape(n, snippet_bins)
    return Snippets(X[idx].astype(np.float32), vel[idx].astype(np.float32), context[idx])
