"""8-direction tuning curves for unimanual and bimanual movement (Fig 1b-c).

Reproduces
----------
The tuning-curve computation of ``tuningAnalysis.m`` (lines 13-118), as a
faithful Python port:

1. Block-wise mean removal of ``tx``, then x50 to get spikes/s.
2. Label each trial [moveType, RH wedge, LH wedge] (bimanual.conditions).
3. Average the 20-ms bins in a window after the go cue (default (15, 35),
   which the authors describe as 300-700 ms) for each of the 8 movement
   directions:
     * unimanual right = trials with code (d, 9); unimanual left = (9, d);
     * bimanual right / left = bimanual trials grouped by the RH / LH wedge.
4. Subtract each channel's mean across the 8 directions, so curves are
   centred on 0.

The output is checked against MATLAB/Octave (step 0.5,
``pixi run compare-reference``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual import conditions as C
from bimanual.epochs import feature_tensor
from bimanual.io import HZ_PER_COUNT, SessionData, remove_block_means

DEFAULT_WINDOW = (15, 35)  # bins after go cue, inclusive (tuningAnalysis.m:27)


@dataclass
class TuningCurves:
    """Each array is (n_channels, 8) in wedge order (see conditions.WEDGE_CENTERS_DEG)."""

    uniR: np.ndarray
    uniL: np.ndarray
    biR: np.ndarray
    biL: np.ndarray
    conditions: np.ndarray  # (n_trials, 3) [moveType, RH wedge, LH wedge]


def _mean_over_bins_and_reps(t: np.ndarray) -> np.ndarray:
    """nanmean over the last two axes (time bins x repetitions) -> (N,).

    Equivalent to MATLAB tiling all bins side by side and calling nanmean.
    """
    flat = t.reshape(t.shape[0], -1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(flat, axis=1)


def tuning_curves(s: SessionData, window: tuple[int, int] = DEFAULT_WINDOW) -> TuningCurves:
    """Compute the four 8-direction tuning curves of one session (port of tuningAnalysis.m)."""
    features = remove_block_means(s.tx, s.block) * HZ_PER_COUNT  # (T, N), spikes/s
    go = s.go
    cond = C.trial_conditions(s.Tp, s.Cp, go + 1, s.trials["move_type"].to_numpy())  # takes 1-based

    # Unimanual: one tensor over the [RH, LH] factors, slices (d, 9) and (9, d).
    fv = feature_tensor(features, cond[:, 1:3], go, window)
    uniR = np.stack([_mean_over_bins_and_reps(fv[:, d, 8]) for d in range(8)], axis=1)
    uniL = np.stack([_mean_over_bins_and_reps(fv[:, 8, d]) for d in range(8)], axis=1)

    # Bimanual: a separate one-factor tensor for each hand.
    bi = np.where(cond[:, 0] == C.BIMANUAL)[0]
    fvR = feature_tensor(features, cond[bi, 1], go[bi], window)
    fvL = feature_tensor(features, cond[bi, 2], go[bi], window)
    biR = np.stack([_mean_over_bins_and_reps(fvR[:, d]) for d in range(8)], axis=1)
    biL = np.stack([_mean_over_bins_and_reps(fvL[:, d]) for d in range(8)], axis=1)

    # Centre each channel's curve on 0 (MATLAB uses mean, not nanmean, here).
    center = lambda tc: tc - tc.mean(axis=1, keepdims=True)
    return TuningCurves(center(uniR), center(uniL), center(biR), center(biL), cond)
