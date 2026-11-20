"""Cut trial-aligned windows of neural features.

Reproduces
----------
``computeFeatureMatrices`` from tuningAnalysis.m:207-266, which tiles
features into an N x F1 x F2 x W x R tensor (channels x factor-1 condition x
factor-2 condition x time bins x repetitions). ``feature_tensor`` is a faithful
port, kept for the MATLAB/Octave cross-check. New analysis code should
normally use the simpler ``extract_windows``.

Window convention (same as MATLAB)
----------------------------------
``window = (w0, w1)`` is INCLUSIVE in bins relative to the event: (15, 35)
means bins event+15 ... event+35, i.e. 21 bins, which tuningAnalysis.m
describes as "300-700 ms after go". Trials whose window runs past either
end of the recording are skipped, as in MATLAB.
"""

from __future__ import annotations

import numpy as np


def extract_windows(features: np.ndarray, events: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Windows of features around events.

    Args:
        features: (n_bins, N).
        events: (n_trials,) 0-based event bins (e.g. go cues).
        window: inclusive (w0, w1) bin offsets relative to each event.

    Returns:
        (n_trials, w1 - w0 + 1, N). Trials whose window would leave the
        recording are filled with NaN, so trial indices stay aligned.
    """
    w0, w1 = window
    W = w1 - w0 + 1
    out = np.full((len(events), W, features.shape[1]), np.nan)
    for i, e in enumerate(np.asarray(events, int)):
        a, b = e + w0, e + w1 + 1
        if a >= 0 and b <= features.shape[0]:
            out[i] = features[a:b]
    return out


def feature_tensor(
    features: np.ndarray, trial_codes: np.ndarray, events: np.ndarray, window: tuple[int, int]
) -> np.ndarray:
    """Faithful port of ``computeFeatureMatrices`` (tuningAnalysis.m:207).

    Args:
        features: (n_bins, N) feature matrix.
        trial_codes: (n_trials,) or (n_trials, n_factors) integer condition
            codes, 1-based like MATLAB (e.g. wedge IDs 1..8, 9 = no move).
        events: (n_trials,) **0-based** event bins.
        window: inclusive (w0, w1) offsets.

    Returns:
        Array of shape (N, C1, ..., Ck, W, maxRep), NaN where empty.
        Condition value v is stored at index v-1 along its factor axis.
        Repetitions keep their MATLAB slot: a trial that is skipped because
        it runs out of range leaves its slot as NaN.

    Notes:
        MATLAB sizes each factor axis by the number of UNIQUE codes but
        indexes it by the code VALUE. If the codes are not exactly 1..n,
        MATLAB silently grows the array. This port sizes the axes by the
        maximum code value instead, which gives identical results whenever
        the MATLAB code runs without error.
    """
    codes = np.asarray(trial_codes)
    if codes.ndim == 1:
        codes = codes[:, None]
    if np.isnan(codes.astype(float)).any():
        raise ValueError("NaN trial codes: MATLAB would error here too")
    codes = codes.astype(int)
    events = np.asarray(events, int)
    w0, w1 = window
    W = w1 - w0 + 1
    N = features.shape[1]

    uniq, inverse = np.unique(codes, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    max_rep = np.bincount(inverse).max()
    sizes = codes.max(axis=0)
    out = np.full((N, *sizes, W, max_rep), np.nan)

    for ci, code in enumerate(uniq):
        trial_idx = np.where(inverse == ci)[0]  # MATLAB find(): ascending order
        for e, t in enumerate(trial_idx):
            a, b = events[t] + w0, events[t] + w1 + 1
            if b > features.shape[0] or a < 0:
                continue  # skipped, but repetition slot e stays (NaN), as in MATLAB
            idx = (slice(None), *(code - 1), slice(None), e)
            out[idx] = features[a:b].T  # (N, W)
    return out
