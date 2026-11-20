"""Demixed variance decomposition: laterality vs direction vs interaction vs time.

Reproduces
----------
Fig 2 (dPCA): "laterality explained 39% of marginalized variance and direction
explained 30%". In motor cortex, which hand moves is encoded more strongly
than where it moves.

Marginalization (Kobak et al. 2016, eLife, "Demixed principal component analysis")
-----------------------------------------------------------------------------------
Condition-averaged data X[n, l, d, t] (neurons x laterality x direction x
time bins) is centred per neuron and split into orthogonal ANOVA-style parts.
Each factor's part also absorbs its interaction with time, as in dPCA:

    time        : X_t                           = <X>_{l,d} (minus grand mean)
    laterality  : X_l + X_lt                    = <X>_{d} - X_t
    direction   : X_d + X_dt                    = <X>_{l} - X_t
    interaction : X_ld + X_ldt                  = X - <X>_d - <X>_l + X_t

(<X>_{..} = mean over the listed axes, broadcast back.) The four parts sum to
the centred data, and their squared norms add up to the total variance.

Cross-validated variance (Methods)
----------------------------------
Noise inflates every marginal's variance, most of all for small effects. The
paper instead splits the trials into two halves, marginalizes each half's
condition averages separately, multiplies the two marginalized arrays
element-wise and sums. Independent noise cancels in expectation, so the
result is an unbiased estimate of the signal variance of each marginal. We
average this over many random splits.
"""

from __future__ import annotations

import numpy as np

MARGINALS = ("laterality", "direction", "interaction", "time")


def marginalize(X: np.ndarray) -> dict[str, np.ndarray]:
    """Split condition averages X (N, L, D, T) into the four marginals (same shape each).

    The input is centred per neuron first. Returns {name: array (N, L, D, T)}.
    """
    X = X - X.mean(axis=(1, 2, 3), keepdims=True)
    m_d = X.mean(axis=2, keepdims=True)  # average over direction -> depends on l, t
    m_l = X.mean(axis=1, keepdims=True)  # average over laterality -> depends on d, t
    m_ld = X.mean(axis=(1, 2), keepdims=True)  # depends on t only
    shape = X.shape
    return {
        "time": np.broadcast_to(m_ld, shape),
        "laterality": np.broadcast_to(m_d - m_ld, shape),
        "direction": np.broadcast_to(m_l - m_ld, shape),
        "interaction": X - m_d - m_l + m_ld,
    }


def condition_average(trials: np.ndarray, lat: np.ndarray, dirs: np.ndarray,
                      n_lat: int = 2, n_dir: int = 4) -> np.ndarray:
    """Average trials into X (N, L, D, T).

    Args:
        trials: (n_trials, T, N) windows of features.
        lat: (n_trials,) laterality index 0..n_lat-1.
        dirs: (n_trials,) direction index 0..n_dir-1.
    """
    n_trials, T, N = trials.shape
    X = np.full((N, n_lat, n_dir, T), np.nan)
    for l in range(n_lat):
        for d in range(n_dir):
            m = (lat == l) & (dirs == d)
            if m.any():
                X[:, l, d, :] = trials[m].mean(axis=0).T
    return X


def marginal_variances(X: np.ndarray) -> dict[str, float]:
    """Plain (not cross-validated) variance of each marginal: biased upward by noise."""
    return {k: float(np.sum(v**2)) for k, v in marginalize(X).items()}


def cv_marginal_variances(trials: np.ndarray, lat: np.ndarray, dirs: np.ndarray,
                          n_splits: int = 100, seed: int = 0) -> dict[str, float]:
    """Cross-validated variance of each marginal (two-split method of the paper).

    For each split, the trials of EVERY condition are halved at random, so
    both halves contain all 8 conditions. Each half is condition-averaged and
    marginalized, and sum(M1 * M2) is taken per marginal. The result is the
    average over splits.
    """
    rng = np.random.default_rng(seed)
    acc = {k: 0.0 for k in MARGINALS}
    cond = lat * 100 + dirs
    for _ in range(n_splits):
        half = np.zeros(len(cond), bool)
        for c in np.unique(cond):
            idx = rng.permutation(np.where(cond == c)[0])
            half[idx[: len(idx) // 2]] = True
        M1 = marginalize(condition_average(trials[half], lat[half], dirs[half]))
        M2 = marginalize(condition_average(trials[~half], lat[~half], dirs[~half]))
        for k in MARGINALS:
            acc[k] += float(np.sum(M1[k] * M2[k]))
    return {k: v / n_splits for k, v in acc.items()}


def fractions(var: dict[str, float]) -> dict[str, float]:
    """Each marginal's share of the summed marginal variance ("% of marginalized variance")."""
    tot = sum(var.values())
    return {k: v / tot for k, v in var.items()}
