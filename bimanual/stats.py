"""Statistics helpers shared across figures.

Reproduces
----------
* The 95% confidence intervals from bootstrap resampling (n = 10,000) used in
  Figs 1, 4 and 5 (Methods, "Statistics").

The ANOVA/FVAF and the less-biased neural distance helpers are added in
Phases 1-2.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval of a statistic.

    Args:
        x: 1-D sample. NaNs are dropped first, e.g. failed trials when the
            statistic is TTA. Handle failures explicitly if they should count.
        stat: the statistic to compute (default: mean).
        n_boot: number of resamples (the paper uses 10,000).
        alpha: 0.05 gives a 95% CI.
        seed: RNG seed for reproducibility.

    Returns:
        (point estimate, CI low, CI high).
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)
