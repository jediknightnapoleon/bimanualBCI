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


def anova_fvaf(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """One-way ANOVA of one electrode's trial-averaged rates across conditions, plus FVAF.

    Used for Supp Fig 1b,c,e (Methods, "Single-electrode tuning"): an electrode
    counts as tuned in a context if p < 1e-5 across the 4 direction quadrants.

    FVAF (fraction of variance accounted for) = SS_dir / SS_total, where
    SS_total = sum_i (f_i - mean)^2 and SS_dir = sum_i (mean_{D(i)} - mean)^2.
    That is the share of trial-to-trial variance explained by the cued
    direction (0..1). This is the per-electrode, scalar form of the paper's
    FVAF.

    Args:
        values: (n_trials,) trial-averaged firing rates of one electrode.
        groups: (n_trials,) condition label of each trial (NaN rows are dropped).

    Returns:
        (F statistic, p value, FVAF).
    """
    from scipy.stats import f_oneway

    values = np.asarray(values, float)
    groups = np.asarray(groups, float)
    ok = ~np.isnan(values) & ~np.isnan(groups)
    v, g = values[ok], groups[ok]
    levels = np.unique(g)
    samples = [v[g == lv] for lv in levels]
    if len(levels) < 2 or np.ptp(v) == 0:
        return (np.nan, 1.0, 0.0)  # constant electrode: no tuning
    F, p = f_oneway(*samples)
    grand = v.mean()
    ss_tot = np.sum((v - grand) ** 2)
    ss_dir = sum(len(s_) * (s_.mean() - grand) ** 2 for s_ in samples)
    return float(F), float(p), float(ss_dir / ss_tot)
