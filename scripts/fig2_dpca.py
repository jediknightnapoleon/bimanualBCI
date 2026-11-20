"""Step 2.4: Fig 2 dPCA, laterality vs direction share of neural variance.

Purpose
-------
Paper: "Laterality explained 39% of marginalized variance and direction
explained 30%. Tuning to laterality was stronger than tuning to movement
direction." This motivates the laterality dimension that the nonlinear
decoders exploit (Fig 3).

Method
------
* Days 1776, 1778, 1792, 1881, 1883, analysed separately; the mean over days
  is reported with a 95% bootstrap CI over days.
* Unimanual open-loop trials. Laterality = uni-R vs uni-L; direction = the 4
  axis-aligned quadrants of the moving hand (register #6).
* Features: z-scored per block (as for the PCA). Time window: bins 15..35
  after the go cue ("300-700 ms", MAIN) and bins 0..49 (0-1000 ms,
  robustness). ASSUMPTION (register #24): the paper doesn't give the dPCA
  window or the scaling.
* Variance per marginal = the two-split cross-validated estimate (Methods),
  averaged over 100 random splits (bimanual.dpca_analysis). Fractions = each
  marginal's share of the summed cross-validated marginal variance.
  Non-cross-validated fractions are reported for comparison.

Inputs : Data/ (5 days)
Outputs: results/fig2/dpca.json; results/figures/fig2_dpca.{png,svg}
Run    : pixi run fig2-dpca            (about 1-2 min; --force to redo)
"""

from __future__ import annotations

import argparse

import numpy as np

from bimanual import conditions as C
from bimanual import dpca_analysis as D
from bimanual import plotting as P
from bimanual.blocksets import FIG2_DAYS
from bimanual.cache import RESULTS_DIR, atomic_write_json, should_skip
from bimanual.epochs import extract_windows
from bimanual.io import load_session, zscore_by_block
from bimanual.stats import bootstrap_ci

WINDOWS = {"300-700ms": (15, 35), "0-1000ms": (0, 49)}
MAIN_WINDOW = "300-700ms"
N_SPLITS = 100
PAPER = {"laterality": 0.39, "direction": 0.30}
OUT_JSON = RESULTS_DIR / "fig2" / "dpca.json"
OUT_FIG = P.FIG_DIR / "fig2_dpca.png"


def day_data(day: int):
    """Z-scored features, go cues, laterality index and direction index of the uni OL trials."""
    s = load_session(day)
    z = zscore_by_block(s.tx, s.block)
    mt = s.trials.move_type.to_numpy()
    q = C.trial_quadrants(s.Tp, s.Cp, s.go, mt)
    keep = (s.trials.kind == "OL").to_numpy() & np.isin(mt, (1, 2))
    lat = (mt == 2).astype(int)  # 0 = uni-R, 1 = uni-L
    qd = np.where(mt == 1, q[:, 0], q[:, 1])  # the moving hand's quadrant
    keep &= ~np.isnan(qd)
    return z, s.go[keep], lat[keep], qd[keep].astype(int) - 1


def main(force=False):
    if should_skip([OUT_JSON, OUT_FIG], force):
        print("outputs exist; --force to redo")
        return
    res = {w: {} for w in WINDOWS}
    for day in FIG2_DAYS:
        z, go, lat, dirs = day_data(day)
        for wname, win in WINDOWS.items():
            tr = extract_windows(z, go, win)
            ok = ~np.isnan(tr).any(axis=(1, 2))
            cv = D.cv_marginal_variances(tr[ok], lat[ok], dirs[ok], n_splits=N_SPLITS, seed=day)
            naive = D.marginal_variances(D.condition_average(tr[ok], lat[ok], dirs[ok]))
            res[wname][day] = dict(n_trials=int(ok.sum()), cv_var=cv, cv_frac=D.fractions(cv),
                                   naive_frac=D.fractions(naive))
        f = res[MAIN_WINDOW][day]["cv_frac"]
        print(f"day {day} ({res[MAIN_WINDOW][day]['n_trials']} trials): " +
              ", ".join(f"{k} {100 * v:.0f}%" for k, v in f.items()))

    summary = {}
    for wname in WINDOWS:
        summary[wname] = {}
        for kind in ("cv_frac", "naive_frac"):
            summary[wname][kind] = {
                m: bootstrap_ci(np.array([res[wname][d][kind][m] for d in FIG2_DAYS])) for m in D.MARGINALS}
    for wname in WINDOWS:
        sm = summary[wname]["cv_frac"]
        print(f"[{wname}] mean over days (CV): " + ", ".join(f"{m} {100 * sm[m][0]:.0f}%" for m in D.MARGINALS)
              + f"   (paper: laterality 39%, direction 30%)")
        sm = summary[wname]["naive_frac"]
        print(f"[{wname}] mean over days (naive): " + ", ".join(f"{m} {100 * sm[m][0]:.0f}%" for m in D.MARGINALS))

    # ---- figure: per-day dots + mean/CI per marginal, both windows side by side
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    colors = dict(laterality=P.SERIES[6], direction=P.SERIES[2], interaction=P.SERIES[3], time=P.INK_MUTED)
    rng = np.random.default_rng(0)
    for ax, wname in zip(axes, WINDOWS):
        sm = summary[wname]["cv_frac"]
        for i, m in enumerate(D.MARGINALS):
            v = [res[wname][d]["cv_frac"][m] for d in FIG2_DAYS]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), np.array(v) * 100, s=16, color=colors[m], alpha=0.6, lw=0)
            mean, lo, hi = sm[m]
            ax.errorbar(i, mean * 100, yerr=[[100 * (mean - lo)], [100 * (hi - mean)]], fmt="o", color=P.INK,
                        capsize=4, ms=6)
            ax.text(i + 0.2, mean * 100, f"{100 * mean:.0f}%", va="center", fontsize=8, color=P.INK)
            if m in PAPER:
                ax.plot(i, PAPER[m] * 100, marker="D", mfc="none", mec=P.INK, ms=8, ls="none",
                        label="paper" if m == "laterality" else None)
        ax.set_xticks(range(4), [m.capitalize() for m in D.MARGINALS])
        ax.set_title(f"Window {wname} after go" + (" (main)" if wname == MAIN_WINDOW else ""), fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of marginalized variance (cross-validated)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Fig 2 dPCA replication: unimanual trials, days 1776/1778/1792/1881/1883 "
                 "(dots = days; mean ± 95% bootstrap CI)", fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "fig2_dpca")
    atomic_write_json(OUT_JSON, dict(windows={k: list(v) for k, v in WINDOWS.items()}, main_window=MAIN_WINDOW,
                                     n_splits=N_SPLITS, paper=PAPER, per_day=res, summary=summary))
    print("wrote", OUT_FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
