"""Step 2.2: Fig 2b-c, Supp Table 1 and Supp Fig 1d (tuning suppression and decorrelation).

Purpose
-------
The paper's central neural finding: during bimanual movement, the tuning of the
non-dominant LEFT hand is suppressed and decorrelated from its unimanual
tuning, while RIGHT-hand tuning stays largely stable. Targets to match:

* Fig 2b: tuning strength in bimanual vs unimanual is 12% lower for the RH and
  34% lower for the LH (two-sample t-test RH vs LH, n = 25 block-pair sets).
* Fig 2c: uni-vs-bi tuning correlation is RH 0.87 (x) / 0.84 (y) and
  LH 0.42 (x) / 0.45 (y). Supp Table 1 gives the pairwise t-test p-values
  between the four correlation types.
* Supp Fig 1d (day 1776): unimanual RH tuning is "almost twice as strong"
  as LH (the LH is 48% weaker).

Method (Methods, "Encoding model"; register #9, #10, #21)
---------------------------------------------------------
* Firing rates: 20-ms bins, block-mean removed, x50 (spikes/s); bins 15..35
  after the go cue ("300-700 ms").
* Regressors: unit cursor->target direction at the go cue for each moving
  hand (conditions.trial_directions).
* Three cvOLS encoding models (5-fold, folds split by trial) per block set:
    uni-R model on uni-R trials:  f = b0 + b_Rx d_Rx + b_Ry d_Ry
    uni-L model on uni-L trials:  f = b0 + b_Lx d_Lx + b_Ly d_Ly
    bi model on bimanual trials:  f = b0 + b_Rx d_Rx + b_Ry d_Ry + b_Lx d_Lx + b_Ly d_Ly
* Tuning strength of a hand = mean of its cross-validated x and y column
  magnitudes (encoding.tuning_strength).
  Suppression ratio = bimanual strength / unimanual strength.
* Correlations (cross-validated Pearson between coefficient columns):
  UniR-BiR, UniL-BiL (same hand, across contexts), UniR-UniL, BiR-BiL
  (across hands, same context), each for x and y separately.
* Block sets: three schemes (bimanual.blocksets). MAIN = "adjacent_all"
  (n = 25 pairs, as in the paper); the others are robustness checks. A set
  is usable only if every context has >= MIN_TRIALS trials (register #22).

Inputs : Data/ (days 1776, 1778, 1792, 1881, 1883)
Outputs: results/fig2/<method>/tuning_change_<scheme>.json (per-set values, summaries, t-tests)
         results/fig2/<method>/supp_fig1d.json
         results/figures/fig2bc_tuning_change_<method>.{png,svg}
Run    : pixi run fig2-tuning [--method splithalf|kfold]   (about 1-2 min; --force to redo)
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
from scipy.stats import ttest_ind

from bimanual import blocksets as BS
from bimanual import conditions as C
from bimanual import encoding as E
from bimanual import plotting as P
from bimanual.cache import RESULTS_DIR, atomic_write_json, should_skip
from bimanual.epochs import extract_windows
from bimanual.io import HZ_PER_COUNT, load_session, remove_block_means
from bimanual.stats import bootstrap_ci
from bimanual.tuning import DEFAULT_WINDOW

MIN_TRIALS = 10  # per context within a set (register #22)
METHOD = "splithalf"  # cvOLS estimator, set from --method (register #23; see encoding.cv_ols)
OUT_DIR = RESULTS_DIR / "fig2"
CORR_TYPES = ("UniR-BiR", "UniL-BiL", "UniR-UniL", "BiR-BiL")
PAPER = dict(supp_R=0.12, supp_L=0.34, corr={"UniR-BiR": (0.87, 0.84), "UniL-BiL": (0.42, 0.45)})


# ----------------------------------------------------------------------------- data per day
def prepare_day(day: int) -> dict:
    """Window the rates of every trial of a day; return arrays used by the models."""
    s = load_session(day)
    rates = remove_block_means(s.tx, s.block) * HZ_PER_COUNT
    win = extract_windows(rates, s.go, DEFAULT_WINDOW)  # (trials, 21, N)
    mt = s.trials.move_type.to_numpy()
    return dict(
        win=win, mt=mt, block=s.trials.block.to_numpy(), kind=s.trials.kind.to_numpy(),
        dirs=C.trial_directions(s.Tp, s.Cp, s.go, mt),
    )


def design(day_data: dict, trials: np.ndarray, cols: list[int]):
    """Bin-level design for cvOLS: X (bins x p), Y (bins x N), groups (trial id per bin)."""
    win = day_data["win"][trials]
    ok = ~np.isnan(win).any(axis=(1, 2))  # drop trials whose window leaves the recording
    win, trials = win[ok], trials[ok]
    n_tr, n_bin, n_ch = win.shape
    d = day_data["dirs"][trials][:, cols]  # constant over the window
    X = np.c_[np.ones(n_tr * n_bin), np.repeat(d, n_bin, axis=0)]
    return X, win.reshape(-1, n_ch), np.repeat(trials, n_bin)


def analyse_set(dd: dict, blocks: tuple[int, int], seed: int) -> dict:
    """Fit the three encoding models on one block set and compute every statistic."""
    in_set = np.isin(dd["block"], blocks) & (dd["kind"] == "OL")
    idx = {mt: np.where(in_set & (dd["mt"] == mt))[0] for mt in (1, 2, 3)}
    n = {k: int(len(v)) for k, v in zip(("uniR", "uniL", "bi"), idx.values())}
    if min(n.values()) < MIN_TRIALS:
        return dict(blocks=list(blocks), n_trials=n, usable=False)

    kw = dict(seed=seed, method=METHOD)
    uR = E.cv_ols(*design(dd, idx[1], [0, 1]), names=("b0", "Rx", "Ry"), **kw)
    uL = E.cv_ols(*design(dd, idx[2], [2, 3]), names=("b0", "Lx", "Ly"), **kw)
    bi = E.cv_ols(*design(dd, idx[3], [0, 1, 2, 3]), names=("b0", "Rx", "Ry", "Lx", "Ly"), **kw)

    st = dict(
        uniR=E.tuning_strength(uR, "Rx", "Ry"), biR=E.tuning_strength(bi, "Rx", "Ry"),
        uniL=E.tuning_strength(uL, "Lx", "Ly"), biL=E.tuning_strength(bi, "Lx", "Ly"),
    )
    corr = {}
    for ax in ("x", "y"):
        R, L = "R" + ax, "L" + ax
        corr[f"UniR-BiR_{ax}"] = E.cv_correlation(uR, R, bi, R)  # different trials: independent fits
        corr[f"UniL-BiL_{ax}"] = E.cv_correlation(uL, L, bi, L)
        corr[f"UniR-UniL_{ax}"] = E.cv_correlation(uR, R, uL, L)
        corr[f"BiR-BiL_{ax}"] = E.cv_correlation(bi, R, None, L)  # same model: cross-fold products
    return dict(
        blocks=list(blocks), n_trials=n, usable=True, strength=st,
        ratio_R=st["biR"] / st["uniR"], ratio_L=st["biL"] / st["uniL"], corr=corr,
    )


# ----------------------------------------------------------------------------- summaries
def summarize(sets: list[dict]) -> dict:
    """Mean and 95% bootstrap CI over sets, plus the t-tests of Fig 2b and Supp Table 1."""
    u = [s for s in sets if s["usable"]]
    rR = np.array([s["ratio_R"] for s in u]); rL = np.array([s["ratio_L"] for s in u])
    out = dict(n_sets=len(sets), n_usable=len(u))
    out["ratio_R"] = bootstrap_ci(rR); out["ratio_L"] = bootstrap_ci(rL)
    out["suppression_R"] = 1 - out["ratio_R"][0]; out["suppression_L"] = 1 - out["ratio_L"][0]
    out["ttest_ratio_R_vs_L_p"] = float(ttest_ind(rR, rL).pvalue)
    out["corr"], out["corr_ttests_p"] = {}, {}
    for ax in ("x", "y"):
        vals = {t: np.array([s["corr"][f"{t}_{ax}"] for s in u], float) for t in CORR_TYPES}
        for t, v in vals.items():
            out["corr"][f"{t}_{ax}"] = bootstrap_ci(v)  # NaNs (no detectable tuning) dropped
        for a, b in combinations(CORR_TYPES, 2):  # Supp Table 1: pairwise two-sample t-tests
            out["corr_ttests_p"][f"{a} vs {b} ({ax})"] = float(
                ttest_ind(vals[a], vals[b], nan_policy="omit").pvalue)
    st = {k: np.array([s["strength"][k] for s in u]) for k in ("uniR", "uniL", "biR", "biL")}
    out["strength_mean"] = {k: float(v.mean()) for k, v in st.items()}
    return out


def supp_fig1d(dd: dict) -> dict:
    """Day 1776: unimanual RH vs LH tuning strength over all OL blocks (Supp Fig 1d)."""
    ol = dd["kind"] == "OL"
    uR = E.cv_ols(*design(dd, np.where(ol & (dd["mt"] == 1))[0], [0, 1]), names=("b0", "Rx", "Ry"), method=METHOD)
    uL = E.cv_ols(*design(dd, np.where(ol & (dd["mt"] == 2))[0], [2, 3]), names=("b0", "Lx", "Ly"), method=METHOD)
    sR, sL = E.tuning_strength(uR, "Rx", "Ry"), E.tuning_strength(uL, "Lx", "Ly")
    return dict(strength_R=sR, strength_L=sL, ratio_R_over_L=sR / sL, L_weaker_by=1 - sL / sR)


# ----------------------------------------------------------------------------- figure
def plot(main: dict, summaries: dict, fig_name: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1, 1.6, 1.1]))
    u = [s for s in main["sets"] if s["usable"]]
    sm = main["summary"]
    rng = np.random.default_rng(0)

    # (b) suppression ratios: one dot per set, with the mean and 95% CI
    ax = axes[0]
    for i, (key, color, lab) in enumerate((("ratio_R", P.UNI, "Right"), ("ratio_L", P.BI, "Left"))):
        v = np.array([s[key] for s in u])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=color, alpha=0.5, lw=0)
        m, lo, hi = sm[key]
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=P.INK, ms=7, capsize=4, lw=1.5)
        ax.text(i + 0.22, m, f"{100 * (1 - m):.0f}% supp.", va="center", fontsize=8, color=P.INK)
    ax.axhline(1, color=P.INK_MUTED, lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["Right hand", "Left hand"]); ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("Tuning strength ratio (bi / uni)")
    ax.set_title(f"(b) Suppression (n={sm['n_usable']} sets)\npaper: RH 12%, LH 34%", fontsize=9)
    ax.grid(axis="x", visible=False)

    # (c) correlations: grouped bars, x and y
    ax = axes[1]
    w = 0.38
    for j, (axn, color) in enumerate((("x", P.SERIES[2]), ("y", P.SERIES[6]))):
        ms = [sm["corr"][f"{t}_{axn}"] for t in CORR_TYPES]
        xs = np.arange(4) + (j - 0.5) * w
        bars = ax.bar(xs, [m[0] for m in ms], width=w - 0.04, color=color, label=f"{axn}-direction")
        ax.errorbar(xs, [m[0] for m in ms], yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]],
                    fmt="none", ecolor=P.INK, capsize=3, lw=1)
        for xi, (m, lo, hi) in zip(xs, ms):  # value label just beyond the CI end, never on the cap
            ax.text(xi, hi + 0.03 if m >= 0 else lo - 0.03, f"{m:.2f}", ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=7, color=P.INK)
    for i, t in enumerate(CORR_TYPES[:2]):  # paper values as hollow markers
        for j, val in enumerate(PAPER["corr"][t]):
            ax.plot(i + (j - 0.5) * w, val, marker="D", mfc="none", mec=P.INK, ms=6, ls="none",
                    label="paper" if (i, j) == (0, 0) else None)
    ax.axhline(0, color=P.INK_MUTED, lw=0.8)
    ax.set_xticks(range(4), CORR_TYPES); ax.set_ylabel("Cross-validated tuning correlation")
    ax.set_title("(c) Tuning correlation between contexts / hands", fontsize=9)
    ax.legend(loc="upper right", fontsize=7); ax.grid(axis="x", visible=False)
    ax.set_ylim(-0.55, 1.12)

    # robustness: suppression for each block-set scheme
    ax = axes[2]
    names = list(summaries)
    for i, sch in enumerate(names):
        for k, (key, color) in enumerate((("ratio_R", P.UNI), ("ratio_L", P.BI))):
            m, lo, hi = summaries[sch][key]
            ax.errorbar(i + (k - 0.5) * 0.25, 1 - m, yerr=[[hi - m], [m - lo]], fmt="o", color=color,
                        capsize=3, label=("Right hand", "Left hand")[k] if i == 0 else None)
    ax.set_xticks(range(len(names)), [f"{n}\n(n={summaries[n]['n_usable']})" for n in names], fontsize=7)
    ax.set_ylabel("Suppression (1 − bi/uni)")
    ax.set_title("Robustness to block-set rule", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="x", visible=False)

    fig.suptitle("Fig 2b-c replication: bimanual suppression and decorrelation of tuning "
                 f"(days 1776, 1778, 1792, 1881, 1883; cvOLS={METHOD}; 95% bootstrap CI)", fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, fig_name)


# ----------------------------------------------------------------------------- main
def main(force=False, method="splithalf"):
    global METHOD
    METHOD = method
    outs = {s: OUT_DIR / method / f"tuning_change_{s}.json" for s in BS.SCHEMES}
    fig_name = f"fig2bc_tuning_change_{method}"
    if should_skip([*outs.values(), P.FIG_DIR / f"{fig_name}.png"], force):
        print("outputs exist; --force to redo")
        return
    days = {d: prepare_day(d) for d in BS.FIG2_DAYS}
    summaries, results = {}, {}
    for scheme in BS.SCHEMES:
        sets = []
        for d in BS.FIG2_DAYS:
            for k, blocks in enumerate(BS.block_sets(d, scheme)):
                r = analyse_set(days[d], blocks, seed=k)
                r["day"] = d
                sets.append(r)
        sm = summarize(sets)
        summaries[scheme], results[scheme] = sm, dict(sets=sets, summary=sm)
        atomic_write_json(outs[scheme], dict(
            scheme=scheme, method=METHOD, min_trials=MIN_TRIALS, window_bins=list(DEFAULT_WINDOW), paper=PAPER,
            unusable_sets=[(s["day"], s["blocks"], s["n_trials"]) for s in sets if not s["usable"]],
            **results[scheme]))
        print(f"\n[{scheme}] usable {sm['n_usable']}/{sm['n_sets']} sets")
        print(f"  suppression RH {100 * sm['suppression_R']:.1f}% (paper 12%)   "
              f"LH {100 * sm['suppression_L']:.1f}% (paper 34%)   t-test p={sm['ttest_ratio_R_vs_L_p']:.2g}")
        for t in CORR_TYPES:
            print(f"  corr {t:10s} x {sm['corr'][t + '_x'][0]:+.2f}  y {sm['corr'][t + '_y'][0]:+.2f}")

    s1d = supp_fig1d(days[1776])
    print(f"\nSupp Fig 1d (day 1776): RH/LH uni tuning strength = {s1d['ratio_R_over_L']:.2f} "
          f"(LH weaker by {100 * s1d['L_weaker_by']:.0f}%; paper: ~2x, 48%)")
    atomic_write_json(OUT_DIR / method / "supp_fig1d.json", s1d)
    plot(results[BS.MAIN_SCHEME], summaries, fig_name)
    print("wrote", P.FIG_DIR / f"{fig_name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--method", choices=["splithalf", "kfold"], default="splithalf",
                    help="cvOLS estimator (default splithalf; kfold = the paper's literal five-fold)")
    a = ap.parse_args()
    main(a.force, a.method)
