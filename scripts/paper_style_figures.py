"""Our replication results re-plotted in the layout of the paper's figures (for docs/figure_comparison.md).

Purpose
-------
The figures made during the replication (results/figures/*.png) use our own layout. To compare with
the published panels side by side, this script redraws the same results in the paper's format: the
same panel arrangement, chart types, colour roles, axis ranges and statistics where the paper states
them. It mostly re-plots saved results; it recomputes from the raw data only where the paper shows
something we never saved:

* Fig 2d: single-trial PCA projections in SD units and the PC1 time course (day 1881).
* Fig 3a-c: per-dimension cursor jitter (the paper reports x and y separately; ours pooled them).
* Fig 4d: decoded traces from the saved RNN checkpoints.
* Fig 4e / 5: MEDIAN time to acquisition with failures counted as 10 s (the paper's Fig 4e axis says
  "Median time to acquisition"; Fig 5 plots timeouts at exactly 10 s, so it is a median with failures
  at 10 s), and angular error per hand on bimanual trials (ours averaged the two hands).

Colours follow the paper's colour roles (e.g. unimanual red / bimanual cyan in Fig 1c) so panels can be
compared by eye; values are ours. Paper-format choices that change a number are noted in the figure
and in docs/figure_comparison.md.

Inputs : results/fig1..fig5, results/cache (saved results), Data/ (for the recomputed panels),
         results/checkpoints (Fig 4d traces)
Outputs: results/figures/paperstyle/ps_*.png (+ .svg); results/paperstyle/values.json (numbers shown)
Run    : pixi run paper-style            (about 2 min)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling figure scripts (fig3_decoding, fig4_rnn)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from bimanual import behavior as B  # noqa: E402
from bimanual.cache import RESULTS_DIR, atomic_write_json, read_jsonl  # noqa: E402
from bimanual.epochs import extract_windows  # noqa: E402
from bimanual.io import load_session, zscore_by_block  # noqa: E402
from bimanual.stats import bootstrap_ci  # noqa: E402
from bimanual.tuning import DEFAULT_WINDOW  # noqa: E402

OUT = RESULTS_DIR / "figures" / "paperstyle"
VALUES = RESULTS_DIR / "paperstyle" / "values.json"
R = RESULTS_DIR

# ---- the paper's colour roles (approximate hues sampled from the published figures)
RED, CYAN = "#d7261e", "#5cc8d0"                     # Fig 1c unimanual / bimanual
BLUE, PURPLE = "#1f6fb4", "#8a2a9c"                  # right / left hand everywhere
LBLUE, LPURPLE = "#9cbcdc", "#c79fd0"                # "altered" / linear (lighter) versions
CRIMSON, GOLD = "#a51c30", "#e8ae1d"                 # Fig 2c UniR-UniL, BiR-BiL; dPCA L x D, time
PC_UNIR, PC_UNIL, PC_BI = "#4b3b9a", "#2ec4b6", "#efb21d"  # Fig 2d
BI_RED = "#a8202a"                                   # Fig 5 bimanual squares
GRAY = "#bdbdbd"
# MATLAB "parula"-like map for the Fig 3d-e heat maps (anchor colours sampled from parula)
PARULA = LinearSegmentedColormap.from_list(
    "parula_like", ["#352a87", "#0f5cdd", "#1481d6", "#06a4ca", "#2eb7a4", "#87bf77", "#d1bb59", "#fec832", "#f9fb0e"])

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "normal", "legend.frameon": False,
                     "svg.fonttype": "none", "font.family": ["Arial", "DejaVu Sans"]})

VALS: dict = {}  # every number drawn, for the comparison document


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT / f"{name}.svg", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def label(ax, text, x=-0.12, y=1.06):
    """Bold lowercase panel letter as in the paper."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def ci_of(x, stat=np.mean):
    m, lo, hi = bootstrap_ci(np.asarray(x, float), stat=stat, n_boot=5000, seed=0)
    return float(m), float(lo), float(hi)


# ============================================================================ Fig 1c
def fig1c():
    d = json.loads((R / "fig1" / "tuning_curves.json").read_text())
    dirs = d["direction_order_deg"]
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.6), sharex=True)
    for j, e in enumerate(d["electrodes"]):
        for i, (hand, uni, bi) in enumerate((("Right hand", "uniR", "biR"), ("Left hand", "uniL", "biL"))):
            ax = axes[i, j]
            for key, col in ((uni, RED), (bi, CYAN)):
                c = d["curves"][key]
                m, lo, hi = (np.array(c[q][j]) for q in ("mean", "lo", "hi"))
                ax.fill_between(range(8), lo, hi, color=col, alpha=0.25, lw=0)
                ax.plot(range(8), m, "-o", color=col, ms=3, lw=1.5)
            ax.set_xticks(range(8), [f"{a}°" for a in dirs], fontsize=7)
            ax.axhline(0, color="#999", lw=0.5)
            if i == 0:
                ax.set_title(f"Electrode no. {e}")
            if j == 0:
                ax.set_ylabel(f"{hand}\nΔ spikes/s")
    # each electrode uses one y-range for both hands, as in the paper
    for j in range(3):
        lim = max(abs(np.array(axes[i, j].get_ylim())).max() for i in range(2))
        for i in range(2):
            axes[i, j].set_ylim(-lim, lim)
    fig.text(0.42, 0.5, "Unimanual", color=RED, ha="right", va="center", fontsize=11)
    fig.text(0.44, 0.5, "Bimanual", color=CYAN, ha="left", va="center", fontsize=11)
    fig.supxlabel("Movement direction (our angle convention; same order as the paper's glyphs)", fontsize=9)
    fig.suptitle("c   Tuning curves (ours, day 1750; shading = 95% bootstrap CI)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(h_pad=2.2)
    save(fig, "ps_fig1c")


# ============================================================================ Fig 2b-c
def fig2bc():
    j = json.loads((R / "fig2" / "splithalf" / "tuning_change_adjacent_all.json").read_text())
    sets = [s for s in j["sets"] if s["usable"]]
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(10.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    out = {}
    for k, (key, lab, col) in enumerate((("ratio_R", "BiR:UniR", BLUE), ("ratio_L", "BiL:UniL", PURPLE))):
        v = np.array([s[key] for s in sets])
        m, lo, hi = ci_of(v)
        out[lab] = dict(mean=m, lo=lo, hi=hi, n=len(v))
        ax.bar(k, m, width=0.6, color=col, edgecolor="black", lw=1.2)
        ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], color="black", capsize=5, lw=1.5)
        ax.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, s=8, color="#666", zorder=3)
        ax.text(k + 0.33, m, f"{m:.2f}", fontsize=8, va="bottom")
    ax.axhline(1, color="black", ls="--", lw=1.2)
    p = j["summary"]["ttest_ratio_R_vs_L_p"]
    ax.plot([0, 0, 1, 1], [1.28, 1.32, 1.32, 1.28], color="black", lw=0.8)
    ax.text(0.5, 1.34, f"p = {p:.1e}", ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["BiR:UniR", "BiL:UniL"]); ax.set_ylim(0, 1.5); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Magnitude ratio"); ax.set_title("Ratio of tuning strength")
    label(ax, "b", -0.3)
    VALS["fig2b"] = dict(out, p=p)

    cols = {"UniR-BiR": BLUE, "UniL-BiL": PURPLE, "UniR-UniL": CRIMSON, "BiR-BiL": GOLD}
    VALS["fig2c"] = {}
    for k, ax_name in enumerate(("x", "y")):
        ax = fig.add_subplot(gs[1 + k])
        for i, (ct, col) in enumerate(cols.items()):
            v = np.array([s["corr"][f"{ct}_{ax_name}"] for s in sets])
            v = v[np.isfinite(v)]
            m, lo, hi = ci_of(v)
            VALS["fig2c"][f"{ct}_{ax_name}"] = dict(mean=m, lo=lo, hi=hi, n=len(v))
            ax.bar(i, m, width=0.65, color=col, edgecolor="black", lw=1.2)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="black", capsize=4, lw=1.5)
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color="#666", zorder=3)
            ax.text(i + 0.35, m, f"{m:.2f}", fontsize=8, va="center")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(4), list(cols), fontsize=8)
        for t, col in zip(ax.get_xticklabels(), cols.values()):
            t.set_color(col)
        ax.set_ylim(-0.5, 1.5); ax.set_ylabel("Correlation")
        ax.set_title(f"{ax_name.upper()}-direction")
        if k == 0:
            label(ax, "c", -0.2)
    fig.suptitle(f"Ours: split-half cvOLS, {len(sets)} block sets from 5 days (bars = mean, error bars = 95% bootstrap CI; "
                 "dots = sets)", fontsize=8, x=0.01, ha="left")
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.98)
    save(fig, "ps_fig2bc")


# ============================================================================ Fig 2d
def fig2d():
    """Single-trial PCA projections, PC1 time course and dPCA marginalized variance."""
    s = load_session(1881)
    z = zscore_by_block(s.tx, s.block)
    ol = (s.trials.kind == "OL").to_numpy()
    X = np.nanmean(extract_windows(z, s.go[ol], DEFAULT_WINDOW), axis=1)
    mt = s.trials.move_type.to_numpy()[ol]
    go = s.go[ol]
    keep = ~np.isnan(X).any(axis=1)
    X, mt, go = X[keep], mt[keep], go[keep]
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    # orient PC1 so that unimanual-left is negative, as in the paper
    sign = -np.sign(((X[mt == 2] - mu) @ Vt[0]).mean())
    Vt[0] *= sign
    Vt[1] *= -np.sign(((X[mt == 2] - mu) @ Vt[1]).mean())  # PC2: unimanual-left negative, as the paper
    sc = (X - mu) @ Vt[:2].T
    # PC1 time course: z-scored activity smoothed with a Gaussian (sigma 2.5 bins = 50 ms), -500..1500 ms
    from scipy.ndimage import gaussian_filter1d
    zs = gaussian_filter1d(z, 2.5, axis=0)
    t_bins = np.arange(-25, 76)
    tc = np.stack([(zs[g + t_bins] - mu) @ Vt[0] for g in go])  # (trials, time)

    groups = ((1, "UniR", PC_UNIR), (2, "UniL", PC_UNIL), (3, "Bi", PC_BI))
    fig = plt.figure(figsize=(11, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 0.95], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    for k, lab, col in groups:
        m = mt == k
        ax.scatter(sc[m, 0], sc[m, 1], s=9, color=col, lw=0, label=lab)
    ax.set_xlabel("PC1 (SD)"); ax.set_ylabel("PC2 (SD)"); ax.grid(True, color="#ddd", lw=0.5)
    ax.set_title("Single trial firing rate projections", fontsize=9); ax.legend(fontsize=8, markerscale=1.8)
    label(ax, "d", -0.25)
    ax = fig.add_subplot(gs[1])
    tms = t_bins * 20
    for k, lab, col in groups:
        m = mt == k
        ax.plot(tms, tc[m].T, color=col, lw=0.3, alpha=0.18)
    for k, lab, col in groups:
        m = mt == k
        mean, sem = tc[m].mean(0), tc[m].std(0) / np.sqrt(m.sum())
        ax.plot(tms, mean, color=col, lw=2.2, label=lab)
        ax.plot(tms, mean + 2 * sem, color=col, lw=0.8); ax.plot(tms, mean - 2 * sem, color=col, lw=0.8)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlim(-500, 1500); ax.set_xlabel("Time (ms)"); ax.set_ylabel("Rate (SD)")
    ax.set_title("Projection onto PC1", fontsize=9); ax.grid(True, color="#ddd", lw=0.5)
    pre = {lab: float(tc[mt == k][:, t_bins < 0].mean()) for k, lab, _ in groups}
    VALS["fig2d_pc1_pre_go_mean"] = pre
    VALS["fig2d_n_trials"] = {lab: int((mt == k).sum()) for k, lab, _ in groups}

    ax = fig.add_subplot(gs[2])
    d = json.loads((R / "fig2" / "dpca.json").read_text())
    per = d["per_day"]["0-1000ms"]
    terms = (("laterality", "Laterality", BLUE), ("direction", "Direction", PURPLE),
             ("interaction", "L x D", CRIMSON), ("time", "Time", GOLD))
    VALS["fig2d_dpca_0_1000"] = {}
    for i, (key, lab, col) in enumerate(terms):
        v = np.array([per[day]["cv_frac"][key] for day in per])
        ax.bar(i, v.mean(), width=0.7, color=col, edgecolor="black", lw=1.2)
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=10, color="#666", zorder=3, marker="P")
        ax.text(i + 0.3, v.mean() + 0.005, f"{v.mean():.2f}", fontsize=8)
        VALS["fig2d_dpca_0_1000"][key] = float(v.mean())
    ax.set_xticks(range(4), [t[1] for t in terms], fontsize=8)
    for t, (_, _, col) in zip(ax.get_xticklabels(), terms):
        t.set_color(col)
    ax.set_ylim(0, 0.5); ax.set_ylabel("Fraction of variance"); ax.set_title("Marginalized variance (0–1000 ms)", fontsize=9)
    fig.suptitle("Ours: day 1881 open-loop trials (PCA on 300–700 ms trial means; PC1 time course smoothed σ = 50 ms, "
                 "mean ± 2 SEM); dPCA: 5 days, cross-validated", fontsize=8, x=0.01, ha="left")
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98)
    save(fig, "ps_fig2d")


# ============================================================================ Fig 3a-c, Supp Fig 2
def fig3_data():
    import fig3_decoding as F3
    s = load_session(F3.DAY)
    z = zscore_by_block(s.tx, s.block)
    ol = (s.trials.kind == "OL").to_numpy()
    mt_all = s.trials.move_type.to_numpy()
    uni = F3.build(s, z, ol & np.isin(mt_all, (1, 2)))
    bi = F3.build(s, z, ol & (mt_all == 3))
    pred = {k: np.load(R / "cache" / f"fig3_direction_{k}.npz")["pred"]
            for k in ("uni_ridge", "uni_ffn", "uni_ridge_nolat", "uni_ffn_nolat", "bi_ridge", "bi_ffn")}
    res = json.loads((R / "fig3" / "decoding_direction.json").read_text())["results"]
    return uni, bi, pred, res


def jitter_per_dim(p, g, mt):
    """Paper's definition per dimension: mean |decoded v| while the hand rests / while it moves."""
    mtb = mt[g]
    out = {}
    for d, (hand, own, other) in enumerate((("R", 1, 2), ("R", 1, 2), ("L", 2, 1), ("L", 2, 1))):
        a = np.abs(p[:, d])
        out[("Rx", "Ry", "Lx", "Ly")[d]] = float(a[mtb == other].mean() / a[mtb == own].mean())
    return out


def fig3abc(uni, pred, res):
    Xu, Yu, gu, mtu, t0u = uni
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.45, wspace=0.25)

    # (a) 8-s traces: the first 20 trials in time order, windows concatenated (as our original figure)
    sub = gs[0, 0].subgridspec(2, 1, hspace=0.35)
    order = np.argsort(t0u)[:20]
    rows = np.concatenate([np.where(gu == k)[0] for k in order])
    t = np.arange(len(rows)) * 0.02
    for i, (dim, col, lcol, nm) in enumerate(((0, BLUE, LBLUE, "Right effector"), (2, PURPLE, LPURPLE, "Left effector"))):
        ax = fig.add_subplot(sub[i])
        ax.plot(t, Yu[rows, dim], color=GRAY, lw=3.5, label="Target Velocity")
        ax.plot(t, pred["uni_ridge"][rows, dim], color=lcol, lw=1, ls="--", label="Linear Regression")
        ax.plot(t, pred["uni_ffn"][rows, dim], color=col, lw=1, label="Feed Forward Network")
        ax.set_ylim(-1.6, 1.6); ax.set_xlim(0, 8); ax.text(7.9, 1.25, nm, color=col, ha="right", fontsize=9)
        if i == 0:
            label(ax, "a", -0.1)
            ax.set_title("Offline decoded velocity traces (single 20-ms bins, unsmoothed)", fontsize=9, loc="right")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s; concatenated 400-ms windows)"); ax.legend(fontsize=7, ncol=3, loc="lower left",
                                                                             bbox_to_anchor=(0, -0.62))
        ax.set_ylabel("X-direction\nvelocity (a.u.)", fontsize=8)

    # (b) r bars: [LR | FFN] x [X | Y] x [RH, RH-lat, LH, LH-lat]
    ax = fig.add_subplot(gs[0, 1])
    VALS["fig3b"] = {}
    x0 = 0
    for dec in ("ridge", "ffn"):
        for dim_xy in (0, 1):
            for k, (hand_dim, col, lat) in enumerate(((dim_xy, BLUE, False), (dim_xy, BLUE, True),
                                                      (2 + dim_xy, PURPLE, False), (2 + dim_xy, PURPLE, True))):
                key = f"uni_{dec}" + ("_nolat" if lat else "")
                r = res[key]["r"][hand_dim]
                VALS["fig3b"][f"{key}_{('Rx', 'Ry', 'Lx', 'Ly')[hand_dim]}"] = r
                x = x0 + k * 0.22
                ax.bar(x, r, width=0.2, color=col if not lat else "white", edgecolor=col, hatch="////" if lat else None, lw=1)
                ax.text(x, r + 0.015, f"{r:.2f}", rotation=60, fontsize=7, ha="left", va="bottom")
            x0 += 1.1
    ax.set_xticks([0.33, 1.43, 2.53, 3.63], ["X-direction", "Y-direction", "X-direction", "Y-direction"], fontsize=8)
    ax.text(0.88, 1.03, "Linear Regression", ha="center", transform=ax.get_xaxis_transform())
    ax.text(3.08, 1.03, "Feed Forward Network", ha="center", transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1); ax.set_ylabel("Correlation"); label(ax, "b", -0.08, 1.1)
    ax.set_title("Offline decoding accuracy of real data", fontsize=9, pad=22)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Right hand"), Patch(color=PURPLE, label="Left hand"),
                       Patch(facecolor="white", edgecolor="black", hatch="////", label="Laterality dim. removed")],
              fontsize=7, loc="upper right", ncol=3, bbox_to_anchor=(1, -0.08))

    # (c) jitter bars, per dimension, paper's layout
    ax = fig.add_subplot(gs[1, 1])
    VALS["fig3c"] = {}
    x0 = 0
    jit = {k: jitter_per_dim(pred[k], gu, mtu) for k in ("uni_ridge", "uni_ridge_nolat", "uni_ffn", "uni_ffn_nolat")}
    for dec in ("ridge", "ffn"):
        for dim_xy in (0, 1):
            for k, (hand, col, lat) in enumerate((("R", BLUE, False), ("R", BLUE, True), ("L", PURPLE, False), ("L", PURPLE, True))):
                key = f"uni_{dec}" + ("_nolat" if lat else "")
                v = jit[key][f"{hand}{'xy'[dim_xy]}"]
                VALS["fig3c"][f"{key}_{hand}{'xy'[dim_xy]}"] = v
                x = x0 + k * 0.22
                ax.bar(x, v, width=0.2, color=col if not lat else "white", edgecolor=col, hatch="////" if lat else None, lw=1)
                ax.text(x, v + 0.015, f"{v:.2f}", rotation=60, fontsize=7, ha="left", va="bottom")
            x0 += 1.1
    ax.set_xticks([0.33, 1.43, 2.53, 3.63], ["X-direction", "Y-direction", "X-direction", "Y-direction"], fontsize=8)
    ax.text(0.88, 1.03, "Linear Regression", ha="center", transform=ax.get_xaxis_transform())
    ax.text(3.08, 1.03, "Feed Forward Network", ha="center", transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1); ax.set_ylabel("Cursor jitter\n(mean |v| at rest / moving)"); label(ax, "c", -0.08, 1.1)
    ax.set_title("Cursor jitter during rest epochs in offline real data (per dimension)", fontsize=9, pad=22)

    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    ax.text(0.02, 0.5, "Ours: day 1883 open-loop unimanual trials, 300–700 ms after go,\n5-fold CV over trials. "
            "Target = unit cursor→target direction (#11).\n\nCursor jitter  =  avg |velocity| during rest\n"
            "                       ÷ avg |velocity| during movement\n(per x / y dimension, as the paper)",
            fontsize=9, va="center")
    fig.suptitle("Unimanual movement decoding (ours, paper layout)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, "ps_fig3abc")


def supp2(uni, bi, pred, res):
    Xu, Yu, gu, mtu, t0u = uni
    Xb, Yb, gb, mtb, t0b = bi
    fig = plt.figure(figsize=(12, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.2, 0.9], wspace=0.3)

    # (a) decoded velocity magnitude per hand, unimanual trials (all bins: moving and resting)
    sub = gs[0].subgridspec(1, 2, wspace=0.3)
    for i, (hand, sl) in enumerate((("Right hand", slice(0, 2)), ("Left hand", slice(2, 4)))):
        ax = fig.add_subplot(sub[i])
        for key, lab, col in (("uni_ridge", "Linear Regression", "#1a9850"), ("uni_ffn", "Feed forward network (FFN)", "#f05a28"),
                              ("uni_ffn_nolat", "FFN (laterality dim. removed)", "#f2b705")):
            sp = np.hypot(*pred[key][:, sl].T)
            w = np.ones_like(sp) / len(sp)
            ax.hist(sp, bins=np.linspace(0, 1.8, 46), weights=w, histtype="step", color=col, lw=1.6, label=lab)
        ax.set_title(hand, fontsize=9); ax.set_xlim(0, 1.8)
        if i == 0:
            ax.set_ylabel("Probability"); label(ax, "a", -0.35)
            ax.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(2.3, -0.2), ncol=3)
        ax.set_xlabel("Decoded velocity magnitude", fontsize=8)

    # (b) bimanual traces
    sub = gs[1].subgridspec(2, 1, hspace=0.3)
    order = np.argsort(t0b)[:20]
    rows = np.concatenate([np.where(gb == k)[0] for k in order])
    t = np.arange(len(rows)) * 0.02
    for i, (dim, col, lcol) in enumerate(((0, BLUE, LBLUE), (2, PURPLE, LPURPLE))):
        ax = fig.add_subplot(sub[i])
        ax.plot(t, Yb[rows, dim], color=GRAY, lw=3)
        ax.plot(t, pred["bi_ridge"][rows, dim], color=lcol, lw=0.9, ls="--")
        ax.plot(t, pred["bi_ffn"][rows, dim], color=col, lw=0.9)
        ax.set_ylim(-1.6, 1.6); ax.set_xlim(0, 8)
        if i == 0:
            label(ax, "b", -0.12); ax.set_title("Offline decoded velocity traces (bimanual)", fontsize=9); ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")
        ax.set_ylabel("X vel. (a.u.)", fontsize=8)

    # (c) bimanual decoding accuracy
    ax = fig.add_subplot(gs[2])
    VALS["supp2c"] = {}
    for i, dim in enumerate(range(4)):
        for k, (key, col) in enumerate((("bi_ridge", (LBLUE, LPURPLE)), ("bi_ffn", (BLUE, PURPLE)))):
            r = res[key]["r"][dim]
            VALS["supp2c"][f"{key}_{('Rx', 'Ry', 'Lx', 'Ly')[dim]}"] = r
            x = i + (k - 0.5) * 0.36
            ax.bar(x, r, width=0.34, color=col[0] if dim < 2 else col[1])
            ax.text(x, r + 0.02, f"{r:.2f}", rotation=60, fontsize=7, va="bottom")
    ax.set_xticks(range(4), ["X-dir.", "Y-dir.", "X-dir.", "Y-dir."], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Correlation"); label(ax, "c", -0.2)
    ax.set_title("Bimanual trials only (light = linear, dark = FFN)", fontsize=8)
    fig.suptitle(f"Supp Fig 2 (ours): day 1883 open-loop; bimanual decoders trained on {len(np.unique(gb))} bimanual "
                 "trials only (5-fold CV)", x=0.01, ha="left", fontsize=9)
    fig.subplots_adjust(top=0.8, bottom=0.3, left=0.05, right=0.99)
    save(fig, "ps_supp2")


# ============================================================================ Fig 3d-e
def fig3de():
    cells = read_jsonl(R / "fig3" / "sim_grid.jsonl")
    rhos = sorted({c["rho"] for c in cells})
    lats = sorted({c["lat"] for c in cells})
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), gridspec_kw=dict(hspace=0.45))
    VALS["fig3de"] = {}
    for row, (metric, cmap_lab) in enumerate((("r_mean", "Decoding performance\n(correlation coefficient)"),
                                              ("jitter", "Cursor jitter"))):
        for col, dec in enumerate(("ridge", "ffn")):
            M = np.full((len(rhos), len(lats)), np.nan)
            for c in cells:
                v = c[dec]["r_mean"] if metric == "r_mean" else 0.5 * (c[dec]["jitter_R"] + c[dec]["jitter_L"])
                M[rhos.index(c["rho"]), lats.index(c["lat"])] = v
            VALS["fig3de"][f"{dec}_{metric}"] = M.tolist()
            ax = axes[row, col]
            im = ax.imshow(M, origin="lower", cmap=PARULA, vmin=0, vmax=1, aspect="auto")
            for i in range(len(rhos)):
                for j in range(len(lats)):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
            ax.set_xticks(range(len(lats)), [f"×{v:g}" for v in lats]); ax.set_yticks(range(len(rhos)), [f"{v:g}" for v in rhos])
            ax.grid(False)
            if row == 0:
                ax.set_title("Linear Regression" if dec == "ridge" else "Feed Forward Network", fontsize=9)
            if col == 0:
                ax.set_ylabel("Correlation between effectors")
            ax.set_xlabel("Laterality scale (× real data)")
        fig.colorbar(im, ax=axes[row, :], label=cmap_lab, fraction=0.025)
        label(axes[row, 0], "de"[row], -0.3)
    fig.suptitle("Simulated data (ours): rows = RH–LH tuning correlation ρ, columns = laterality as a multiple of the real "
                 "size; jitter = mean of both hands", x=0.01, ha="left", fontsize=8)
    save(fig, "ps_fig3de")


# ============================================================================ Fig 4d
def fig4d():
    import fig4_rnn as F4
    from bimanual import rnn as RN
    summ = json.loads((R / "fig4" / "summary.json").read_text())
    recs = read_jsonl(F4.TRIALS)
    best_raw = next(r for r in recs if r["run_id"] == summ["best"]["raw"]["run_id"])
    longclip = [r for r in recs if r["tag"] == "longclip"]
    data = F4.Data()
    X, V = data.eval_arrays("test")
    # 10-s window (50-bin steps) where the hand that moves least still moves most: maximise
    # min over hands of the number of bins with |x target| > 0.5
    a0 = max(range(0, len(V) - 500, 50),
             key=lambda a: min((np.abs(V[a:a + 500, 0]) > 0.5).sum(), (np.abs(V[a:a + 500, 2]) > 0.5).sum()))
    sl = slice(a0, a0 + 500)
    tt = np.arange(500) * 0.02
    VALS["fig4d_window_start_bin"] = int(a0)
    fig = plt.figure(figsize=(10, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.3)
    sub = gs[0].subgridspec(2, 1, hspace=0.25)
    dec = {}
    for name, rec in (("raw", best_raw), ("aug", longclip[0])):
        m = RN.load_from_record(rec, X.shape[1], F4.CKPT_DIR)
        dec[name], _ = RN.decode(m, X)
    for i, (dim, col, lcol, nm) in enumerate(((0, BLUE, LBLUE, "Right effector"), (2, PURPLE, LPURPLE, "Left effector"))):
        ax = fig.add_subplot(sub[i])
        ax.plot(tt, V[sl, dim], color=GRAY, lw=3.5, label="Target Velocity")
        ax.plot(tt, dec["raw"][sl, dim], color=col, lw=1.2, label="Raw training data")
        ax.plot(tt, dec["aug"][sl, dim], color=lcol, lw=1.2, label="Altered training data (2–8-s clips)")
        ax.set_ylim(-1.1, 1.1); ax.set_xlim(0, 10); ax.text(9.9, 0.8, nm, color=col, ha="right")
        ax.set_ylabel("X vel. (a.u.)", fontsize=8)
        if i == 0:
            label(ax, "d", -0.12); ax.set_xticklabels([])
            ax.set_title("Offline unimanual decoding, trial day 1869 (held-out block 7)", fontsize=9)
        else:
            ax.set_xlabel("Time (s)"); ax.legend(fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.3))
    ax = fig.add_subplot(gs[1])
    raw = summ["seed_spread"]["raw"]; aug = summ["aug_longclip_2to8s"]; short = summ["seed_spread"]["aug"]
    VALS["fig4d"] = {}
    xs = 0
    for hand, dark, light in (("R", BLUE, LBLUE), ("L", PURPLE, LPURPLE)):
        for k, (lab, src, col, hatch) in enumerate((("raw", raw, dark, None), ("altered 2–8 s", aug, light, None),
                                                    ("altered 0.2–0.8 s", short, "white", "////"))):
            v = float(np.mean(src[f"test_r_{hand}"]))
            VALS["fig4d"][f"{lab}_{hand}"] = v
            ax.bar(xs + k * 0.3, v, width=0.28, color=col, edgecolor=light if hatch else None, hatch=hatch, lw=1)
            ax.text(xs + k * 0.3, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        xs += 1.2
    ax.set_xticks([0.3, 1.5], ["Right hand", "Left hand"]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Avg. correlation across dims."); ax.set_title("Offline performance (mean of 3 seeds)", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Raw training data"), Patch(color=LBLUE, label="Altered, 2–8-s clips (main)"),
                       Patch(facecolor="white", edgecolor=LBLUE, hatch="////", label="Altered, 200–800-ms clips")],
              fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2)
    fig.subplots_adjust(top=0.9, bottom=0.25, left=0.07, right=0.98)
    save(fig, "ps_fig4d")


# ============================================================================ Fig 4e, Fig 5
def online_table() -> pd.DataFrame:
    """One row per closed-loop trial with per-hand angular errors (the saved table averaged them)."""
    import fig5_behavior as F5
    from diagnose_differences import net_angle  # angle of the NET displacement over 300-500 ms
    rows = []
    for day in F5.CL_DAYS:
        s = load_session(day)
        t = s.trials
        for r in t[t.kind == "CL"].itertuples():
            ok = r.move_bins < F5.TIMEOUT_BINS
            rows.append(dict(day=day, block=r.block, decoder=r.decoder, gain=r.gain, variant=r.variant,
                             uni_only=r.uni_only, move_type=F5.MT_NAME[r.move_type], success=bool(ok),
                             tta_s=r.move_bins * 0.02 if ok else np.nan, tta10=min(r.move_bins * 0.02, 10.0) if ok else 10.0,
                             ang_R=B.angular_error(s.Cp, s.Tp, r.go, "R") if r.move_type in (1, 3) else np.nan,
                             ang_L=B.angular_error(s.Cp, s.Tp, r.go, "L") if r.move_type in (2, 3) else np.nan,
                             net_R=net_angle(s.Cp, s.Tp, r.go, "R") if r.move_type in (1, 3) else np.nan,
                             net_L=net_angle(s.Cp, s.Tp, r.go, "L") if r.move_type in (2, 3) else np.nan,
                             go=int(r.go)))
    return pd.DataFrame(rows)


def med(x):
    """Median TTA (failures = 10 s) with a 95% bootstrap CI."""
    return ci_of(x, stat=np.median)


def fig4e(df):
    fig = plt.figure(figsize=(9.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
    sub = gs[0].subgridspec(2, 1, hspace=0.35)
    s = load_session(1869)
    for i, (var, col) in enumerate((("raw", BLUE), ("augmented", LBLUE))):
        d = df[(df.day == 1869) & (df.variant == var) & (df.move_type == "uniR")]
        g0 = int(d.go.iloc[0])
        step = np.array([np.hypot(*(s.Cp[0:2, t + 1] - s.Cp[0:2, t])) for t in range(g0 - 25, g0 + 575)])
        # a jump of > 50 position units in one 20-ms bin (2500 units/s, about 6x the fastest decoded movement)
        # is the task resetting the cursor between trials, not decoded movement: blank it
        sp = np.where(step > 50, np.nan, step / 0.02)
        ax = fig.add_subplot(sub[i])
        ax.plot(np.arange(len(sp)) * 0.02, sp / np.nanpercentile(sp, 99.5), color=col, lw=1)
        ax.set_ylim(0, 1.15); ax.set_xlim(0, 12); ax.set_ylabel("Cursor speed\n(a.u.)", fontsize=8)
        if i == 0:
            label(ax, "e", -0.15); ax.set_xticklabels([])
            ax.set_title("Example online right-cursor speed, day 1869 (12 s from a uniR go cue)", fontsize=9)
        else:
            ax.set_xlabel("Time (s)")
        ax.text(11.9, 1.0, "Raw training data" if var == "raw" else "Altered training data", ha="right", color=col, fontsize=8)
    ax = fig.add_subplot(gs[1])
    VALS["fig4e_median_tta10"] = {}
    for day, filled in ((1867, True), (1869, False)):
        for j, (mtp, dark, light) in enumerate((("uniR", BLUE, LBLUE), ("uniL", PURPLE, LPURPLE))):
            pts = []
            for k, (var, col) in enumerate((("raw", dark), ("augmented", light))):
                x = df[(df.day == day) & (df.variant == var) & (df.move_type == mtp)].tta10
                m, lo, hi = med(x)
                VALS["fig4e_median_tta10"][f"{day}_{mtp}_{var}"] = dict(median=m, lo=lo, hi=hi, n=int(len(x)))
                xpos = j * 2 + k * 0.6 + (0.08 if not filled else 0)
                ax.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt="o", color=col, mfc=col if filled else "white",
                            ms=7, capsize=0, lw=1.2)
                pts.append((xpos, m))
            ax.plot(*zip(*pts), color="black", lw=0.8)
    ax.axhline(10, color=GRAY, ls="--", lw=1); ax.text(1.3, 10.15, "Timeout", color="#888", fontsize=8, ha="center")
    ax.set_xticks([0.3, 2.3], ["Right hand", "Left hand"]); ax.set_ylim(0, 10.8)
    ax.set_ylabel("Median time to acquisition (s)\n(failures = 10 s)")
    ax.set_title("Online performance (● 1867, ○ 1869; dark = raw, light = altered)", fontsize=8)
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.98)
    save(fig, "ps_fig4e")


def fig5(df):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.25)
    # Fig 5a sessions: the 4 standard RNN evaluation days plus the SIMULTANEOUS-strategy blocks of 1881 and
    # 1883. Six sessions, as the paper plots; their values match the paper's six points (see
    # docs/figure_comparison.md), and Supp Table 2 lists 1881/1883 under Fig 5A (register #29).
    sessions = ((1776, None), (1778, None), (1790, None), (1792, None), (1881, "simultaneous"), (1883, "simultaneous"))

    def sess(day, var):
        d = df[(df.day == day) & (df.decoder == "RNN")]
        return d if var is None else d[d.variant == var]

    # (a) TTA per session: uniR, uniL, bi
    ax = fig.add_subplot(gs[0, 0])
    VALS["fig5a"] = {}
    for d_i, (day, var) in enumerate(sessions):
        pts = []
        for j, (mtp, col, mk) in enumerate((("uniR", BLUE, "o"), ("uniL", PURPLE, "o"), ("bi", BI_RED, "s"))):
            x = sess(day, var)
            x = x[x.move_type == mtp]
            m, lo, hi = med(x.tta10)
            mean_s = float(x.tta_s.mean())
            VALS["fig5a"][f"{day}{'_sim' if var else ''}_{mtp}"] = dict(median_tta10=m, lo=lo, hi=hi, mean_successful=mean_s,
                                                 success=float(x.success.mean()), n=int(len(x)))
            xp = j + (d_i - 2.5) * 0.07
            ax.errorbar(xp, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=6, lw=1)
            pts.append((xp, m))
        ax.plot(*zip(*pts), color=GRAY, lw=1.2, zorder=0)
    ax.axhline(10, color=GRAY, ls=":", lw=1); ax.text(-0.35, 10.1, "(Timeout)", fontsize=8)
    ax.set_xticks(range(3), ["Right only", "Left only", "Both"]); ax.set_xlim(-0.4, 2.5); ax.set_ylim(0, 10.6)
    ax.set_ylabel("Median time to acquisition (s)\n(failures = 10 s)")
    ax.set_title("RNN two-cursor control: 1776, 1778, 1790, 1792 + 1881/1883 simultaneous", fontsize=9); label(ax, "a", -0.15)

    # (a) angular error per hand: uni (circles) vs bi (squares)
    ax = fig.add_subplot(gs[0, 1])
    VALS["fig5a_ang"] = {}
    for d_i, (day, var) in enumerate(sessions):
        for h, (hand, uni_mt, col, x0) in enumerate((("R", "uniR", BLUE, 0), ("L", "uniL", PURPLE, 2.2))):
            pts = []
            for k, (mtp, mk) in enumerate(((uni_mt, "o"), ("bi", "s"))):
                x = sess(day, var)
                ours_def = x[x.move_type == mtp][f"ang_{hand}"].dropna()
                x = x[x.move_type == mtp][f"net_{hand}"].dropna()
                m, lo, hi = ci_of(x, stat=np.median)  # paper-consistent: net displacement, median over trials
                VALS["fig5a_ang"][f"{day}{'_sim' if var else ''}_{mtp}_{hand}"] = dict(
                    net_median=m, lo=lo, hi=hi, ours_per_bin_mean=float(ours_def.mean()), n=int(len(x)))
                xp = x0 + k + (d_i - 2.5) * 0.07
                ax.errorbar(xp, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=6, lw=1)
                pts.append((xp, m))
            ax.plot(*zip(*pts), color=GRAY, lw=1.2, zorder=0)
    ax.set_xticks([0, 1, 2.2, 3.2], ["Right, uni", "Right, bi", "Left, uni", "Left, bi"])
    ax.set_ylim(0, 90); ax.set_ylabel("Angular error (deg), median ± 95% CI")
    ax.set_title("Angle of net cursor displacement 300–500 ms vs target (● uni, ■ bi)", fontsize=9)

    # (b) sequential vs simultaneous, per day
    sub = gs[1, 0].subgridspec(1, 2, wspace=0.15)
    VALS["fig5b"] = {}
    for i, day in enumerate((1881, 1883)):
        ax = fig.add_subplot(sub[i])
        for var, col, ls in (("simultaneous", GRAY, "--"), ("sequential", "black", "-")):
            pts = []
            for j, (mtp, mcol, mk) in enumerate((("uniR", BLUE, "o"), ("uniL", PURPLE, "o"), ("bi", BI_RED, "s"))):
                x = df[(df.day == day) & (df.variant == var) & (df.move_type == mtp)].tta10
                m, lo, hi = med(x)
                VALS["fig5b"][f"{day}_{var}_{mtp}"] = dict(median=m, lo=lo, hi=hi, n=int(len(x)))
                xp = j + (0.06 if var == "sequential" else -0.06)
                ax.errorbar(xp, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=mcol, ms=6, lw=1)
                pts.append((xp, m))
            ax.plot(*zip(*pts), color=col, ls=ls, lw=1.5, zorder=0, label=f"{var.capitalize()} {'bimanual' if var == 'simultaneous' else 'unimanual'}")
        ax.set_title(f"Trial day {day}", fontsize=9); ax.set_ylim(0, 10.6); ax.set_xlim(-0.4, 2.4)
        ax.set_xticks(range(3), ["R", "L", "Both"])
        if i == 0:
            ax.set_ylabel("Median TTA (s), failures = 10 s"); label(ax, "b", -0.3)
            ax.legend(fontsize=7, loc="upper left")
        else:
            ax.set_yticklabels([])

    # (c) linear vs RNN, per day; bimanual task (solid) vs unimanual task (dashed); pooled over gains
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.15)
    VALS["fig5c"] = {}
    for i, day in enumerate((1853, 1855)):
        ax = fig.add_subplot(sub[i])
        for dec, col in (("linear", "#d7261e"), ("RNN", "black")):
            for uo, ls in ((False, "-"), (True, "--")):
                d = df[(df.day == day) & (df.decoder == dec) & (df.uni_only == uo)]
                if d.empty:
                    continue
                pts = []
                for j, (mtp, mcol, mk) in enumerate((("uniR", BLUE, "o"), ("uniL", PURPLE, "o"), ("bi", BI_RED, "s"))):
                    x = d[d.move_type == mtp].tta10
                    if x.empty:
                        continue
                    m, lo, hi = med(x)
                    VALS["fig5c"][f"{day}_{dec}_{'unitask' if uo else 'bitask'}_{mtp}"] = dict(
                        median=m, lo=lo, hi=hi, n=int(len(x)), gains=sorted(d.gain.dropna().unique().tolist()))
                    xp = j + (0.06 if dec == "RNN" else -0.06)
                    ax.errorbar(xp, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=mcol, ms=6, lw=1)
                    pts.append((xp, m))
                ax.plot(*zip(*pts), color=col, ls=ls, lw=1.5, zorder=0,
                        label=f"{'Linear regression' if dec == 'linear' else 'RNN'}, {'unimanual' if uo else 'bimanual'} task")
        ax.set_title(f"Trial day {day}", fontsize=9); ax.set_ylim(0, 10.6); ax.set_xlim(-0.4, 2.4)
        ax.set_xticks(range(3), ["R", "L", "Both"])
        if i == 0:
            ax.set_ylabel("Median TTA (s), failures = 10 s"); label(ax, "c", -0.3)
        else:
            ax.set_yticklabels([]); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(-1.1, -0.12), ncol=2)
    fig.suptitle("Fig 5 (ours, from the closed-loop logs): medians with 95% bootstrap CIs; Fig 5c pools all decoder "
                 "gains of each block type", x=0.01, ha="left", fontsize=9)
    save(fig, "ps_fig5")


# ============================================================================ Supp Fig 1c-e
def supp1(df_sets=None):
    a = json.loads((R / "fig1" / "supp_fig1_anova.json").read_text())
    j = json.loads((R / "fig2" / "splithalf" / "tuning_change_adjacent_all.json").read_text())
    fig = plt.figure(figsize=(10, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.55, 1.3], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    order = (("uniR", "UniR", "#a39441"), ("biR", "BiR", "#79a964"), ("uniL", "UniL", "#8080d0"), ("biL", "BiL", "#b65fae"))
    for i, (k, lab, col) in enumerate(order):
        v = a["tuned_counts"][k]
        ax.bar(i, v, color=col, width=0.75); ax.text(i, v + 1.5, str(v), ha="center", fontsize=9)
    ax.set_xticks(range(4), [o[1] for o in order], rotation=45); ax.set_ylim(0, 100)
    ax.set_ylabel("Number of channels"); ax.set_title(f"Channels with significant tuning\n(p < {a['p_threshold']:g})", fontsize=9)
    label(ax, "c", -0.3)
    ax = fig.add_subplot(gs[1])
    v = np.array([s["strength"]["uniR"] / s["strength"]["uniL"] for s in j["sets"] if s["usable"]])
    m = float(v.mean())
    d1776 = json.loads((R / "fig2" / "splithalf" / "supp_fig1d.json").read_text())["ratio_R_over_L"]
    VALS["supp1d"] = dict(per_set_mean=m, per_set_median=float(np.median(v)), n_sets=int(len(v)), day1776=d1776)
    ax.bar(0, m, color="#9c8178", width=0.6)
    ax.scatter(np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=6, color="#333", zorder=3)
    ax.axhline(1, color="black", ls="--", lw=0.8); ax.text(0.33, m, f"{m:.2f}", fontsize=8)
    ax.set_xticks([0], ["UniR:UniL"]); ax.set_xlim(-0.6, 0.8); ax.set_ylabel("Magnitude ratio")
    ax.set_title("Ratio of unimanual\ntuning strength", fontsize=9); label(ax, "d", -0.6)
    ax = fig.add_subplot(gs[2])
    n = a["n_electrodes_by_n_contexts"]
    pie_cols = ["#3b2a9a", "#2f80ed", "#17b3b0", "#c7c13a", "#f4ef2a"]
    ax.pie(n, colors=pie_cols, startangle=90, counterclock=False, labels=[str(x) for x in n], labeldistance=0.7,
           textprops=dict(color="white", fontsize=9, fontweight="bold"))
    ax.set_title("Channels tuned to 0–4 movement sets\n(UniR, UniL, BiR, BiL)", fontsize=9)
    ax.legend([f"{i}" for i in range(5)], title="sets", fontsize=7, title_fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    label(ax, "e", -0.35, 1.12)
    fig.suptitle("Supp Fig 1c–e (ours): day 1750 one-way ANOVA over 4 direction quadrants, 300–700 ms; "
                 "d = per block set, 5 days", x=0.01, ha="left", fontsize=8)
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.07, right=0.93)
    save(fig, "ps_supp1")


# ============================================================================ Supp Fig 4a
def supp4():
    d = json.loads((R / "fig3" / "sim_real_stats.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2))
    for row, (src, lab, fb, fw) in enumerate((("scatter", "Real unimanual right data", d["fsnr_single_bin"], d["fsnr_window"]),
                                              ("sim_scatter", "Simulated unimanual right data",
                                               d["sim_fsnr_single_bin"], d["sim_fsnr_window"]))):
        for col, (yk, pk, f, nm) in enumerate((("y_bin", "p_bin", fb, "Single-bin decoding"),
                                               ("y_win", "p_win", fw, "Window-average decoding"))):
            ax = axes[row, col]
            y, p = np.array(d[src][yk]), np.array(d[src][pk])
            ang = np.degrees(np.arctan2(y[:, 1], y[:, 0])) % 360
            n = min(len(p), 900)
            idx = np.random.default_rng(0).choice(len(p), n, replace=False)
            ax.scatter(p[idx, 0], p[idx, 1], c=ang[idx], cmap="hsv", vmin=0, vmax=360, s=5, lw=0)
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
            ax.text(-1.9, 1.75, f"fSNR = {f:.2f}", fontsize=8)
            th = np.linspace(0, 2 * np.pi, 60)
            ax.scatter(1.6 + 0.25 * np.cos(th), 1.6 + 0.25 * np.sin(th), c=np.degrees(th), cmap="hsv", s=2)
            if row == 0:
                ax.set_title(nm, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{lab}\nDecoded y-velocity", fontsize=8)
            if row == 1:
                ax.set_xlabel("Decoded x-velocity", fontsize=8)
    label(axes[0, 0], "a", -0.35)
    fig.suptitle("Supp Fig 4a (ours): day 1883, cross-validated linear decoding; 900 random points per panel", fontsize=8,
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "ps_supp4")


def main():
    fig1c(); fig2bc(); fig2d()
    uni, bi, pred, res = fig3_data()
    fig3abc(uni, pred, res); supp2(uni, bi, pred, res); fig3de()
    fig4d()
    df = online_table()
    (R / "paperstyle").mkdir(parents=True, exist_ok=True)
    df.to_csv(R / "paperstyle" / "online_trials_per_hand.csv", index=False)
    fig4e(df); fig5(df)
    supp1(); supp4()
    atomic_write_json(VALUES, VALS)
    print("wrote", VALUES)


if __name__ == "__main__":
    main()
