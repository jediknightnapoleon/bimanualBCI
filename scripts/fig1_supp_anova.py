"""Step 1.2: Supp Fig 1b,c,e, single-electrode tuning to each movement context (day 1750).

Purpose
-------
The paper's Supplementary Fig 1 shows that tuning to unimanual and bimanual
movement is intermixed across electrodes:

* (b) the tuning strength (FVAF) of every electrode in each context;
* (c) the number of electrodes significantly tuned in each context. The text
  says these show "greater preference for right hand tuning across both
  movement contexts";
* (e) how many of the 4 contexts each electrode is tuned to (0-4).

Method (Methods, "Single-electrode tuning")
-------------------------------------------
* Contexts: uniR (RH moves alone), uniL (LH alone), biR (bimanual trials,
  grouped by the RH direction), biL (bimanual trials, grouped by the LH
  direction).
* Each trial gives one rate per electrode: the mean over bins 15..35 after the
  go cue ("300-700 ms"), block-mean removed, in spikes/s.
* A 1-way ANOVA across the 4 direction quadrants. An electrode is "tuned" if
  p < 1e-5.
* FVAF = SS_direction / SS_total (stats.anova_fvaf).

ASSUMPTION (register #6): quadrants are axis-aligned, from the cursor->target
vector at the go cue.
ASSUMPTION (register #18): the same 300-700 ms window as the encoding
analyses; the paper doesn't state the window for this ANOVA.
DEVIATION (register #19): Supp Fig 1b's anatomical array layout can't be
drawn because the data has no electrode positions. FVAF is shown as a
192-electrode x 4-context heat map instead.

Inputs : Data/t5_06_02_2021.mat
Outputs: results/figures/supp_fig1_tuning_counts.{png,svg}
         results/fig1/supp_fig1_anova.json
Run    : pixi run fig1-anova           (about 15 s; --force to redo)
"""

from __future__ import annotations

import argparse

import numpy as np

from bimanual import conditions as C
from bimanual import plotting as P
from bimanual.cache import RESULTS_DIR, atomic_write_json, should_skip
from bimanual.epochs import extract_windows
from bimanual.io import HZ_PER_COUNT, load_session, remove_block_means
from bimanual.stats import anova_fvaf
from bimanual.tuning import DEFAULT_WINDOW

P_THRESH = 1e-5
CONTEXTS = ("uniR", "uniL", "biR", "biL")
OUT_JSON = RESULTS_DIR / "fig1" / "supp_fig1_anova.json"
OUT_FIG = P.FIG_DIR / "supp_fig1_tuning_counts.png"


def main(force=False):
    if should_skip([OUT_JSON, OUT_FIG], force):
        print("outputs exist; --force to redo")
        return
    s = load_session(1750)
    rates = remove_block_means(s.tx, s.block) * HZ_PER_COUNT
    tr = np.nanmean(extract_windows(rates, s.go, DEFAULT_WINDOW), axis=1)  # (trials, 192)
    mt = s.trials.move_type.to_numpy()
    quad = C.trial_quadrants(s.Tp, s.Cp, s.go, mt)  # (trials, 2): RH, LH quadrant

    sel = {  # context -> (trial mask, quadrant column)
        "uniR": (mt == C.UNI_RIGHT, 0), "uniL": (mt == C.UNI_LEFT, 1),
        "biR": (mt == C.BIMANUAL, 0), "biL": (mt == C.BIMANUAL, 1),
    }
    n_ch = tr.shape[1]
    p = np.ones((n_ch, 4)); fvaf = np.zeros((n_ch, 4)); n_trials = {}
    for j, ctx in enumerate(CONTEXTS):
        mask, col = sel[ctx]
        n_trials[ctx] = np.bincount(quad[mask, col][~np.isnan(quad[mask, col])].astype(int), minlength=5)[1:].tolist()
        for c in range(n_ch):
            _, p[c, j], fvaf[c, j] = anova_fvaf(tr[mask, c], quad[mask, col])
    tuned = p < P_THRESH
    counts = dict(zip(CONTEXTS, tuned.sum(0).tolist()))
    n_sets = np.bincount(tuned.sum(1), minlength=5).tolist()  # electrodes tuned to 0..4 contexts
    print("trials per quadrant:", n_trials)
    print("tuned electrodes per context (p<1e-5):", counts)
    print("electrodes tuned to 0..4 contexts:", n_sets)

    # ---- figure: (left) counts per context, (middle) 0-4 histogram, (right) FVAF heat map
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1, 1.3]))
    ax = axes[0]
    colors = [P.UNI, P.UNI, P.BI, P.BI]
    bars = ax.bar(range(4), [counts[c] for c in CONTEXTS], color=colors, width=0.6)
    for b, h in zip(bars, (None, "//", None, "//")):  # hatch = left hand (identity not by colour alone)
        if h:
            b.set_hatch(h); b.set_edgecolor("white")
    ax.bar_label(bars, color=P.INK, fontsize=8)
    ax.set_xticks(range(4), ["Uni R", "Uni L", "Bi R", "Bi L"])
    ax.set_ylabel(f"Tuned electrodes (p < {P_THRESH:g})")
    ax.set_title("(c) Tuned electrodes per context")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    bars = ax.bar(range(5), n_sets, color=P.SERIES[2], width=0.6)
    ax.bar_label(bars, color=P.INK, fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xlabel("Number of contexts tuned to")
    ax.set_ylabel("Electrodes")
    ax.set_title("(e) Mixed tuning across contexts")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    im = ax.imshow(fvaf.T, aspect="auto", cmap="Blues", vmin=0, interpolation="nearest")
    ax.set_yticks(range(4), ["Uni R", "Uni L", "Bi R", "Bi L"])
    ax.set_xlabel("Electrode (1–192)")
    ax.set_title("(b, as heat map) FVAF by direction")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="FVAF", fraction=0.05)
    fig.suptitle("Supp Fig 1 replication: single-electrode direction tuning, day 1750", fontsize=9, color=P.INK)
    fig.tight_layout()
    png = P.save(fig, "supp_fig1_tuning_counts")

    atomic_write_json(OUT_JSON, dict(
        p_threshold=P_THRESH, window_bins=list(DEFAULT_WINDOW), contexts=list(CONTEXTS),
        trials_per_quadrant=n_trials, tuned_counts=counts, n_electrodes_by_n_contexts=n_sets,
        fvaf_median=dict(zip(CONTEXTS, np.median(fvaf, 0).tolist())),
        fvaf_median_tuned_only={c: float(np.median(fvaf[tuned[:, j], j])) if tuned[:, j].any() else None
                                for j, c in enumerate(CONTEXTS)},
        p=p.tolist(), fvaf=fvaf.tolist(),
    ))
    print("wrote", png, "and", OUT_JSON)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
