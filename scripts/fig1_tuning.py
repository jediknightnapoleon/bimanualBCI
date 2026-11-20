"""Step 1.1: Fig 1b-c, tuning curves of electrodes 97, 13 and 23 (day 1750).

Purpose
-------
The paper's Fig 1 shows example electrodes whose direction tuning changes
between the unimanual and the bimanual context:

* electrode 97 keeps its tuning for both hands;
* electrode 13 has suppressed tuning for both hands during bimanual movement;
* electrode 23 has suppressed LEFT-hand tuning during bimanual movement.

For each electrode and hand, this script plots the 8-direction tuning curve in
the unimanual context (blue) and the bimanual context (orange), with
bootstrap 95% CIs (10,000 resamples of trials, as in Methods "Statistics").

Method (same as the authors' tuningAnalysis.m)
----------------------------------------------
* Firing rate = block-mean-removed threshold crossings x 50 (spikes/s).
* Each trial's rate = mean over bins 15..35 after the go cue (inclusive,
  "300-700 ms").
* Trials are grouped by the 8 direction wedges of the moving hand:
  uni-R = moveType 1 by RH wedge, uni-L = moveType 2 by LH wedge,
  bi-R / bi-L = moveType 3 by RH / LH wedge.
* Each curve is centred on its mean across the 8 directions, as in the
  authors' script. The CIs are shifted by the same amount.
* Electrode numbers are 1-based channel indices in the data (as in "CH. c"
  of tuningAnalysis.m), so electrode 97 = row 96.

Inputs : Data/t5_06_02_2021.mat
Outputs: results/figures/fig1_tuning_curves.{png,svg}
         results/fig1/tuning_curves.json (curves + CIs + check vs the port)
Run    : pixi run fig1-tuning          (about 20 s; --force to redo)
"""

from __future__ import annotations

import argparse

import numpy as np

from bimanual import conditions as C
from bimanual import plotting as P
from bimanual.cache import RESULTS_DIR, atomic_write_json, should_skip
from bimanual.epochs import extract_windows
from bimanual.io import HZ_PER_COUNT, load_session, remove_block_means
from bimanual.tuning import DEFAULT_WINDOW, tuning_curves

ELECTRODES = (97, 13, 23)  # 1-based, as named in the paper
N_BOOT = 10_000
OUT_JSON = RESULTS_DIR / "fig1" / "tuning_curves.json"
OUT_FIG = P.FIG_DIR / "fig1_tuning_curves.png"
# x-axis order: wedge IDs 1..8 = 180, 225, 270, 315, 0, 45, 90, 135 deg (as in the paper)
XLABELS = [f"{int(a)}" for a in C.WEDGE_CENTERS_DEG]


def boot_mean_ci(x: np.ndarray, rng, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap the mean over trials (axis 0) of x (n_trials, n_ch) -> (lo, hi) per channel."""
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)  # (n_boot, n_ch)
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)


def main(force=False):
    if should_skip([OUT_JSON, OUT_FIG], force):
        print("outputs exist; --force to redo")
        return
    s = load_session(1750)
    rates = remove_block_means(s.tx, s.block) * HZ_PER_COUNT
    trial_rates = np.nanmean(extract_windows(rates, s.go, DEFAULT_WINDOW), axis=1)  # (trials, N)
    cond = C.trial_conditions(s.Tp, s.Cp, s.go + 1, s.trials.move_type.to_numpy())
    ch = np.array(ELECTRODES) - 1

    # context -> (trial mask, wedge column of the moving hand)
    contexts = {
        "uniR": (cond[:, 0] == C.UNI_RIGHT, 1),
        "uniL": (cond[:, 0] == C.UNI_LEFT, 2),
        "biR": (cond[:, 0] == C.BIMANUAL, 1),
        "biL": (cond[:, 0] == C.BIMANUAL, 2),
    }
    rng = np.random.default_rng(0)
    out = {}
    for name, (mask, col) in contexts.items():
        mean = np.zeros((len(ch), 8)); lo = np.zeros_like(mean); hi = np.zeros_like(mean); n = []
        for d in range(1, 9):
            x = trial_rates[mask & (cond[:, col] == d)][:, ch]
            x = x[~np.isnan(x).any(axis=1)]
            mean[:, d - 1] = x.mean(axis=0)
            lo[:, d - 1], hi[:, d - 1] = boot_mean_ci(x, rng)
            n.append(len(x))
        center = mean.mean(axis=1, keepdims=True)
        out[name] = dict(mean=mean - center, lo=lo - center, hi=hi - center, n_trials_per_dir=n)

    # Check: the curves must equal the faithful port of tuningAnalysis.m (verified against Octave).
    port = tuning_curves(s)
    max_diff = max(float(np.max(np.abs(out[k]["mean"] - getattr(port, k)[ch]))) for k in contexts)
    print(f"max |difference| vs the tuningAnalysis.m port: {max_diff:.2e}")
    assert max_diff < 1e-9, "trial-mean curves differ from the reference port"

    # ---- figure: rows = electrodes, columns = right hand / left hand
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ch), 2, figsize=(7.0, 2.2 * len(ch)), sharex=True)
    x = np.arange(8)
    for r, e in enumerate(ELECTRODES):
        for c, (hand, uni, bi) in enumerate((("Right hand", "uniR", "biR"), ("Left hand", "uniL", "biL"))):
            ax = axes[r, c]
            for key, color, label in ((uni, P.UNI, "Unimanual"), (bi, P.BI, "Bimanual")):
                ax.fill_between(x, out[key]["lo"][r], out[key]["hi"][r], color=color, alpha=0.18, lw=0)
                ax.plot(x, out[key]["mean"][r], "-o", color=color, label=label)
            ax.axhline(0, color=P.INK_MUTED, lw=0.6)
            if r == 0:
                ax.set_title(hand)
            if c == 0:
                ax.set_ylabel(f"Electrode {e}\nΔ rate (spikes/s)")
            ax.set_xticks(x, XLABELS)
    for ax in axes[-1]:
        ax.set_xlabel("Movement direction (deg)")
    fig.suptitle("Fig 1b-c replication: tuning curves, day 1750 (300–700 ms after go; 95% bootstrap CI)",
                 fontsize=9, color=P.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # legend in its own strip under the title so it never covers data
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
    png = P.save(fig, "fig1_tuning_curves")

    def tolist(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    atomic_write_json(OUT_JSON, dict(
        electrodes=list(ELECTRODES), window_bins=list(DEFAULT_WINDOW),
        direction_order_deg=C.WEDGE_CENTERS_DEG.tolist(), n_boot=N_BOOT,
        max_diff_vs_port=max_diff, curves={k: tolist(v) for k, v in out.items()},
    ))
    print("wrote", png, "and", OUT_JSON)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
