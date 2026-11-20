"""Step 2.3: Fig 2 PCA of trial-averaged population activity (day 1881).

Purpose
-------
The paper visualises the population state of every open-loop trial with PCA
and reports that bimanual activity most closely resembles unimanual
RIGHT-hand activity in the top PCs. The dominant hemisphere's motor cortex
represents a bimanual movement much like a movement of the contralateral
(right) hand. A strong "laterality" axis separates uni-R from uni-L.

Method (Methods, "PCA")
-----------------------
* Day 1881, all open-loop (OL) trials.
* Features: 20-ms threshold crossings, block-mean removed and z-scored per
  block (io.zscore_by_block), averaged over bins 15..35 after the go cue
  ("300-700 ms"), giving one 192-dim vector per trial.
* PCA on the trials x 192 matrix (centred), by SVD.
* Quantified as distances between condition centroids (uni-R, uni-L,
  bimanual) in the top-2 and top-3 PCs and in the full 192-dim space. The
  claim holds if d(bi, uniR) < d(bi, uniL).

Inputs : Data/t5_10_11_2021.mat
Outputs: results/figures/fig2_pca.{png,svg}; results/fig2/pca.json
Run    : pixi run fig2-pca             (about 10 s; --force to redo)
"""

from __future__ import annotations

import argparse

import numpy as np

from bimanual import plotting as P
from bimanual.cache import RESULTS_DIR, atomic_write_json, should_skip
from bimanual.epochs import extract_windows
from bimanual.io import load_session, zscore_by_block
from bimanual.tuning import DEFAULT_WINDOW

DAY = 1881
OUT_JSON = RESULTS_DIR / "fig2" / "pca.json"
OUT_FIG = P.FIG_DIR / "fig2_pca.png"
# colours follow the entity: uni-R = slot 1, bimanual = slot 2 (as elsewhere), uni-L = slot 3
GROUPS = {1: ("Unimanual right", P.SERIES[0]), 2: ("Unimanual left", P.SERIES[2]), 3: ("Bimanual", P.SERIES[1])}


def main(force=False):
    if should_skip([OUT_JSON, OUT_FIG], force):
        print("outputs exist; --force to redo")
        return
    s = load_session(DAY)
    z = zscore_by_block(s.tx, s.block)
    ol = (s.trials.kind == "OL").to_numpy()
    X = np.nanmean(extract_windows(z, s.go[ol], DEFAULT_WINDOW), axis=1)  # (trials, 192)
    mt = s.trials.move_type.to_numpy()[ol]
    keep = ~np.isnan(X).any(axis=1)
    X, mt = X[keep], mt[keep]

    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S
    var_expl = S**2 / np.sum(S**2)

    cent = {k: scores[mt == k].mean(axis=0) for k in GROUPS}
    cent_full = {k: Xc[mt == k].mean(axis=0) for k in GROUPS}
    dist = {}
    for name, dims, cc in (("top2", 2, cent), ("top3", 3, cent), ("full", None, cent_full)):
        sl = slice(0, dims)
        d = lambda a, b: float(np.linalg.norm(cc[a][sl] - cc[b][sl]))
        dist[name] = {"bi-uniR": d(3, 1), "bi-uniL": d(3, 2), "uniR-uniL": d(1, 2)}
    claim = all(v["bi-uniR"] < v["bi-uniL"] for v in dist.values())
    n = {GROUPS[k][0]: int(np.sum(mt == k)) for k in GROUPS}
    print("trials:", n)
    print("variance explained PC1-3:", np.round(var_expl[:3], 3))
    for k, v in dist.items():
        print(f"centroid distances ({k}):", {a: round(b, 2) for a, b in v.items()})
    print("bimanual closer to uni-R than to uni-L in all spaces:", claim)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (i, j) in zip(axes, ((0, 1), (0, 2))):
        for k, (lab, col) in GROUPS.items():
            m = mt == k
            ax.scatter(scores[m, i], scores[m, j], s=12, color=col, alpha=0.45, lw=0, label=f"{lab} (n={m.sum()})")
            ax.scatter(cent[k][i], cent[k][j], s=110, color=col, edgecolor="white", lw=1.5, zorder=3)
        ax.set_xlabel(f"PC{i + 1} ({100 * var_expl[i]:.1f}% var)")
        ax.set_ylabel(f"PC{j + 1} ({100 * var_expl[j]:.1f}% var)")
    axes[0].legend(loc="best", fontsize=7, markerscale=1.5)
    t = dist["top2"]
    axes[1].set_title(f"centroid distance (PC1-2): bi–uniR {t['bi-uniR']:.1f}, bi–uniL {t['bi-uniL']:.1f}", fontsize=8)
    axes[0].set_title("Trial-averaged activity, 300–700 ms after go", fontsize=8)
    fig.suptitle(f"Fig 2 PCA replication, day {DAY} open-loop trials (large dots = condition means)",
                 fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "fig2_pca")
    atomic_write_json(OUT_JSON, dict(day=DAY, n_trials=n, var_explained_top10=var_expl[:10],
                                     centroid_distances=dist, bimanual_closest_to_uniR=claim))
    print("wrote", OUT_FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
