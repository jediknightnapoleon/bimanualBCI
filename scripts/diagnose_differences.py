"""Diagnose the main remaining differences between the paper's figures and ours (docs/figure_comparison.md).

Each check changes ONE analysis choice and reports how far that moves our number toward or away from
the paper's. None of these variants replaces a main result; they are evidence for the explanations in
the comparison document.

D1  Supp Fig 1c/e tuned-electrode counts (paper 90 / 97 / 60 / 52; ours 83 / 87 / 50 / 28):
    significance threshold x direction grouping (our 4 quadrants, #6, vs the 8 target directions of
    Fig 1) x window (300-700 ms, #18, vs 0-1000 ms).
D2  Fig 5a angular error (ours about 15-25 deg above the paper): our per-bin definition (#8) vs the
    angle of the net cursor displacement over 300-500 ms, with mean and median over trials.
D3  Fig 3c / Supp Fig 2 FFN (ours jitterier, more dependent on laterality, below ridge on bimanual
    trials): FFN training variants (weight decay, early stopping, epochs) on the same folds.
D4  Fig 2b-c ratios (paper 0.88 / 0.66; ours 0.81 / 0.59): cvOLS estimator (split-half, #23, vs the
    paper's 5-fold) x regressors (unit direction at go, #21, vs the paper's Eq. 2 per-time-step
    target - cursor position).

Inputs : Data/, results/fig1, results/cache
Outputs: results/paperstyle/diagnostics.json
Run    : pixi run python scripts/diagnose_differences.py [--only D1,D2,D3,D4]   (about 5 min, GPU for D3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bimanual import behavior as B  # noqa: E402
from bimanual import blocksets as BS  # noqa: E402
from bimanual import conditions as C  # noqa: E402
from bimanual import decoders as Dc  # noqa: E402
from bimanual.cache import RESULTS_DIR, atomic_write_json  # noqa: E402
from bimanual.epochs import extract_windows  # noqa: E402
from bimanual.io import HZ_PER_COUNT, load_session, remove_block_means, zscore_by_block  # noqa: E402
from bimanual.stats import anova_fvaf  # noqa: E402

OUT = RESULTS_DIR / "paperstyle" / "diagnostics.json"


# ============================================================================ D1
def d1_supp1():
    """Tuned-electrode counts on day 1750 under different thresholds, groupings and windows."""
    s = load_session(1750)
    rates = remove_block_means(s.tx, s.block) * HZ_PER_COUNT
    mt = s.trials.move_type.to_numpy()
    groupings = {
        "quadrant(4)": C.trial_quadrants(s.Tp, s.Cp, s.go, mt),
        # Fig 1's 8 target directions (tuningAnalysis.m wedges); 9 = idle hand -> NaN
        "direction(8)": np.where(C.trial_conditions(s.Tp, s.Cp, s.go + 1, mt)[:, 1:] == C.NO_MOVE, np.nan,
                                 C.trial_conditions(s.Tp, s.Cp, s.go + 1, mt)[:, 1:]),
    }
    windows = {"300-700ms": (15, 35), "0-1000ms": (0, 49)}
    sel = {"uniR": (mt == 1, 0), "biR": (mt == 3, 0), "uniL": (mt == 2, 1), "biL": (mt == 3, 1)}
    out = {}
    for wname, win in windows.items():
        tr = np.nanmean(extract_windows(rates, s.go, win), axis=1)
        for gname, grp in groupings.items():
            p = np.ones((tr.shape[1], 4))
            for j, (ctx, (mask, col)) in enumerate(sel.items()):
                for c in range(tr.shape[1]):
                    _, p[c, j], _ = anova_fvaf(tr[mask, c], grp[mask, col])
            for thr in (1e-5, 1e-3, 1e-2, 0.05):
                tuned = p < thr
                key = f"{wname} | {gname} | p<{thr:g}"
                out[key] = dict(counts=dict(zip(sel, tuned.sum(0).tolist())),
                                n_contexts=np.bincount(tuned.sum(1), minlength=5).tolist())
    paper = dict(counts=dict(uniR=90, biR=97, uniL=60, biL=52), n_contexts=[82, 16, 33, 27, 34])

    def dist(v):  # summed absolute difference from the paper's 4 counts
        return sum(abs(v["counts"][k] - paper["counts"][k]) for k in paper["counts"])
    for k, v in out.items():
        v["abs_diff_from_paper_counts"] = dist(v)
    print("D1 Supp Fig 1 counts (uniR, biR, uniL, biL | 0-4 contexts), paper 90 97 60 52 | 82 16 33 27 34")
    for k, v in sorted(out.items(), key=lambda kv: kv[1]["abs_diff_from_paper_counts"]):
        c = v["counts"]
        print(f"  {k:38s} {c['uniR']:3d} {c['biR']:3d} {c['uniL']:3d} {c['biL']:3d} | {v['n_contexts']}  "
              f"(|diff| {v['abs_diff_from_paper_counts']})")
    return dict(paper=paper, variants=out)


# ============================================================================ D2
def net_angle(Cp, Tp, go, hand, window_s=(0.3, 0.5), target_at="go"):
    """|angle(net cursor displacement over the window) - angle(cursor->target)|, degrees."""
    rows = B.HAND_ROWS[hand]
    b0 = go + int(round(window_s[0] / B.BIN_S)); b1 = go + int(round(window_s[1] / B.BIN_S))
    ref = go if target_at == "go" else b0
    to_t = Tp[rows, ref] - Cp[rows, ref]
    disp = Cp[rows, b1] - Cp[rows, b0]
    if not np.any(to_t) or not np.any(disp):
        return np.nan
    d = np.degrees(np.arctan2(disp[1], disp[0]) - np.arctan2(to_t[1], to_t[0]))
    return float(abs((d + 180) % 360 - 180))


def d2_angular_error():
    rows = []
    # the six Fig 5a sessions: four evaluation days + the simultaneous blocks of 1881 / 1883
    for day, var in ((1776, None), (1778, None), (1790, None), (1792, None), (1881, "simultaneous"), (1883, "simultaneous")):
        s = load_session(day)
        t = s.trials
        t = t[(t.kind == "CL") & (t.decoder == "RNN")]
        t = t if var is None else t[t.variant == var]
        for r in t.itertuples():
            for hand, own in (("R", 1), ("L", 2)):
                if r.move_type not in (own, 3):
                    continue
                rows.append(dict(day=day, hand=hand, ctx="bi" if r.move_type == 3 else "uni",
                                 per_bin=B.angular_error(s.Cp, s.Tp, r.go, hand),
                                 net=net_angle(s.Cp, s.Tp, r.go, hand),
                                 net_target_at_300ms=net_angle(s.Cp, s.Tp, r.go, hand, target_at="300ms")))
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    for (hand, ctx), g in df.groupby(["hand", "ctx"]):
        per_day = g.groupby("day")
        out[f"{ctx}{hand}"] = {m: dict(mean_by_day=per_day[m].mean().round(1).tolist(),
                                       median_by_day=per_day[m].median().round(1).tolist())
                               for m in ("per_bin", "net", "net_target_at_300ms")}
    print("D2 angular error by session (1776, 1778, 1790, 1792, 1881 sim, 1883 sim); paper ranges: uniR 11-25, biR 20-40, uniL 13-49, biL 30-65")
    for k, v in out.items():
        print(f"  {k:5s} per-bin mean {v['per_bin']['mean_by_day']}  net mean {v['net']['mean_by_day']}  "
              f"net median {v['net']['median_by_day']}")
    return out


# ============================================================================ D3
def d3_ffn():
    import fig3_decoding as F3
    s = load_session(F3.DAY)
    z = zscore_by_block(s.tx, s.block)
    ol = (s.trials.kind == "OL").to_numpy()
    mt_all = s.trials.move_type.to_numpy()
    Xu, Yu, gu, mtu, _ = F3.build(s, z, ol & np.isin(mt_all, (1, 2)))
    Xb, Yb, gb, mtb, _ = F3.build(s, z, ol & (mt_all == 3))
    folds_u = Dc.stratified_trial_folds(mtu, 5)
    folds_b = Dc.stratified_trial_folds(mtb, 5)
    lat_tf = F3.laterality_transform(np.where(mtu == 1, 1.0, -1.0))
    variants = {
        "baseline (ours: wd 1e-4, early stop)": dict(),
        "no weight decay": dict(weight_decay=0.0),
        "no early stopping (60 epochs)": dict(patience=10**6),
        "no early stopping, no wd, 150 epochs": dict(patience=10**6, weight_decay=0.0, max_epochs=150),
    }

    def cv(X, Y, g, folds, kw, transform=None):
        pred = np.zeros_like(Y, dtype=float)
        fob = folds[g]
        for k in np.unique(folds):
            te = fob == k
            Xtr, Xte = X[~te], X[te]
            if transform is not None:
                ap = transform(Xtr, Y[~te], g[~te]); Xtr, Xte = ap(Xtr), ap(Xte)
            pred[te] = Dc.ffn_fit_predict(Xtr, Y[~te], Xte, g[~te], seed=0, **kw)
        return pred

    def jit(p):
        m = mtu[gu]
        return {d: float(np.abs(p[m == other, i]).mean() / np.abs(p[m == own, i]).mean())
                for i, (d, own, other) in enumerate((("Rx", 1, 2), ("Ry", 1, 2), ("Lx", 2, 1), ("Ly", 2, 1)))}

    cached = np.load(RESULTS_DIR / "cache" / "fig3_direction_uni_ffn.npz")["pred"]
    out = dict(cached_baseline=dict(r=Dc.pearson_per_dim(Yu, cached).tolist(), jitter=jit(cached)))
    rb = Dc.pearson_per_dim(Yb, np.load(RESULTS_DIR / "cache" / "fig3_direction_bi_ridge.npz")["pred"])
    out["bimanual_ridge_r"] = rb.tolist()
    print("D3 FFN variants (r Rx Ry Lx Ly | LH r with laterality removed | jitter Rx Ry Lx Ly | bimanual r)")
    for name, kw in variants.items():
        p = cv(Xu, Yu, gu, folds_u, kw)
        pn = cv(Xu, Yu, gu, folds_u, kw, lat_tf)
        pbi = cv(Xb, Yb, gb, folds_b, kw)
        r, rn, rbi = (Dc.pearson_per_dim(Y, q) for Y, q in ((Yu, p), (Yu, pn), (Yb, pbi)))
        out[name] = dict(r=r.tolist(), r_nolat=rn.tolist(), jitter=jit(p), jitter_nolat=jit(pn), r_bimanual=rbi.tolist())
        print(f"  {name:38s} r {np.round(r, 2)} | no-lat LH {np.round(rn[2:], 2)} | jitter "
              f"{np.round(list(jit(p).values()), 2)} | bi {np.round(rbi, 2)}")
    print(f"  bimanual ridge r {np.round(rb, 2)}")
    return out


# ============================================================================ D4
def d4_tuning_change():
    import fig2_tuning_change as F2
    from bimanual import encoding as E
    from bimanual.tuning import DEFAULT_WINDOW

    def prepare(day):
        dd = F2.prepare_day(day)
        s = load_session(day)
        # Eq. 2 regressors: target - cursor position of each hand, per 20-ms bin of the window
        dd["diffs"] = extract_windows((s.Tp - s.Cp).T, s.go, DEFAULT_WINDOW)  # (trials, 21, 4)
        return dd

    def design_perstep(dd, trials, cols):
        win, dif = dd["win"][trials], dd["diffs"][trials][:, :, cols]
        ok = ~np.isnan(win).any(axis=(1, 2)) & ~np.isnan(dif).any(axis=(1, 2))
        win, dif, trials = win[ok], dif[ok], trials[ok]
        n_tr, n_bin, n_ch = win.shape
        X = np.c_[np.ones(n_tr * n_bin), dif.reshape(-1, len(cols))]
        return X, win.reshape(-1, n_ch), np.repeat(trials, n_bin)

    days = {d: prepare(d) for d in BS.FIG2_DAYS}
    unit_design = F2.design
    out = {}
    for reg_name, des in (("unit direction at go (ours, #21)", unit_design), ("Eq. 2 per-step target-cursor", design_perstep)):
        for method in ("splithalf", "kfold"):
            F2.design, F2.METHOD = des, method  # analyse_set looks both up at call time
            sets = []
            for d in BS.FIG2_DAYS:
                for k, blocks in enumerate(BS.block_sets(d, BS.MAIN_SCHEME)):
                    r = F2.analyse_set(days[d], blocks, seed=k)
                    if r["usable"]:
                        sets.append(r)
            rR = np.array([x["ratio_R"] for x in sets]); rL = np.array([x["ratio_L"] for x in sets])
            corr = {c: float(np.nanmean([x["corr"][c] for x in sets])) for c in sets[0]["corr"]}
            key = f"{reg_name} | {method}"
            out[key] = dict(n=len(sets), ratio_R_mean=float(rR.mean()), ratio_L_mean=float(rL.mean()),
                            ratio_R_sd=float(rR.std()), ratio_L_sd=float(rL.std()), corr=corr)
            print(f"D4 {key:48s} ratio R {rR.mean():.2f} (sd {rR.std():.2f})  L {rL.mean():.2f} (sd {rL.std():.2f})  "
                  f"UniL-BiL x/y {corr['UniL-BiL_x']:.2f}/{corr['UniL-BiL_y']:.2f}")
    F2.design, F2.METHOD = unit_design, "splithalf"
    out["paper"] = dict(ratio_R=0.88, ratio_L=0.66, UniL_BiL=(0.42, 0.45))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="D1,D2,D3,D4")
    only = ap.parse_args().only.split(",")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key, fn in (("D1", d1_supp1), ("D2", d2_angular_error), ("D3", d3_ffn), ("D4", d4_tuning_change)):
        if key in only:
            res[key] = fn()
            atomic_write_json(OUT, res)  # save after each check
    print("wrote", OUT)


if __name__ == "__main__":
    main()
