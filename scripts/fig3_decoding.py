"""Steps 3.1-3.2: Fig 3a-c and Supp Fig 2, linear vs nonlinear single-bin decoding (day 1883).

Purpose
-------
The paper's case for nonlinear decoders, on real data:

* (Fig 3) On unimanual trials, a one-layer FFN decodes both cursors'
  velocities better than ridge regression, especially the LEFT hand. Its
  decoded speeds are also larger (Supp Fig 2a).
* (Fig 3) Removing the laterality dimension from the neural data hurts the
  FFN (lower r, more cursor jitter) but leaves ridge unchanged. So the FFN's
  advantage comes from using the "which hand" signal.
* (Supp Fig 2b-c) On bimanual trials, RH decoding beats LH decoding.

Method
------
* Data: day 1883, OPEN-LOOP trials. Unimanual analysis: all 491 uni-R/uni-L
  OL trials. ASSUMPTION (register #12): the paper says 482; this is the
  closest natural subset.
* Window: bins 15..34 after the go cue (400 ms, "300-700 ms"). Each 20-ms
  bin is one sample.
* Inputs: 192 channels, z-scored per block.
* Targets (``--target``, register #11):
  - "direction" (MAIN): the unit cursor->target direction of each moving hand
    at the go cue, held constant over the window; 0 for the idle hand. In
    this window it equals the saturated velocity targets the paper's online
    decoders were trained on (direction x a profile that plateaus after a
    200-ms reaction time).
  - "cursor" (alternative): the recorded open-loop cursor velocity,
    (Cp[t+1] - Cp[t]) x 50 in units/s. In open-loop trials the cursor only
    starts moving about 300 ms after the go cue, so in this window it is
    mostly the acceleration ramp, scaled by the target distance (30-750
    units). That ramp isn't in the neural signal, so every r is capped low.
* Five-fold CV with folds split by trial and balanced by movement type.
  Ridge alpha is the grid value with the best mean CV Pearson r ("swept to
  maximize Pearson correlation").
* Laterality removal: fit Eq. 4 (with the c_lat regressor) on the TRAINING
  fold only, then project out b_lat from train and test inputs.
* Metrics:
  - Pearson r per output dimension over all held-out bins, with a 95% CI by
    bootstrapping trials.
  - Jitter per hand: mean decoded speed of that cursor while the OTHER hand
    moves (it should rest), divided by its mean decoded speed while it
    moves. Lower = steadier idle cursor.

Inputs : Data/t5_10_13_2021.mat
Outputs: results/fig3/decoding_<target>.json (r, CIs, jitter, paired bootstrap tests)
         results/cache/fig3_<target>_<cond>.npz (out-of-fold predictions, so a rerun can resume)
         results/figures/fig3_decoding_<target>.{png,svg}
Run    : pixi run fig3-decoding [--target direction|cursor]   (about 2-4 min on the GPU; --force to redo)
"""

from __future__ import annotations

import argparse

import numpy as np

from bimanual import conditions as C
from bimanual import decoders as Dc
from bimanual import encoding as E
from bimanual import plotting as P
from bimanual.cache import RESULTS_DIR, atomic_write_json, atomic_write_npz, should_skip
from bimanual.io import HZ_PER_COUNT, load_session, zscore_by_block

DAY = 1883
WINDOW = (15, 34)  # inclusive -> 20 bins = 400 ms
DIMS = ("R x", "R y", "L x", "L y")
N_BOOT = 1000
OUT_DIR = RESULTS_DIR / "fig3"
CACHE = RESULTS_DIR / "cache"


# ----------------------------------------------------------------------------- data
TARGET = "direction"  # set from --target


def build(s, z, trial_mask):
    """Bin-level samples for the selected trials.

    Returns X (bins, 192), Y (bins, 4) decoding targets (see TARGET), g (bins,)
    trial index 0..n-1, mt (n,) movement type of each selected trial, and
    t0 (n,) the absolute first bin of each trial's window (for plotting in
    time order).
    """
    idx = np.where(trial_mask)[0]
    X, Y, g = [], [], []
    vel = np.diff(s.Cp, axis=1, append=s.Cp[:, -1:]).T * HZ_PER_COUNT  # (T, 4): velocity of bin t
    dirs = C.trial_directions(s.Tp, s.Cp, s.go, s.trials.move_type.to_numpy())  # (trials, 4) unit vectors
    t0 = []
    for k, i in enumerate(idx):
        a, b = s.go[i] + WINDOW[0], s.go[i] + WINDOW[1] + 1
        y = vel[a:b] if TARGET == "cursor" else np.repeat(dirs[i][None], b - a, axis=0)
        X.append(z[a:b]); Y.append(y); g += [k] * (b - a); t0.append(a)
    return np.concatenate(X), np.concatenate(Y), np.array(g), s.trials.move_type.to_numpy()[idx], np.array(t0)


def laterality_transform(lat_of_trial):
    """Training-fold-only laterality removal (Eq. 4 regressors: 1, 4 velocities, c_lat)."""

    def fit(Xtr, Ytr, gtr):
        c_lat = lat_of_trial[gtr]
        D = np.c_[np.ones(len(Xtr)), Ytr / (np.abs(Ytr).max() + 1e-9), c_lat]
        u = E.laterality_axis(D, Xtr, lat_col=D.shape[1] - 1)
        return lambda Z: E.remove_axis(Z, u)

    return fit


# ----------------------------------------------------------------------------- metrics
def boot_r(Y, P_, g, rng, n_boot=N_BOOT):
    """Pearson r per dimension with a 95% CI from resampling trials."""
    r = Dc.pearson_per_dim(Y, P_)
    trials = np.unique(g)
    rows = [np.where(g == t)[0] for t in trials]
    boots = np.empty((n_boot, Y.shape[1]))
    for i in range(n_boot):
        pick = np.concatenate([rows[j] for j in rng.integers(0, len(trials), len(trials))])
        boots[i] = Dc.pearson_per_dim(Y[pick], P_[pick])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return r, lo, hi


def jitter(P_, g, mt):
    """Idle-cursor speed / active-cursor speed, per hand (lower = steadier)."""
    sp_R, sp_L = np.hypot(P_[:, 0], P_[:, 1]), np.hypot(P_[:, 2], P_[:, 3])
    mtb = mt[g]
    return dict(R=float(sp_R[mtb == 2].mean() / sp_R[mtb == 1].mean()),
                L=float(sp_L[mtb == 1].mean() / sp_L[mtb == 2].mean()))


def paired_tests(preds, Y, g, rng, n_boot=N_BOOT):
    """Paired bootstrap over trials for the key comparisons.

    For each comparison A - B: diff = mean Pearson r(A) - mean r(B), over all
    4 dims and over the 2 LH dims. Trials are resampled with replacement, with
    the same resample for A and B (paired). Two-sided p = 2 x min(P(diff <= 0),
    P(diff >= 0)).
    """
    comps = {"FFN - Ridge": ("uni_ffn", "uni_ridge"),
             "FFN - FFN_nolat": ("uni_ffn", "uni_ffn_nolat"),
             "Ridge - Ridge_nolat": ("uni_ridge", "uni_ridge_nolat")}
    trials = np.unique(g)
    rows = [np.where(g == t)[0] for t in trials]
    res = {}
    for name, (a, b) in comps.items():
        for scope, dims in (("all", slice(0, 4)), ("LH", slice(2, 4))):
            d0 = (Dc.pearson_per_dim(Y, preds[a]) - Dc.pearson_per_dim(Y, preds[b]))[dims].mean()
            boots = np.empty(n_boot)
            for i in range(n_boot):
                pick = np.concatenate([rows[j] for j in rng.integers(0, len(trials), len(trials))])
                boots[i] = (Dc.pearson_per_dim(Y[pick], preds[a][pick])
                            - Dc.pearson_per_dim(Y[pick], preds[b][pick]))[dims].mean()
            p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            res[f"{name} ({scope})"] = dict(diff=float(d0), lo=float(lo), hi=float(hi), p_two_sided=float(p))
            print(f"  paired {name:22s} {scope:3s}: diff {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    return {"paired_tests": res}


# ----------------------------------------------------------------------------- runs
def run_condition(name, X, Y, g, folds, decoder, force, transform=None):
    """Out-of-fold predictions for one decoder/condition, cached to npz for resume."""
    path = CACHE / f"fig3_{TARGET}_{name}.npz"
    if path.exists() and not force:
        d = np.load(path)
        return d["pred"], float(d["alpha"])
    alpha = np.nan
    if decoder == "ridge":  # sweep alpha, keep the one with the best mean CV r
        best = (-np.inf, None, None)
        for a in Dc.RIDGE_ALPHAS:
            p = Dc.cross_validate(X, Y, g, folds, "ridge", alpha=a, transform=transform)
            r = np.nanmean(Dc.pearson_per_dim(Y, p))
            if r > best[0]:
                best = (r, a, p)
        _, alpha, pred = best
    else:
        pred = Dc.cross_validate(X, Y, g, folds, "ffn", transform=transform)
    atomic_write_npz(path, pred=pred, alpha=np.array(alpha))
    print(f"  {name}: done (alpha={alpha})")
    return pred, alpha


def main(force=False, target="direction"):
    global TARGET
    TARGET = target
    out_json = OUT_DIR / f"decoding_{target}.json"
    if should_skip([out_json], force):
        print("outputs exist; --force to redo")
        return
    s = load_session(DAY)
    z = zscore_by_block(s.tx, s.block)
    ol = (s.trials.kind == "OL").to_numpy()
    mt_all = s.trials.move_type.to_numpy()
    rng = np.random.default_rng(0)
    out = {}

    # ---- unimanual: ridge / FFN, with and without the laterality dimension
    Xu, Yu, gu, mtu, t0u = build(s, z, ol & np.isin(mt_all, (1, 2)))
    folds_u = Dc.stratified_trial_folds(mtu, 5)
    lat_u = np.where(mtu == 1, 1.0, -1.0)
    print(f"unimanual: {len(mtu)} trials ({np.sum(mtu == 1)} uniR, {np.sum(mtu == 2)} uniL), {len(Xu)} bins")
    preds = {}
    for cond, dec, tf in (("uni_ridge", "ridge", None), ("uni_ffn", "ffn", None),
                          ("uni_ridge_nolat", "ridge", laterality_transform(lat_u)),
                          ("uni_ffn_nolat", "ffn", laterality_transform(lat_u))):
        p, a = run_condition(cond, Xu, Yu, gu, folds_u, dec, force, tf)
        preds[cond] = p
        r, lo, hi = boot_r(Yu, p, gu, rng)
        out[cond] = dict(r=r, r_lo=lo, r_hi=hi, alpha=a, jitter=jitter(p, gu, mtu),
                         mean_speed_active=dict(
                             R=float(np.hypot(*p[mtu[gu] == 1][:, :2].T).mean()),
                             L=float(np.hypot(*p[mtu[gu] == 2][:, 2:].T).mean())))
        print(f"{cond:16s} r " + " ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, r)) +
              f"  jitter R={out[cond]['jitter']['R']:.2f} L={out[cond]['jitter']['L']:.2f}")

    # ---- bimanual (Supp Fig 2b-c)
    Xb, Yb, gb, mtb, t0b = build(s, z, ol & (mt_all == 3))
    folds_b = Dc.stratified_trial_folds(mtb, 5)
    print(f"bimanual: {len(mtb)} trials, {len(Xb)} bins")
    for cond, dec in (("bi_ridge", "ridge"), ("bi_ffn", "ffn")):
        p, a = run_condition(cond, Xb, Yb, gb, folds_b, dec, force)
        preds[cond] = p
        r, lo, hi = boot_r(Yb, p, gb, rng)
        out[cond] = dict(r=r, r_lo=lo, r_hi=hi, alpha=a)
        print(f"{cond:16s} r " + " ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, r)))

    out.update(paired_tests(preds, Yu, gu, rng))
    atomic_write_json(out_json, dict(day=DAY, target=TARGET, window_bins=list(WINDOW), n_uni_trials=int(len(mtu)),
                                     n_bi_trials=int(len(mtb)), dims=list(DIMS), results=out))
    plot(out, preds, Yu, gu, mtu, t0u, Yb, gb, t0b)


# ----------------------------------------------------------------------------- figures
def plot(out, preds, Yu, gu, mtu, t0u, Yb, gb, t0b):
    import matplotlib.pyplot as plt

    cols = {"ridge": P.SERIES[0], "ffn": P.SERIES[1]}

    def traces(ax, Y, pr, g, t0, dim, title):
        """8-s example: the first 20 test trials in time order, windows concatenated."""
        order = np.argsort(t0)[:20]
        rows = np.concatenate([np.where(g == k)[0] for k in order])
        t = np.arange(len(rows)) * 0.02
        ax.plot(t, Y[rows, dim], color=P.INK, lw=1.2, label="target")
        for name, key in (("Ridge", "ridge"), ("FFN", "ffn")):
            ax.plot(t, pr[key][rows, dim], color=cols[key], lw=1.2, alpha=0.9, label=name)
        ax.set_title(title, fontsize=8); ax.set_xlabel("time (s, concatenated 400-ms windows)")

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3)
    for j, dim in enumerate((0, 2)):
        ax = fig.add_subplot(gs[0, j])
        traces(ax, Yu, {"ridge": preds["uni_ridge"], "ffn": preds["uni_ffn"]}, gu, t0u, dim,
               f"(a) {'Right' if dim == 0 else 'Left'} cursor x-velocity, unimanual trials")
        ax.set_ylabel("target / decoded" if TARGET == "direction" else "velocity (units/s)")
        if j == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper left")

    # (b) r per dimension: ridge vs FFN, with and without laterality
    ax = fig.add_subplot(gs[0, 2])
    w = 0.2
    for k, (cond, lab, col, hatch) in enumerate((
            ("uni_ridge", "Ridge", cols["ridge"], None), ("uni_ridge_nolat", "Ridge − lat", cols["ridge"], "//"),
            ("uni_ffn", "FFN", cols["ffn"], None), ("uni_ffn_nolat", "FFN − lat", cols["ffn"], "//"))):
        r, lo, hi = (np.array(out[cond][q]) for q in ("r", "r_lo", "r_hi"))
        xs = np.arange(4) + (k - 1.5) * w
        bars = ax.bar(xs, r, width=w - 0.02, color=col, label=lab, hatch=hatch, edgecolor="white" if hatch else None)
        ax.errorbar(xs, r, yerr=[r - lo, hi - r], fmt="none", ecolor=P.INK, lw=0.8, capsize=2)
    ax.set_xticks(range(4), DIMS); ax.set_ylim(0, 1); ax.set_ylabel("Pearson r (5-fold CV)")
    ax.set_title("(b) Decoding accuracy, unimanual", fontsize=8); ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(axis="x", visible=False)

    # (c) jitter
    ax = fig.add_subplot(gs[1, 0])
    for k, (cond, lab, col, hatch) in enumerate((
            ("uni_ridge", "Ridge", cols["ridge"], None), ("uni_ridge_nolat", "Ridge − lat", cols["ridge"], "//"),
            ("uni_ffn", "FFN", cols["ffn"], None), ("uni_ffn_nolat", "FFN − lat", cols["ffn"], "//"))):
        v = [out[cond]["jitter"]["R"], out[cond]["jitter"]["L"]]
        xs = np.arange(2) + (k - 1.5) * w
        bars = ax.bar(xs, v, width=w - 0.02, color=col, label=lab, hatch=hatch, edgecolor="white" if hatch else None)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, color=P.INK)
    ax.set_xticks(range(2), ["Right cursor", "Left cursor"])
    ax.set_ylabel("idle speed / active speed"); ax.set_title("(c) Cursor jitter (lower = steadier)", fontsize=8)
    ax.legend(fontsize=6, ncol=2); ax.grid(axis="x", visible=False)

    # Supp Fig 2a: decoded speed distributions of the ACTIVE cursor
    ax = fig.add_subplot(gs[1, 1])
    act = mtu[gu]
    for cond, lab, col, ls in (("uni_ridge", "Ridge", cols["ridge"], "-"), ("uni_ffn", "FFN", cols["ffn"], "-"),
                               ("uni_ffn_nolat", "FFN − lat", cols["ffn"], "--")):
        p = preds[cond]
        sp = np.r_[np.hypot(*p[act == 1][:, :2].T), np.hypot(*p[act == 2][:, 2:].T),
                   np.hypot(*p[act == 1][:, 2:].T), np.hypot(*p[act == 2][:, :2].T)]
        ax.hist(sp, bins=60, histtype="step", color=col, ls=ls, lw=1.5, density=True, label=lab)
    ax.set_xlabel("decoded speed, both cursors"); ax.set_ylabel("density")
    ax.set_title("(Supp 2a) Decoded speed distribution", fontsize=8); ax.legend(fontsize=7)

    # Supp Fig 2c: bimanual r
    ax = fig.add_subplot(gs[1, 2])
    for k, (cond, lab, col) in enumerate((("bi_ridge", "Ridge", cols["ridge"]), ("bi_ffn", "FFN", cols["ffn"]))):
        r, lo, hi = (np.array(out[cond][q]) for q in ("r", "r_lo", "r_hi"))
        xs = np.arange(4) + (k - 0.5) * 0.36
        ax.bar(xs, r, width=0.34, color=col, label=lab)
        ax.errorbar(xs, r, yerr=[r - lo, hi - r], fmt="none", ecolor=P.INK, lw=0.8, capsize=2)
    ax.set_xticks(range(4), DIMS); ax.set_ylim(0, 1); ax.set_ylabel("Pearson r")
    ax.set_title("(Supp 2c) Decoding accuracy, bimanual trials", fontsize=8); ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)

    fig.suptitle(f"Fig 3 / Supp Fig 2 replication: single-bin decoding, day {DAY} open-loop, "
                 f"target={TARGET} (95% CI from bootstrapping trials)", fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, f"fig3_decoding_{TARGET}")
    print("wrote", P.FIG_DIR / f"fig3_decoding_{TARGET}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--target", choices=["direction", "cursor"], default="direction",
                    help="decoding target (default: unit direction; 'cursor' = recorded OL cursor velocity)")
    a = ap.parse_args()
    main(a.force, a.target)
