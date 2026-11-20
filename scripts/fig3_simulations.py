"""Step 3.3: Fig 3d-e simulations and the Supp Fig 4 fSNR analysis.

Purpose
-------
Why do nonlinear decoders help? The paper's simulations make the argument:

* As the RH and LH tuning vectors become more CORRELATED, a linear decoder
  can no longer tell which hand is moving, so its accuracy drops and the idle
  cursor jitters.
* A nonlinear decoder (FFN) can use the LATERALITY dimension to gate its
  outputs. It stays accurate if laterality is large enough.

Supp Fig 4 shows the noise model is adequate for SINGLE-BIN decoding: the
simulated single-bin fSNR matches the real one (2.03 for uni-R in the
paper). The window-averaged fSNR does not match, because real noise is
correlated over time and simulated noise is not.

Method
------
1. REAL DATA (day 1883, OL unimanual trials, bins 15..34 after go):
   * baseline rates b0 = channel means (Hz);
   * tuning magnitudes ||b_Rx||, ||b_Ry|| (uni-R model), ||b_Lx||, ||b_Ly||
     (uni-L model) and ||b_lat|| (Eq. 4 model on all uni trials), from
     split-half cvOLS on block-mean-removed rates;
   * fSNR of a cross-validated linear (ridge) decoder of the RH direction on
     uni-R trials, for single 20-ms bins and for 400-ms window averages.
2. CALIBRATION: find the noise SD sigma for which a simulated population
   (rho = 0, laterality x1) reaches the real single-bin fSNR (bisection on
   log sigma).
3. SWEEP: rho (RH-LH tuning correlation) in RHOS x laterality scale in
   LATS. For each cell: draw a population, simulate 2000 uni-R + 2000 uni-L
   trials x 20 bins, z-score with training statistics, and train ridge (alpha
   chosen on a validation 10% of the training trials) and the FFN on 80% of
   the trials. Report Pearson r on the held-out 20%, plus jitter.

ASSUMPTION (register #27): 20 noisy bins per simulated trial (like the
400-ms real window); 8 wedge directions; a single population draw per cell.

Inputs : Data/t5_10_13_2021.mat
Outputs: results/fig3/sim_real_stats.json (real magnitudes, fSNR, sigma)
         results/fig3/sim_grid.jsonl (one line per finished cell; resumable)
         results/figures/fig3_simulations.{png,svg}, supp_fig4_fsnr.{png,svg}
Run    : pixi run fig3-sim          (about 15-25 min on the GPU; resumes after an interruption)
"""

from __future__ import annotations

import argparse

import numpy as np

from bimanual import conditions as C
from bimanual import decoders as Dc
from bimanual import encoding as E
from bimanual import plotting as P
from bimanual import simulate as S
from bimanual.cache import RESULTS_DIR, append_jsonl, atomic_write_json, read_jsonl
from bimanual.epochs import extract_windows
from bimanual.io import HZ_PER_COUNT, load_session, remove_block_means

DAY = 1883
WINDOW = (15, 34)
N_PER_HAND, BINS = 2000, 20
RHOS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
LATS = (0.0, 0.5, 1.0, 2.0, 4.0)
OUT_DIR = RESULTS_DIR / "fig3"
REAL_JSON = OUT_DIR / "sim_real_stats.json"
GRID = OUT_DIR / "sim_grid.jsonl"


# ----------------------------------------------------------------------------- helpers
def zscore_train_test(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - m) / s, (Xte - m) / s


def ridge_val(Xtr, Ytr, gtr, Xte, rng):
    """Ridge with alpha picked on a validation 10% of the TRAINING trials, then refit on all of them."""
    trials = np.unique(gtr)
    val = np.isin(gtr, rng.choice(trials, max(1, len(trials) // 10), replace=False))
    best = max(Dc.RIDGE_ALPHAS, key=lambda a: np.nanmean(Dc.pearson_per_dim(
        Ytr[val], Dc.ridge_fit_predict(Xtr[~val], Ytr[~val], Xtr[val], a))))
    return Dc.ridge_fit_predict(Xtr, Ytr, Xte, best), float(best)


def cv_linear_fsnr(X, Y2, g, trial_type, rng):
    """fSNR of a 5-fold cross-validated ridge decoder (alpha chosen by CV r) of a 2-D target."""
    folds = Dc.stratified_trial_folds(trial_type, 5)
    best = (-np.inf, None)
    for a in Dc.RIDGE_ALPHAS:
        p = Dc.cross_validate(X, Y2, g, folds, "ridge", alpha=a)
        r = np.nanmean(Dc.pearson_per_dim(Y2, p))
        if r > best[0]:
            best = (r, p)
    return S.fsnr(Y2, best[1]), best[1]


# ----------------------------------------------------------------------------- 1. real data
def real_stats():
    if REAL_JSON.exists():
        return read_real()
    s = load_session(DAY)
    ol = (s.trials.kind == "OL").to_numpy()
    mt = s.trials.move_type.to_numpy()
    rates = remove_block_means(s.tx, s.block) * HZ_PER_COUNT
    b0 = s.tx[np.isin(s.block, s.trials.block[ol].unique())].mean(0) * HZ_PER_COUNT  # raw mean rate (Hz)
    dirs = C.trial_directions(s.Tp, s.Cp, s.go, mt)
    win = extract_windows(rates, s.go, WINDOW)  # (trials, 20, N)
    nb = WINDOW[1] - WINDOW[0] + 1

    def design(idx, cols, extra=None):
        d = dirs[idx][:, cols]
        if extra is not None:
            d = np.c_[d, extra]
        X = np.c_[np.ones(len(idx) * nb), np.repeat(d, nb, axis=0)]
        return X, win[idx].reshape(-1, win.shape[2]), np.repeat(idx, nb)

    iR, iL = np.where(ol & (mt == 1))[0], np.where(ol & (mt == 2))[0]
    iU = np.r_[iR, iL]
    fR = E.cv_ols(*design(iR, [0, 1]), method="splithalf", names=("b0", "Rx", "Ry"))
    fL = E.cv_ols(*design(iL, [2, 3]), method="splithalf", names=("b0", "Lx", "Ly"))
    c_lat = np.where(mt[iU] == 1, 1.0, -1.0)
    fU = E.cv_ols(*design(iU, [0, 1, 2, 3], extra=c_lat), method="splithalf",
                  names=("b0", "Rx", "Ry", "Lx", "Ly", "lat"))
    mags = dict(Rx=E.cv_magnitude(fR, "Rx"), Ry=E.cv_magnitude(fR, "Ry"),
                Lx=E.cv_magnitude(fL, "Lx"), Ly=E.cv_magnitude(fL, "Ly"), lat=E.cv_magnitude(fU, "lat"))
    corr = dict(x=E.cv_correlation(fR, "Rx", fL, "Lx"), y=E.cv_correlation(fR, "Ry", fL, "Ly"))

    # real fSNR of RH decoding on uni-R trials: single bin and 400-ms window average
    rng = np.random.default_rng(0)
    Xb = win[iR].reshape(-1, win.shape[2]); Yb = np.repeat(dirs[iR][:, :2], nb, axis=0)
    gb = np.repeat(np.arange(len(iR)), nb)
    f_bin, p_bin = cv_linear_fsnr(Xb, Yb, gb, np.ones(len(iR), int), rng)
    Xw = win[iR].mean(1); Yw = dirs[iR][:, :2]
    f_win, p_win = cv_linear_fsnr(Xw, Yw, np.arange(len(iR)), np.ones(len(iR), int), rng)
    out = dict(day=DAY, n_uniR=len(iR), n_uniL=len(iL), b0=b0, mags=mags, real_rh_lh_corr=corr,
               fsnr_single_bin=f_bin, fsnr_window=f_win,
               scatter=dict(y_bin=Yb, p_bin=p_bin, y_win=Yw, p_win=p_win))
    atomic_write_json(REAL_JSON, out)
    print(f"real: mags {({k: round(v, 1) for k, v in mags.items()})}, RH-LH corr x {corr['x']:.2f} y {corr['y']:.2f}")
    print(f"real fSNR uni-R: single-bin {f_bin:.2f} (paper 2.03), window-average {f_win:.2f}")
    return read_real()


def read_real():
    import json
    d = json.loads(REAL_JSON.read_text())
    d["b0"] = np.array(d["b0"])
    return d


# ----------------------------------------------------------------------------- 2. calibration
def sim_fsnr(pop, sigma, seed, window=False):
    """Single-bin (or window-average) fSNR of RH decoding on simulated uni-R trials."""
    rng = np.random.default_rng(seed)
    X, Y, g, hand = S.simulate_trials(pop, 600, BINS, sigma, rng)
    m = hand[g] == 1
    X, Y, g = X[m], Y[m][:, :2], g[m]
    if window:
        tr = np.unique(g)
        X = np.stack([X[g == t].mean(0) for t in tr]); Y = np.stack([Y[g == t][0] for t in tr]); g = np.arange(len(tr))
    _, gi = np.unique(g, return_inverse=True)
    f, p = cv_linear_fsnr(X, Y, gi, np.ones(gi.max() + 1, int), rng)
    return f, Y, p


def calibrate_sigma(real, target):
    if "sigma" in real:
        return real["sigma"]
    pop = S.make_population(real["b0"], real["mags"], 0.0, 1.0, np.random.default_rng(1))
    lo, hi = np.log(1.0), np.log(2000.0)  # fSNR falls as sigma grows
    for it in range(14):
        mid = 0.5 * (lo + hi)
        f, _, _ = sim_fsnr(pop, np.exp(mid), seed=it)
        print(f"  calibrate: sigma={np.exp(mid):7.2f} Hz -> fSNR {f:.3f} (target {target:.3f})")
        lo, hi = (mid, hi) if f > target else (lo, mid)
    sigma = float(np.exp(0.5 * (lo + hi)))
    fw, yw, pw = sim_fsnr(pop, sigma, seed=99, window=True)
    fb, yb, pb = sim_fsnr(pop, sigma, seed=98)
    import json
    d = json.loads(REAL_JSON.read_text())
    d.update(sigma=sigma, sim_fsnr_single_bin=fb, sim_fsnr_window=fw,
             sim_scatter=dict(y_bin=yb, p_bin=pb, y_win=yw, p_win=pw))
    atomic_write_json(REAL_JSON, d)
    print(f"sigma = {sigma:.2f} Hz; simulated fSNR single-bin {fb:.2f}, window-average {fw:.2f}")
    return sigma


# ----------------------------------------------------------------------------- 3. sweep
def run_cell(real, sigma, rho, lat, seed):
    rng = np.random.default_rng(seed)
    pop = S.make_population(real["b0"], real["mags"], rho, lat, rng)
    X, Y, g, hand = S.simulate_trials(pop, N_PER_HAND, BINS, sigma, rng)
    fold = Dc.stratified_trial_folds(hand, 5, seed=seed)  # fold 0 = 20% test
    te = fold[g] == 0
    Xtr, Xte = zscore_train_test(X[~te], X[te])
    out = dict(rho=rho, lat=lat, seed=seed)
    for name in ("ridge", "ffn"):
        if name == "ridge":
            pred, alpha = ridge_val(Xtr, Y[~te], g[~te], Xte, rng)
        else:
            pred, alpha = Dc.ffn_fit_predict(Xtr, Y[~te], Xte, g[~te], seed=seed), None
        r = Dc.pearson_per_dim(Y[te], pred)
        hb = hand[g[te]]
        spR, spL = np.hypot(pred[:, 0], pred[:, 1]), np.hypot(pred[:, 2], pred[:, 3])
        out[name] = dict(r=r.tolist(), r_mean=float(np.mean(r)), r_LH=float(np.mean(r[2:])), alpha=alpha,
                         jitter_R=float(spR[hb == 2].mean() / spR[hb == 1].mean()),
                         jitter_L=float(spL[hb == 1].mean() / spL[hb == 2].mean()))
    return out


def main(force=False):
    if force:
        for f in (REAL_JSON, GRID):
            f.unlink(missing_ok=True)
    real = real_stats()
    sigma = calibrate_sigma(real, real["fsnr_single_bin"])
    done = {(c["rho"], c["lat"]) for c in read_jsonl(GRID)}
    for i, rho in enumerate(RHOS):
        for j, lat in enumerate(LATS):
            if (rho, lat) in done:
                continue
            c = run_cell(real, sigma, rho, lat, seed=1000 + 10 * i + j)
            append_jsonl(GRID, c)
            print(f"cell rho={rho:.2f} lat={lat:.1f}: r ridge {c['ridge']['r_mean']:.3f} "
                  f"(LH {c['ridge']['r_LH']:.3f})  FFN {c['ffn']['r_mean']:.3f} (LH {c['ffn']['r_LH']:.3f})")
    plot_grid(read_jsonl(GRID), read_real())


# ----------------------------------------------------------------------------- figures
def plot_grid(cells, real):
    import matplotlib.pyplot as plt

    # sequential (one-hue) ramp for the ordinal laterality scale: light -> dark
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for k, lat in enumerate(LATS):
        cs = sorted([c for c in cells if c["lat"] == lat], key=lambda c: c["rho"])
        rho = [c["rho"] for c in cs]
        for ax, key in ((axes[0], "ridge"), (axes[1], "ffn")):
            ax.plot(rho, [c[key]["r_mean"] for c in cs], "-o", color=ramp[k], label=f"laterality ×{lat:g}")
        axes[2].plot(rho, [c["ffn"]["jitter_L"] - c["ridge"]["jitter_L"] for c in cs], "-o", color=ramp[k])
    # magnitude of the real RH-LH correlation (x is negative, y positive; |rho| is what hurts a linear decoder)
    real_rho = np.mean([abs(real["real_rh_lh_corr"]["x"]), abs(real["real_rh_lh_corr"]["y"])])
    for ax, title in ((axes[0], "Ridge (linear)"), (axes[1], "FFN (nonlinear)")):
        ax.set_ylim(0, 1); ax.set_xlabel("RH–LH tuning correlation ρ"); ax.set_ylabel("held-out Pearson r (mean of 4 dims)")
        ax.set_title(title, fontsize=9)
        ax.axvline(abs(real_rho), color=P.INK_MUTED, ls=":", lw=1)
    axes[0].text(abs(real_rho) + 0.01, 0.05, f"real mean |ρ| = {real_rho:.2f}", fontsize=7, color=P.INK_MUTED)
    axes[1].legend(fontsize=7, loc="lower left")
    axes[2].axhline(0, color=P.INK_MUTED, lw=0.8)
    axes[2].set_xlabel("RH–LH tuning correlation ρ"); axes[2].set_ylabel("FFN − ridge, LH jitter")
    axes[2].set_title("Idle-cursor jitter difference (< 0: FFN steadier)", fontsize=9)
    fig.suptitle(f"Fig 3 simulation replication: 192 synthetic neurons, noise matched to real single-bin fSNR "
                 f"({real['fsnr_single_bin']:.2f}); 20% held-out trials", fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "fig3_simulations")

    # Supp Fig 4: decoded x/y scatter coloured by target direction, real vs simulated, bin vs window
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
    for row, (src, lab, fb, fw) in enumerate((
            ("scatter", "real", real["fsnr_single_bin"], real["fsnr_window"]),
            ("sim_scatter", "simulated", real["sim_fsnr_single_bin"], real["sim_fsnr_window"]))):
        for col, (yk, pk, f, nm) in enumerate((("y_bin", "p_bin", fb, "single 20-ms bin"),
                                               ("y_win", "p_win", fw, "400-ms window average"))):
            ax = axes[row, col]
            y, p = np.array(real[src][yk]), np.array(real[src][pk])
            ang = np.degrees(np.arctan2(y[:, 1], y[:, 0])) % 360
            n = min(len(p), 4000)
            idx = np.random.default_rng(0).choice(len(p), n, replace=False)
            ax.scatter(p[idx, 0], p[idx, 1], c=ang[idx], cmap="twilight", vmin=0, vmax=360, s=4, alpha=0.5, lw=0)
            ax.set_title(f"{lab}, {nm}: fSNR {f:.2f}", fontsize=8)
            ax.set_aspect("equal"); ax.set_xlabel("decoded x"); ax.set_ylabel("decoded y")
    fig.suptitle("Supp Fig 4 replication: decoded RH direction, coloured by true target angle "
                 "(cyclic colour map)", fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "supp_fig4_fsnr")
    print("wrote", P.FIG_DIR / "fig3_simulations.png", "and supp_fig4_fsnr.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="discard cached real stats and grid cells")
    main(ap.parse_args().force)
