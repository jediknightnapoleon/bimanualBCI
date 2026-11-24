"""Phase 11 (optional extension): threshold crossings (TX) vs spike-band power (SBP) in T5.

Question
--------
The target system (8 flexible shanks x 32 electrodes) gives SBP only. Does SBP decode movement as
well as TX in the same participant and arrays, does it degrade the same way with fewer or muted
channels, and how much of SBP is a copy of TX?

Data: Rigotti-Thompson et al. 2026 (Dryad doi:10.5061/dryad.brv15dvr9), ``fig1_T5_move.h5`` (main)
and ``fig1_T5_delay.h5`` (preparatory period). See ``bimanual/sbp.py`` for the format and caveats:
open-loop radial-8 wrist task, 48 of 192 channels (rate > 2 Hz), ACAUSAL Gaussian smoothing
(sigma 100 ms), 3 blocks (7, 8, 9), about 340 trials, one session.

Method (CPU only; nothing here may slow the Phase 10 GPU queue)
----------------------------------------------------------------
* Folds: leave-one-block-out (3 folds). Every choice (ridge alpha, greedy channel sets, the
  TX->SBP residual model, z-scoring statistics) uses the training blocks of the fold only; alpha
  by an inner leave-one-block-out over the 2 training blocks.
* Features (z-scored per channel with training-block statistics):
  TX; SBP (raw a.u.); logSBP; both (TX + logSBP); resid = logSBP minus the part predicted from
  the SAME electrode's TX (5 shifts, fitted on training blocks).
  # ASSUMPTION: logSBP is the main SBP feature for items 2-3 (power is right-skewed; chosen a
  priori, raw SBP is reported alongside in item 1).
* Targets:
  - velocity: per-bin cursor velocity (central difference, px/bin) for t in [-200, 500) ms
    after the go cue. The cursor follows a cued bell-shaped trajectory out to the target
    (mean speed rises from 0 at ~50 ms, peaks ~250 ms, is ~0 by 500 ms) and jumps back to the
    centre at ~600 ms, which the window excludes. Metric: Pearson r per x/y, averaged, pooled
    over all held-out bins.
  - class8: target identity (radial 8) from the mean features over [0, 400) ms. Ridge on
    one-hot targets, argmax. Metric: accuracy (chance 0.125).
  - delay8: target identity from the delay period of the delay file, [200, 700) ms after the
    delay start, trials with delay >= 1.0 s only (>= 300 ms before the go cue, i.e. 3 sigma of
    the smoothing kernel).
* Decoders: ridge (all items); FFN (bimanual.decoders.ffn_fit_predict, 3 seeds) for velocity in
  item 1. Lag-0 features everywhere; causal lags (0, 40, 80 ms) are tested for ridge in item 1.
* CIs: paired bootstrap over trials (2000 resamples) of SBP - TX, using each trial's held-out
  prediction. For channel count and muting the spread is over subsets / masks.

Items
-----
1. Feature comparison (velocity: ridge, ridge+lags, FFN; class8 and delay8: ridge).
2. Channel count: 200 random subsets per n (the SAME subsets for TX and SBP): all 48 channels
   n in {4, 8, 16, 32, 48}; array A n in {4, 8, 16, 20}; array B n in {4, 8, 16, 28}. Plus greedy
   best-n (forward selection on the inner leave-one-block-out velocity r, within each fold).
3. Muting at test time, f in {0, ..., 0.6}, 20 masks (seed 7000 + 100 f, as register #42), modes
   zero / dead / noisy; models trained normally and with channel-dropout augmentation.
4. Overlap: per-channel TX-SBP correlation, held-out R^2 of zlogSBP from the own electrode's TX and
   from all 48 TX channels; decoding from the residual (item 1).

Outputs: results/ext_sbp/summary.json; then scripts/ext_sbp_plot.py draws
results/figures/ext_sbp.{png,svg}.
With ``--metric r2 --out results/r2/sbp.json`` (run by scripts/rescore_r2.py) the velocity scores are
R^2 instead of r (register #47); models and selections are the same.
Run    : CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 pixi run python -I scripts/ext_sbp.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only: the GPU belongs to the Phase 10 queue
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

torch.set_num_threads(4)

from bimanual import sbp as S  # noqa: E402
from bimanual.cache import atomic_write_json  # noqa: E402
from bimanual.decoders import ffn_fit_predict, r2_per_dim  # noqa: E402

ALPHAS = np.logspace(-1, 5, 13)
VEL_WIN = (-200, 500)
CLS_WIN = (0, 400)
DELAY_WIN, DELAY_MIN_S = (200, 700), 1.0
LAGS = (0, 2, 4)  # causal lags in bins (0, 40, 80 ms) for the "ridge+lags" variant
FFN_SEEDS = (0, 1, 2)
N_BOOT = 2000
N_SUB = 200
POOLS = {"all": (4, 8, 16, 32, 48), "A": (4, 8, 16, 20), "B": (4, 8, 16, 28)}
MUTE_FRACS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
N_MASKS = 20
DROPOUT_COPIES, DROPOUT_MAX = 4, 0.5
# Reported velocity metric: "r" (default, Pearson r) or "r2" (1 - SSE/SST, set by --metric r2 for the R^2
# rescoring). Model selection (ridge alpha, greedy channel order) always uses r, so both runs fit the
# same models and only the reported score differs.
REPORT_METRIC = "r"
MAIN = ("TX", "logSBP", "SBP")  # features compared in items 2-3 (logSBP = main SBP feature, chosen a priori)


# ----------------------------------------------------------------------------- per-fold features
def fold_features(d: S.RTTrials, train: np.ndarray) -> tuple[dict, dict]:
    """z-scored features for all trials, with statistics from ``train`` trials. Also returns the
    z-value of a dead channel per feature (for the 'dead' muting mode)."""
    n_tx = S.Norm.fit(d.tx[train])
    n_sbp = S.Norm.fit(d.sbp[train])
    n_log = S.Norm.fit(d.sbp[train], log=True)
    z_tx, z_sbp, z_log = n_tx(d.tx), n_sbp(d.sbp), n_log(d.sbp)
    resid, _, _ = S.own_channel_residual(z_tx, z_log, train)
    n_res = S.Norm.fit(resid[train])
    feats = dict(TX=z_tx, SBP=z_sbp, logSBP=z_log, both=np.concatenate([z_tx, z_log], 2), resid=n_res(resid))
    # dead channel: TX = 0 counts; SBP = the channel's training-block 1st percentile (its floor).
    # ASSUMPTION: a disconnected / flat SBP channel sits at its noise floor, approximated by the
    # 1st percentile of its smoothed training-block values.
    floor = np.percentile(d.sbp[train], 1, axis=(0, 1))
    dead = dict(TX=n_tx.z_of_raw(np.zeros(d.tx.shape[2])), logSBP=n_log.z_of_raw(floor), SBP=n_sbp.z_of_raw(floor))
    return feats, dead


def folds(d: S.RTTrials):
    for b in np.unique(d.block):
        yield b, d.block != b, d.block == b


def inner_folds(d: S.RTTrials, train: np.ndarray):
    for b in np.unique(d.block[train]):
        yield train & (d.block != b), train & (d.block == b)


# ----------------------------------------------------------------------------- task models
class VelTask:
    """Per-bin velocity regression in a window; rows are (trial, bin) pairs."""

    def __init__(self, d: S.RTTrials, win=VEL_WIN, lags=(0,)):
        self.w = S.time_window(d.t, *win)
        self.lags = lags
        self.Y = S.cursor_velocity(d.cursor)[:, self.w]  # (trials, W, 2)

    def X(self, F):
        if self.lags == (0,):
            return F[:, self.w]
        return np.concatenate([S.shift_bins(F, k)[:, self.w] for k in self.lags], -1)

    def score(self, y, yhat):  # y, yhat: (trials, W, 2); the reported metric
        if REPORT_METRIC == "r2":
            return float(r2_per_dim(y.reshape(-1, 2), yhat.reshape(-1, 2)).mean())
        return S.mean_r(y.reshape(-1, 2), yhat.reshape(-1, 2))

    def select_score(self, y, yhat):  # model selection: always Pearson r
        return S.mean_r(y.reshape(-1, 2), yhat.reshape(-1, 2))


class ClsTask:
    """8-way target classification from mean features in a window; ridge on one-hot targets."""

    def __init__(self, d: S.RTTrials, win=CLS_WIN, trials: np.ndarray | None = None):
        self.w = S.time_window(d.t, *win)
        self.cls = S.target_class(d.target)
        self.Y = S.one_hot(self.cls)
        self.keep = np.ones(len(self.cls), bool) if trials is None else trials

    def X(self, F):
        return F[:, self.w].mean(1)

    def score(self, y, yhat):
        return float(np.mean(y.argmax(-1) == yhat.argmax(-1)))

    select_score = score


def _rows(task, A, m):
    """Rows of A (trials x ...) for trial mask m, restricted to the task's trials."""
    keep = getattr(task, "keep", None)
    mm = m if keep is None else (m & keep)
    return A[mm].reshape(-1, A.shape[-1]), mm


class FoldModel:
    """Gram-based ridge for one fold and one feature: outer + inner Grams, alpha chosen per subset."""

    def __init__(self, task, X, d, train):
        self.task, self.X = task, X
        Xtr, _ = _rows(task, X, train)
        Ytr, _ = _rows(task, task.Y, train)
        self.outer = S.GramRidge(Xtr, Ytr)
        self.inner = []
        for itr, iva in inner_folds(d, train):
            Xi, _ = _rows(task, X, itr)
            Yi, _ = _rows(task, task.Y, itr)
            Xv, mv = _rows(task, X, iva)
            self.inner.append((S.GramRidge(Xi, Yi), Xv, task.Y[mv]))

    def inner_score(self, s, alpha):
        out = []
        for g, Xv, Yv in self.inner:
            p = g.predict(Xv, s, alpha).reshape(Yv.shape)
            out.append(self.task.select_score(Yv, p))
        return float(np.mean(out))

    def best_alpha(self, s):
        sc = [self.inner_score(s, a) for a in ALPHAS]
        return float(ALPHAS[int(np.argmax(sc))])

    def predict(self, Xte, s, alpha=None):
        alpha = self.best_alpha(s) if alpha is None else alpha
        return self.outer.predict(Xte, s, alpha), alpha


def heldout_predict(d, task, feat_by_fold, s=None, alpha_log=None):
    """Held-out predictions for every trial (each from the model of its own fold); returns
    (pred (trials, ...) aligned with task.Y, kept-trial mask)."""
    pred = np.full(task.Y.shape, np.nan)
    for (b, train, test), F in zip(folds(d), feat_by_fold):
        X = task.X(F)
        fm = FoldModel(task, X, d, train)
        ss = np.arange(X.shape[-1]) if s is None else s
        Xte, mte = _rows(task, X, test)
        p, a = fm.predict(Xte, ss)
        pred[mte] = p.reshape(task.Y[mte].shape)
        if alpha_log is not None:
            alpha_log.append(a)
    keep = getattr(task, "keep", np.ones(len(task.Y), bool))
    return pred, keep


def boot_compare(task, y, pa, pb, keep, seed=0):
    idx_all = np.flatnonzero(keep)
    fa = lambda i: task.score(y[idx_all[i]], pa[idx_all[i]])
    fb = lambda i: task.score(y[idx_all[i]], pb[idx_all[i]])
    return S.paired_bootstrap(fa, fb, len(idx_all), N_BOOT, seed)


# ----------------------------------------------------------------------------- item 1
def item1(d, dd, fb, fb_delay, log):
    out = {}
    feats = ["TX", "SBP", "logSBP", "both", "resid"]
    tasks = {"velocity_ridge": VelTask(d), "velocity_ridge_lags": VelTask(d, lags=LAGS), "class8_ridge": ClsTask(d)}
    keep_delay = dd.delay >= DELAY_MIN_S
    tasks_delay = {"delay8_ridge": ClsTask(dd, DELAY_WIN, trials=keep_delay)}
    for name, task, fbx, dat in [(k, v, fb, d) for k, v in tasks.items()] + [(k, v, fb_delay, dd) for k, v in tasks_delay.items()]:
        preds, alphas = {}, {}
        for f in feats:
            al = []
            preds[f], keep = heldout_predict(dat, task, [F[f] for F in fbx], alpha_log=al)
            alphas[f] = al
        res = {f: dict(score=task.score(task.Y[keep], preds[f][keep]), alphas=alphas[f]) for f in feats}
        for f in ("SBP", "logSBP", "both", "resid"):
            res[f]["vs_TX"] = boot_compare(task, task.Y, preds["TX"], preds[f], keep, seed=1)
        res["n_trials"] = int(keep.sum())
        out[name] = res
        log(f"item1 {name}: " + "  ".join(f"{f} {res[f]['score']:.3f}" for f in feats))
    # FFN on velocity (lag 0), 3 seeds; bootstrap of the seed-mean score
    task = VelTask(d)
    preds = {f: np.full((len(FFN_SEEDS),) + task.Y.shape, np.nan) for f in feats}
    for (b, train, test), F in zip(folds(d), fb):
        tr_trials = np.flatnonzero(train)
        groups = np.repeat(tr_trials, task.w.sum())
        for f in feats:
            X = task.X(F[f])
            for i, sd in enumerate(FFN_SEEDS):
                p = ffn_fit_predict(X[train].reshape(-1, X.shape[-1]), task.Y[train].reshape(-1, 2),
                                    X[test].reshape(-1, X.shape[-1]), groups, seed=sd)
                preds[f][i, test] = p.reshape(task.Y[test].shape)
    keep = np.ones(len(task.Y), bool)
    res = {}
    for f in feats:
        per_seed = [task.score(task.Y, preds[f][i]) for i in range(len(FFN_SEEDS))]
        res[f] = dict(score=float(np.mean(per_seed)), per_seed=per_seed)
    seedmean = lambda P: (lambda idx: np.mean([task.score(task.Y[idx], P[i][idx]) for i in range(len(FFN_SEEDS))]))
    for f in ("SBP", "logSBP", "both", "resid"):
        res[f]["vs_TX"] = S.paired_bootstrap(seedmean(preds["TX"]), seedmean(preds[f]), len(task.Y), N_BOOT, 1)
    out["velocity_ffn"] = res
    log("item1 velocity_ffn: " + "  ".join(f"{f} {res[f]['score']:.3f}" for f in feats))
    return out


# ----------------------------------------------------------------------------- item 2
def item2(d, fb, log):
    chan_pool = {"all": np.arange(len(d.chans)), "A": np.flatnonzero(d.array == "A"), "B": np.flatnonzero(d.array == "B")}
    tasks = {"velocity": VelTask(d), "class8": ClsTask(d)}
    out = {}
    for tname, task in tasks.items():
        # per fold, per feature: one FoldModel (Grams of all 48 channels) reused for every subset
        fms = {f: [] for f in MAIN}
        for (b, train, test), F in zip(folds(d), fb):
            for f in MAIN:
                X = task.X(F[f])
                Xte, mte = _rows(task, X, test)
                fms[f].append((FoldModel(task, X, d, train), Xte, mte))

        def heldout(f, s):
            pred = np.full(task.Y.shape, np.nan)
            for fm, Xte, mte in fms[f]:
                p, _ = fm.predict(Xte, s)
                pred[mte] = p.reshape(task.Y[mte].shape)
            return task.score(task.Y, pred)

        res = {}
        for pi, (pool, ns) in enumerate(POOLS.items()):
            res[pool] = {}
            for n in ns:
                subs = S.random_subsets(chan_pool[pool], n, N_SUB, seed=11000 + 100 * pi + n)
                sc = {f: np.array([heldout(f, s) for s in subs]) for f in MAIN}
                res[pool][str(n)] = dict(
                    n_subsets=len(subs),
                    **{f: dict(median=float(np.median(sc[f])), p5=float(np.percentile(sc[f], 5)),
                              p95=float(np.percentile(sc[f], 95)), mean=float(sc[f].mean())) for f in MAIN})
                for f in ("logSBP", "SBP"):  # paired over the same subsets
                    diff = sc[f] - sc["TX"]
                    res[pool][str(n)][f"diff_{f}_minus_TX"] = dict(
                        mean=float(diff.mean()), median=float(np.median(diff)), p2_5=float(np.percentile(diff, 2.5)),
                        p97_5=float(np.percentile(diff, 97.5)), frac_SBP_better=float((diff > 0).mean()))
            log(f"item2 {tname} {pool}: " + "  ".join(
                f"n{n} TX {res[pool][str(n)]['TX']['median']:.3f} SBP {res[pool][str(n)]['logSBP']['median']:.3f}" for n in ns))
        # greedy best-n (velocity only): selection on the inner leave-one-block-out score of each fold
        if tname == "velocity":
            gres = {}
            for pool, ns in POOLS.items():
                gres[pool] = {}
                for f in MAIN:
                    pred = {n: np.full(task.Y.shape, np.nan) for n in ns}
                    sets = []
                    for fm, Xte, mte in fms[f]:
                        a_all = fm.best_alpha(chan_pool[pool])
                        chosen, _ = S.greedy_select(lambda s: fm.inner_score(s, a_all), chan_pool[pool], max(ns))
                        sets.append([int(d.chans[c]) for c in chosen])
                        for n in ns:
                            p, _ = fm.predict(Xte, np.array(chosen[:n]))
                            pred[n][mte] = p.reshape(task.Y[mte].shape)
                    gres[pool][f] = dict(score={str(n): task.score(task.Y, pred[n]) for n in ns},
                                         order_by_fold_electrode_ids=sets)
                log(f"item2 greedy {pool}: " + "  ".join(
                    f"{f} " + ",".join(f"{gres[pool][f]['score'][str(n)]:.3f}" for n in ns) for f in MAIN))
            res["greedy"] = gres
        out[tname] = res
    return out


# ----------------------------------------------------------------------------- item 3
def item3(d, fb, dead_by_fold, log):
    tasks = {"velocity": VelTask(d), "class8": ClsTask(d)}
    C = len(d.chans)
    out = {}
    for tname, task in tasks.items():
        res = {}
        for f in MAIN:
            # fit per fold: normal ridge (Gram, alpha by inner CV) and a channel-dropout ridge
            models = []
            for (b, train, test), F, dead in zip(folds(d), fb, dead_by_fold):
                X = task.X(F[f])
                fm = FoldModel(task, X, d, train)
                s = np.arange(C)
                alpha = fm.best_alpha(s)
                W = fm.outer.coef(s, alpha)
                # dropout augmentation: copies of each training trial with p ~ U(0, 0.5) of its
                # channels set to 0 (z units). DEVIATION: alpha reused from the normal model.
                rng = np.random.default_rng(500 + int(b))
                Xtr = X[train]
                Ytr = task.Y[train]
                aug_X, aug_Y = [Xtr], [Ytr]
                for _ in range(DROPOUT_COPIES):
                    p = rng.uniform(0, DROPOUT_MAX, size=(len(Xtr),) + (1,) * (Xtr.ndim - 1))
                    keep = rng.random((len(Xtr),) + (1,) * (Xtr.ndim - 2) + (C,)) >= p
                    aug_X.append(Xtr * keep); aug_Y.append(Ytr)
                gd = S.GramRidge(np.concatenate(aug_X).reshape(-1, C), np.concatenate(aug_Y).reshape(-1, Ytr.shape[-1]))
                Wd = gd.coef(s, alpha)
                models.append(dict(test=test, F=F[f][test], dead=dead[f], normal=(fm.outer.mx, W, fm.outer.my), dropout=(gd.mx, Wd, gd.my)))
            res[f] = {}
            for kind in ("normal", "dropout"):
                res[f][kind] = {}
                for mode in S.MUTE_MODES:
                    curve = []
                    for frac in MUTE_FRACS:
                        masks = S.random_masks(C, frac, N_MASKS, seed=7000 + int(round(100 * frac)))
                        scores = []
                        for mi, m in enumerate(masks):
                            pred = np.full(task.Y.shape, np.nan)
                            for md in models:
                                # mute the per-bin features, then build the task input (so a noisy
                                # channel is N(0, 1) in every bin, also before window averaging)
                                rng = np.random.default_rng(90000 + 1000 * mi + int(round(100 * frac)))
                                Xm = task.X(S.mute(md["F"], m, mode, dead_z=md["dead"], rng=rng))
                                mx, W, my = md[kind]
                                pred[md["test"]] = (Xm - mx) @ W + my
                            scores.append(task.score(task.Y, pred))
                            if frac == 0:
                                break  # every mask is empty at f = 0
                        scores = np.array(scores)
                        curve.append(dict(f=frac, mean=float(scores.mean()), p5=float(np.percentile(scores, 5)),
                                          p95=float(np.percentile(scores, 95))))
                    res[f][kind][mode] = curve
            log(f"item3 {tname} {f}: " + "  ".join(
                f"{k}/{m} f0.3 {res[f][k][m][3]['mean']:.3f}" for k in ("normal", "dropout") for m in S.MUTE_MODES))
        out[tname] = res
    return out


# ----------------------------------------------------------------------------- item 4
def item4(d, log):
    C = len(d.chans)
    corr_log = [float(np.corrcoef(d.tx[..., c].ravel(), np.log(d.sbp[..., c]).ravel())[0, 1]) for c in range(C)]
    corr_raw = [float(np.corrcoef(d.tx[..., c].ravel(), d.sbp[..., c].ravel())[0, 1]) for c in range(C)]
    own_pred = np.full(d.sbp.shape, np.nan)
    pop_pred = np.full(d.sbp.shape, np.nan)
    z_all = np.full(d.sbp.shape, np.nan)
    for b, train, test in folds(d):
        n_tx, n_log = S.Norm.fit(d.tx[train]), S.Norm.fit(d.sbp[train], log=True)
        z_tx, z_log = n_tx(d.tx), n_log(d.sbp)
        _, pred, _ = S.own_channel_residual(z_tx, z_log, train)
        own_pred[test] = pred[test]
        g = S.GramRidge(z_tx[train].reshape(-1, C), z_log[train].reshape(-1, C))
        pop_pred[test] = g.predict(z_tx[test].reshape(-1, C), np.arange(C), 10.0).reshape(z_tx[test].shape)
        z_all[test] = z_log[test]
    r2_own = [S.r2(z_all[..., c].ravel(), own_pred[..., c].ravel()) for c in range(C)]
    r2_pop = [S.r2(z_all[..., c].ravel(), pop_pred[..., c].ravel()) for c in range(C)]
    summ = lambda v: dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))
    out = dict(per_channel=dict(electrode=d.chans.tolist(), array=d.array.tolist(), corr_tx_logsbp=corr_log,
                                corr_tx_sbp=corr_raw, heldout_r2_own_tx=r2_own, heldout_r2_all_tx=r2_pop,
                                tx_mean_count_per_bin=d.tx.mean((0, 1)).tolist()),
               corr_tx_logsbp=summ(corr_log), corr_tx_sbp=summ(corr_raw),
               heldout_r2_own_tx=summ(r2_own), heldout_r2_all_tx=summ(r2_pop))
    log(f"item4: corr(TX,logSBP) median {out['corr_tx_logsbp']['median']:.2f}; held-out R2 own-TX "
        f"{out['heldout_r2_own_tx']['median']:.2f}, all-TX {out['heldout_r2_all_tx']['median']:.2f}")
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", default=str(S.MOVE_FILE))
    ap.add_argument("--delay", default=str(S.DELAY_FILE))
    ap.add_argument("--items", default="1234")
    ap.add_argument("--metric", choices=["r", "r2"], default="r", help="reported velocity metric (selection uses r)")
    ap.add_argument("--out", default=str(S.OUT_DIR / "summary.json"))
    args = ap.parse_args()
    global REPORT_METRIC
    REPORT_METRIC = args.metric
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(out_path.parent / ("run_log.txt" if args.metric == "r" else f"run_log_{args.metric}.txt"), "a")
    t0 = time.time()

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    d, dd = S.load_rt(Path(args.move)), S.load_rt(Path(args.delay))
    log(f"move: {d.tx.shape} ({d.n_dropped} dropped); delay: {dd.tx.shape}; blocks {np.unique(d.block).tolist()}; "
        f"A {int((d.array == 'A').sum())} / B {int((d.array == 'B').sum())} channels")
    fb, dead = zip(*[fold_features(d, train) for _, train, _ in folds(d)])
    fb_delay = [fold_features(dd, train)[0] for _, train, _ in folds(dd)]

    summary = json.loads(out_path.read_text()) if out_path.exists() else {}
    summary["meta"] = dict(
        data=[d.meta, dd.meta], n_trials_move=int(len(d.tx)), n_trials_delay_used=int((dd.delay >= DELAY_MIN_S).sum()),
        n_dropped_nan=[d.n_dropped, dd.n_dropped], electrodes=d.chans.tolist(),
        n_chan=dict(all=int(len(d.chans)), A=int((d.array == "A").sum()), B=int((d.array == "B").sum())),
        blocks=np.unique(d.block).tolist(), folds="leave-one-block-out (3); alpha by inner leave-one-block-out",
        velocity_window_ms=VEL_WIN, class8_window_ms=CLS_WIN, delay8_window_ms=DELAY_WIN, delay_min_s=DELAY_MIN_S,
        alphas=ALPHAS.tolist(), lags_bins=LAGS, ffn_seeds=FFN_SEEDS, n_boot=N_BOOT, n_subsets=N_SUB, pools=POOLS,
        mute_fracs=MUTE_FRACS, n_masks=N_MASKS, dropout=dict(copies=DROPOUT_COPIES, max_p=DROPOUT_MAX),
        main_sbp_feature="logSBP", chance_class8=0.125, velocity_metric=REPORT_METRIC)
    if "1" in args.items:
        summary["item1_features"] = item1(d, dd, fb, fb_delay, log)
        atomic_write_json(out_path, summary)
    if "4" in args.items:
        summary["item4_overlap"] = item4(d, log)
        atomic_write_json(out_path, summary)
    if "2" in args.items:
        summary["item2_channel_count"] = item2(d, fb, log)
        atomic_write_json(out_path, summary)
    if "3" in args.items:
        summary["item3_muting"] = item3(d, fb, dead, log)
    summary["meta"]["runtime_s"] = round(time.time() - t0, 1)
    atomic_write_json(out_path, summary)
    log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
