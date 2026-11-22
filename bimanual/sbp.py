"""Phase 11 (optional extension): spike-band power (SBP) vs threshold crossings (TX) in T5.

Why
---
The target system for a possible follow-up (8 flexible shanks x 32 electrodes) only gives SBP,
high-pass filtered at about 300 Hz. The paper's T5 data has threshold crossings (TX) only. So we
ask, on public T5 data that has BOTH features on the same electrodes and bins: does SBP decode
movement as well as TX, does it degrade the same way with fewer or muted channels, and how much
of it is just a copy of TX?

Data
----
Rigotti-Thompson et al. 2026, Dryad doi:10.5061/dryad.brv15dvr9, files
``Data/ext/fig1_T5_{move,delay}.h5`` (README copy: ``Data/ext/RT_README.md``):

* open-loop instructed-delay radial-8 task, attempted wrist movements; the cursor follows a cued
  trajectory (it is not decoded), so "cursor velocity" is the instructed movement;
* ``spike_data_all`` (trials, bins, chans): TX counts per bin, smoothed;
  ``sbp_data_all``: SBP (a.u.), same channels and bins;
* band-pass 250-5000 Hz (acausal 4th-order Butterworth), 20-ms bins, ACAUSAL Gaussian smoothing
  (sigma = 100 ms), channels with mean rate < 2 Hz dropped (48 of 192 kept, ``keep_chans_all``);
* ``move`` file: aligned to the go cue (t = 0); ``delay`` file: aligned to the delay start.

DEVIATION (register #44): the planned T11 (MINDFUL) comparison and the TX->SBP "stand-in" for the
bimanual sessions are dropped. The MINDFUL data could not be downloaded in time, and a raw-bin
stand-in cannot be fitted from acausally smoothed, trial-aligned data. The T5 files answer the
main question (TX vs SBP in the same participant and arrays) directly.

Contents
--------
* ``load_rt``: the h5 loader (``RTTrials``), dropping trials with NaNs.
* ``target_class``, ``cursor_velocity``: task variables.
* ``Norm``: per-channel z-scoring with TRAINING-block statistics (optionally log-transformed).
* ``GramRidge``: ridge regression from precomputed Gram matrices, so any channel subset can be
  fitted in microseconds (channel-count and greedy-selection analyses).
* ``mute``: test-time channel muting in three modes (register #42 analogue).
* ``greedy_select``: forward selection on an inner leave-one-block-out score.
* ``paired_bootstrap``: CI of a paired difference over trials.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "ext"
OUT_DIR = ROOT / "results" / "ext_sbp"
MOVE_FILE = DATA_DIR / "fig1_T5_move.h5"
DELAY_FILE = DATA_DIR / "fig1_T5_delay.h5"

BIN_MS = 20
N_ARRAY = 96  # channels 0-95 = array A, 96-191 = array B (as in the bimanual data)


# ----------------------------------------------------------------------------- loading
@dataclass
class RTTrials:
    tx: np.ndarray       # (trials, bins, chans) smoothed TX counts per bin
    sbp: np.ndarray      # (trials, bins, chans) smoothed SBP, a.u.
    cursor: np.ndarray   # (trials, bins, 2) px
    target: np.ndarray   # (trials, 2) px
    block: np.ndarray    # (trials,) int
    delay: np.ndarray    # (trials,) s
    t: np.ndarray        # (bins,) ms relative to the alignment event
    chans: np.ndarray    # (chans,) 0-based electrode ids out of 192
    n_dropped: int = 0   # trials dropped for NaNs
    meta: dict = field(default_factory=dict)

    @property
    def array(self) -> np.ndarray:
        """'A' or 'B' per channel."""
        return np.where(self.chans < N_ARRAY, "A", "B")


def file_meta(path: Path) -> dict:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(file=Path(path).name, bytes=Path(path).stat().st_size, sha256=h)


def load_rt(path: Path = MOVE_FILE) -> RTTrials:
    """Load a Rigotti-Thompson fig1 T5 file; trials with any NaN in TX/SBP/cursor/target are dropped."""
    import h5py

    with h5py.File(path, "r") as f:
        g = {k: f[k][()] for k in f.keys()}
    tx, sbp, cur, tgt = g["spike_data_all"], g["sbp_data_all"], g["cursor_data_all"], g["target_data_all"]
    ok = ~(np.isnan(tx).any((1, 2)) | np.isnan(sbp).any((1, 2)) | np.isnan(cur).any((1, 2))
           | np.isnan(tgt).any(1))
    # ASSUMPTION: all included trials are go trials (no_go_bool_all is 0 or NaN in both T5 files).
    if "no_go_bool_all" in g:
        ok &= ~(g["no_go_bool_all"] == 1)
    return RTTrials(tx=tx[ok].astype(np.float64), sbp=sbp[ok].astype(np.float64), cursor=cur[ok],
                    target=tgt[ok], block=g["block_id_data_all"][ok].astype(int),
                    delay=g["delay_data_all"][ok], t=g["t_data"].astype(int),
                    chans=g["keep_chans_all"].astype(int), n_dropped=int((~ok).sum()),
                    meta=file_meta(path))


def target_class(target: np.ndarray) -> np.ndarray:
    """Radial-8 target index 0..7 (0 = right, counter-clockwise in 45-degree steps)."""
    ang = np.arctan2(target[:, 1], target[:, 0])
    return np.round(ang / (np.pi / 4)).astype(int) % 8


def cursor_velocity(cursor: np.ndarray) -> np.ndarray:
    """Central-difference cursor velocity in px per bin, (trials, bins, 2)."""
    return np.gradient(cursor, axis=1)


def time_window(t: np.ndarray, start_ms: float, stop_ms: float) -> np.ndarray:
    return (t >= start_ms) & (t < stop_ms)


# ----------------------------------------------------------------------------- features
@dataclass
class Norm:
    """Per-channel z-scoring with statistics from training trials only (all their bins)."""

    mean: np.ndarray
    sd: np.ndarray
    log: bool = False

    @classmethod
    def fit(cls, x: np.ndarray, log: bool = False) -> "Norm":
        v = np.log(x) if log else x
        return cls(v.mean((0, 1)), v.std((0, 1)) + 1e-9, log)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((np.log(x) if self.log else x) - self.mean) / self.sd

    def z_of_raw(self, raw: np.ndarray) -> np.ndarray:
        """z-value of a raw per-channel value (e.g. 0 counts for a dead TX channel)."""
        return ((np.log(raw) if self.log else raw) - self.mean) / self.sd


def shift_bins(x: np.ndarray, k: int) -> np.ndarray:
    """Shift (trials, bins, ...) by k bins along time within each trial (k > 0: past), edge-padded."""
    if k == 0:
        return x
    out = np.empty_like(x)
    if k > 0:
        out[:, k:] = x[:, :-k]
        out[:, :k] = x[:, :1]
    else:
        out[:, :k] = x[:, -k:]
        out[:, k:] = x[:, -1:]
    return out


def own_channel_residual(z_tx: np.ndarray, z_sbp: np.ndarray, train: np.ndarray,
                         shifts=(-4, -2, 0, 2, 4), alpha: float = 1.0):
    """Per channel, regress zSBP on its OWN electrode's zTX at several shifts (fit on train trials).

    Shifts go both ways because the data are acausally smoothed anyway.
    Returns (residual (trials, bins, chans), prediction, coefficients (chans, n_shifts + 1)).
    """
    T, B, C = z_tx.shape
    lag = np.stack([shift_bins(z_tx, k) for k in shifts], -1)  # (T, B, C, S)
    pred = np.empty_like(z_sbp)
    coefs = np.zeros((C, len(shifts) + 1))
    for c in range(C):
        X = lag[train, :, c].reshape(-1, len(shifts))
        y = z_sbp[train, :, c].ravel()
        mx, my = X.mean(0), y.mean()
        w = np.linalg.solve((X - mx).T @ (X - mx) + alpha * np.eye(len(shifts)), (X - mx).T @ (y - my))
        pred[:, :, c] = (lag[:, :, c] - mx) @ w + my
        coefs[c] = np.r_[w, my - mx @ w]
    return z_sbp - pred, pred, coefs


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ----------------------------------------------------------------------------- ridge
class GramRidge:
    """Ridge regression (unpenalised intercept) fitted from the Gram matrix of the training data.

    ``fit(X, Y)`` stores X'X and X'Y of the centred training data once; ``coef(s, alpha)`` solves
    for any channel subset ``s`` cheaply, which makes 200 random subsets x many n affordable.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.mx, self.my = X.mean(0), Y.mean(0)
        Xc = X - self.mx
        self.G = Xc.T @ Xc
        self.C = Xc.T @ (Y - self.my)

    def coef(self, s: np.ndarray, alpha: float) -> np.ndarray:
        Gs = self.G[np.ix_(s, s)]
        return np.linalg.solve(Gs + alpha * np.eye(len(s)), self.C[s])

    def predict(self, X: np.ndarray, s: np.ndarray, alpha: float) -> np.ndarray:
        return (X[..., s] - self.mx[s]) @ self.coef(s, alpha) + self.my


def mean_r(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean over output dimensions of the Pearson r (pooled over all rows)."""
    return float(np.mean([np.corrcoef(y[:, d], yhat[:, d])[0, 1] for d in range(y.shape[1])]))


def one_hot(c: np.ndarray, k: int = 8) -> np.ndarray:
    return np.eye(k)[c]


# ----------------------------------------------------------------------------- muting
MUTE_MODES = ("zero", "dead", "noisy")


def mute(z: np.ndarray, mask: np.ndarray, mode: str, dead_z: np.ndarray | None = None,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """Replace the channels in ``mask`` (bool, chans) of z-scored data (..., chans).

    * zero : the channel is detected as bad and removed -> its training mean (0 in z units);
    * dead : undetected dead channel -> ``dead_z`` (z-value of a flat channel at its floor);
    * noisy: undetected noisy channel -> N(0, 1) per bin.
    """
    out = z.copy()
    if not mask.any():
        return out
    if mode == "zero":
        out[..., mask] = 0.0
    elif mode == "dead":
        out[..., mask] = dead_z[mask]
    elif mode == "noisy":
        out[..., mask] = rng.standard_normal(out[..., mask].shape)
    else:
        raise ValueError(mode)
    return out


def random_masks(n_chan: int, frac: float, n_masks: int, seed: int) -> np.ndarray:
    """(n_masks, n_chan) bool; exactly round(frac * n_chan) channels muted per mask."""
    rng = np.random.default_rng(seed)
    k = int(round(frac * n_chan))
    m = np.zeros((n_masks, n_chan), bool)
    for i in range(n_masks):
        m[i, rng.choice(n_chan, k, replace=False)] = True
    return m


def random_subsets(pool: np.ndarray, n: int, n_sub: int, seed: int) -> np.ndarray:
    """(n_sub, n) channel indices drawn from ``pool``; a single subset if n == len(pool)."""
    if n >= len(pool):
        return pool[None, :].copy()
    rng = np.random.default_rng(seed)
    return np.stack([np.sort(rng.choice(pool, n, replace=False)) for _ in range(n_sub)])


# ----------------------------------------------------------------------------- selection
def greedy_select(score_fn, pool: np.ndarray, n_max: int) -> tuple[list[int], list[float]]:
    """Forward selection: repeatedly add the channel in ``pool`` that maximises ``score_fn(subset)``."""
    chosen: list[int] = []
    scores: list[float] = []
    left = list(pool)
    for _ in range(min(n_max, len(pool))):
        best = max(left, key=lambda c: score_fn(np.array(chosen + [c])))
        chosen.append(best)
        left.remove(best)
        scores.append(score_fn(np.array(chosen)))
    return chosen, scores


# ----------------------------------------------------------------------------- statistics
def paired_bootstrap(stat_a, stat_b, n_units: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap over units (trials) of stat_b(idx) - stat_a(idx); returns point, CI, both points."""
    rng = np.random.default_rng(seed)
    full = np.arange(n_units)
    a0, b0 = stat_a(full), stat_b(full)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_units, n_units)
        d[i] = stat_b(idx) - stat_a(idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(a=float(a0), b=float(b0), diff=float(b0 - a0), ci=[float(lo), float(hi)])
