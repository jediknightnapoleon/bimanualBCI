"""Closed-loop proxy: score an offline decoder on the neural data of recorded closed-loop blocks.

Why
---
The paper's Fig 4 point is that offline r on open-loop data does not predict online
performance. The raw-trained and augmented RNNs differ by only about 0.03 r offline (2–8 s
clips), yet online the augmented RNN acquired 59–86% of targets vs 7–16% for the raw one
(re-scored in Phase 5). Live control can't be re-run, so this module adds a cheap
"closed-loop-like" test. It decodes the neural activity T5 produced DURING closed-loop control
(day 1869 has 12 CL blocks) and compares the output with his inferred intention. That data has
none of the rigid open-loop timing (delay -> go -> fixed-length movement), so a decoder that
learned the open-loop timing should do worse here.

Intention target (register #35, ReFIT-style, as in Gilja et al. 2012)
--------------------------------------------------------------------
For each CL trial and each ACTIVE hand: the unit vector from cursor to target, per bin, from
200 ms after the go cue (the reaction time used by the training targets) until the pursuit
ends. Zero everywhere else: delay, inter-trial gaps, the idle hand, and the final 500-ms dwell
of successful trials.

ASSUMPTION: the target radius isn't stored in the data (register #7), so "the pursuit ends" is
read from the log. Successful trials (``move_bins`` < 499) end with the 500-ms dwell, so the
target is 0 for their last 25 bins. Timed-out trials are pursuit until ``move_end``. Earlier
brief target entries are not detected; they are rare and count as pursuit.

Metrics, per online-decoder variant (blocks driven by the "raw" vs the "augmented" RNN)
---------------------------------------------------------------------------------------
* ``r``: Pearson r per output dim [R x, R y, L x, L y] over all bins of the blocks; ``r_R``
  and ``r_L`` are the means over x and y. Each block is decoded from a zero state.
* ``ang_err_deg``: mean absolute angle between the active hand's decoded velocity, averaged over
  300–500 ms after go, and the cursor->target direction at go. This is the decoded counterpart
  of the Fig 5 angular error.
* ``sustain``: a direct check for the "pulse, then stop" failure of Fig 4e. It is the mean
  decoded velocity projected on the intention direction 1–3 s after go, divided by that over
  0.3–1 s. Only trials still in pursuit at 3 s are used. 1 = sustained; well below 1 = it
  fades.
* ``jitter``: mean decoded speed of the idle hand / mean decoded speed of the active hand,
  over pursuit bins (lower = steadier idle cursor, as in Fig 3c).

This is a proxy, not online performance. The neural activity was shaped by feedback from the
decoder that ran online, which is why results are reported separately per variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bimanual.conditions import BIMANUAL, UNI_LEFT, UNI_RIGHT
from bimanual.decoders import pearson_per_dim

RT_BINS = 10  # 200-ms reaction time (as in bimanual.targets)
DWELL_BINS = 25  # 500-ms dwell that ends a successful trial
TIMEOUT_BINS = 499  # logged move_bins of a 10-s timeout
EARLY = (15, 50)  # 0.3-1.0 s after go (bins), "sustain" denominator
LATE = (50, 150)  # 1.0-3.0 s after go, "sustain" numerator
ANG = (15, 25)  # 300-500 ms after go (bins), angular-error window

HAND_COLS = {"R": slice(0, 2), "L": slice(2, 4)}


def _active(move_type: int) -> tuple[bool, bool]:
    return move_type in (UNI_RIGHT, BIMANUAL), move_type in (UNI_LEFT, BIMANUAL)


@dataclass
class ProxyData:
    """Intention targets for the CL blocks of one session.

    Attributes:
        V: (n_bins, 4) intention target for the whole session (0 outside CL trials).
        pursuit: (n_bins, 2) bool, whether the R / L hand is in pursuit (the target is nonzero by
            construction, unless the cursor is within 1 unit of its target).
        segments: {block: (a, b)} bin range of each CL block (contiguous).
        variant: {block: "raw" | "augmented" | ...}, the decoder that ran online in that block.
        trials: the CL trials (DataFrame from ``io.load_session``) plus ``pursuit_end``.
    """

    V: np.ndarray
    pursuit: np.ndarray
    segments: dict
    variant: dict
    trials: pd.DataFrame


def build(s, rt_bins: int = RT_BINS, dwell_bins: int = DWELL_BINS) -> ProxyData:
    """Intention targets for every closed-loop (kind == "CL") trial of session ``s``."""
    t = s.trials
    cl = t[t.kind == "CL"].copy()
    n = s.Tp.shape[1]
    V = np.zeros((n, 4), np.float32)
    pursuit = np.zeros((n, 2), bool)
    ends = []
    for r in cl.itertuples():
        a = int(r.go) + rt_bins
        b = int(r.move_end) - (dwell_bins if int(r.move_bins) < TIMEOUT_BINS else 0)
        ends.append(b)
        if b <= a:
            continue
        for j, (hand, on) in enumerate(zip("RL", _active(int(r.move_type)))):
            if not on:
                continue
            cols = HAND_COLS[hand]
            err = s.Tp[cols, a:b] - s.Cp[cols, a:b]  # (2, b - a)
            norm = np.hypot(*err)
            ok = norm >= 1.0
            V[a:b, cols] = np.where(ok, err / np.maximum(norm, 1e-9), 0.0).T
            pursuit[a:b, j] = ok
    cl["pursuit_end"] = ends
    segs, var = {}, {}
    for blk in sorted(cl.block.unique()):
        idx = np.where(s.block == blk)[0]
        assert np.all(np.diff(idx) == 1), "blocks are contiguous in time"
        segs[int(blk)] = (int(idx[0]), int(idx[-1]) + 1)
        var[int(blk)] = str(cl.loc[cl.block == blk, "variant"].iloc[0])
    return ProxyData(V, pursuit, segs, var, cl)


def decode_blocks(decode_fn, X: np.ndarray, pdata: ProxyData) -> np.ndarray:
    """Decode each CL block separately (zero initial state) into a session-length array.

    Args:
        decode_fn: callable(X_block (T, N)) -> decoded velocity (T, 4).
        X: (n_bins, N) z-scored inputs of the whole session.
    """
    out = np.zeros((len(X), 4), np.float32)
    for a, b in pdata.segments.values():
        out[a:b] = decode_fn(X[a:b])
    return out


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Absolute angle (degrees) between 2-D vectors u and v."""
    d = np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]))
    return float(abs((d + 180.0) % 360.0 - 180.0))


def score(vhat: np.ndarray, pdata: ProxyData) -> dict:
    """Proxy metrics per online-decoder variant (see the module docstring), plus "all".

    Args:
        vhat: (n_bins, 4) decoded velocity from ``decode_blocks``.
    """
    out = {}
    variants = sorted(set(pdata.variant.values()))
    for var in variants + ["all"]:
        blocks = [b for b, v in pdata.variant.items() if var in (v, "all")]
        idx = np.concatenate([np.arange(*pdata.segments[b]) for b in blocks])
        r = pearson_per_dim(pdata.V[idx], vhat[idx])
        tr = pdata.trials[pdata.trials.block.isin(blocks)]
        angs, sus_num, sus_den, idle_sp, act_sp = [], [], [], [], []
        for row in tr.itertuples():
            g, pe = int(row.go), int(row.pursuit_end)
            for j, (hand, on) in enumerate(zip("RL", _active(int(row.move_type)))):
                cols = HAND_COLS[hand]
                other = HAND_COLS["L" if hand == "R" else "R"]
                if not on:
                    continue
                d0 = pdata.V[g + RT_BINS, cols] if g + RT_BINS < pe else np.zeros(2)
                if np.hypot(*d0) > 0:
                    angs.append(_angle_deg(d0, vhat[g + ANG[0]:g + ANG[1], cols].mean(0)))
                if pe >= g + LATE[1]:  # still in pursuit 3 s after go
                    proj = (vhat[g:pe, cols] * pdata.V[g:pe, cols]).sum(1)
                    sus_num.append(proj[LATE[0]:LATE[1]].mean()); sus_den.append(proj[EARLY[0]:EARLY[1]].mean())
                pur = pdata.pursuit[g:pe, j]
                if pur.any() and int(row.move_type) != BIMANUAL:
                    act_sp.append(np.hypot(*vhat[g:pe][pur][:, cols].T).mean())
                    idle_sp.append(np.hypot(*vhat[g:pe][pur][:, other].T).mean())
        den = float(np.mean(sus_den)) if sus_den else np.nan
        out[var] = dict(
            r=r.tolist(), r_R=float(np.mean(r[:2])), r_L=float(np.mean(r[2:])), r_mean=float(np.mean(r)),
            ang_err_deg=float(np.mean(angs)) if angs else np.nan,
            sustain=float(np.mean(sus_num) / den) if sus_den and den > 0 else np.nan,
            n_sustain_trials=len(sus_num),
            jitter=float(np.mean(idle_sp) / np.mean(act_sp)) if act_sp else np.nan,
            n_blocks=len(blocks), n_trials=int(len(tr)),
        )
    return out


def score_model(model, X: np.ndarray, pdata: ProxyData, decode=None) -> dict:
    """Convenience wrapper: decode the CL blocks with an RNN ``model`` and score them."""
    if decode is None:
        from bimanual.rnn import decode
    vhat = decode_blocks(lambda x: decode(model, x)[0], X, pdata)
    return score(vhat, pdata)
