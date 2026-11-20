"""Load a session file and apply the standard feature preprocessing.

Reproduces
----------
The data loading and preprocessing used throughout the paper (Methods,
"Neural features"):

* 20-ms bins of threshold crossings (``tx``, 192 channels).
* Block-wise mean removal: the paper removes each block's mean firing rate to
  correct for slow drift between blocks (also tuningAnalysis.m:13-21).
* Conversion to spikes/s: counts per 20-ms bin x 50 (tuningAnalysis.m:23).
* Optional per-block z-scoring for the decoders ("block-wise mean removed,
  Z-scored").

Conventions
-----------
* Arrays are TIME-MAJOR: ``tx`` is (n_bins, n_channels). The .mat files store
  (n_channels, n_bins); the loader transposes.
* All bin indices on a :class:`SessionData` are **0-based**. The files store
  1-based MATLAB indices in ``goCue``; this module is the ONLY place that
  converts them.

Usage
-----
    from bimanual.io import load_session, remove_block_means
    s = load_session(1750)                    # or load_session("06_02_2021")
    rates = remove_block_means(s.tx, s.block) * 50   # spikes/s, block-mean removed
    s.trials                                  # pandas DataFrame, one row per trial
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.io as sio

from bimanual import sessions

BIN_S = 0.020
HZ_PER_COUNT = 1.0 / BIN_S  # 50: counts per 20-ms bin -> spikes/s


@dataclass
class SessionData:
    """One experiment day, as loaded from ``Data/t5_<date>.mat``.

    Attributes:
        info:  catalogue entry (trial day, figures, block labels).
        tx:    (n_bins, 192) float64 threshold-crossing counts per 20-ms bin.
        block: (n_bins,) block number of each bin.
        Cp,Tp: (4, n_bins) cursor / target positions [R x, R y, L x, L y]
               (kept in the file's row layout so they match the readme).
        trials: DataFrame with one row per trial and columns:
               go (0-based go-cue bin), start (first delay bin),
               move_end (exclusive end of the move period), move_type (1/2/3),
               delay_bins, move_bins, block, kind ("OL"/"CLR"/"CL"),
               decoder, gain, variant, uni_only.
    """

    info: sessions.Session
    tx: np.ndarray
    block: np.ndarray
    Cp: np.ndarray
    Tp: np.ndarray
    trials: pd.DataFrame

    @property
    def go(self) -> np.ndarray:
        """0-based go-cue bin of every trial."""
        return self.trials["go"].to_numpy()


def load_session(day_or_date: int | str) -> SessionData:
    """Load one session by trial day (e.g. 1750) or date string ("06_02_2021")."""
    info = sessions.get(day_or_date)
    d = sio.loadmat(info.path, squeeze_me=True)
    go = d["goCue"].astype(int) - 1  # 1-based (MATLAB) -> 0-based
    delay = d["durationBinnedDelay"].astype(int)
    move = d["durationBinnedMove"].astype(int)
    block = d["blockNum"].astype(int)
    trial_block = block[go]

    rows = []
    for g, dl, mv, b, mt in zip(go, delay, move, trial_block, d["moveType"].astype(int)):
        lab = info.blocks.get(int(b))
        rows.append(dict(
            go=g, start=g - dl, move_end=g + mv, move_type=mt, delay_bins=dl, move_bins=mv,
            block=int(b),
            kind=lab.kind if lab else None, decoder=lab.decoder if lab else None,
            gain=lab.gain if lab else None, variant=lab.variant if lab else None,
            uni_only=lab.uni_only if lab else None,
        ))
    return SessionData(
        info=info,
        tx=d["tx"].T.astype(np.float64),
        block=block,
        Cp=np.asarray(d["Cp"], float),
        Tp=np.asarray(d["Tp"], float),
        trials=pd.DataFrame(rows),
    )


def remove_block_means(X: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Subtract each block's mean (over all its bins) from every channel.

    Port of tuningAnalysis.m:13-21, where the mean is taken over ALL bins of
    the block, including the inter-trial periods. NaNs are ignored in the mean.

    Args:
        X: (n_bins, n_channels).
        block: (n_bins,) block number of each bin.
    """
    out = np.empty_like(X, dtype=np.float64)
    for b in np.unique(block):
        m = block == b
        out[m] = X[m] - np.nanmean(X[m], axis=0, keepdims=True)
    return out


def zscore_by_block(X: np.ndarray, block: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Block-wise mean removal, then division by each block's per-channel SD.

    Used for decoder inputs ("block-wise mean removed, Z-scored"). Channels
    with zero variance in a block are left at 0, not divided by 0.
    """
    out = remove_block_means(X, block)
    for b in np.unique(block):
        m = block == b
        sd = np.nanstd(out[m], axis=0, keepdims=True)
        out[m] = out[m] / np.where(sd > eps, sd, 1.0)
    return out
