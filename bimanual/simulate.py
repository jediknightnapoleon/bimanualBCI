"""Synthetic neural populations for the Fig 3 simulations, and the fSNR metric (Supp Fig 4).

Reproduces
----------
Methods, "Simulations", and Supp Fig 4:

* 192 synthetic neurons follow the laterality encoding model (Eq. 4):
      f = b0 + b_Rx d_Rx + b_Ry d_Ry + b_Lx d_Lx + b_Ly d_Ly + b_lat c_lat
  with c_lat = +1 for unimanual right and -1 for unimanual left. Tuning
  coefficients are standard-normal vectors scaled to the magnitudes measured
  in real data.
* The LH tuning columns are given a chosen correlation rho with the RH
  columns (swept). The laterality column b_lat gets a chosen size (swept).
* Each 20-ms bin gets independent Gaussian noise of SD sigma, the same for
  all channels (Eq. 3). sigma is tuned until the simulated single-bin fSNR
  matches the real data. Negative rates are clipped to 0.
* Unimanual trials only: about 2000 uni-R and 2000 uni-L, with targets in
  the 8 wedge directions.

fSNR (functional signal-to-noise ratio, Methods / Supp Fig 4)
--------------------------------------------------------------
Fit decoded = D @ true + offset + eps, with a 2x2 matrix D. Then
fSNR = mean(D11, D22) / std(eps): the decoder's gain along the true
direction divided by its noise. It measures how separable the directions
are, whatever the units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- fSNR
def fsnr(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Functional SNR of a 2-D decoder output.

    Args:
        y_true: (n, 2) true x/y targets.
        y_hat: (n, 2) decoded x/y.

    Returns:
        0.5 * (D[0,0] + D[1,1]) / std(eps), from the least-squares fit
        y_hat = y_true @ D.T + c + eps. std(eps) is pooled over both dims.
    """
    A = np.c_[y_true, np.ones(len(y_true))]
    coef, *_ = np.linalg.lstsq(A, y_hat, rcond=None)  # (3, 2)
    D = coef[:2].T
    eps = y_hat - A @ coef
    return float(0.5 * (D[0, 0] + D[1, 1]) / eps.std())


# ----------------------------------------------------------------------------- population
@dataclass
class Population:
    """Tuning coefficients of a synthetic population (Eq. 4). All vectors are (N,), in Hz."""

    b0: np.ndarray
    bRx: np.ndarray
    bRy: np.ndarray
    bLx: np.ndarray
    bLy: np.ndarray
    blat: np.ndarray


def _unit(v):
    return v / np.linalg.norm(v)


def make_population(b0: np.ndarray, mags: dict, rho: float, lat_scale: float, rng) -> Population:
    """Draw a population with a given RH-LH tuning correlation and laterality size.

    Args:
        b0: (N,) baseline firing rates (Hz), e.g. the real channel means.
        mags: real column magnitudes (Hz per unit regressor) with keys
            "Rx", "Ry", "Lx", "Ly", "lat".
        rho: target correlation between the LH and RH columns (applied to x
            and to y separately).
        lat_scale: multiplier on the real laterality magnitude (0 = no
            laterality signal).
        rng: numpy Generator.
    """
    N = len(b0)

    def pair(mag_r, mag_l):
        zr, zo = rng.standard_normal(N), rng.standard_normal(N)
        zr, zo = zr - zr.mean(), zo - zo.mean()
        zo = zo - (zo @ zr) / (zr @ zr) * zr  # exactly orthogonal, so the realized correlation = rho
        r = _unit(zr)
        l = rho * r + np.sqrt(max(1 - rho**2, 0)) * _unit(zo)
        return mag_r * r, mag_l * _unit(l)

    bRx, bLx = pair(mags["Rx"], mags["Lx"])
    bRy, bLy = pair(mags["Ry"], mags["Ly"])
    blat = lat_scale * mags["lat"] * _unit(rng.standard_normal(N))
    return Population(b0, bRx, bRy, bLx, bLy, blat)


def simulate_trials(pop: Population, n_per_hand: int, bins_per_trial: int, sigma: float, rng):
    """Unimanual trials with 8 wedge directions and i.i.d. Gaussian bin noise.

    Returns:
        X: (n_trials * bins, N) rates in Hz, clipped at 0.
        Y: (n_trials * bins, 4) unit-direction targets [Rx, Ry, Lx, Ly]; 0 for
           the idle hand.
        g: (n_trials * bins,) trial index.
        hand: (n_trials,) 1 = uni-R, 2 = uni-L.
    """
    n = 2 * n_per_hand
    hand = np.repeat([1, 2], n_per_hand)
    ang = np.radians(45.0 * rng.integers(0, 8, n))
    d = np.c_[np.cos(ang), np.sin(ang)]
    Ytr = np.zeros((n, 4))
    Ytr[hand == 1, 0:2] = d[hand == 1]
    Ytr[hand == 2, 2:4] = d[hand == 2]
    c_lat = np.where(hand == 1, 1.0, -1.0)
    mean = (pop.b0[None] + Ytr[:, [0]] * pop.bRx + Ytr[:, [1]] * pop.bRy + Ytr[:, [2]] * pop.bLx
            + Ytr[:, [3]] * pop.bLy + c_lat[:, None] * pop.blat)  # (n, N)
    X = np.repeat(mean, bins_per_trial, axis=0)
    X = np.maximum(X + rng.normal(0, sigma, X.shape), 0.0)  # Eq. 3 noise, then clip negative rates
    return X, np.repeat(Ytr, bins_per_trial, axis=0), np.repeat(np.arange(n), bins_per_trial), hand
