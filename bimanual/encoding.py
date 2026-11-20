"""Linear encoding models with cross-validated (less biased) tuning statistics.

Reproduces
----------
The encoding-model analyses behind Fig 2b-c, plus the laterality dimension
used in Fig 2 and Fig 3:

* Eq. 1-2 : f = E @ [1, d_rx, d_ry, d_lx, d_ly]^T, with E of shape (N x 5),
  fit by five-fold cross-validated OLS ("cvOLS") on 20-ms bins.
* Tuning strength: magnitude of a coefficient column across channels,
  ||E[:, j]||. The x and y magnitudes of each hand are averaged.
* Tuning correlation: Pearson correlation between corresponding coefficient
  columns of two models, for example uni-R x vs bi-R x.
* Eq. 4 : the laterality regressor c_lat (+1 = uni-R, -1 = uni-L), its
  coefficient column b_lat, and removal of that axis from the neural data.

Why cross-validation?
---------------------
A coefficient vector estimated from noisy data is b_hat = b + e. Its squared
norm is biased upward:

    E[ ||b_hat||^2 ] = ||b||^2 + E[ ||e||^2 ]

so a channel with no tuning still looks "tuned". Noise also pulls naive
correlations toward 0. Two estimates fitted on DISJOINT data have independent
noise, so

    E[ <b_hat_1, b_hat_2> ] = <b, b> = ||b||^2

is unbiased. cvOLS uses this: for each fold k, fit one model on fold k alone
("test") and one on the other folds ("train"). The average of
<train_k, test_k> over folds is an unbiased estimate of the squared norm. This
follows the approach of Willett et al. (2020, Cell, "Hand knob area of
premotor cortex represents the whole body in a compositional way").
DEVIATION: we reimplemented it from that description, not from their code.

* Two models fitted on DIFFERENT trials (e.g. the unimanual vs the bimanual
  context) already have independent noise, so their cross product needs no
  folds; only each model's own squared norm (the denominator) is
  cross-validated.

Inputs
------
This module is deliberately generic. It takes a design matrix ``X`` (samples x
regressors) and neural data ``Y`` (samples x channels). The phase scripts
build ``X`` from the task variables, using the per-hand position errors of
Eq. 2.

Samples are 20-ms bins, and bins from the same trial are not independent.
Folds therefore split by TRIAL through the ``groups`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- folds
def group_folds(groups: np.ndarray, n_folds: int = 5, seed: int | None = 0) -> list[np.ndarray]:
    """Split samples into ``n_folds`` folds, keeping each group (trial) whole.

    Args:
        groups: shape (n_samples,), trial ID of each sample (20-ms bin).
        n_folds: number of folds (the paper uses 5).
        seed: RNG seed for shuffling the trials before assigning folds.

    Returns:
        List of ``n_folds`` boolean masks over samples (True = in that fold).
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size < n_folds:
        raise ValueError(f"need at least {n_folds} trials, got {uniq.size}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    fold_of_group = {g: i % n_folds for i, g in enumerate(order)}
    fold_id = np.array([fold_of_group[g] for g in groups])
    return [fold_id == k for k in range(n_folds)]


# ----------------------------------------------------------------------------- OLS
def ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ordinary least squares coefficients, shape (n_regressors, n_channels).

    Uses the minimum-norm least-squares solution, so an all-zero regressor
    column (e.g. LH velocity in a uni-R-only fit) gets coefficient 0 rather
    than raising an error.
    """
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return B


@dataclass
class CVFit:
    """Result of :func:`cv_ols`. Coefficient arrays are indexed [regressor, channel].

    Attributes:
        full:  (p, N) fit on all samples (point estimate).
        train: (K, p, N) fit on all folds except k.
        test:  (K, p, N) fit on fold k only.
        names: optional regressor names, for readability.
    """

    full: np.ndarray
    train: np.ndarray
    test: np.ndarray
    names: tuple[str, ...] = ()

    def col(self, j: int | str) -> int:
        """Column index from an int or a regressor name."""
        return self.names.index(j) if isinstance(j, str) else j


def cv_ols(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int | None = 0,
    names: tuple[str, ...] = (),
) -> CVFit:
    """Five-fold cross-validated OLS (cvOLS), as in Methods "Encoding model".

    Args:
        X: (n_samples, p) design matrix; include a column of ones for the
            baseline (b0 in Eq. 1).
        Y: (n_samples, N) firing rates (block-mean removed, per Methods).
        groups: (n_samples,) trial IDs, so folds don't split a trial.
        n_folds: number of folds K.
        seed: fold-assignment seed.
        names: optional names for the p regressors.
    """
    folds = group_folds(groups, n_folds, seed)
    train = np.stack([ols(X[~m], Y[~m]) for m in folds])
    test = np.stack([ols(X[m], Y[m]) for m in folds])
    return CVFit(full=ols(X, Y), train=train, test=test, names=tuple(names))


# ----------------------------------------------------------------------------- statistics
def _center(v: np.ndarray, center: bool) -> np.ndarray:
    """Subtract the mean across channels (last axis) if ``center``. Pearson needs it."""
    return v - v.mean(axis=-1, keepdims=True) if center else v


def cv_dot(
    fit_a: CVFit,
    col_a: int | str,
    fit_b: CVFit | None = None,
    col_b: int | str | None = None,
    center: bool = False,
) -> float:
    """Unbiased estimate of the inner product <b_a, b_b> of two coefficient columns.

    * Same model (``fit_b`` is None): average over folds of the cross-fold
      products <train_k[a], test_k[b]> and <train_k[b], test_k[a]>. The noise in
      train and test is independent, so the expectation equals the true <b_a, b_b>.
      With col_a == col_b this is the unbiased squared norm.
    * Different models (``fit_b`` given): the models must come from disjoint
      trials. Their full fits have independent noise, so <full_a, full_b> is
      already unbiased.

    Args:
        center: subtract the across-channel mean first (needed for Pearson r).
    """
    if fit_b is None:
        fit_b = fit_a
        col_b = col_a if col_b is None else col_b
        ia, ib = fit_a.col(col_a), fit_a.col(col_b)
        tr, te = _center(fit_a.train, center), _center(fit_a.test, center)
        prods = np.einsum("kn,kn->k", tr[:, ia], te[:, ib]) + np.einsum(
            "kn,kn->k", tr[:, ib], te[:, ia]
        )
        return float(prods.mean() / 2.0)
    ia, ib = fit_a.col(col_a), fit_b.col(col_b if col_b is not None else col_a)
    return float(_center(fit_a.full[ia], center) @ _center(fit_b.full[ib], center))


def cv_sq_norm(fit: CVFit, col: int | str, center: bool = False) -> float:
    """Unbiased squared norm ||b||^2 of one coefficient column. Can be < 0 by chance."""
    return cv_dot(fit, col, center=center)


def cv_magnitude(fit: CVFit, col: int | str, center: bool = False) -> float:
    """Signed square root of the unbiased squared norm: the tuning strength of one column.

    The value is negative when the unbiased estimate is negative, which only
    happens for (near-)untuned data. The sign is kept so averages over
    repeats stay unbiased around 0.
    """
    s = cv_sq_norm(fit, col, center)
    return float(np.sign(s) * np.sqrt(abs(s)))


def tuning_strength(fit: CVFit, x_col: int | str, y_col: int | str) -> float:
    """Tuning strength of one hand: mean of its x and y column magnitudes (Methods)."""
    return 0.5 * (cv_magnitude(fit, x_col) + cv_magnitude(fit, y_col))


def cv_correlation(
    fit_a: CVFit,
    col_a: int | str,
    fit_b: CVFit | None = None,
    col_b: int | str | None = None,
) -> float:
    """Cross-validated Pearson correlation between two coefficient columns.

    r = <a, b> / sqrt(||a||^2 ||b||^2), with every term estimated without bias
    on channel-centered vectors.

    Unlike a naive correlation, this is not pulled toward 0 by estimation
    noise. Because it is a ratio of noisy estimates, it can slightly exceed
    +/-1 for very well-correlated vectors. It is NaN if a denominator
    estimate is <= 0, meaning no detectable tuning.
    """
    num = cv_dot(fit_a, col_a, fit_b, col_b, center=True)
    da = cv_sq_norm(fit_a, col_a, center=True)
    if fit_b is None:
        db = cv_sq_norm(fit_a, col_b if col_b is not None else col_a, center=True)
    else:
        db = cv_sq_norm(fit_b, col_b if col_b is not None else col_a, center=True)
    if da <= 0 or db <= 0:
        return np.nan
    return float(num / np.sqrt(da * db))


# ----------------------------------------------------------------------------- laterality
def laterality_axis(X: np.ndarray, Y: np.ndarray, lat_col: int) -> np.ndarray:
    """Unit-norm laterality direction in neural space (Eq. 4).

    Fit the encoding model with the laterality regressor c_lat (+1 for uni-R
    trials, -1 for uni-L trials) and return the normalized coefficient column
    b_lat / ||b_lat||, shape (N,).

    Args:
        X: (n_samples, p) design matrix that includes the c_lat column.
        Y: (n_samples, N) firing rates.
        lat_col: index of c_lat in ``X``.
    """
    b_lat = ols(X, Y)[lat_col]
    return b_lat / np.linalg.norm(b_lat)


def remove_axis(Y: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Project out the unit direction ``u`` from every sample: Y - (Y u) u^T.

    This removes the laterality dimension (Methods, "Laterality removal"). A
    refit of Eq. 4 on the result gives b_lat ~ 0, which the paper uses as its
    check.
    """
    u = u / np.linalg.norm(u)
    return Y - np.outer(Y @ u, u)
