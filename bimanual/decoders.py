"""Single-bin velocity decoders: ridge regression and a one-layer feed-forward net.

Reproduces
----------
Fig 3 and Supp Fig 2 (Methods, "Single-bin decoders"):

* Input: the firing-rate vector of ONE 20-ms bin (192 channels, z-scored).
* Output: x/y velocity of both cursors (4 values).
* Ridge regression: the ridge parameter is "swept to maximize Pearson
  correlation" (paper: scikit-learn RidgeCV).
* Feed-forward network (FFN): one dense hidden layer of 512 units (paper:
  TensorFlow v1). Here: PyTorch, ReLU, Adam, MSE loss, early stopping on a
  held-out 10% of the training trials. DEVIATION (register #25): the paper
  gives no activation, optimizer or training details, so these are standard
  choices.
* Five-fold cross-validation, "balanced by trial type": folds split by TRIAL
  and stratified by movement type (``stratified_trial_folds``).

Why the FFN can beat ridge
--------------------------
When left- and right-hand tuning vectors are correlated, a linear decoder
can't tell "right hand moves up" from "left hand moves up". A nonlinear
decoder can gate its outputs using the laterality dimension: a population
signal that says WHICH hand is moving. Removing that dimension (Fig 3,
``encoding.remove_axis``) should therefore hurt the FFN but not ridge.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RIDGE_ALPHAS = np.logspace(-1, 5, 13)


# ----------------------------------------------------------------------------- folds
def stratified_trial_folds(trial_type: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index (0..n_folds-1) per TRIAL, balanced across trial types.

    Within each trial type, the trials are shuffled and dealt round-robin into
    folds, so every fold has nearly the same mix of movement types.
    """
    rng = np.random.default_rng(seed)
    fold = np.empty(len(trial_type), int)
    for t in np.unique(trial_type):
        idx = rng.permutation(np.where(trial_type == t)[0])
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def pearson_per_dim(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Pearson r between true and decoded values, separately for each output column."""
    yc, hc = y - y.mean(0), yhat - yhat.mean(0)
    return (yc * hc).sum(0) / np.sqrt((yc**2).sum(0) * (hc**2).sum(0))


# ----------------------------------------------------------------------------- ridge
def ridge_fit_predict(Xtr, Ytr, Xte, alpha):
    """Closed-form ridge with an unpenalized intercept; returns predictions for Xte."""
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A = Xtr - mx
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Ytr - my))
    return (Xte - mx) @ W + my


# ----------------------------------------------------------------------------- FFN
class FFN(nn.Module):
    """Input -> Dense(512) -> ReLU -> Dense(4)."""

    def __init__(self, n_in: int, n_out: int = 4, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def ffn_fit_predict(Xtr, Ytr, Xte, groups_tr, seed=0, max_epochs=60, batch=512, lr=1e-3,
                    weight_decay=1e-4, patience=6, val_frac=0.1):
    """Train an FFN with early stopping on held-out TRAINING trials; predict Xte.

    Targets are standardized per output for training, and predictions are
    mapped back to the original units.

    Args:
        Xtr, Ytr: (n_bins, n_in), (n_bins, 4) training bins.
        Xte: (n_bins_test, n_in).
        groups_tr: (n_bins,) trial ID of each training bin. The early-stopping
            set is a random 10% of the training TRIALS, never the test data.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    trials = np.unique(groups_tr)
    val_trials = rng.choice(trials, max(1, int(round(val_frac * len(trials)))), replace=False)
    vm = np.isin(groups_tr, val_trials)

    ym, ys = Ytr[~vm].mean(0), Ytr[~vm].std(0) + 1e-8
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    Xa, Ya = f(Xtr[~vm]), f((Ytr[~vm] - ym) / ys)
    Xv, Yv = f(Xtr[vm]), f((Ytr[vm] - ym) / ys)

    model = FFN(Xtr.shape[1], Ytr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xa), generator=g)
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch].to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xa[b]), Ya[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xv), Yv).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(f(Xte)).cpu().numpy()
    return pred * ys + ym


# ----------------------------------------------------------------------------- CV driver
def cross_validate(X, Y, groups, folds_by_trial, decoder: str, alpha=None, seed=0, transform=None):
    """Out-of-fold predictions for every bin.

    Args:
        X: (n_bins, n_in); Y: (n_bins, 4); groups: (n_bins,) trial index into
            ``folds_by_trial``.
        folds_by_trial: (n_trials,) fold of each trial.
        decoder: "ridge" (needs ``alpha``) or "ffn".
        transform: optional ``fn(Xtr, Ytr, groups_tr) -> (fn_apply)``, fitted
            on the TRAINING fold only and applied to both train and test
            (used for laterality removal, so no test data leaks into it).

    Returns:
        (n_bins, 4) out-of-fold predictions.
    """
    pred = np.zeros_like(Y, dtype=float)
    fold_of_bin = folds_by_trial[groups]
    for k in np.unique(folds_by_trial):
        te = fold_of_bin == k
        Xtr, Xte = X[~te], X[te]
        if transform is not None:
            apply = transform(Xtr, Y[~te], groups[~te])
            Xtr, Xte = apply(Xtr), apply(Xte)
        if decoder == "ridge":
            pred[te] = ridge_fit_predict(Xtr, Y[~te], Xte, alpha)
        elif decoder == "ffn":
            pred[te] = ffn_fit_predict(Xtr, Y[~te], Xte, groups[~te], seed=seed + int(k))
        else:
            raise ValueError(decoder)
    return pred
