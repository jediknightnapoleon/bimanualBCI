"""Best-n channel selection with a fast linear surrogate decoder (Phase 10, register #43).

Question
--------
Which n channels of one array decode best? The full search is combinatorial
(C(96, 8) ~ 1e11 subsets), so we use a cheap surrogate decoder and two search strategies:

* **Surrogate:** causal linear (ridge) decoder on a multi-timescale history of each channel: its
  current z-scored bin plus causal exponential smoothings with time constants ``TAUS`` (2-75 bins =
  40 ms-1.5 s), 6 features per channel; targets are the RNN's 4 velocity dims.
  DEVIATION (#43): the plan said "10 raw lags (200 ms)". On day 1869 (array A, all 96 channels) that
  surrogate reaches only val r 0.35 vs about 0.88-0.93 for the LSTMs; the multi-timescale basis
  reaches about 0.61 with fewer features (6 vs 10 per channel), so it is a much closer stand-in for
  a recurrent decoder that integrates seconds of history. Raw lags remain available
  (``spec=("lags", n)``) and are used in the unit tests.
  Fitted on the training blocks, scored as the mean over the 4 dims of the Pearson r
  on the validation block (the same number used to select the RNNs, ``val.r_mean``).
* **Greedy forward selection + swap refinement** (main method): add the channel that raises the
  validation r most, then, for a fixed n, try every (selected, unselected) swap and accept the best
  improving one per pass until none improves. Greedy has no optimality guarantee (near-optimal only
  when the score has diminishing returns; Das & Kempe 2011); the swaps measure how far it is from a
  local optimum.
* **Group lasso** (joint benchmark): the same lagged linear model with one group per channel
  (its LAGS x 4 weights) and penalty lambda * sum_c ||W_c||_F, solved by FISTA along a lambda path;
  the lambda giving exactly n active channels is found by bisection, then plain ridge is refitted on
  those channels and scored like the other methods.

Everything works on covariance matrices precomputed once per channel pool (train: Cxx, Cxy; val:
Vxx, Vxy, var(y)), so a subset fit is a small (k * LAGS)-dim solve and its validation r needs no
prediction pass:  r_d = w_d' Vxy_d / sqrt(w_d' Vxx w_d * var(y_d)).

Test-block guard: every function that reads data goes through ``guard_ranges``, which raises if a
bin range touches a block outside ``SEL_BLOCKS`` (blocks 1-6 of day 1869). Block 7 is never read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAUS = (2, 5, 12, 30, 75)  # ASSUMPTION (#43): causal smoothing time constants in 20-ms bins
SPEC = ("exp", TAUS)  # default feature spec: raw bin + one causal exponential smoothing per tau
LAGS = 10  # length of the raw-lag alternative (``spec=("lags", LAGS)``), kept for comparison/tests
SEL_BLOCKS = (1, 2, 3, 4, 5, 6)  # day 1869: train blocks 1-5 + validation block 6; NEVER test block 7
ALPHA_GRID = np.logspace(-5, 2, 15)  # ridge alpha in covariance units (features are z-scored, unit scale)
ARRAYS = {"A": np.arange(0, 96), "B": np.arange(96, 192)}  # register #36


# ----------------------------------------------------------------------------- data access
def guard_ranges(block: np.ndarray, ranges, allowed=SEL_BLOCKS):
    """Raise ValueError if any [a, b) bin range contains a block outside ``allowed`` (the test guard)."""
    allowed = set(int(x) for x in allowed)
    for a, b in ranges:
        found = set(np.unique(block[a:b]).astype(int).tolist())
        if not found <= allowed:
            raise ValueError(f"bins [{a}, {b}) touch blocks {sorted(found - allowed)}, outside the "
                             f"selection blocks {sorted(allowed)} (the test block must never be used)")


def lagged(x: np.ndarray, lags: int = LAGS) -> np.ndarray:
    """Causal lag matrix of one contiguous block.

    x: (T, k). Returns (T, k * lags); column ``j * lags + a`` holds channel j delayed by a bins.
    Bins before the block start are zero (= the channel mean in z-units), so no lag crosses a
    block boundary.
    """
    T, k = x.shape
    out = np.zeros((T, k, lags), np.float64)
    for a in range(lags):
        out[a:, :, a] = x[: T - a] if a else x
    return out.reshape(T, k * lags)


def exp_basis(x: np.ndarray, taus=TAUS) -> np.ndarray:
    """Multi-timescale causal history of one contiguous block.

    x: (T, k). Returns (T, k * (1 + len(taus))); for channel j, column ``j * nf`` is the raw bin and
    columns ``j * nf + 1 ...`` are causal exponential smoothings y_t = a y_{t-1} + (1 - a) x_t with
    a = exp(-1 / tau). Filters start from 0 (the channel mean in z-units) at the block start, so no
    history crosses a block boundary.
    """
    from scipy.signal import lfilter
    T, k = x.shape
    feats = [x]
    for tau in taus:
        a = np.exp(-1.0 / tau)
        feats.append(lfilter([1 - a], [1, -a], x, axis=0))
    return np.stack(feats, 2).reshape(T, k * (1 + len(taus)))


def n_features(spec) -> int:
    kind, par = spec
    return int(par) if kind == "lags" else 1 + len(par)


def featurize(x: np.ndarray, spec=SPEC) -> np.ndarray:
    """Per-channel causal features, channel-major columns (channel j owns columns j*nf ... j*nf+nf-1)."""
    kind, par = spec
    if kind == "lags":
        return lagged(x, int(par))
    if kind == "exp":
        return exp_basis(x, par)
    raise ValueError(f"unknown feature spec {spec}")


def _moments(z, y, ranges, channels, spec):
    """Centered covariances over the given block ranges: (C_xx, C_xy, var_y) with N normalization."""
    P = len(channels) * n_features(spec)
    n, sx, sy = 0, np.zeros(P), np.zeros(y.shape[1])
    Sxx, Sxy, Syy = np.zeros((P, P)), np.zeros((P, y.shape[1])), np.zeros(y.shape[1])
    for a, b in ranges:
        X = featurize(z[a:b][:, channels].astype(np.float64), spec)
        Y = y[a:b].astype(np.float64)
        n += len(X); sx += X.sum(0); sy += Y.sum(0)
        Sxx += X.T @ X; Sxy += X.T @ Y; Syy += (Y ** 2).sum(0)
    mx, my = sx / n, sy / n
    Cxx = Sxx / n - np.outer(mx, mx)
    Cxy = Sxy / n - np.outer(mx, my)
    vy = Syy / n - my ** 2
    return Cxx, Cxy, vy


@dataclass
class Surrogate:
    """Lagged-ridge surrogate over a pool of channels (precomputed train/val covariances).

    ``channels`` are the pool's GLOBAL channel indices; methods take LOCAL indices into the pool.
    """

    channels: np.ndarray
    spec: tuple
    Cxx: np.ndarray
    Cxy: np.ndarray
    Vxx: np.ndarray
    Vxy: np.ndarray
    vy: np.ndarray

    @classmethod
    def build(cls, z, y, block, train_ranges, val_ranges, channels, spec=SPEC, allowed=SEL_BLOCKS):
        guard_ranges(block, list(train_ranges) + list(val_ranges), allowed)
        channels = np.asarray(channels, int)
        Cxx, Cxy, _ = _moments(z, y, train_ranges, channels, spec)
        Vxx, Vxy, vy = _moments(z, y, val_ranges, channels, spec)
        return cls(channels, tuple(spec), Cxx, Cxy, Vxx, Vxy, vy)

    @classmethod
    def from_data(cls, data, channels, spec=SPEC):
        """Pool from a ``fig4_rnn.Data`` object: train = data.seg['train'], val = data.seg['val']."""
        return cls.build(data.z, data.vel, data.block, data.seg["train"], data.seg["val"], channels, spec)

    @property
    def nf(self) -> int:
        return n_features(self.spec)

    @property
    def k(self) -> int:
        return len(self.channels)

    def cols(self, local) -> np.ndarray:
        local = np.asarray(local, int)
        nf = self.nf
        return (local[:, None] * nf + np.arange(nf)[None, :]).ravel()

    def fit(self, local, alpha):
        c = self.cols(local)
        A = self.Cxx[np.ix_(c, c)] + alpha * np.eye(len(c))
        return c, np.linalg.solve(A, self.Cxy[c])

    def score(self, local, alpha) -> float:
        """Mean over the 4 target dims of the validation Pearson r of the ridge fit on ``local``."""
        if len(local) == 0:
            return 0.0
        c, W = self.fit(local, alpha)
        num = np.einsum("pd,pd->d", W, self.Vxy[c])
        var = np.einsum("pd,pq,qd->d", W, self.Vxx[np.ix_(c, c)], W)
        r = num / np.sqrt(np.maximum(var, 1e-300) * self.vy)
        return float(np.mean(r))

    def choose_alpha(self, grid=ALPHA_GRID) -> float:
        """ASSUMPTION (#43): alpha chosen ONCE on the whole pool (best validation r) and reused for subsets."""
        allc = np.arange(self.k)
        return float(max(grid, key=lambda a: self.score(allc, a)))


# ----------------------------------------------------------------------------- search strategies
def greedy(sur: Surrogate, alpha: float, n_max: int = 64):
    """Greedy forward selection. Returns (order (local idx), val r after each addition)."""
    sel, rem, vals = [], list(range(sur.k)), []
    for _ in range(min(n_max, sur.k)):
        scores = [sur.score(sel + [c], alpha) for c in rem]
        j = int(np.argmax(scores))
        sel.append(rem.pop(j)); vals.append(float(scores[j]))
    return sel, vals


def swap_refine(sur: Surrogate, start, alpha: float, tol: float = 1e-6, max_passes: int = 200):
    """Best-improvement 1-swap local search from ``start`` (local idx).

    Each pass evaluates every (selected, unselected) swap and applies the single best one if it
    improves the validation r by more than ``tol``; stops when no swap improves.
    Returns (final set, final score, number of swaps applied, gain over the start).
    """
    S = list(start)
    s0 = best = sur.score(S, alpha)
    n_swaps = 0
    for _ in range(max_passes):
        out = [c for c in range(sur.k) if c not in S]
        cand, cand_s = None, best + tol
        for i in range(len(S)):
            for c in out:
                T = S[:i] + [c] + S[i + 1:]
                s = sur.score(T, alpha)
                if s > cand_s:
                    cand, cand_s = T, s
        if cand is None:
            break
        S, best = cand, cand_s
        n_swaps += 1
    assert best >= s0 - 1e-12, "swap refinement must never lower the score"
    return S, float(best), n_swaps, float(best - s0)


def _fista(Cxx, Cxy, lam, alpha, groups, W0, L, max_iter=3000, tol=1e-7):
    """Group-lasso (+ ridge alpha) on covariances by FISTA.

    min_W 1/2 tr(W' Cxx W) - tr(W' Cxy) + alpha/2 ||W||^2 + lam * sum_g ||W_g||_F
    (the same objective as 1/(2N) ||Yc - Xc W||^2 + penalties, up to a constant).
    """
    W, Z, t = W0.copy(), W0.copy(), 1.0
    step = 1.0 / L
    for _ in range(max_iter):
        G = Cxx @ Z + alpha * Z - Cxy
        V = Z - step * G
        nrm = np.sqrt((V.reshape(groups, -1) ** 2).sum(1))
        shrink = np.maximum(0.0, 1.0 - step * lam / np.maximum(nrm, 1e-300))
        Wn = (V.reshape(groups, -1) * shrink[:, None]).reshape(V.shape)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Z = Wn + ((t - 1) / tn) * (Wn - W)
        diff = np.abs(Wn - W).max()
        W, t = Wn, tn
        if diff < tol * max(1.0, np.abs(W).max()):
            break
    return W


def _group_norms(W, k):
    return np.sqrt((W.reshape(k, -1) ** 2).sum(1))


def group_lasso_sets(sur: Surrogate, alpha: float, ns=(8, 16, 32, 64), n_path: int = 80,
                     n_bisect: int = 30, eps: float = 1e-10):
    """Group-lasso channel sets of exactly n channels for each n (local idx).

    The lambda path runs from lambda_max (all groups zero) down geometrically to 1e-3 * lambda_max
    with warm starts. For each n, the first path lambda with >= n active groups and its predecessor
    bracket the target; log-lambda bisection then looks for exactly n active groups. If it is never
    exact, the top n groups by norm at the bracketing lambda with > n active groups are kept
    (documented per n in ``exact``).
    Returns {n: dict(channels, lambda, exact)}.
    """
    k = sur.k
    # Rows of W are ordered (channel, lag), so W.reshape(k, -1) gives one row per channel group
    # (its nf features x 4 weights).
    Cxx, Cxy = sur.Cxx, sur.Cxy
    L = float(np.linalg.eigvalsh(Cxx)[-1]) + alpha
    lam_max = float(_group_norms(Cxy, k).max())
    lams = lam_max * np.logspace(0, -3, n_path)
    W = np.zeros_like(Cxy)
    path = []  # (lam, W, n_active)
    for lam in lams:
        W = _fista(Cxx, Cxy, lam, alpha, k, W, L)
        na = int((_group_norms(W, k) > eps).sum())
        path.append((lam, W.copy(), na))
    out = {}
    for n in ns:
        hit = next((i for i, (_, _, na) in enumerate(path) if na >= n), None)
        if hit is None:  # never reached n groups: take the top n at the smallest lambda
            lam, Wn, _ = path[-1]
            top = np.argsort(-_group_norms(Wn, k))[:n]
            out[n] = dict(channels=sorted(top.tolist()), **{"lambda": float(lam)}, exact=False)
            continue
        lam_lo, W_lo, na_lo = path[hit]
        if na_lo == n:
            act = np.where(_group_norms(W_lo, k) > eps)[0]
            out[n] = dict(channels=act.tolist(), **{"lambda": float(lam_lo)}, exact=True)
            continue
        lam_hi, W_hi, _ = path[hit - 1] if hit > 0 else (lam_max, np.zeros_like(Cxy), 0)
        found = None
        lo, hi, Wc = np.log(lam_lo), np.log(lam_hi), W_hi.copy()
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            Wm = _fista(Cxx, Cxy, float(np.exp(mid)), alpha, k, Wc, L)
            na = int((_group_norms(Wm, k) > eps).sum())
            if na == n:
                found = (float(np.exp(mid)), Wm)
                break
            if na > n:
                lo, W_lo = mid, Wm
            else:
                hi, Wc = mid, Wm
        if found is not None:
            act = np.where(_group_norms(found[1], k) > eps)[0]
            out[n] = dict(channels=act.tolist(), **{"lambda": found[0]}, exact=True)
        else:  # DEVIATION (#43): no lambda gives exactly n groups; keep the top n at the bracket above n
            top = np.argsort(-_group_norms(W_lo, k))[:n]
            out[n] = dict(channels=sorted(top.tolist()), **{"lambda": float(np.exp(lo))}, exact=False)
    return out


def random_baseline(sur: Surrogate, alpha: float, n: int, n_sets: int, seed: int):
    """Surrogate validation r of ``n_sets`` random n-subsets of the pool (default_rng(seed))."""
    rng = np.random.default_rng(seed)
    return [sur.score(np.sort(rng.choice(sur.k, n, replace=False)), alpha) for _ in range(n_sets)]


def percentile(value: float, dist) -> float:
    """Percent of the random distribution strictly below ``value`` (ties count half)."""
    d = np.asarray(dist)
    return float(100.0 * ((d < value).sum() + 0.5 * (d == value).sum()) / len(d))


def ablation_importance(score_fn, X: np.ndarray):
    """Single-channel ablation: importance_c = score(X) - score(X with channel c set to 0).

    ``score_fn(X) -> float`` evaluates a trained decoder on the (validation) recording X (T, k).
    Setting a channel to 0 = its mean in z-units. Redundancy-blind: two channels that carry the same
    information each look unimportant, so this is a cross-check only (register #43).
    """
    base = score_fn(X)
    imp = np.empty(X.shape[1])
    for c in range(X.shape[1]):
        Xc = X.copy(); Xc[:, c] = 0.0
        imp[c] = base - score_fn(Xc)
    return float(base), imp


# ----------------------------------------------------------------------------- public helper
_POOLS: dict = {}


def surrogate_score(data, channels, alpha=None) -> float:
    """Surrogate validation r of a GLOBAL channel set on a ``fig4_rnn.Data`` object.

    If all channels lie in one array, the cached pool of that array is used and, when ``alpha`` is
    None, the array's alpha (chosen on all 96 channels, as in the selection). Otherwise a pool of
    exactly these channels is built and alpha is chosen on it.
    """
    ch = np.asarray(channels, int)
    for name, pool in ARRAYS.items():
        if np.isin(ch, pool).all():
            key = (id(data), name)
            if key not in _POOLS:
                sur = Surrogate.from_data(data, pool)
                _POOLS[key] = (sur, sur.choose_alpha())
            sur, a = _POOLS[key]
            local = np.searchsorted(pool, ch)
            return sur.score(local, a if alpha is None else alpha)
    sur = Surrogate.from_data(data, ch)
    return sur.score(np.arange(len(ch)), sur.choose_alpha() if alpha is None else alpha)
