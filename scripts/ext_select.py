"""Phase 10: best-n channel selection within each array (day 1869), register #43.

For arrays A (channels 0-95) and B (96-191), using the lagged-ridge surrogate of
``bimanual/select.py`` (raw bin + causal exponential smoothings at 2-75 bins per channel; train
blocks 1-5, scored on validation block 6; test block 7 never read):
  1. alpha chosen once on all 96 channels;
  2. greedy forward selection up to 64 channels;
  3. swap refinement of the greedy prefixes, n = 8, 16, 32, 64  -> the "greedy_swap" sets that
     the GPU queue trains (first write of selection.json, final from then on);
  4. group lasso (joint benchmark), n = 8, 16, 32, 64;
  5. random baseline: 1000 random n-subsets per n (default_rng(5000+n) for A, 6000+n for B);
  6. LSTM ablation ranking from the Phase 9 one-array LSTMs (2-8 s clips, 3 seeds), on block 6.
Each method's set is also reported as a percentile of the random baseline.

Output: results/ext_feas/selection.json (+ select_log.txt). Channel indices are GLOBAL (0-191).
Run   : pixi run python scripts/ext_select.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_rnn as F  # noqa: E402

from bimanual import select as S  # noqa: E402
from bimanual.cache import RESULTS_DIR, atomic_write_json, read_jsonl  # noqa: E402

OUT = RESULTS_DIR / "ext_feas"
SEL = OUT / "selection.json"
LOG = OUT / "select_log.txt"
NS = (8, 16, 32, 64)
N_RANDOM = 1000
RAND_SEED = {"A": 5000, "B": 6000}
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.1f} s] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def glob(pool, local):
    return [int(pool[i]) for i in local]


def ablation(arr):
    """Ablation importance (block 6) averaged over the 3 Phase 9 LSTM one-array models of ``arr``."""
    from bimanual import rnn as R
    recs = [r for r in read_jsonl(RESULTS_DIR / "ext_arrays" / "trials.jsonl")
            if r.get("cell") == "lstm" and r.get("kind_ext") == "augL" and r.get("channel_set") == arr]
    assert len(recs) == 3, f"expected 3 Phase 9 LSTM models for array {arr}, got {len(recs)}"
    (a, b), = data.seg["val"]
    S.guard_ranges(data.block, [(a, b)])
    imps, bases = [], []
    for rec in recs:
        ch = np.asarray(rec["channels"], int)
        assert np.array_equal(ch, S.ARRAYS[arr])
        X, V = data.z[a:b][:, ch], data.vel[a:b]
        m = R.load_from_record(rec, 96, F.CKPT_DIR)
        base, imp = S.ablation_importance(lambda Xm: float(np.mean(R.evaluate(m, Xm, V)[0])), X)
        assert abs(base - rec["val"]["r_mean"]) < 1e-3, (base, rec["val"]["r_mean"])  # intact = logged val r
        imps.append(imp); bases.append(base)
        log(f"  ablation {rec['run_id']}: intact val r {base:.4f}, max drop {imp.max():.4f}")
    imp = np.mean(imps, 0)
    order = np.argsort(-imp)
    return dict(order=glob(S.ARRAYS[arr], order), importance=[float(imp[i]) for i in order],
                intact_val_r=float(np.mean(bases)))


def main():
    global data
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    data = F.Data()
    res = {"meta": dict(day=F.DAY, lags=None, features=dict(kind=S.SPEC[0], taus_bins=list(S.SPEC[1]),
                                                          per_channel=S.n_features(S.SPEC)),
                        blocks_used=list(S.SEL_BLOCKS), alpha={}, n_random=N_RANDOM,
                        swap_rule="best-improvement 1-swap per pass, tol 1e-6", seconds=None)}
    sur = {}
    # ---- 1-3: alpha, greedy, swap refinement (the sets the GPU queue needs first)
    for arr, pool in S.ARRAYS.items():
        s = S.Surrogate.from_data(data, pool)
        alpha = s.choose_alpha()
        full = s.score(np.arange(96), alpha)
        sur[arr] = (s, alpha)
        res["meta"]["alpha"][arr] = alpha
        log(f"array {arr}: alpha {alpha:.4g}, all-96 surrogate val r {full:.4f}")
        order, vals = S.greedy(s, alpha, n_max=max(NS))
        log(f"  greedy done: r at n=8/16/32/64 = " + " / ".join(f"{vals[n - 1]:.4f}" for n in NS))
        sw = {}
        for n in NS:
            fin, sc, k, gain = S.swap_refine(s, order[:n], alpha)
            sw[str(n)] = dict(channels=sorted(glob(pool, fin)), val_r=sc, n_swaps=k, gain=gain)
            log(f"  swap n={n}: {k} swaps, gain {gain:+.4f} -> {sc:.4f}")
        res[arr] = dict(full_val_r=full, greedy=dict(order=glob(pool, order), val_r=vals), greedy_swap=sw)
    res["sets"] = {"greedy_swap": {arr: {n: res[arr]["greedy_swap"][n]["channels"] for n in map(str, NS)}
                                   for arr in S.ARRAYS}}
    res["meta"]["seconds"] = time.time() - T0
    atomic_write_json(SEL, res)
    log("wrote selection.json with the final greedy_swap sets")

    # ---- 4-6: group lasso, random baseline, ablation
    res["sets"]["group_lasso"], res["sets"]["ablation"] = {}, {}
    for arr, pool in S.ARRAYS.items():
        s, alpha = sur[arr]
        gl = S.group_lasso_sets(s, alpha, ns=NS)
        res[arr]["group_lasso"] = {str(n): dict(channels=sorted(glob(pool, v["channels"])),
                                                val_r=s.score(sorted(v["channels"]), alpha),
                                                **{"lambda": v["lambda"]}, exact=v["exact"])
                                   for n, v in gl.items()}
        log(f"  group lasso: " + "  ".join(f"n={n} r {res[arr]['group_lasso'][str(n)]['val_r']:.4f}"
                                           f"{'' if gl[n]['exact'] else ' (top-n)'}" for n in NS))
        res[arr]["random"] = {str(n): dict(val_r=S.random_baseline(s, alpha, n, N_RANDOM, RAND_SEED[arr] + n))
                              for n in NS}
        log("  random: " + "  ".join(f"n={n} median {np.median(res[arr]['random'][str(n)]['val_r']):.4f}"
                                     for n in NS))
        res[arr]["ablation"] = ablation(arr)
        abl_local = np.searchsorted(pool, res[arr]["ablation"]["order"])
        res[arr]["ablation"]["sets"] = {str(n): dict(channels=sorted(res[arr]["ablation"]["order"][:n]),
                                                     val_r=s.score(np.sort(abl_local[:n]), alpha)) for n in NS}
        res["sets"]["group_lasso"][arr] = {str(n): res[arr]["group_lasso"][str(n)]["channels"] for n in NS}
        res["sets"]["ablation"][arr] = {str(n): res[arr]["ablation"]["sets"][str(n)]["channels"] for n in (8, 16)}
        pct = {}
        for meth, get in (("greedy", lambda n: res[arr]["greedy"]["val_r"][n - 1]),
                          ("greedy_swap", lambda n: res[arr]["greedy_swap"][str(n)]["val_r"]),
                          ("group_lasso", lambda n: res[arr]["group_lasso"][str(n)]["val_r"]),
                          ("ablation", lambda n: res[arr]["ablation"]["sets"][str(n)]["val_r"])):
            pct[meth] = {str(n): S.percentile(get(n), res[arr]["random"][str(n)]["val_r"]) for n in NS}
        res[arr]["percentiles"] = pct
        res["meta"]["seconds"] = time.time() - T0
        atomic_write_json(SEL, res)
        log(f"array {arr} complete")
    log("done")


if __name__ == "__main__":
    main()
