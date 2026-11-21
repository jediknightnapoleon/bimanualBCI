"""Phase 9 (extension): how much RNN decoding survives with only ONE of T5's two Utah arrays?

Question
--------
T5 has two 96-electrode arrays in the hand knob of left precentral gyrus, 192 channels in all.
If one array failed or were never implanted, how much of the offline decoding performance
(Fig 4d setup, day 1869) would remain? Does the answer differ between hands? The left
(ipsilateral) hand's signal is weaker and more context-dependent.

Design
------
* ``check`` stage (9.1). ASSUMPTION (register #36): channels 1-96 are one array and 97-192 the
  other; the data has no electrode map. This is tested on the correlation structure of
  block-mean-removed 20-ms counts (days 1869 and 1750). Shared array-level noise (reference,
  common-mode artifacts) should make channels correlate more within an array than across.
  Clustering the channels into 2 groups should recover the contiguous split. Which array is
  medial and which lateral stays unknown (see docs/paper_check.md).
* ``run`` stage (9.3). Every model uses the best 192-channel hyperparameters of its
  (cell, kind) from Phase 8 (DEVIATION #37: no new search per channel set):
    - array A only (channels 0-95) and array B only (96-191): GRU and LSTM x raw and augL x
      3 seeds (0, 101, 102) = 24 models;
    - random-96 control: 3 random 96-channel subsets drawn from both arrays, GRU, raw + augL,
      seed 0 = 6 models. This separates "half the channels" from "which array".
  The 192-channel reference is the 3-seed set from Phase 8 (``ext_lstm.seed_records``).
* ``zero`` stage: the 192-channel models with one array set to 0 at test, no retraining.
  0 is the block mean in z-scored units (#38). Evaluation only.
* Every model is also scored on the closed-loop proxy (bimanual.proxy).
GPU cap: 2 h of summed train time. Priority: GRU per array -> LSTM per array -> random-96.

Stages: check | run | zero | plot | all
Inputs : Data/t5_09_29_2021.mat, Data/t5_06_02_2021.mat; results/ext_lstm/trials.jsonl
Outputs: results/ext_arrays/{array_check.json, trials.jsonl, zero_out.json, summary.json};
         results/figures/ext_arrays_check.png, ext_arrays.{png,svg}
Run    : pixi run ext-arrays [--stage ...]   (about 1.5 GPU-h after Phase 8; resumable)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_lstm as XL  # noqa: E402
import fig4_rnn as F  # noqa: E402

from bimanual import proxy as PX  # noqa: E402
from bimanual import rnn as R  # noqa: E402
from bimanual.cache import RESULTS_DIR, atomic_write_json, read_jsonl  # noqa: E402
from bimanual.io import load_session, remove_block_means  # noqa: E402

OUT = RESULTS_DIR / "ext_arrays"
TRIALS = OUT / "trials.jsonl"
GPU_BUDGET_S = 2 * 3600
ARRAYS = {"A": np.arange(0, 96), "B": np.arange(96, 192)}
SEEDS = (0, 101, 102)
N_RANDOM = 3
CHECK_DAYS = (1869, 1750, 1883)


# ----------------------------------------------------------------------------- 9.1 array check
def array_check():
    """Is there a two-block correlation structure consistent with channels 1-96 / 97-192?

    Tests per day, on block-mean-removed 20-ms counts:
    * contrast = mean within-half r - mean across-half r, vs 200 RANDOM 96/96 splits (null);
    * own > other: fraction of channels correlating more with their own half;
    * shift rank: rank of the 1-96 split among all 96 contiguous (circularly shifted) 96/96
      splits. Correlation is local in channel index, so any contiguous split beats a random
      one; only this comparison tests where the boundary is.
    Explored and dropped (uninformative): average-linkage / spectral clustering (the r values are
    about 0.01 and the clusters come out degenerate); a firing-rate step at channel 96 (none).

    ``passes`` (the plan's stop rule): a two-block structure beyond the random-split null on
    every day. ``strength`` summarizes the shift ranks.
    """
    res, mats = {}, {}
    for day in CHECK_DAYS:
        s = load_session(day)
        X = remove_block_means(s.tx, s.block)
        ok = X.std(0) > 0
        full = np.full((192, 192), np.nan); full[np.ix_(ok, ok)] = np.corrcoef(X[:, ok].T)
        np.fill_diagonal(full, np.nan)

        def contrast_of(g):
            return (np.nanmean(full[np.ix_(g, g)]) + np.nanmean(full[np.ix_(~g, ~g)])) / 2 - np.nanmean(full[np.ix_(g, ~g)])

        isB = np.arange(192) >= 96
        a, b = ARRAYS["A"], ARRAYS["B"]
        within_A = np.nanmean(full[np.ix_(a, a)]); within_B = np.nanmean(full[np.ix_(b, b)])
        across = np.nanmean(full[np.ix_(a, b)])
        contrast = contrast_of(isB)
        rng = np.random.default_rng(0)
        null = [contrast_of(np.isin(np.arange(192), rng.permutation(192)[:96])) for _ in range(200)]
        own = np.array([np.nanmean(full[i, isB == isB[i]]) for i in range(192)])
        oth = np.array([np.nanmean(full[i, isB != isB[i]]) for i in range(192)])
        shifts = []
        for p in range(96):
            g = np.zeros(192, bool); g[p:p + 96] = True
            shifts.append(contrast_of(g))
        shift_rank = int((np.array(shifts) > shifts[0]).sum()) + 1
        res[day] = dict(within_A=float(within_A), within_B=float(within_B), across=float(across),
                        contrast=float(contrast), null_contrast_max=float(np.max(null)),
                        own_gt_other=float(np.mean((own > oth)[ok])), shift_rank=shift_rank, n_shifts=96,
                        n_silent=int((~ok).sum()), passes=bool(contrast > np.max(null)))
        mats[day] = full
        print(f"day {day}: within A {within_A:.4f}, B {within_B:.4f}, across {across:.4f}; contrast {contrast:.4f} "
              f"(random-split max {np.max(null):.4f}); own>other {res[day]['own_gt_other']:.2f}; "
              f"shift rank {shift_rank}/96")
    ranks = [v["shift_rank"] for v in res.values()]
    strength = "strong" if max(ranks) <= 5 else "moderate" if min(ranks) <= 5 else "weak"
    out = dict(days=res, passes=all(v["passes"] for v in res.values()), strength=strength,
               note="Consistent with (not proof of) channels 1-96 / 97-192 being separate arrays; see register #36.")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "array_check.json", out)
    np.savez_compressed(OUT / "corr_matrices.npz", **{str(k): v for k, v in mats.items()})
    _plot_check(mats, res)
    print("ARRAY CHECK", "PASSES" if out["passes"] else "FAILS", f"(strength: {strength})")
    return out


def _plot_check(mats, res):
    import matplotlib.pyplot as plt

    from bimanual import plotting as P

    fig, axes = plt.subplots(1, len(mats), figsize=(5 * len(mats), 4.4))
    for ax, (day, M) in zip(np.atleast_1d(axes), mats.items()):
        v = np.nanpercentile(np.abs(M), 99)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.axhline(95.5, color=P.INK, lw=0.8); ax.axvline(95.5, color=P.INK, lw=0.8)
        r = res[day]
        ax.set_title(f"Day {day}: channel correlation (block means removed)\nwithin {(r['within_A'] + r['within_B']) / 2:.4f}"
                     f" vs across {r['across']:.4f}; shift rank {r['shift_rank']}/96", fontsize=9)
        ax.set_xlabel("channel"); ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    P.save(fig, "ext_arrays_check")


# ----------------------------------------------------------------------------- 9.3 one-array training
def _spent():
    return sum(r["train_seconds"] for r in read_jsonl(TRIALS))


def channel_sets():
    """{name: 0-based channel indices}: the two arrays and the random-96 controls."""
    rng = np.random.default_rng(96)
    sets = dict(ARRAYS)
    for i in range(N_RANDOM):
        sets[f"rand{i}"] = np.sort(rng.choice(192, 96, replace=False))
    return sets


def stage_run():
    if not (OUT / "array_check.json").exists() or not __import__("json").loads(
            (OUT / "array_check.json").read_text())["passes"]:
        print("array check missing or failed: no one-array training (see the briefing)")
        return
    best = {(c, k): XL.best_records(c, k) for c in ("gru", "lstm") for k in ("raw", "augL")}
    missing = [ck for ck, v in best.items() if v is None]
    if missing:
        print("Phase 8 best settings missing for", missing, "- run ext-lstm first"); return
    sets = channel_sets()
    plan = []  # (set name, cell, kind, seed), in priority order
    for cell in ("gru", "lstm"):
        for arr in ("A", "B"):
            for kind in ("raw", "augL"):
                plan += [(arr, cell, kind, sd) for sd in SEEDS]
    plan += [(f"rand{i}", "gru", kind, 0) for i in range(N_RANDOM) for kind in ("raw", "augL")]
    skipped = []
    for name in dict.fromkeys(p[0] for p in plan):  # one data build per channel set
        todo = [p for p in plan if p[0] == name
                and not any(r["run_id"] == _rid(*p) for r in read_jsonl(TRIALS))]
        if not todo:
            continue
        if _spent() >= GPU_BUDGET_S:
            skipped += [_rid(*p) for p in todo]; continue
        ch = sets[name]
        data = F.Data("clip", channels=ch)
        assert data.z.shape[1] == 96
        pdata = PX.build(data.session)
        fns = XL.batch_fns(data)
        post = lambda m: dict(proxy=PX.score_model(m, data.z, pdata))  # noqa: E731
        for (_, cell, kind, sd) in todo:
            if _spent() >= GPU_BUDGET_S:
                skipped.append(_rid(name, cell, kind, sd)); continue
            hp = R.HParams(**best[(cell, kind)]["hp"])
            F.train_one(_rid(name, cell, kind, sd), kind="raw" if kind == "raw" else "aug", hp=hp, data=data,
                        batch_fns={"raw": fns["raw"], "aug": fns[kind]}, seed=sd, tag="arrays", cell=cell,
                        trials=TRIALS, extra=dict(kind_ext=kind, channel_set=name, channels=ch.tolist()),
                        post_score=post)
        del data, fns
    atomic_write_json(OUT / "budget.json", dict(spent_hours=_spent() / 3600, cap_hours=GPU_BUDGET_S / 3600,
                                                skipped=skipped))
    print(f"one-array runs done: {_spent() / 3600:.2f} GPU-h; skipped {skipped}")


def _rid(name, cell, kind, sd):
    return f"arr{name}_{cell}_{kind}_seed{sd}"


# ----------------------------------------------------------------------------- zero one array at test
def stage_zero():
    """192-channel 3-seed models, evaluated with array A or B set to 0 (no retraining)."""
    data = F.Data("clip")
    pdata = PX.build(data.session)
    out = {}
    for cell in ("gru", "lstm"):
        for kind in ("raw", "augL"):
            for rec in XL.seed_records(cell, kind):
                m = R.load_from_record(rec, 192, F.CKPT_DIR)
                row = {}
                for drop in ("none", "A", "B"):
                    z = data.z.copy()
                    if drop != "none":
                        z[:, ARRAYS[drop]] = 0.0
                    a, b = data.seg["test"][0]
                    r, vhat, _ = R.evaluate(m, z[a:b], data.vel[a:b])
                    row[drop] = dict(r=r.tolist(), r_R=float(np.mean(r[:2])), r_L=float(np.mean(r[2:])),
                                     roughness=F.roughness(vhat), proxy=PX.score_model(m, z, pdata)["all"])
                # sanity: nothing zeroed reproduces the logged test score
                assert abs(row["none"]["r_R"] - rec["test"]["r_R"]) < 1e-3, rec["run_id"]
                out[rec["run_id"]] = dict(cell=cell, kind=kind, **row)
                print(f"zero-out {rec['run_id']}: none {row['none']['r_R']:.2f}/{row['none']['r_L']:.2f}  "
                      f"A off {row['A']['r_R']:.2f}/{row['A']['r_L']:.2f}  B off {row['B']['r_R']:.2f}/{row['B']['r_L']:.2f}")
    atomic_write_json(OUT / "zero_out.json", out)
    return out


def main(stage="all"):
    OUT.mkdir(parents=True, exist_ok=True)
    if stage in ("all", "check"):
        array_check()
    if stage in ("all", "run"):
        stage_run()
    if stage in ("all", "zero"):
        stage_zero()
    if stage in ("all", "plot"):
        import ext_arrays_plot
        ext_arrays_plot.summarize_and_plot()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "check", "run", "zero", "plot"], default="all")
    main(ap.parse_args().stage)
