"""Phase 8 (extension): swap the paper's GRU for an LSTM, in the Fig 4d setup (day 1869).

Question
--------
Does an LSTM core decode as well as the paper's GRU, raw-trained and with temporal-fracturing
augmentation? The comparison is matched: same data, splits, targets, augmentation, training
loop and random-search space as the Fig 4 replication (``scripts/fig4_rnn.py``).

Design
------
* Data and metrics come from ``fig4_rnn``: train on blocks 1-5, select on block 6, report
  block 7; Pearson r per hand; roughness.
* Three training-data kinds:
    raw   original timing (random contiguous 10-s windows)
    augL  2-8-s clips (the MAIN augmentation, register #14)
    augS  200-800-ms clips (the replication search's original augmentation)
* PAIRED search (register #33): the same 10 hyperparameter draws as the GRU search
  (``default_rng(2021)`` + ``fig4_rnn.sample_hp``, step scale 1.0), the same per-setting seeds,
  and the same fixed snippet sets. Setting k of the LSTM and of the GRU differ only in the cell.
  The GRU's raw and augS runs already exist (``results/rnn_search``). Its augL search is new
  here (register #34, resolving #31).
* Seeds: the best setting per (cell, kind), picked on validation r, plus seeds 101 and 102.
* Parameter-matched LSTM: ``rnn.matched_hidden()`` = 433 units, with recurrent parameters about
  equal to GRU-512 (1.086M vs 1.084M). Trained at the best LSTM-512 settings, raw and augL,
  3 seeds each.
* Every new model is also scored on the closed-loop proxy (``bimanual.proxy``); the GRU
  baseline checkpoints are scored in stage ``proxy``.

Priority order under the GPU cap (``GPU_BUDGET_S``, 3 h of summed train time):
  1 LSTM raw + augL search -> 2 GRU augL search -> 3 seeds of those bests ->
  4 LSTM augS search + seeds -> 5 parameter-matched LSTM.
Whatever doesn't fit is listed in ``budget.json`` as skipped.

Stages: smoke | search (all training, in priority order) | proxy (scores the GRU baselines) |
        plot (summary + figure) | all
Inputs : Data/t5_09_29_2021.mat; results/rnn_search/trials.jsonl (GRU baseline)
Outputs: results/ext_lstm/trials.jsonl, budget.json, proxy_gru_baseline.json, summary.json;
         results/checkpoints/<run>.pt; results/figures/ext_lstm_vs_gru.{png,svg}
Run    : pixi run ext-lstm [--stage ...]   (about 3 GPU-h; resumable: finished models are skipped)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_rnn as F  # noqa: E402

from bimanual import proxy as PX  # noqa: E402
from bimanual import rnn as R  # noqa: E402
from bimanual.cache import RESULTS_DIR, atomic_write_json, read_jsonl  # noqa: E402

OUT = RESULTS_DIR / "ext_lstm"
TRIALS = OUT / "trials.jsonl"
GPU_BUDGET_S = 3 * 3600
N_CONFIGS = F.N_CONFIGS  # 10, as in the GRU search
EXTRA_SEEDS = (101, 102)
CLIP_BINS = {"augS": (10, 40), "augL": (100, 400)}
KINDS = ("raw", "augL", "augS")


# ----------------------------------------------------------------------------- shared helpers
def search_hps():
    """The GRU search's 10 settings, re-drawn exactly as fig4_rnn.main does (scale 1.0)."""
    rng = np.random.default_rng(2021)
    return [F.sample_hp(rng, 1.0) for _ in range(N_CONFIGS)]


def batch_fns(data):
    """{kind: batch_fn}. Snippet sets use seed 0, the same fixed sets as the GRU runs."""
    return {"raw": data.raw_batch_fn(),
            "augS": data.aug_batch_fn(seed=0, clip_bins=CLIP_BINS["augS"]),
            "augL": data.aug_batch_fn(seed=0, clip_bins=CLIP_BINS["augL"])}


def gru_baseline_search():
    """{(kind, k): record} of the GRU search models: replication runs (raw, augS) + new augL runs."""
    out = {}
    for r in read_jsonl(F.TRIALS):
        if r["tag"] == "search":
            out[("raw" if r["kind"] == "raw" else "augS", int(r["run_id"][1:3]))] = r
    for r in read_jsonl(TRIALS):
        if r.get("cell") == "gru" and r["tag"] == "search":
            out[(r["kind_ext"], r["setting"])] = r
    return out


def lstm_search():
    return {(r["kind_ext"], r["setting"]): r for r in read_jsonl(TRIALS)
            if r.get("cell") == "lstm" and r["tag"] == "search"}


def best_records(cell: str, kind: str):
    """Best search record (validation r_mean) for (cell, kind) at the 512-unit width, or None."""
    pool = gru_baseline_search() if cell == "gru" else lstm_search()
    recs = [r for (k_, _), r in pool.items() if k_ == kind]
    return max(recs, key=lambda r: r["val"]["r_mean"]) if recs else None


def seed_records(cell: str, kind: str, hidden: int | None = None):
    """The best search model of (cell, kind) plus its extra-seed runs (the '3 seeds')."""
    best = best_records(cell, kind)
    if best is None:
        return []
    hidden = hidden or 512
    if hidden != 512:  # parameter-matched runs
        return [r for r in read_jsonl(TRIALS) if r["tag"] == "match" and r.get("cell") == cell
                and r["kind_ext"] == kind and r["hp"]["hidden"] == hidden]
    seeds = [r for r in read_jsonl(TRIALS) if r["tag"] == "seed" and r.get("cell") == cell and r["kind_ext"] == kind]
    if cell == "gru" and kind in ("raw", "augS"):  # replication's extra seeds
        old_kind = "raw" if kind == "raw" else "aug"
        seeds = [r for r in read_jsonl(F.TRIALS) if r["tag"] == "seed" and r["kind"] == old_kind]
    return [best] + seeds


def _spent():
    return sum(r["train_seconds"] for r in read_jsonl(TRIALS))


# ----------------------------------------------------------------------------- stages
def stage_smoke(data, fns):
    rows = {}
    for cell, h in (("gru", 512), ("lstm", 512), ("lstm", R.matched_hidden())):
        sps = F.smoke_seconds_per_step(data, {"aug": fns["augL"]}, cell=cell, hidden=h)
        m = R.RNNDecoder(n_in=192, hidden=h, cell=cell)
        rows[f"{cell}{h}"] = dict(sec_per_step=sps, n_params=m.n_params(), n_params_recurrent=m.n_params(True))
        print(f"smoke {cell}-{h}: {sps * 1000:.0f} ms/step at batch 32, {m.n_params() / 1e6:.3f} M params")
    est = rows["lstm512"]["sec_per_step"] * 3000 * (3 * N_CONFIGS + 6 + 6) + rows["gru512"]["sec_per_step"] * 3000 * (
        N_CONFIGS + 2)
    budget = dict(models=rows, est_hours_all=est / 3600, cap_hours=GPU_BUDGET_S / 3600)
    atomic_write_json(OUT / "budget.json", budget)
    print(f"estimated {est / 3600:.1f} GPU-h for everything (cap {GPU_BUDGET_S / 3600:.0f} h)")
    return budget


def stage_search(data, fns, pdata):
    hps = search_hps()
    skipped = []
    post = lambda m: dict(proxy=PX.score_model(m, data.z, pdata))  # noqa: E731

    def run(run_id, cell, kind, hp, seed, tag, setting=None):
        if any(r["run_id"] == run_id for r in read_jsonl(TRIALS)):
            return
        if _spent() >= GPU_BUDGET_S:
            skipped.append(run_id)
            return
        F.train_one(run_id, kind="raw" if kind == "raw" else "aug", hp=hp, data=data, batch_fns={
            "raw": fns["raw"], "aug": fns[kind]}, seed=seed, tag=tag, cell=cell, trials=TRIALS,
            extra=dict(kind_ext=kind, setting=setting), post_score=post)

    def seeds_of(cell, kind):
        best = best_records(cell, kind)
        if best is None:
            return
        hp = R.HParams(**best["hp"])
        for sd in EXTRA_SEEDS:
            run(f"{cell}_{kind}_best_seed{sd}", cell, kind, hp, sd, "seed", setting=best.get("setting"))

    # 1. LSTM raw + augL search (paired with the GRU's settings and per-setting seeds)
    for k, hp in enumerate(hps):
        for kind in ("raw", "augL"):
            run(f"lstm_{kind}_s{k:02d}", "lstm", kind, hp, k, "search", setting=k)
    # 2. GRU augL search
    for k, hp in enumerate(hps):
        run(f"gru_augL_s{k:02d}", "gru", "augL", hp, k, "search", setting=k)
    # 3. seeds of the bests (the GRU raw/augS seeds already exist in the replication)
    seeds_of("lstm", "raw"); seeds_of("lstm", "augL"); seeds_of("gru", "augL")
    # 4. LSTM augS search + seeds
    for k, hp in enumerate(hps):
        run(f"lstm_augS_s{k:02d}", "lstm", "augS", hp, k, "search", setting=k)
    seeds_of("lstm", "augS")
    # 5. parameter-matched LSTM at the best LSTM-512 settings
    h = R.matched_hidden()
    for kind in ("raw", "augL"):
        best = best_records("lstm", kind)
        if best is None:
            continue
        hp = R.HParams(**{**best["hp"], "hidden": h})
        for sd in (best["seed"],) + EXTRA_SEEDS:
            run(f"lstm{h}_{kind}_seed{sd}", "lstm", kind, hp, sd, "match", setting=best.get("setting"))

    b = read_jsonl_budget()
    b.update(spent_hours=_spent() / 3600, skipped=skipped)
    atomic_write_json(OUT / "budget.json", b)
    print(f"search done: {_spent() / 3600:.2f} GPU-h spent; skipped {len(skipped)} runs {skipped}")


def read_jsonl_budget():
    import json
    p = OUT / "budget.json"
    return json.loads(p.read_text()) if p.exists() else {}


def stage_proxy(data, pdata):
    """Closed-loop proxy for every GRU baseline checkpoint of the replication (evaluation only)."""
    path = OUT / "proxy_gru_baseline.json"
    done = {}
    if path.exists():
        import json
        done = json.loads(path.read_text())
    for r in read_jsonl(F.TRIALS):
        if r["run_id"] in done or r.get("sat_mode", "clip") != "clip":
            continue
        m = R.load_from_record(r, data.z.shape[1], F.CKPT_DIR)
        done[r["run_id"]] = PX.score_model(m, data.z, pdata)
        print(f"proxy {r['run_id']}: all r {done[r['run_id']]['all']['r_mean']:.3f}")
    atomic_write_json(path, done)
    return done


def main(stage="all"):
    OUT.mkdir(parents=True, exist_ok=True); F.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data = F.Data("clip")
    pdata = PX.build(data.session)
    if stage in ("all", "smoke", "search", "proxy"):
        fns = batch_fns(data) if stage != "proxy" else None
        if stage in ("all", "smoke") or not (OUT / "budget.json").exists():
            stage_smoke(data, fns or batch_fns(data))
        if stage in ("all", "proxy", "search"):
            stage_proxy(data, pdata)
        if stage in ("all", "search"):
            stage_search(data, fns, pdata)
    if stage in ("all", "plot"):
        import ext_lstm_plot
        ext_lstm_plot.summarize_and_plot(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "smoke", "search", "proxy", "plot"], default="all")
    main(ap.parse_args().stage)
