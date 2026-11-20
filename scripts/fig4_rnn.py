"""Phase 4: Fig 4b-d, RNN decoders trained on raw vs temporally fractured (augmented) data.

Purpose
-------
The paper's Fig 4 argues that the RNN has to be trained on time-warped,
shuffled clips ("temporal fracturing"). Otherwise it learns the rigid timing
of the open-loop task (delay -> go -> 1-2 s movement) and, online, makes
pulse-like movements. Offline (Fig 4d), the raw-trained RNN reaches
r = 0.90 (RH) / 0.83 (LH), and the augmented RNN is slightly LOWER, with
noisier velocities. The augmented RNN pays off only online (Fig 4e, Phase 5).

Method (day 1869, open-loop blocks 1-7, all unimanual)
------------------------------------------------------
* Inputs: 192 channels, z-scored per block.
* Targets (bimanual.targets): saturated min-jerk-like velocity profiles that
  start 200 ms after the go cue, along the unit target direction; 4-way
  context labels. ``--sat-mode clip`` (MAIN, register #13) or ``hold``.
* Split by BLOCK: train = blocks 1-5, validation = block 6 (selection only),
  test = block 7 (reported). The splits are made before augmentation (Methods).
* RAW training data: random contiguous 10-s windows of the training blocks
  (the original timing is kept).
  AUGMENTED training data: 2000 fixed 10-s snippets from SnippetSampler
  (clips of 200-800 ms, x0.5-2 time warp, shuffled, balanced over strata;
  register #14, #15).
* Model: bimanual.rnn.GRUDecoder (day-affine -> GRU 512 -> velocity + context).
* Random search (register #16, #28): N_CONFIGS settings of lr, batch,
  steps, L2, input noise and mean drift. EACH setting is trained on raw AND
  on augmented data (a paired comparison). The best of each kind is chosen on
  the validation block and reported on the test block.
* Metrics on a continuous decode of a whole block from a zero state:
  - Pearson r per dim over all bins; per hand = mean of x and y;
  - "roughness" = mean |v[t+1] - v[t]| / std(v), the bin-to-bin jitter of the
    decoded velocity, averaged over dims (higher = noisier).

Stages (all resumable): data -> smoke (timing) -> search -> extras (alternative
target reading, extra seeds) -> plot.
Inputs : Data/t5_09_29_2021.mat
Outputs: results/rnn_search/trials.jsonl (one line per trained model)
         results/checkpoints/<run>.pt (model + optimizer + RNG; gitignored)
         results/fig4/summary.json; results/figures/fig4_rnn.{png,svg}
Run    : pixi run fig4-rnn            (the whole search runs in the background; about 1-3 h on an RTX 3050)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from bimanual import augment as A
from bimanual import conditions as C
from bimanual import plotting as P
from bimanual import rnn as R
from bimanual import targets as T
from bimanual.cache import RESULTS_DIR, append_jsonl, atomic_write_json, read_jsonl
from bimanual.io import load_session, zscore_by_block

DAY = 1869
TRAIN_BLOCKS, VAL_BLOCKS, TEST_BLOCKS = (1, 2, 3, 4, 5), (6,), (7,)
N_SNIPPETS = 2000
N_CONFIGS = 10
GPU_BUDGET_S = 3 * 3600  # cap for the whole search (user instruction, 2026-11-21)
PAPER = dict(raw_R=0.90, raw_L=0.83)
SEARCH_DIR = RESULTS_DIR / "rnn_search"
CKPT_DIR = RESULTS_DIR / "checkpoints"
TRIALS = SEARCH_DIR / "trials.jsonl"
OUT_JSON = RESULTS_DIR / "fig4" / "summary.json"


# ----------------------------------------------------------------------------- data
class Data:
    """Day-1869 OL data: z-scored inputs, targets, and block splits (built once per process)."""

    def __init__(self, sat_mode="clip"):
        s = load_session(DAY)
        self.z = zscore_by_block(s.tx, s.block).astype(np.float32)
        t = s.trials
        mt = t.move_type.to_numpy()
        dirs = C.trial_directions(s.Tp, s.Cp, s.go, mt)
        quad = C.trial_quadrants(s.Tp, s.Cp, s.go, mt)
        ol = (t.kind == "OL").to_numpy()
        specs = [T.TrialSpec(start_bin=int(r.start), go_bin=int(r.go), move_end_bin=int(r.move_end),
                             end_bin=int(r.move_end), move_type=int(r.move_type),
                             dir_R=dirs[i, :2], dir_L=dirs[i, 2:],
                             quadrant_R=None if np.isnan(quad[i, 0]) else int(quad[i, 0]),
                             quadrant_L=None if np.isnan(quad[i, 1]) else int(quad[i, 1]))
                 for i, r in enumerate(t.itertuples()) if ol[i]]
        self.vel, self.ctx, self.stratum = T.build_targets(
            len(self.z), specs, sat=0.5, hold_to_end=(sat_mode == "hold"))
        self.block = s.block
        self.seg = {name: [self._block_range(b) for b in blocks]
                    for name, blocks in (("train", TRAIN_BLOCKS), ("val", VAL_BLOCKS), ("test", TEST_BLOCKS))}

    def _block_range(self, b):
        idx = np.where(self.block == b)[0]
        assert np.all(np.diff(idx) == 1), "blocks are contiguous in time"
        return int(idx[0]), int(idx[-1]) + 1

    def eval_arrays(self, split):
        a, b = self.seg[split][0]
        return self.z[a:b], self.vel[a:b]

    def raw_batch_fn(self, T_=A.SNIPPET_BINS):
        """Random contiguous 10-s windows of the training blocks (the original timing is kept)."""
        segs = self.seg["train"]
        w = np.array([b - a - T_ for a, b in segs], float)

        def fn(rng, B):
            X = np.empty((B, T_, self.z.shape[1]), np.float32); V = np.empty((B, T_, 4), np.float32)
            Cx = np.empty((B, T_), int)
            for i in range(B):
                a, b = segs[rng.choice(len(segs), p=w / w.sum())]
                s0 = int(rng.integers(a, b - T_))
                X[i], V[i], Cx[i] = self.z[s0:s0 + T_], self.vel[s0:s0 + T_], self.ctx[s0:s0 + T_]
            return X, V, Cx, np.zeros(B, int)

        return fn

    def aug_batch_fn(self, seed=0, clip_bins=A.CLIP_BINS):
        """Batches drawn from 2000 fixed augmented snippets (generated once).

        ``clip_bins``: (min, max) clip length in bins. The default (10, 40) =
        200-800 ms (register #14 main reading); (100, 400) = 0.2-0.8 x the
        500-bin snippet (the alternative reading).
        """
        sampler = A.SnippetSampler(self.z, self.vel, self.ctx, self.stratum, self.seg["train"], seed=seed,
                                   clip_bins=clip_bins)
        t0 = time.time()
        sn = sampler.make_snippets(N_SNIPPETS)
        print(f"  generated {N_SNIPPETS} augmented snippets in {time.time() - t0:.0f} s "
              f"({len(sampler.strata)} strata)")
        X, V, Cx = sn.X.astype(np.float16), sn.vel, sn.context  # float16 storage: about 380 MB

        def fn(rng, B):
            idx = rng.integers(0, len(X), B)
            return X[idx].astype(np.float32), V[idx], Cx[idx], np.zeros(B, int)

        return fn


# ----------------------------------------------------------------------------- metrics
def roughness(v):
    """Mean |v[t+1] - v[t]| / std(v), averaged over dims (decoded-velocity jitter)."""
    return float(np.mean(np.abs(np.diff(v, axis=0)).mean(0) / (v.std(0) + 1e-9)))


def score(model, data, split):
    X, V = data.eval_arrays(split)
    r, vhat, _ = R.evaluate(model, X, V)
    return dict(r=r.tolist(), r_R=float(np.mean(r[:2])), r_L=float(np.mean(r[2:])), r_mean=float(np.mean(r)),
                roughness=roughness(vhat))


# ----------------------------------------------------------------------------- search
def sample_hp(rng, steps_scale=1.0):
    """One random setting (the paper searched input noise, mean drift, lr, batch size, #batches, L2)."""
    return R.HParams(
        lr=float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3))),
        batch=int(rng.choice([16, 32, 48])),
        steps=int(round(rng.choice([2000, 3000, 4000]) * steps_scale)),
        l2=float(10 ** rng.uniform(-6, -4)),
        noise_sd=float(rng.uniform(0.1, 0.6)),
        drift_sd=float(rng.uniform(0.0, 0.3)),
    )


def train_one(run_id, kind, hp, data, batch_fns, seed, sat_mode="clip", tag="search"):
    """Train (or resume) one model, score it on val and test, and log one jsonl line."""
    if any(r["run_id"] == run_id for r in read_jsonl(TRIALS)):
        return
    torch.manual_seed(seed)
    model = R.GRUDecoder(n_in=data.z.shape[1], hidden=hp.hidden)
    print(f"[{run_id}] kind={kind} {hp}")
    info = R.train(model, batch_fns[kind], hp, ckpt_path=str(CKPT_DIR / f"{run_id}.pt"), seed=seed, log_every=500)
    rec = dict(run_id=run_id, tag=tag, kind=kind, sat_mode=sat_mode, seed=seed, hp=hp.__dict__,
               train_seconds=info["seconds"], val=score(model, data, "val"), test=score(model, data, "test"))
    append_jsonl(TRIALS, rec)
    print(f"[{run_id}] val r R {rec['val']['r_R']:.3f} L {rec['val']['r_L']:.3f} | test r R {rec['test']['r_R']:.3f} "
          f"L {rec['test']['r_L']:.3f} | rough {rec['test']['roughness']:.3f} | {info['seconds']:.0f} s")


def smoke_seconds_per_step(data, batch_fns):
    """Time 60 steps at batch 32 on augmented data (used to fit the search into the GPU budget)."""
    model = R.GRUDecoder(n_in=data.z.shape[1])
    hp = R.HParams(batch=32, steps=60)
    info = R.train(model, batch_fns["aug"], hp, log=lambda *a: None)
    return info["seconds"] / hp.steps


def main(stage="all"):
    SEARCH_DIR.mkdir(parents=True, exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data = Data("clip")
    batch_fns = {"raw": data.raw_batch_fn(), "aug": data.aug_batch_fn(seed=0)}

    if stage in ("all", "smoke", "search"):
        sps = smoke_seconds_per_step(data, batch_fns)
        mean_steps = 3000 * (32 / 32)
        est = 2 * N_CONFIGS * mean_steps * sps
        scale = min(1.0, 0.8 * GPU_BUDGET_S / est)
        print(f"smoke: {sps * 1000:.0f} ms/step (batch 32) -> full search about {est / 3600:.1f} h; "
              f"step scale {scale:.2f} to fit the {GPU_BUDGET_S / 3600:.0f}-h budget")
        atomic_write_json(SEARCH_DIR / "budget.json", dict(sec_per_step=sps, est_hours=est / 3600, steps_scale=scale))
        if stage == "smoke":
            return
        rng = np.random.default_rng(2021)
        hps = [sample_hp(rng, scale) for _ in range(N_CONFIGS)]  # same settings for raw and aug (paired)
        for k, hp in enumerate(hps):
            for kind in ("raw", "aug"):
                train_one(f"s{k:02d}_{kind}", kind, hp, data, batch_fns, seed=k)

    if stage in ("all", "extras"):
        recs = [r for r in read_jsonl(TRIALS) if r["tag"] == "search"]
        best = {kind: max((r for r in recs if r["kind"] == kind), key=lambda r: r["val"]["r_mean"])
                for kind in ("raw", "aug")}
        # extra seeds of the best settings, to see the run-to-run spread
        for kind in ("raw", "aug"):
            hp = R.HParams(**best[kind]["hp"])
            for sd in (101, 102):
                train_one(f"best_{kind}_seed{sd}", kind, hp, data, batch_fns, seed=sd, tag="seed")
        # alternative reading of "saturated" targets (register #13): rise, then hold to the end of the move period
        data_h = Data("hold")
        fns_h = {"raw": data_h.raw_batch_fn(), "aug": data_h.aug_batch_fn(seed=0)}
        for kind in ("raw", "aug"):
            train_one(f"best_{kind}_hold", kind, R.HParams(**best[kind]["hp"]), data_h, fns_h, seed=0,
                      sat_mode="hold", tag="sat_alt")

    if stage in ("all", "clips"):
        # Register #14: the alternative reading of the clip length, "0.2-0.8 x snippet length" = 2-8 s clips.
        # The 200-800-ms clips gave a much larger offline drop than the paper's "slightly lower".
        recs = [r for r in read_jsonl(TRIALS) if r["tag"] == "search"]
        best_aug = max((r for r in recs if r["kind"] == "aug"), key=lambda r: r["val"]["r_mean"])
        fns_long = {"aug": data.aug_batch_fn(seed=0, clip_bins=(100, 400))}
        for sd in (0, 101, 102):
            train_one(f"best_aug_longclip_seed{sd}", "aug", R.HParams(**best_aug["hp"]), data, fns_long, seed=sd,
                      tag="longclip")

    if stage in ("all", "extras", "clips", "plot"):
        summarize_and_plot(data)


# ----------------------------------------------------------------------------- summary + figure
def summarize_and_plot(data):
    import matplotlib.pyplot as plt

    recs = read_jsonl(TRIALS)
    search = [r for r in recs if r["tag"] == "search"]
    best = {k: max((r for r in search if r["kind"] == k), key=lambda r: r["val"]["r_mean"]) for k in ("raw", "aug")}
    seeds = {k: [best[k]] + [r for r in recs if r["tag"] == "seed" and r["kind"] == k] for k in ("raw", "aug")}
    alt = {r["kind"]: r for r in recs if r["tag"] == "sat_alt"}
    longclip = [r for r in recs if r["tag"] == "longclip"]
    summ = dict(paper=PAPER, n_search_models=len(search),
                best={k: dict(run_id=v["run_id"], hp=v["hp"], val=v["val"], test=v["test"]) for k, v in best.items()},
                seed_spread={k: dict(test_r_R=[r["test"]["r_R"] for r in v], test_r_L=[r["test"]["r_L"] for r in v],
                                     roughness=[r["test"]["roughness"] for r in v]) for k, v in seeds.items()},
                sat_alternative={k: dict(test=v["test"]) for k, v in alt.items()},
                aug_longclip_2to8s=dict(test_r_R=[r["test"]["r_R"] for r in longclip],
                                        test_r_L=[r["test"]["r_L"] for r in longclip],
                                        roughness=[r["test"]["roughness"] for r in longclip]),
                paired_search=[dict(cfg=r["run_id"][:3],
                                    raw=next(q["val"]["r_mean"] for q in search if q["run_id"] == r["run_id"][:3] + "_raw"),
                                    aug=r["val"]["r_mean"]) for r in search if r["kind"] == "aug"
                               and any(q["run_id"] == r["run_id"][:3] + "_raw" for q in search)])
    atomic_write_json(OUT_JSON, summ)
    for k in ("raw", "aug"):
        t = best[k]["test"]
        print(f"BEST {k}: test r RH {t['r_R']:.3f} LH {t['r_L']:.3f} (paper raw: 0.90 / 0.83), roughness {t['roughness']:.3f}")

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3)

    # (b) target profiles
    ax = fig.add_subplot(gs[0, 0])
    n = 90
    tt = (np.arange(10 + n) * 0.02)
    for prof, lab, col in ((T.minjerk_speed(n), "minimum-jerk", P.INK_MUTED),
                           (T.saturated_speed(n, 0.5), "saturated (clip, main)", P.SERIES[0]),
                           (T.saturated_speed(n, 0.5, hold_to_end=True), "saturated (hold, alt.)", P.SERIES[2])):
        ax.plot(tt, np.r_[np.zeros(10), prof], color=col, label=lab)
    ax.axvline(0.2, color=P.INK_MUTED, ls=":", lw=1)
    ax.text(0.22, 0.9, "200-ms\nreaction time", fontsize=7, color=P.INK_MUTED, va="top")
    ax.set_xlabel("time after go (s)"); ax.set_ylabel("target speed (norm.)")
    ax.set_title("(b) Velocity-target profiles", fontsize=9); ax.legend(fontsize=7)

    # (d) best test r per hand, raw vs aug, seeds as dots, paper as diamonds
    ax = fig.add_subplot(gs[0, 1])
    groups = [("Raw", P.SERIES[0], seeds["raw"]), ("Augmented, 200–800-ms clips", P.SERIES[1], seeds["aug"])]
    if longclip:
        groups.append(("Augmented, 2–8-s clips", P.SERIES[3], longclip))
    w = 0.8 / len(groups)
    for i, (lab, col, runs) in enumerate(groups):
        for j, hand in enumerate(("R", "L")):
            vals = [r["test"][f"r_{hand}"] for r in runs]
            x = j + (i - (len(groups) - 1) / 2) * w
            ax.bar(x, np.mean(vals), width=w - 0.03, color=col, label=lab if j == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, color=P.INK, s=10, zorder=3)
            ax.text(x, np.mean(vals) + 0.03, f"{np.mean(vals):.2f}", ha="center", fontsize=7, color=P.INK)
    x_raw = (0 - (len(groups) - 1) / 2) * w
    ax.plot([x_raw, 1 + x_raw], [PAPER["raw_R"], PAPER["raw_L"]], "D", mfc="none", mec=P.INK, ms=7, ls="none",
            label="paper (raw)")
    ax.set_xticks([0, 1], ["Right hand", "Left hand"]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("test-block Pearson r (mean of x, y)")
    ax.set_title("(d) Offline decoding, held-out block 7 (bars = mean of 3 seeds)", fontsize=9, pad=40)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)

    # paired search: validation r raw vs aug per setting
    ax = fig.add_subplot(gs[0, 2])
    ps = summ["paired_search"]
    for p_ in ps:
        ax.plot([0, 1], [p_["raw"], p_["aug"]], "-o", color=P.INK_MUTED, lw=1, ms=4)
    ax.set_xticks([0, 1], ["Raw", "Augmented"]); ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("validation r (mean of 4 dims)")
    ax.set_xticklabels(["Raw", "Augmented\n(200–800-ms clips)"])
    ax.set_title(f"Random search: {len(ps)} paired settings", fontsize=9); ax.grid(axis="x", visible=False)

    # decoded traces on the test block (best models): RH x and LH x
    X, V = data.eval_arrays("test")
    a0 = 600; sl = slice(a0, a0 + 1500)
    tt = np.arange(sl.stop - sl.start) * 0.02
    for j, (dim, name) in enumerate(((0, "Right cursor x-velocity"), (2, "Left cursor x-velocity"))):
        ax = fig.add_subplot(gs[1, 0:2] if j == 0 else gs[1, 2])
        ax.plot(tt, V[sl, dim], color=P.INK, lw=1.2, label="target")
        runs = [(best["raw"], P.SERIES[0], "Raw RNN", 1.0)]
        if longclip:
            runs.append((longclip[0], P.SERIES[3], "Augmented, 2–8-s clips (main)", 1.0))
        runs.append((best["aug"], P.SERIES[1], "Augmented, 200–800-ms clips", 0.4))
        for rec, col, lab, alpha in runs:
            m = R.GRUDecoder(n_in=X.shape[1], hidden=rec["hp"]["hidden"]).to(R.DEVICE)
            m.load_state_dict(torch.load(CKPT_DIR / f"{rec['run_id']}.pt", map_location=R.DEVICE,
                                         weights_only=False)["model"])
            v, _ = R.decode(m, X)
            ax.plot(tt, v[sl, dim], color=col, lw=1, alpha=alpha, label=lab)
        ax.set_title(f"{name}, test block (30 s excerpt)", fontsize=9); ax.set_xlabel("time (s)")
        if j == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper left"); ax.set_ylabel("velocity target / decoded")
    fig.suptitle(f"Fig 4 replication: RNN decoders, day {DAY} open-loop (train blocks 1–5, select on 6, test on 7)",
                 fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "fig4_rnn")
    print("wrote", P.FIG_DIR / "fig4_rnn.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "smoke", "search", "extras", "clips", "plot"], default="all")
    main(ap.parse_args().stage)
