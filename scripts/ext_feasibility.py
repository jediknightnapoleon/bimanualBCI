"""Phase 10 (extension): feasibility tests for a smaller implant (8 shanks x 32 electrodes, about one
Utah array's area of hand knob, one hemisphere), run on T5's threshold crossings.

Questions
---------
1. Muted channels: how much does decoding lose when random channels go bad, and does training
   with random channel dropout (register #39) make the decoder tolerant?
2. Channel count: r with n = 8 / 16 / 32 / 64 channels drawn at random WITHIN one array (a
   neuron-dropping curve), and with the best n channels chosen by greedy forward selection +
   swaps, by group lasso, or by LSTM ablation (selection by Fork A: ``scripts/ext_select.py``,
   ``bimanual/select.py``, register #43). n = 96 (one array) and 192 are known from Phases 8-9.
3. Training data: r with the last k = 1..4 of the 5 training blocks (#40).
4. Other days: day 1750 (mixed one- and two-hand trials; r on two-hand trials too) and day 1867
   (one-hand), with all channels / array A / array B (#41).

Setup common to all runs: LSTM-512 at the best Phase 8 setting for 2-8-s clips
(``ext_lstm.best_records("lstm", "augL")``), trained on 2-8-s augmented clips; the same metrics
as Fig 4 (``fig4_rnn.score``) plus the closed-loop proxy on day 1869.

Execution: a job queue (``bimanual/jobs.py``) shared by N worker processes on one GPU; N is
chosen by the ``smoke`` stage from measured throughput. Each record carries ``worker``,
``t_start``, ``t_end``; ``timing.json`` (updated after every run) has the throughput and the
ETA of the queue against the planned estimate.

Stages: smoke | jobs (write the job list) | drive (jobs + N workers + mute) | worker | mute | plot
Outputs: results/ext_feas/{jobs.json, trials_w*.jsonl, timing.json, smoke.json, mute.json,
         summary.json}; results/checkpoints/<run>.pt; results/figures/ext_feasibility.{png,svg}
Run    : pixi run ext-feas-driver   (resumable: rerun it; finished runs are skipped)
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_lstm as XL  # noqa: E402
import fig4_rnn as F  # noqa: E402

from bimanual import proxy as PX  # noqa: E402
from bimanual import rnn as R  # noqa: E402
from bimanual import sessions as S  # noqa: E402
from bimanual.cache import RESULTS_DIR, append_jsonl, atomic_write_json  # noqa: E402
from bimanual.decoders import DEVICE, pearson_per_dim  # noqa: E402
from bimanual.jobs import Queue  # noqa: E402

OUT = RESULTS_DIR / "ext_feas"
SELECTION = OUT / "selection.json"
SEL_REL = "results/ext_feas/selection.json"  # for job prerequisites (relative to the repo root)
ARRAYS = {"A": list(range(96)), "B": list(range(96, 192))}
NS = (8, 16, 32, 64)
QUEUE = Queue(OUT)


# ----------------------------------------------------------------------------- setup helpers
def best_setting():
    """(HParams, seed) of the best Phase 8 LSTM on 2-8-s clips; seeds = (seed, 101, 102) as in Phase 8."""
    best = XL.best_records("lstm", "augL")
    assert best is not None, "Phase 8 results missing (results/ext_lstm/trials.jsonl)"
    return best["hp"], int(best["seed"])


def ol_split(day: int):
    """ASSUMPTION (#41): for days other than 1869, the open-loop blocks in time order are split as
    test = last 2, validation = the 2 before, train = the rest (day 1750: 12-25 / 26-27 / 28-29)."""
    ol = sorted(b for b, blk in S.get(day).blocks.items() if blk.kind == "OL")
    return tuple(ol[:-4]), tuple(ol[-4:-2]), tuple(ol[-2:])


def random_subsets():
    """{(n, array, i): sorted global channel indices}; 3 subsets per (n, array), ``default_rng(1000+n)``."""
    out = {}
    for n in NS:
        rng = np.random.default_rng(1000 + n)
        for arr in ("A", "B"):
            for i in range(3):
                out[(n, arr, i)] = sorted(int(c) for c in rng.choice(ARRAYS[arr], n, replace=False))
    return out


# Timing adjustment 1 (2026-11-22, after the smoke test): one worker only (2-3 concurrent processes
# gave LESS total throughput: 12.3-14.4 vs 16.0 steps/s, the GPU is near saturation and RAM swaps).
# At about 4.7 min per run, the extra seeds below move to an "extras" tier that runs after
# everything else (and after Phase 11 jobs, priority 10), so they are what gets dropped if the
# 12-h window runs short.
EXTRA_PRIO = 20


def _extra(run_id: str) -> bool:
    """Runs demoted to the extras tier (timing adjustment 1)."""
    return (run_id.startswith("d1867_") and not run_id.endswith(f"seed{best_setting()[1]}")) or         (run_id.startswith("data") and run_id.endswith("seed102")) or         (run_id.startswith(("greedy32", "greedy64")) and run_id.endswith("seed101")) or         (run_id.startswith("d1750_") and run_id.endswith("seed102"))


def build_jobs() -> list[dict]:
    """The Phase 10 job list in priority order (see the plan)."""
    _, s0 = best_setting()
    seeds = (s0, 101, 102)
    jobs, order = [], 0

    def add(run_id, prio, group, **kw):
        nonlocal order
        jobs.append(dict(run_id=run_id, priority=EXTRA_PRIO if _extra(run_id) else prio, order=order, group=group,
                         **kw)); order += 1

    # 1 random channel dropout (192 ch, day 1869)
    for sd in seeds:
        add(f"drop_seed{sd}", 1, "dropout", seed=sd, chan_dropout=0.5, channels=dict(type="all"))
    # 2 random n within each array
    for (n, arr, i), ch in random_subsets().items():
        add(f"rand{n}{arr}{i}", 2, "random_n", seed=s0, channels=dict(type="list", array=arr, n=n, i=i, list=ch))
    # 3 best n: greedy + swap (waits for Fork A's selection.json)
    for n in NS:
        for arr in ("A", "B"):
            for sd in seeds[:2]:
                add(f"greedy{n}{arr}_seed{sd}", 3, "greedy_swap", seed=sd,
                    channels=dict(type="select", method="greedy_swap", array=arr, n=n),
                    requires=[[SEL_REL, "sets.greedy_swap"]])
    # 4 training-data amount: the last k of the 5 training blocks (#40)
    for k in (1, 2, 3, 4):
        for sd in seeds:
            add(f"data{k}_seed{sd}", 4, "data_amount", seed=sd, channels=dict(type="all"), k_blocks=k,
                blocks=[list(F.TRAIN_BLOCKS[-k:]), list(F.VAL_BLOCKS), list(F.TEST_BLOCKS)])
    # 5 day 1750 (two-hand trials included)
    for which in ("all", "A", "B"):
        for sd in seeds:
            add(f"d1750_{which}_seed{sd}", 5, "day1750", seed=sd, day=1750, blocks=[list(b) for b in ol_split(1750)],
                channels=dict(type="all") if which == "all" else dict(type="array", array=which))
    # 6 group-lasso sets (joint benchmark); 7 LSTM-ablation top-n (cross-check)
    for n in NS:
        for arr in ("A", "B"):
            add(f"glasso{n}{arr}_seed{s0}", 6, "group_lasso", seed=s0,
                channels=dict(type="select", method="group_lasso", array=arr, n=n),
                requires=[[SEL_REL, "sets.group_lasso"]])
    for n in (8, 16):
        for arr in ("A", "B"):
            add(f"ablat{n}{arr}_seed{s0}", 7, "ablation", seed=s0,
                channels=dict(type="select", method="ablation", array=arr, n=n),
                requires=[[SEL_REL, "sets.ablation"]])
    # 8 day 1867 (one-hand)
    for which in ("all", "A", "B"):
        for sd in seeds:
            add(f"d1867_{which}_seed{sd}", 8, "day1867", seed=sd, day=1867, blocks=[list(b) for b in ol_split(1867)],
                channels=dict(type="all") if which == "all" else dict(type="array", array=which))
    return jobs


def resolve_channels(spec: dict):
    """Global channel indices for a job (None = all 192)."""
    t = spec["type"]
    if t == "all":
        return None
    if t == "array":
        return ARRAYS[spec["array"]]
    if t == "list":
        return list(spec["list"])
    if t == "select":
        sel = json.loads(SELECTION.read_text())
        return sorted(int(c) for c in sel["sets"][spec["method"]][spec["array"]][str(spec["n"])])
    raise ValueError(spec)


class DataCache:
    """Per-worker cache of loaded data + its 2-8-s augmented batch function (all 192 channels).

    Channel subsets are views: ``z`` sliced from the full z-scored data, and batches sliced from
    the same 2000 full-channel snippets, so every subset of a day sees identical training snippets.
    Holds at most ``maxsize`` entries (each about 0.6-0.8 GB of RAM).
    """

    def __init__(self, maxsize: int = 2):
        self.maxsize, self.items = maxsize, OrderedDict()

    def get(self, day: int, blocks):
        key = (day, None if blocks is None else tuple(tuple(b) for b in blocks))
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        while len(self.items) >= self.maxsize:
            self.items.popitem(last=False)
        data = F.Data("clip", day=day, blocks=key[1])
        entry = dict(data=data, fn=data.aug_batch_fn(seed=0, clip_bins=XL.CLIP_BINS["augL"]),
                     pdata=PX.build(data.session) if day == F.DAY else None)
        self.items[key] = entry
        return entry


def subset_view(data, ch):
    if ch is None:
        return data
    v = copy.copy(data)
    v.channels = np.asarray(ch, int)
    v.z = np.ascontiguousarray(data.z_all[:, v.channels])
    return v


def sliced_fn(fn, ch):
    if ch is None:
        return fn
    idx = np.asarray(ch, int)

    def g(rng, B):
        X, V, Cx, d = fn(rng, B)
        return np.ascontiguousarray(X[:, :, idx]), V, Cx, d

    return g


# ----------------------------------------------------------------------------- one job
def run_job(job: dict, worker: int, cache: DataCache) -> None:
    trials = QUEUE.trials_path(worker)
    if "runner" in job:  # jobs owned by another script (e.g. Phase 11: "ext_sbp:run_job")
        mod, fn = job["runner"].split(":")
        getattr(importlib.import_module(mod), fn)(job, worker, trials)
        return
    hp_d, _ = best_setting()
    hp = R.HParams(**{**hp_d, "chan_dropout": float(job.get("chan_dropout", 0.0))})
    day = int(job.get("day", F.DAY))
    entry = cache.get(day, job.get("blocks"))
    ch = resolve_channels(job["channels"])
    if job["channels"]["type"] == "select" and job["channels"]["method"] != "greedy_swap":
        g = resolve_channels(dict(job["channels"], method="greedy_swap"))
        if g == ch:  # identical to the greedy set: no need to train it again
            append_jsonl(trials, dict(run_id=job["run_id"], tag="dup", group=job["group"], same_as="greedy_swap",
                                      channels=ch, worker=worker, train_seconds=0.0))
            print(f"[{job['run_id']}] identical to the greedy+swap set; not trained")
            return
    data = subset_view(entry["data"], ch)
    n_in = 192 if ch is None else len(ch)
    assert data.z.shape[1] == n_in, (data.z.shape, n_in)
    pdata = entry["pdata"]
    post = (lambda m: dict(proxy=PX.score_model(m, data.z, pdata))) if pdata is not None else None
    extra = dict(group=job["group"], kind_ext="augL", day=day, channel_spec={k: v for k, v in job["channels"].items()
                                                                           if k != "list"},
                 channels=ch, n_ch=n_in, worker=worker)
    for k in ("k_blocks", "blocks"):
        if k in job:
            extra[k] = job[k]
    F.train_one(job["run_id"], kind="aug", hp=hp, data=data, batch_fns={"aug": sliced_fn(entry["fn"], ch)},
                seed=int(job["seed"]), tag=job["group"], cell="lstm", trials=trials, extra=extra, post_score=post)


# ----------------------------------------------------------------------------- timing
def write_timing() -> dict:
    """Measured throughput and the queue's ETA vs the planned estimate (updated after every run)."""
    q = QUEUE.read()
    meta = q["meta"]
    recs = [r for r in QUEUE.records() if "t_start" in r and r.get("tag") not in ("dup", "failed")]
    t0 = meta.get("t_drive", time.time())
    recent = [r for r in recs if r["t_start"] >= t0 - 1]
    now = time.time()
    rem = QUEUE.remaining()
    by_group = {}
    for r in recs:
        by_group.setdefault(r["group"], []).append(r["t_end"] - r["t_start"])
    thr = len(recent) / (now - t0) if recent and now > t0 else None  # runs per second, all workers
    out = dict(updated=now, workers=meta.get("workers"), runs_done=len(recs), runs_done_this_drive=len(recent),
               runs_remaining=len(rem), remaining_by_priority={str(p): sum(j["priority"] == p for j in rem)
                                                               for p in sorted({j["priority"] for j in rem})},
               mean_wall_s_per_run=float(np.mean([r["t_end"] - r["t_start"] for r in recent])) if recent else None,
               mean_train_s_per_run=float(np.mean([r["train_seconds"] for r in recent])) if recent else None,
               throughput_runs_per_hour=thr * 3600 if thr else None,
               eta_hours=(len(rem) / thr / 3600) if thr else None,
               elapsed_hours=(now - t0) / 3600,
               planned_queue_hours=meta.get("planned_queue_hours"),
               wall_s_by_group={g: float(np.mean(v)) for g, v in by_group.items()})
    atomic_write_json(OUT / "timing.json", out)
    return out


# ----------------------------------------------------------------------------- stages
def stage_worker(worker: int, max_wait_s: float = 4 * 3600) -> None:
    cache = DataCache()
    waited = 0.0
    print(f"worker {worker} started (pid {__import__('os').getpid()})", flush=True)
    while True:
        job, status = QUEUE.next_job(worker)
        if job is None:
            if status == "empty":
                break
            # blocked: worth waiting only if some unclaimed job waits on a prerequisite
            claimed = QUEUE.claimed()
            if not [j for j in QUEUE.remaining() if j["run_id"] not in claimed] or waited > max_wait_s:
                break
            time.sleep(60); waited += 60
            continue
        waited = 0.0
        try:
            run_job(job, worker, cache)
        except Exception as e:  # noqa: BLE001 - log and move on; the summary lists failures
            traceback.print_exc()
            append_jsonl(QUEUE.trials_path(worker), dict(run_id=job["run_id"], tag="failed", group=job.get("group"),
                                                         error=repr(e), worker=worker, train_seconds=0.0))
        finally:
            QUEUE.release(job["run_id"])
            torch.cuda.empty_cache()
        t = write_timing()
        if t["eta_hours"] is not None:
            print(f"[timing] done {t['runs_done_this_drive']} this drive, {t['runs_remaining']} left, "
                  f"{t['throughput_runs_per_hour']:.1f} runs/h, ETA {t['eta_hours']:.2f} h "
                  f"(planned queue {t['planned_queue_hours']})", flush=True)
    print(f"worker {worker} finished", flush=True)


def stage_smoke_one(n: int, k: int, steps: int = 150) -> None:
    """One process of the concurrency test: load data, wait for the others, time ``steps`` steps."""
    d = OUT / "smoke" / f"n{n}"
    d.mkdir(parents=True, exist_ok=True)
    data = F.Data("clip")
    fn = data.aug_batch_fn(seed=0, clip_bins=XL.CLIP_BINS["augL"])
    (d / f"ready{k}").write_text("1")
    t_wait = time.time()
    while len(list(d.glob("ready*"))) < n and time.time() - t_wait < 600:
        time.sleep(0.5)
    hp_d, _ = best_setting()
    hp = R.HParams(**{**hp_d, "steps": steps})
    torch.manual_seed(k)
    m = R.RNNDecoder(n_in=192, hidden=hp.hidden, cell="lstm")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    info = R.train(m, fn, hp, seed=k, log=lambda *a: None)
    import psutil
    out = dict(n=n, k=k, sec_per_step=info["seconds"] / steps, batch=hp.batch,
               gpu_peak_mb=torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == "cuda" else None,
               rss_mb=psutil.Process().memory_info().rss / 1e6)
    atomic_write_json(d / f"result{k}.json", out)


def stage_smoke(max_n: int = 3) -> dict:
    """Throughput with 1..max_n concurrent training processes; picks N (register #39 timing note)."""
    import psutil
    import shutil
    res = {}
    for n in range(1, max_n + 1):
        avail = psutil.virtual_memory().available / 1e9
        if n > 1 and avail < 2.5:
            print(f"smoke n={n}: skipped, only {avail:.1f} GB RAM available")
            break
        d = OUT / "smoke" / f"n{n}"
        shutil.rmtree(d, ignore_errors=True)
        procs = [subprocess.Popen([sys.executable, "-u", __file__, "--stage", "smoke_one", "--n", str(n),
                                   "--k", str(k)]) for k in range(n)]
        min_avail = avail
        while any(p.poll() is None for p in procs):
            min_avail = min(min_avail, psutil.virtual_memory().available / 1e9)
            time.sleep(1)
        rows = [json.loads(p.read_text()) for p in sorted(d.glob("result*.json"))]
        if len(rows) < n:
            print(f"smoke n={n}: {n - len(rows)} process(es) failed; stopping here")
            break
        sps = float(np.mean([r["sec_per_step"] for r in rows]))
        res[n] = dict(sec_per_step=sps, steps_per_s_total=n / sps, gpu_peak_mb=[r["gpu_peak_mb"] for r in rows],
                      rss_mb=[r["rss_mb"] for r in rows], min_ram_available_gb=min_avail)
        print(f"smoke n={n}: {sps * 1000:.0f} ms/step per process -> {n / sps:.1f} steps/s total; "
              f"GPU peak {max(r['gpu_peak_mb'] or 0 for r in rows):.0f} MB/proc; RSS "
              f"{max(r['rss_mb'] for r in rows):.0f} MB/proc; min RAM available {min_avail:.1f} GB", flush=True)
    # pick the largest N that still gains >= 10% throughput over N-1
    best_n = 1
    for n in sorted(res):
        if n > 1 and res[n]["steps_per_s_total"] >= 1.10 * res[best_n]["steps_per_s_total"]:
            best_n = n
    hp_d, _ = best_setting()
    run_s = hp_d["steps"] * res[best_n]["sec_per_step"] + 30  # + data/snippet/scoring overhead (estimate)
    out = dict(results={str(k): v for k, v in res.items()}, chosen_workers=best_n, est_wall_s_per_run=run_s,
               steps_per_run=hp_d["steps"])
    atomic_write_json(OUT / "smoke.json", out)
    print(f"chosen workers: {best_n}; estimated {run_s:.0f} s per run per worker")
    return out


def stage_jobs(workers: int) -> None:
    """Write jobs.json (keeps jobs owned by other scripts, e.g. Phase 11 runners, if present)."""
    jobs = build_jobs()
    old = QUEUE.read()
    ids = {j["run_id"] for j in jobs}
    jobs += [j for j in old["jobs"] if "runner" in j and j["run_id"] not in ids]
    smoke = json.loads((OUT / "smoke.json").read_text())
    done = QUEUE.done()
    n_left = sum(j["run_id"] not in done for j in jobs)
    planned = n_left * smoke["est_wall_s_per_run"] / workers / 3600
    meta = dict(old["meta"], workers=workers, t_drive=time.time(), planned_queue_hours=planned,
                est_wall_s_per_run=smoke["est_wall_s_per_run"])
    QUEUE.write_jobs(jobs, meta)
    print(f"{len(jobs)} jobs ({n_left} to run); planned {planned:.2f} h with {workers} workers")


def stage_drive(workers: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True); F.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUT / "smoke.json").exists():
        stage_smoke()
    workers = workers or json.loads((OUT / "smoke.json").read_text())["chosen_workers"]
    print(f"cleared {QUEUE.clear_claims()} stale claims")
    stage_jobs(workers)
    procs = []
    for k in range(workers):
        log = open(OUT / f"worker{k}.log", "a")
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, "--stage", "worker", "--worker", str(k)],
                                      stdout=log, stderr=subprocess.STDOUT))
        time.sleep(30)  # stagger data loading (RAM peaks)
    for p in procs:
        p.wait()
    t = write_timing()
    print(f"queue finished: {t['runs_done_this_drive']} runs this drive in {t['elapsed_hours']:.2f} h "
          f"(planned {t['planned_queue_hours']:.2f} h)", flush=True)
    stage_mute()
    print("drive done", flush=True)


# ----------------------------------------------------------------------------- muting evaluation
@torch.no_grad()
def decode_batch(model, Xb: np.ndarray, chunk: int = 5000) -> np.ndarray:
    """``rnn.decode`` for a batch of recordings Xb (B, T, N), each from a zero state."""
    model.eval()
    h, outs = None, []
    d = torch.zeros(len(Xb), dtype=torch.long, device=DEVICE)
    for i in range(0, Xb.shape[1], chunk):
        x = torch.as_tensor(Xb[:, i:i + chunk], dtype=torch.float32, device=DEVICE)
        v, _, h = model(x, d, h)
        outs.append(v.cpu().numpy())
    return np.concatenate(outs, axis=1)


FRACS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
N_MASKS = 20
MODES = ("zero", "dead", "noisy")


def stage_mute() -> dict:
    """Test-block r with a random fraction f of channels muted (no retraining), register #42.

    Modes: "zero"  detected bad channel set to 0 (its mean in z-units);
           "dead"  undetected dead channel: the z-value of 0 counts under the training blocks'
                   statistics (mean and SD of the raw counts over blocks 1-5);
           "noisy" undetected noisy channel: N(0, 1) white noise.
    The same masks (``default_rng(7000 + 100 f)``) are used for every model, so comparisons are paired.
    """
    data = F.Data("clip")
    pdata = PX.build(data.session)
    (a, b), = data.seg["test"]
    Xt, Vt = data.z[a:b], data.vel[a:b]
    tr = np.concatenate([data.session.tx[x:y] for x, y in data.seg["train"]]).astype(np.float64)
    dead_val = (-tr.mean(0) / np.where(tr.std(0) > 0, tr.std(0), 1.0)).astype(np.float32)
    models = [("normal", r) for r in XL.seed_records("lstm", "augL")]
    models += [("dropout", r) for r in QUEUE.records() if r.get("group") == "dropout" and "test" in r]
    out = {}
    for kind, rec in models:
        m = R.load_from_record(rec, 192, F.CKPT_DIR)
        v0 = decode_batch(m, Xt[None])[0]
        r0 = pearson_per_dim(Vt, v0)
        assert abs(np.mean(r0[:2]) - rec["test"]["r_R"]) < 2e-3 and abs(np.mean(r0[2:]) - rec["test"]["r_L"]) < 2e-3, \
            (rec["run_id"], r0, rec["test"])
        row = dict(kind=kind, f={}, proxy={})
        for f in FRACS:
            rng = np.random.default_rng(7000 + int(round(100 * f)))
            masks = np.zeros((N_MASKS, 192), bool)
            for i in range(N_MASKS):
                masks[i, rng.choice(192, int(round(f * 192)), replace=False)] = True
            noise_seeds = rng.integers(0, 2**31, N_MASKS)
            row["f"][str(f)] = {}
            for mode in MODES:
                rs = []
                for i0 in range(0, N_MASKS, 5):  # 5 masks per batch (RAM)
                    mk = masks[i0:i0 + 5, None, :]
                    Xb = np.repeat(Xt[None], len(mk), 0)
                    if f > 0 and mode == "zero":
                        Xb = np.where(mk, np.float32(0), Xb)
                    elif f > 0 and mode == "dead":
                        Xb = np.where(mk, dead_val[None, None, :], Xb)
                    elif f > 0:  # noisy: same noise per mask index for every model
                        nz = np.stack([np.random.default_rng(int(s)).standard_normal(Xt.shape).astype(np.float32)
                                       for s in noise_seeds[i0:i0 + 5]])
                        Xb = np.where(mk, nz, Xb)
                    vh = decode_batch(m, Xb)
                    rs += [pearson_per_dim(Vt, vh[i]) for i in range(len(mk))]
                rs = np.array(rs)
                rR, rL = rs[:, :2].mean(1), rs[:, 2:].mean(1)
                row["f"][str(f)][mode] = dict(r_R=float(rR.mean()), r_L=float(rL.mean()), r_R_sd=float(rR.std()),
                                              r_L_sd=float(rL.std()))
                if f == 0:
                    break  # all modes are identical with nothing muted
            if f in (0.0, 0.3, 0.5):
                prs = []
                for i in range(5 if f > 0 else 1):
                    Xf = data.z.copy()
                    Xf[:, masks[i]] = 0.0
                    prs.append(PX.score_model(m, Xf, pdata)["all"]["r_mean"])
                row["proxy"][str(f)] = prs
        out[rec["run_id"]] = row
        print(f"mute {rec['run_id']} ({kind}): " + "  ".join(
            f"f={f}: {row['f'][str(f)]['zero']['r_R']:.2f}/{row['f'][str(f)]['zero']['r_L']:.2f}" for f in FRACS), flush=True)
        del m; torch.cuda.empty_cache()
    atomic_write_json(OUT / "mute.json", dict(fracs=FRACS, n_masks=N_MASKS, modes=MODES, models=out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "smoke_one", "jobs", "drive", "worker", "mute", "plot"], default="drive")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); F.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if a.stage == "smoke":
        stage_smoke()
    elif a.stage == "smoke_one":
        stage_smoke_one(a.n, a.k)
    elif a.stage == "jobs":
        stage_jobs(a.workers or json.loads((OUT / "smoke.json").read_text())["chosen_workers"])
    elif a.stage == "drive":
        stage_drive(a.workers)
    elif a.stage == "worker":
        stage_worker(a.worker)
    elif a.stage == "mute":
        stage_mute()
    elif a.stage == "plot":
        import ext_feasibility_plot
        ext_feasibility_plot.summarize_and_plot()


if __name__ == "__main__":
    main()
