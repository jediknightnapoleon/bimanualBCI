"""R^2 rescoring: every velocity-decoding result of the replication and Phases 8-11, as R^2.

Why
---
Every decoding number in this repository is Pearson r, because that is what Deo et al. report.
Many groups report the coefficient of determination instead, R^2 = 1 - SSE/SST. The two differ:
r ignores the decoder's gain and offset, R^2 does not (a decoded velocity at the right direction
but half the speed has r = 1 and R^2 = 0.75; at twice the speed, R^2 = 0). R^2 is therefore not
r squared and cannot be derived from the saved r values; it needs the predictions.

Definition (register #47)
-------------------------
* R^2 per velocity dimension, ``decoders.r2_per_dim``: 1 - SSE/SST, SST around that dimension's
  mean over the evaluated bins; pooled over all held-out bins, exactly where r was computed.
* Per hand = mean of the x and y R^2 (as r was averaged).
* DEVIATION: only the reported metric changes. Every model, split, setting and channel set is the
  one chosen with r (ridge alpha, random-search winner, greedy channel order), so the R^2 numbers
  describe the same decoders as the r numbers. Where choosing by R^2 could change a decoder
  (Fig 3 ridge alpha), the R^2-chosen alternative is reported alongside.

Guard: every re-run must reproduce its logged r (within 2e-3 for the RNNs, as in the muting
analysis; to 1e-9 for the cached Fig 3 predictions) before its R^2 is used.

Stages (each writes one file in results/r2/; ``all`` runs them in order and resumes ``rnn``):
  fig3     day-1883 single-bin decoders from the cached out-of-fold predictions (CPU, seconds)
  rnn      every trained RNN/LSTM record (Phases 4, 8, 9, 10): val + test R^2 (GPU, ~10 min)
  zero     Phase 9: 192-channel models with one array zeroed at test (GPU, ~1 min)
  mute     Phase 10: muted-channel evaluation, the same masks as ext_feasibility (GPU, ~10 min)
  sbp      Phase 11 TX vs SBP: scripts/ext_sbp.py --metric r2 (CPU, ~7 min)
  summary  the tables of the report as R^2 -> results/r2/summary.json

Inputs : Data/ (days 1750, 1867, 1869, 1883; Data/ext for the SBP data), results/checkpoints/,
         results/cache/fig3_direction_*.npz, the trials/summary files of each phase.
Outputs: results/r2/{fig3.json, rnn.jsonl, zero_out.json, mute.json, sbp.json, summary.json}
Run    : pixi run rescore-r2 [--stage STAGE]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_arrays as XA  # noqa: E402
import ext_feasibility as E  # noqa: E402
import ext_lstm as XL  # noqa: E402
import fig3_decoding as FD  # noqa: E402
import fig4_rnn as F  # noqa: E402

from bimanual import decoders as Dc  # noqa: E402
from bimanual import rnn as R  # noqa: E402
from bimanual.cache import RESULTS_DIR, append_jsonl, atomic_write_json, read_jsonl  # noqa: E402
from bimanual.decoders import pearson_per_dim, r2_per_dim  # noqa: E402
from bimanual.io import load_session, zscore_by_block  # noqa: E402
from bimanual.stats import bootstrap_ci  # noqa: E402

OUT = RESULTS_DIR / "r2"
RNN_JSONL = OUT / "rnn.jsonl"
R_TOL = 2e-3
N_BOOT = 1000


def _hands(v):
    v = np.asarray(v, float)
    return float(v[:2].mean()), float(v[2:].mean())


# ----------------------------------------------------------------------------- Fig 3 (day 1883)
def stage_fig3():
    FD.TARGET = "direction"
    s = load_session(FD.DAY)
    z = zscore_by_block(s.tx, s.block)
    ol = (s.trials.kind == "OL").to_numpy()
    mt_all = s.trials.move_type.to_numpy()
    logged = json.loads((RESULTS_DIR / "fig3" / "decoding_direction.json").read_text())["results"]
    rng = np.random.default_rng(0)
    sets = {"uni": FD.build(s, z, ol & np.isin(mt_all, (1, 2))), "bi": FD.build(s, z, ol & (mt_all == 3))}
    out, preds = {}, {}
    for cond in ("uni_ridge", "uni_ffn", "uni_ridge_nolat", "uni_ffn_nolat", "bi_ridge", "bi_ffn"):
        X, Y, g, mt, _ = sets[cond[:cond.index("_")]]
        d = np.load(RESULTS_DIR / "cache" / f"fig3_direction_{cond}.npz")
        p = d["pred"]
        r = pearson_per_dim(Y, p)
        assert np.allclose(r, logged[cond]["r"], atol=1e-9), (cond, r, logged[cond]["r"])
        r2 = r2_per_dim(Y, p)
        rows = [np.where(g == t)[0] for t in np.unique(g)]
        boots = np.empty((N_BOOT, 4))
        for i in range(N_BOOT):
            pick = np.concatenate([rows[j] for j in rng.integers(0, len(rows), len(rows))])
            boots[i] = r2_per_dim(Y[pick], p[pick])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        preds[cond] = p
        out[cond] = dict(r=r.tolist(), r2=r2.tolist(), r2_lo=lo.tolist(), r2_hi=hi.tolist(),
                         alpha=float(d["alpha"]), r2_R=_hands(r2)[0], r2_L=_hands(r2)[1])
        print(f"fig3 {cond:16s} R2 " + " ".join(f"{v:.3f}" for v in r2) + "   r " + " ".join(f"{v:.3f}" for v in r))

    # paired bootstrap over trials (same comparisons as fig3_decoding.paired_tests), in R^2
    X, Y, g, mt, _ = sets["uni"]
    rows = [np.where(g == t)[0] for t in np.unique(g)]
    tests = {}
    for name, (a, b) in {"FFN - Ridge": ("uni_ffn", "uni_ridge"), "FFN - FFN_nolat": ("uni_ffn", "uni_ffn_nolat"),
                         "Ridge - Ridge_nolat": ("uni_ridge", "uni_ridge_nolat")}.items():
        for scope, dims in (("all", slice(0, 4)), ("LH", slice(2, 4))):
            d0 = (r2_per_dim(Y, preds[a]) - r2_per_dim(Y, preds[b]))[dims].mean()
            bs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                pick = np.concatenate([rows[j] for j in rng.integers(0, len(rows), len(rows))])
                bs[i] = (r2_per_dim(Y[pick], preds[a][pick]) - r2_per_dim(Y[pick], preds[b][pick]))[dims].mean()
            tests[f"{name} ({scope})"] = dict(diff=float(d0), lo=float(np.percentile(bs, 2.5)),
                                              hi=float(np.percentile(bs, 97.5)),
                                              p_two_sided=float(2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))
    out["paired_tests_r2"] = tests

    # Ridge alpha: the paper sweeps it to maximise r. Shrinkage lowers R^2 without changing r much,
    # so also report each ridge decoder with alpha chosen by CV R^2 (sensitivity, not the main number).
    lat_u = np.where(sets["uni"][3] == 1, 1.0, -1.0)
    alt = {}
    for cond, key, tf in (("uni_ridge", "uni", None), ("uni_ridge_nolat", "uni", FD.laterality_transform(lat_u)),
                          ("bi_ridge", "bi", None)):
        X, Y, g, mt, _ = sets[key]
        folds = Dc.stratified_trial_folds(mt, 5)
        sweep = []
        for a in Dc.RIDGE_ALPHAS:
            p = Dc.cross_validate(X, Y, g, folds, "ridge", alpha=a, transform=tf)
            sweep.append(dict(alpha=float(a), r_mean=float(np.mean(pearson_per_dim(Y, p))),
                              r2=r2_per_dim(Y, p).tolist(), r2_mean=float(np.mean(r2_per_dim(Y, p)))))
        best = max(sweep, key=lambda q: q["r2_mean"])
        alt[cond] = dict(alpha_by_r=out[cond]["alpha"], alpha_by_r2=best["alpha"], r2_at_alpha_by_r2=best["r2"],
                         sweep=sweep)
        print(f"fig3 {cond}: alpha by r {out[cond]['alpha']:g}, by R2 {best['alpha']:g} -> R2 "
              + " ".join(f"{v:.3f}" for v in best["r2"]))
    out["ridge_alpha_by_r2"] = alt
    atomic_write_json(OUT / "fig3.json", dict(day=FD.DAY, target="direction", dims=list(FD.DIMS), results=out))


# ----------------------------------------------------------------------------- RNN records
def rnn_records():
    """[(source, record)] of every trained model with a checkpoint, across the four phases."""
    src = [("fig4", read_jsonl(F.TRIALS)), ("ext_lstm", read_jsonl(XL.TRIALS)),
           ("ext_arrays", read_jsonl(XA.TRIALS)), ("ext_feas", E.QUEUE.records())]
    out = [(s, r) for s, recs in src for r in recs if "test" in r]
    ids = [r["run_id"] for _, r in out]
    assert len(ids) == len(set(ids)), "run_ids must be unique across phases"
    return out


class DataCache:
    def __init__(self, maxsize=3):
        self.maxsize, self.items = maxsize, OrderedDict()

    def get(self, day, sat, blocks):
        key = (day, sat, json.dumps(blocks))
        if key not in self.items:
            while len(self.items) >= self.maxsize:
                self.items.popitem(last=False)
            self.items[key] = F.Data(sat, day=day, blocks=None if blocks is None else tuple(tuple(b) for b in blocks))
        self.items.move_to_end(key)
        return self.items[key]


def _data_key(rec):
    return (int(rec.get("day", F.DAY)), rec.get("sat_mode", "clip"), rec.get("blocks"))


def score_r2(model, data, split):
    """r and R^2 per dim on a split (each block decoded from a zero state), plus two-hand-trial bins."""
    V, vh, mt = F.decode_split(model, data, split)
    r, r2 = pearson_per_dim(V, vh), r2_per_dim(V, vh)
    o = dict(r=r.tolist(), r2=r2.tolist(), r_R=_hands(r)[0], r_L=_hands(r)[1], r2_R=_hands(r2)[0], r2_L=_hands(r2)[1])
    bi = mt == 3
    if bi.sum() > 500:  # the same rule as fig4_rnn.score
        rb, r2b = pearson_per_dim(V[bi], vh[bi]), r2_per_dim(V[bi], vh[bi])
        o.update(r_bi_R=_hands(rb)[0], r_bi_L=_hands(rb)[1], r2_bi_R=_hands(r2b)[0], r2_bi_L=_hands(r2b)[1])
    return o


def stage_rnn():
    import torch

    done = {r["run_id"] for r in read_jsonl(RNN_JSONL)}
    todo = sorted([(s, r) for s, r in rnn_records() if r["run_id"] not in done], key=lambda sr: str(_data_key(sr[1])))
    print(f"rnn: {len(todo)} models to rescore ({len(done)} done)")
    cache = DataCache()
    for i, (src, rec) in enumerate(todo):
        day, sat, blocks = _data_key(rec)
        data = E.subset_view(cache.get(day, sat, blocks), rec.get("channels"))
        n_in = data.z.shape[1]
        assert n_in == rec.get("n_in", 192), (rec["run_id"], n_in, rec.get("n_in"))
        m = R.load_from_record(rec, n_in, F.CKPT_DIR)
        row = dict(source=src, run_id=rec["run_id"])
        for split in ("val", "test"):
            sc = score_r2(m, data, split)
            for h in ("R", "L"):
                assert abs(sc[f"r_{h}"] - rec[split][f"r_{h}"]) < R_TOL, (rec["run_id"], split, sc, rec[split])
            if "r_bi_R" in rec[split]:
                assert abs(sc["r_bi_R"] - rec[split]["r_bi_R"]) < R_TOL, (rec["run_id"], "bi")
            row[split] = sc
        append_jsonl(RNN_JSONL, row)
        t = row["test"]
        print(f"[{i + 1}/{len(todo)}] {src:10s} {rec['run_id']:28s} test r {t['r_R']:.3f}/{t['r_L']:.3f}  "
              f"R2 {t['r2_R']:.3f}/{t['r2_L']:.3f}", flush=True)
        del m
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- Phase 9: one array zeroed
def stage_zero():
    data = F.Data("clip")
    logged = json.loads((XA.OUT / "zero_out.json").read_text())
    (a, b), = data.seg["test"]
    V = data.vel[a:b]
    out = {}
    for cell in ("gru", "lstm"):
        for kind in ("raw", "augL"):
            for rec in XL.seed_records(cell, kind):
                m = R.load_from_record(rec, 192, F.CKPT_DIR)
                row = {}
                for drop in ("none", "A", "B"):
                    z = data.z[a:b].copy()
                    if drop != "none":
                        z[:, XA.ARRAYS[drop]] = 0.0
                    vh = R.decode(m, z)[0]
                    r, r2 = pearson_per_dim(V, vh), r2_per_dim(V, vh)
                    assert abs(_hands(r)[0] - logged[rec["run_id"]][drop]["r_R"]) < R_TOL, (rec["run_id"], drop)
                    row[drop] = dict(r_R=_hands(r)[0], r_L=_hands(r)[1], r2_R=_hands(r2)[0], r2_L=_hands(r2)[1])
                out[rec["run_id"]] = dict(cell=cell, kind=kind, **row)
                print(f"zero {rec['run_id']}: " + "  ".join(
                    f"{d} R2 {row[d]['r2_R']:.2f}/{row[d]['r2_L']:.2f}" for d in row))
    atomic_write_json(OUT / "zero_out.json", out)


# ----------------------------------------------------------------------------- Phase 10: muting
def stage_mute():
    """``ext_feasibility.stage_mute`` with R^2 added (same masks and noise; no proxy)."""
    import torch

    data = F.Data("clip")
    (a, b), = data.seg["test"]
    Xt, Vt = data.z[a:b], data.vel[a:b]
    tr = np.concatenate([data.session.tx[x:y] for x, y in data.seg["train"]]).astype(np.float64)
    dead_val = (-tr.mean(0) / np.where(tr.std(0) > 0, tr.std(0), 1.0)).astype(np.float32)
    logged = json.loads((E.OUT / "mute.json").read_text())["models"]
    models = [("normal", r) for r in XL.seed_records("lstm", "augL")]
    models += [("dropout", r) for r in E.QUEUE.records() if r.get("group") == "dropout" and "test" in r]
    out = {}
    for kind, rec in models:
        m = R.load_from_record(rec, 192, F.CKPT_DIR)
        row = dict(kind=kind, f={})
        for f in E.FRACS:
            rng = np.random.default_rng(7000 + int(round(100 * f)))
            masks = np.zeros((E.N_MASKS, 192), bool)
            for i in range(E.N_MASKS):
                masks[i, rng.choice(192, int(round(f * 192)), replace=False)] = True
            noise_seeds = rng.integers(0, 2**31, E.N_MASKS)
            row["f"][str(f)] = {}
            for mode in E.MODES:
                rs, r2s = [], []
                for i0 in range(0, E.N_MASKS, 5):
                    mk = masks[i0:i0 + 5, None, :]
                    Xb = np.repeat(Xt[None], len(mk), 0)
                    if f > 0 and mode == "zero":
                        Xb = np.where(mk, np.float32(0), Xb)
                    elif f > 0 and mode == "dead":
                        Xb = np.where(mk, dead_val[None, None, :], Xb)
                    elif f > 0:
                        nz = np.stack([np.random.default_rng(int(s)).standard_normal(Xt.shape).astype(np.float32)
                                       for s in noise_seeds[i0:i0 + 5]])
                        Xb = np.where(mk, nz, Xb)
                    vh = E.decode_batch(m, Xb)
                    rs += [pearson_per_dim(Vt, vh[i]) for i in range(len(mk))]
                    r2s += [r2_per_dim(Vt, vh[i]) for i in range(len(mk))]
                rs, r2s = np.array(rs), np.array(r2s)
                ent = dict(r_R=float(rs[:, :2].mean()), r_L=float(rs[:, 2:].mean()),
                           r2_R=float(r2s[:, :2].mean()), r2_L=float(r2s[:, 2:].mean()),
                           r2_R_sd=float(r2s[:, :2].mean(1).std()), r2_L_sd=float(r2s[:, 2:].mean(1).std()))
                lg = logged[rec["run_id"]]["f"][str(f)][mode]
                assert abs(ent["r_R"] - lg["r_R"]) < R_TOL and abs(ent["r_L"] - lg["r_L"]) < R_TOL, (rec["run_id"], f, mode)
                row["f"][str(f)][mode] = ent
                if f == 0:
                    break
        out[rec["run_id"]] = row
        print(f"mute {rec['run_id']} ({kind}) zero-mode R2: " + "  ".join(
            f"f={f}: {row['f'][str(f)]['zero']['r2_R']:.2f}/{row['f'][str(f)]['zero']['r2_L']:.2f}" for f in E.FRACS),
            flush=True)
        del m
        torch.cuda.empty_cache()
    atomic_write_json(OUT / "mute.json", dict(fracs=E.FRACS, n_masks=E.N_MASKS, modes=E.MODES, models=out))


# ----------------------------------------------------------------------------- Phase 11: SBP
def stage_sbp():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    script = Path(__file__).resolve().parent / "ext_sbp.py"
    subprocess.run([sys.executable, "-I", str(script), "--metric", "r2", "--items", "123",
                    "--out", str(OUT / "sbp.json")], check=True, env=env)


# ----------------------------------------------------------------------------- summary
def _mean_hands(rows, split="test", m="r2"):
    return dict(R=float(np.mean([r[split][f"{m}_R"] for r in rows])), L=float(np.mean([r[split][f"{m}_L"] for r in rows])),
                n=len(rows), run_ids=[r["run_id"] for r in rows])


def _both(rows, split="test"):
    """R^2 and r means per hand for a list of rescored rows."""
    return dict(r2=_mean_hands(rows, split, "r2"), r=_mean_hands(rows, split, "r"))


def stage_summary():
    rs = {r["run_id"]: r for r in read_jsonl(RNN_JSONL)}
    S = {}

    def rows(recs):
        miss = [r["run_id"] for r in recs if r["run_id"] not in rs]
        assert not miss, f"not rescored: {miss}"
        return [rs[r["run_id"]] for r in recs]

    # Fig 3
    S["fig3"] = json.loads((OUT / "fig3.json").read_text())["results"]

    # Fig 4 (GRU, day 1869)
    f4 = read_jsonl(F.TRIALS)
    search = [r for r in f4 if r["tag"] == "search"]
    best = {k: max((r for r in search if r["kind"] == k), key=lambda r: r["val"]["r_mean"]) for k in ("raw", "aug")}
    seeds = {k: [best[k]] + [r for r in f4 if r["tag"] == "seed" and r["kind"] == k] for k in ("raw", "aug")}
    S["fig4"] = dict(raw=_both(rows(seeds["raw"])), aug_200_800ms=_both(rows(seeds["aug"])),
                     aug_2_8s=_both(rows([r for r in f4 if r["tag"] == "longclip"])),
                     sat_hold={r["kind"]: _both(rows([r])) for r in f4 if r["tag"] == "sat_alt"})

    # Phase 8: LSTM vs GRU (paired validation over the search settings; best-setting test, 3 seeds)
    from scipy.stats import wilcoxon
    g_s, l_s = XL.gru_baseline_search(), XL.lstm_search()
    paired = {}
    for kind in XL.KINDS:
        ks = sorted(k for (kk, k) in l_s if kk == kind and (kind, k) in g_s)
        g = np.array([np.mean(rs[g_s[(kind, k)]["run_id"]]["val"]["r2"]) for k in ks])
        l_ = np.array([np.mean(rs[l_s[(kind, k)]["run_id"]]["val"]["r2"]) for k in ks])
        d = l_ - g
        est, lo, hi = bootstrap_ci(d)
        paired[kind] = dict(settings=ks, gru_val_r2=g.tolist(), lstm_val_r2=l_.tolist(), diff_mean=float(est),
                            diff_ci=[float(lo), float(hi)], wilcoxon_p=float(wilcoxon(d).pvalue),
                            lstm_wins=int((d > 0).sum()))
    best8 = {f"{c}_{k}": _both(rows(XL.seed_records(c, k))) for c in ("gru", "lstm") for k in XL.KINDS}
    h = R.matched_hidden()
    for k in ("raw", "augL"):
        best8[f"lstm{h}_{k}"] = _both(rows(XL.seed_records("lstm", k, hidden=h)))
    S["phase8"] = dict(paired_val=paired, best_test=best8)

    # Phase 9: one array (day 1869)
    arr = read_jsonl(XA.TRIALS)
    p9 = {}
    for c in ("gru", "lstm"):
        for k in ("raw", "augL"):
            ent = dict(both=_both(rows(XL.seed_records(c, k))))
            for a in ("A", "B"):
                ent[a] = _both(rows([r for r in arr if r["channel_set"] == a and r["cell"] == c and r["kind_ext"] == k]))
            for a in ("A", "B"):
                ent[f"{a}_pct"] = {h_: 100 * ent[a]["r2"][h_] / ent["both"]["r2"][h_] for h_ in ("R", "L")}
            p9[f"{c}_{k}"] = ent
    rnd = [r for r in arr if r["channel_set"].startswith("rand")]
    p9["rand96_gru"] = {k: _both(rows([r for r in rnd if r["kind_ext"] == k])) for k in ("raw", "augL")}
    z = json.loads((OUT / "zero_out.json").read_text())
    p9["zero_one_array"] = {rid: {d: {q: v[d][q] for q in ("r2_R", "r2_L", "r_R", "r_L")} for d in ("none", "A", "B")}
                            for rid, v in z.items()}
    S["phase9"] = p9

    # Phase 10
    q = [r for r in E.QUEUE.records() if "test" in r]
    by = {r["run_id"]: r for r in E.QUEUE.records()}
    ref = _both(rows(XL.seed_records("lstm", "augL")))
    p10 = dict(ref192=ref)
    cc = {}
    for a in ("A", "B"):
        cc[a] = {}
        for n in E.NS:
            ent = dict(random=_both(rows([r for r in q if r.get("group") == "random_n"
                                         and r["channel_spec"]["array"] == a and r["channel_spec"]["n"] == n])))
            ent["random_each"] = [dict(run_id=r["run_id"], r2_R=rs[r["run_id"]]["test"]["r2_R"],
                                       r2_L=rs[r["run_id"]]["test"]["r2_L"])
                                  for r in q if r.get("group") == "random_n" and r["channel_spec"]["array"] == a
                                  and r["channel_spec"]["n"] == n]
            for meth, pref in (("greedy_swap", "greedy"), ("group_lasso", "glasso"), ("ablation", "ablat")):
                mr = [r for r in E.QUEUE.records() if r["run_id"].startswith(f"{pref}{n}{a}_")]
                if not mr:
                    continue
                dup = any(r.get("tag") == "dup" for r in mr)
                if dup:  # identical set to greedy + swap: report the greedy runs
                    mr = [r for r in q if r["run_id"].startswith(f"greedy{n}{a}_")]
                ent[meth] = dict(**_both(rows([r for r in mr if "test" in r])), dup_of_greedy=dup)
            cc[a][str(n)] = ent
        cc[a]["96"] = _both(rows([r for r in arr if r["channel_set"] == a and r["cell"] == "lstm"
                                  and r["kind_ext"] == "augL"]))
    p10["channel_count"] = cc
    p10["data_amount"] = {str(k): _both(rows([r for r in q if r.get("group") == "data_amount" and r["k_blocks"] == k]))
                          for k in (1, 2, 3, 4)}
    p10["data_amount"]["5"] = ref
    days = {}
    for day in (1750, 1867):
        for w in ("all", "A", "B"):
            rr = rows([r for r in q if r["run_id"].startswith(f"d{day}_{w}_")])
            ent = _both(rr)
            if "r2_bi_R" in rr[0]["test"]:
                ent["bi"] = dict(r2=dict(R=float(np.mean([r["test"]["r2_bi_R"] for r in rr])),
                                         L=float(np.mean([r["test"]["r2_bi_L"] for r in rr]))),
                                 r=dict(R=float(np.mean([r["test"]["r_bi_R"] for r in rr])),
                                        L=float(np.mean([r["test"]["r_bi_L"] for r in rr]))))
            days[f"{day}_{w}"] = ent
    p10["days"] = days
    p10["dropout_unmuted"] = _both(rows([r for r in q if r.get("group") == "dropout"]))
    mu = json.loads((OUT / "mute.json").read_text())
    agg = {}
    for kind in ("normal", "dropout"):
        mrows = [v for v in mu["models"].values() if v["kind"] == kind]
        agg[kind] = {mode: {f: {k_: float(np.mean([v["f"][f].get(mode, v["f"][f]["zero"])[k_] for v in mrows]))
                                for k_ in ("r2_R", "r2_L", "r_R", "r_L")}
                            for f in mrows[0]["f"]} for mode in mu["modes"]}
    p10["mute"] = agg
    S["phase10"] = p10

    # Phase 11 (TX vs SBP): the R^2 run of ext_sbp
    sb = json.loads((OUT / "sbp.json").read_text())
    S["phase11"] = dict(item1={k: v for k, v in sb["item1_features"].items() if k.startswith("velocity")},
                        item2_velocity=sb["item2_channel_count"]["velocity"],
                        item3_velocity=sb["item3_muting"]["velocity"])
    S["definition"] = ("R^2 = 1 - SSE/SST per velocity dimension over the held-out bins, mean of x and y per hand; "
                       "models and settings as selected with r (register #47)")
    atomic_write_json(OUT / "summary.json", S)
    print("wrote", OUT / "summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "fig3", "rnn", "zero", "mute", "sbp", "summary"], default="all")
    st = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig3", stage_fig3), ("rnn", stage_rnn), ("zero", stage_zero), ("mute", stage_mute),
                     ("sbp", stage_sbp), ("summary", stage_summary)):
        if st in ("all", name):
            fn()


if __name__ == "__main__":
    main()
