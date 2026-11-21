"""Phase 8 summary + figure: LSTM vs GRU (called by ``ext_lstm.py --stage plot``).

Reads results/ext_lstm/trials.jsonl, results/rnn_search/trials.jsonl and
results/ext_lstm/proxy_gru_baseline.json. Writes results/ext_lstm/summary.json and
results/figures/ext_lstm_vs_gru.{png,svg}.

Statistics:
* Paired search: for each data kind, LSTM - GRU validation r (mean of the 4 dims) over the
  settings both cells trained. Wilcoxon signed-rank test + bootstrap 95% CI of the mean difference.
* Best models: the mean and individual values of the test r over the 3 seeds of the best setting.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

import ext_lstm as XL
import fig4_rnn as F
from bimanual import plotting as P
from bimanual.cache import atomic_write_json, read_jsonl
from bimanual.stats import bootstrap_ci

KIND_LABEL = {"raw": "Raw", "augL": "Aug. 2–8-s clips", "augS": "Aug. 200–800-ms clips"}


def _proxy(rec, base):
    if "extra_scores" in rec:
        return rec["extra_scores"]["proxy"]["all"]
    p = base.get(rec["run_id"])
    return p["all"] if p else None


def _group(recs, base):
    """Mean and per-seed values of the test metrics and the proxy for a list of seed records."""
    if not recs:
        return None
    px = [_proxy(r, base) for r in recs]
    px = [p for p in px if p]
    return dict(run_ids=[r["run_id"] for r in recs], hp=recs[0]["hp"],
                test_r_R=[r["test"]["r_R"] for r in recs], test_r_L=[r["test"]["r_L"] for r in recs],
                roughness=[r["test"]["roughness"] for r in recs],
                mean_r_R=float(np.mean([r["test"]["r_R"] for r in recs])),
                mean_r_L=float(np.mean([r["test"]["r_L"] for r in recs])),
                mean_roughness=float(np.mean([r["test"]["roughness"] for r in recs])),
                proxy_r=[p["r_mean"] for p in px], proxy_ang=[p["ang_err_deg"] for p in px],
                proxy_sustain=[p["sustain"] for p in px], proxy_jitter=[p["jitter"] for p in px],
                train_seconds=float(np.mean([r["train_seconds"] for r in recs])))


def summarize():
    path = XL.OUT / "proxy_gru_baseline.json"
    base = json.loads(path.read_text()) if path.exists() else {}
    g_s, l_s = XL.gru_baseline_search(), XL.lstm_search()
    paired = {}
    for kind in XL.KINDS:
        ks = sorted(k for (kk, k) in l_s if kk == kind and (kind, k) in g_s)
        if not ks:
            continue
        g = np.array([g_s[(kind, k)]["val"]["r_mean"] for k in ks]); l_ = np.array([l_s[(kind, k)]["val"]["r_mean"] for k in ks])
        d = l_ - g
        est, lo, hi = bootstrap_ci(d)
        p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else float("nan")
        paired[kind] = dict(settings=ks, gru_val=g.tolist(), lstm_val=l_.tolist(), diff_mean=float(est),
                            diff_ci=[float(lo), float(hi)], wilcoxon_p=p, lstm_wins=int((d > 0).sum()))
    best = {}
    for cell in ("gru", "lstm"):
        for kind in XL.KINDS:
            best[f"{cell}_{kind}"] = _group(XL.seed_records(cell, kind), base)
    from bimanual.rnn import matched_hidden
    h = matched_hidden()
    for kind in ("raw", "augL"):
        best[f"lstm{h}_{kind}"] = _group(XL.seed_records("lstm", kind, hidden=h), base)
    budget = json.loads((XL.OUT / "budget.json").read_text()) if (XL.OUT / "budget.json").exists() else {}
    summ = dict(paired_search=paired, best=best, budget=budget, paper_raw=F.PAPER)
    atomic_write_json(XL.OUT / "summary.json", summ)
    for kind, v in paired.items():
        print(f"paired {kind}: LSTM - GRU val r = {v['diff_mean']:+.3f} [{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}], "
              f"Wilcoxon p {v['wilcoxon_p']:.3g}, LSTM better in {v['lstm_wins']}/{len(v['settings'])}")
    for k, v in best.items():
        if v:
            print(f"{k:12s} test r RH {v['mean_r_R']:.3f} LH {v['mean_r_L']:.3f} rough {v['mean_roughness']:.3f} "
                  f"proxy r {np.mean(v['proxy_r']) if v['proxy_r'] else float('nan'):.3f} (n={len(v['run_ids'])})")
    return summ


def summarize_and_plot(data=None):
    import matplotlib.pyplot as plt

    summ = summarize()
    paired, best = summ["paired_search"], summ["best"]
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3)
    kcol = {"raw": P.SERIES[0], "augL": P.SERIES[3], "augS": P.SERIES[1]}

    # (a) paired search: validation r, GRU (x) vs LSTM (y)
    ax = fig.add_subplot(gs[0, 0])
    for kind, v in paired.items():
        ax.scatter(v["gru_val"], v["lstm_val"], s=22, color=kcol[kind],
                   label=f"{KIND_LABEL[kind]} (Δ {v['diff_mean']:+.3f}, p={v['wilcoxon_p']:.2g})")
    lo = min(min(v["gru_val"] + v["lstm_val"]) for v in paired.values()) - 0.02
    ax.plot([lo, 1], [lo, 1], color=P.INK_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("GRU validation r"); ax.set_ylabel("LSTM validation r")
    ax.set_title("(a) Paired search: same 10 settings", fontsize=9); ax.legend(fontsize=6.5, loc="upper left")

    # (b) test r per hand, best setting x 3 seeds
    keys = [k for k in best if best[k]]
    labels = {k: k.replace("_", " ").replace("augL", "2–8 s").replace("augS", "0.2–0.8 s") for k in keys}
    ax = fig.add_subplot(gs[0, 1:])
    w = 0.38
    for i, k in enumerate(keys):
        v = best[k]
        for j, hand in enumerate(("R", "L")):
            vals = v[f"test_r_{hand}"]
            x = i + (j - 0.5) * w
            ax.bar(x, np.mean(vals), width=w - 0.04, color=P.SERIES[0] if hand == "R" else P.SERIES[2],
                   label=("Right hand" if hand == "R" else "Left hand") if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, s=8, color=P.INK, zorder=3)
            ax.text(x, np.mean(vals) + 0.015, f"{np.mean(vals):.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(keys)), [labels[k] for k in keys], fontsize=7, rotation=20)
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("test-block r (mean of x, y)")
    ax.axhline(F.PAPER["raw_R"], color=P.SERIES[0], ls=":", lw=0.8); ax.axhline(F.PAPER["raw_L"], color=P.SERIES[2], ls=":", lw=0.8)
    ax.set_title("(b) Offline decoding, held-out block 7: best setting × 3 seeds (dotted: paper's raw GRU)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=1); ax.grid(axis="x", visible=False)

    # (c) roughness, (d) proxy r, (e) proxy angular error
    for col, (field, title, ylab) in enumerate((("roughness", "(c) Decoded-velocity roughness (lower = smoother)", "roughness"),
                                                ("proxy_r", "(d) Closed-loop proxy r (weak metric, #35)", "proxy r (mean of 4 dims)"),
                                                ("proxy_ang", "(e) Closed-loop proxy angular error", "degrees (300–500 ms)"))):
        ax = fig.add_subplot(gs[1, col])
        for i, k in enumerate(keys):
            vals = best[k][field]
            if not vals:
                continue
            c = kcol["raw" if "raw" in k else "augL" if "augL" in k else "augS"]
            ax.bar(i, np.mean(vals), color=c, alpha=1.0 if k.startswith("gru") else 0.6,
                   hatch=None if k.startswith("gru") else "//", edgecolor="white" if k.startswith("gru") else P.INK_MUTED)
            ax.scatter(np.full(len(vals), i), vals, s=8, color=P.INK, zorder=3)
        ax.set_xticks(range(len(keys)), [labels[k] for k in keys], fontsize=6.5, rotation=35, ha="right")
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    fig.suptitle("Extension: LSTM vs GRU decoders, day 1869 (same data, splits, targets and search settings; "
                 "solid = GRU, hatched = LSTM)", fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "ext_lstm_vs_gru")
    print("wrote", P.FIG_DIR / "ext_lstm_vs_gru.png")
