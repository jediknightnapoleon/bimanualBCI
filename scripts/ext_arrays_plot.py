"""Phase 9 summary + figure: decoding with one array (called by ``ext_arrays.py --stage plot``).

Reads results/ext_arrays/{trials.jsonl, zero_out.json, array_check.json, corr_matrices.npz} and
the 192-channel reference models of Phase 8 (``ext_lstm.seed_records``). Writes
results/ext_arrays/summary.json and results/figures/ext_arrays.{png,svg}.

"Retained" = mean test r of the one-array models / mean test r of the 192-channel models of the
same (cell, kind), per hand. The CI comes from a bootstrap over the seed-level ratios (each
one-array seed against the 192-channel seed mean); with 3 seeds it is only indicative.
"""

from __future__ import annotations

import json

import numpy as np

import ext_arrays as XA
import ext_lstm as XL
from bimanual import plotting as P
from bimanual.cache import atomic_write_json, read_jsonl

SETS = ("192", "A", "B", "rand")
SET_LABEL = {"192": "both arrays (192)", "A": "array A only (ch 1–96)", "B": "array B only (ch 97–192)",
             "rand": "random 96 (both arrays)"}
SET_COL = {"192": P.INK_MUTED, "A": P.SERIES[0], "B": P.SERIES[1], "rand": P.SERIES[2]}
COMBOS = [("gru", "raw"), ("gru", "augL"), ("lstm", "raw"), ("lstm", "augL")]


def _vals(recs):
    if not recs:
        return None
    px = [r["extra_scores"]["proxy"]["all"]["r_mean"] if "extra_scores" in r else None for r in recs]
    return dict(run_ids=[r["run_id"] for r in recs],
                r_R=[r["test"]["r_R"] for r in recs], r_L=[r["test"]["r_L"] for r in recs],
                roughness=[r["test"]["roughness"] for r in recs], proxy_r=[p for p in px if p is not None])


def summarize():
    recs = read_jsonl(XA.TRIALS)
    base_px = XL.OUT / "proxy_gru_baseline.json"
    base_px = json.loads(base_px.read_text()) if base_px.exists() else {}
    out = {}
    for cell, kind in COMBOS:
        ref = XL.seed_records(cell, kind)
        row = {"192": _vals(ref)}
        if row["192"] is not None:
            row["192"]["proxy_r"] = [(r["extra_scores"]["proxy"]["all"]["r_mean"] if "extra_scores" in r
                                      else base_px.get(r["run_id"], {}).get("all", {}).get("r_mean"))
                                     for r in ref]
            row["192"]["proxy_r"] = [p for p in row["192"]["proxy_r"] if p is not None]
        for s in ("A", "B", "rand"):
            row[s] = _vals([r for r in recs if r.get("cell") == cell and r["kind_ext"] == kind
                            and (r["channel_set"] == s or (s == "rand" and r["channel_set"].startswith("rand")))])
        if row["192"]:
            for s in ("A", "B", "rand"):
                if row[s]:
                    for h in ("R", "L"):
                        ratios = np.array(row[s][f"r_{h}"]) / np.mean(row["192"][f"r_{h}"])
                        row[s][f"retained_{h}"] = float(np.mean(ratios))
                        row[s][f"retained_{h}_range"] = [float(ratios.min()), float(ratios.max())]
        out[f"{cell}_{kind}"] = row
    zpath = XA.OUT / "zero_out.json"
    zero = {}
    if zpath.exists():
        z = json.loads(zpath.read_text())
        for cell, kind in COMBOS:
            rows = [v for v in z.values() if v["cell"] == cell and v["kind"] == kind]
            if rows:
                zero[f"{cell}_{kind}"] = {d: dict(r_R=float(np.mean([v[d]["r_R"] for v in rows])),
                                                  r_L=float(np.mean([v[d]["r_L"] for v in rows])),
                                                  proxy_r=float(np.mean([v[d]["proxy"]["r_mean"] for v in rows])))
                                          for d in ("none", "A", "B")}
    check = json.loads((XA.OUT / "array_check.json").read_text())
    budget = json.loads((XA.OUT / "budget.json").read_text()) if (XA.OUT / "budget.json").exists() else {}
    summ = dict(trained=out, zero_out=zero, array_check=check, budget=budget)
    atomic_write_json(XA.OUT / "summary.json", summ)
    for k, row in out.items():
        line = [k]
        for s in SETS:
            v = row.get(s)
            if v:
                line.append(f"{s}: {np.mean(v['r_R']):.3f}/{np.mean(v['r_L']):.3f}"
                            + (f" ({v['retained_R'] * 100:.0f}%/{v['retained_L'] * 100:.0f}%)" if "retained_R" in v else ""))
        print("  ".join(line))
    for k, z in zero.items():
        print(f"zero-out {k}: " + "  ".join(f"{d}: {v['r_R']:.2f}/{v['r_L']:.2f}" for d, v in z.items()))
    return summ


def summarize_and_plot():
    import matplotlib.pyplot as plt

    summ = summarize()
    tr, zero = summ["trained"], summ["zero_out"]
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    # (a) correlation matrix, day 1869
    ax = fig.add_subplot(gs[0, 0])
    M = np.load(XA.OUT / "corr_matrices.npz")["1869"]
    v = np.nanpercentile(np.abs(M), 99)
    ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.axhline(95.5, color=P.INK, lw=0.8); ax.axvline(95.5, color=P.INK, lw=0.8)
    c = summ["array_check"]["days"]["1869"]
    ax.set_title(f"(a) Channel correlations, day 1869 (block means removed)\nwithin-half {(c['within_A'] + c['within_B']) / 2:.4f}"
                 f" vs across {c['across']:.4f}; split support: {summ['array_check']['strength']}", fontsize=8.5)
    ax.set_xlabel("channel"); ax.set_ylabel("channel")

    # (b) test r per hand for each channel set
    ax = fig.add_subplot(gs[0, 1:])
    keys = [k for k in tr if tr[k].get("192")]
    w = 0.8 / len(SETS)
    for i, k in enumerate(keys):
        for j, hand in enumerate(("R", "L")):
            base_x = i * 2.2 + j
            for si, s in enumerate(SETS):
                vv = tr[k].get(s)
                if not vv:
                    continue
                x = base_x + (si - (len(SETS) - 1) / 2) * w
                vals = vv[f"r_{hand}"]
                ax.bar(x, np.mean(vals), width=w - 0.02, color=SET_COL[s],
                       label=SET_LABEL[s] if (i == 0 and j == 0) else None)
                ax.scatter(np.full(len(vals), x), vals, s=6, color=P.INK, zorder=3)
    ax.set_xticks([i * 2.2 + j for i in range(len(keys)) for j in (0, 1)],
                  [f"{k.replace('_', ' ').replace('augL', 'aug 2–8 s')}\n{'RH' if j == 0 else 'LH'}" for k in keys
                   for j in (0, 1)], fontsize=7)
    ax.set_ylabel("test-block r (mean of x, y)")
    ax.set_title("(b) Offline decoding (day 1869, block 7) when trained on one array, 3 seeds each", fontsize=9)
    ax.set_ylim(0.4, 1.06); ax.legend(fontsize=7, ncol=4, loc="upper center", framealpha=1); ax.grid(axis="x", visible=False)

    # (c) retained %
    ax = fig.add_subplot(gs[1, 0])
    for si, s in enumerate(("A", "B", "rand")):
        for i, k in enumerate(keys):
            vv = tr[k].get(s)
            if not vv or "retained_R" not in vv:
                continue
            ax.scatter(vv["retained_R"] * 100, vv["retained_L"] * 100, color=SET_COL[s], s=30,
                       marker="o" if k.startswith("gru") else "s", label=SET_LABEL[s] if i == 0 else None)
    ax.plot([80, 104], [80, 104], ls=":", color=P.INK_MUTED, lw=0.8); ax.set_xlim(80, 104); ax.set_ylim(80, 104)
    ax.set_xlabel("RH r retained (% of 192-ch)"); ax.set_ylabel("LH r retained (%)")
    ax.set_title("(c) Retained performance (circles GRU, squares LSTM)", fontsize=9); ax.legend(fontsize=6.5)

    # (d) zero one array at test (no retraining)
    ax = fig.add_subplot(gs[1, 1])
    zk = list(zero)
    for i, k in enumerate(zk):
        for di, (d, col) in enumerate((("none", P.INK_MUTED), ("A", P.SERIES[0]), ("B", P.SERIES[1]))):
            x = i + (di - 1) * 0.27
            ax.bar(x, zero[k][d]["r_R"], width=0.12, color=col, label=f"{'nothing' if d == 'none' else 'array ' + d} zeroed (RH)" if i == 0 else None)
            ax.bar(x + 0.12, zero[k][d]["r_L"], width=0.12, color=col, alpha=0.5,
                   label=f"... (LH)" if i == 0 and d == "none" else None)
    ax.set_xticks(range(len(zk)), [k.replace("_", " ").replace("augL", "aug 2–8 s") for k in zk], fontsize=7)
    ax.set_ylabel("test r (light = LH)"); ax.set_ylim(0, 1.3)
    ax.set_title("(d) 192-ch models, one array zeroed at test", fontsize=9); ax.legend(fontsize=6.5, loc="upper center", ncol=2, framealpha=1)
    ax.grid(axis="x", visible=False)

    # (e) closed-loop proxy r
    ax = fig.add_subplot(gs[1, 2])
    for i, k in enumerate(keys):
        for si, s in enumerate(SETS):
            vv = tr[k].get(s)
            if not vv or not vv["proxy_r"]:
                continue
            x = i + (si - (len(SETS) - 1) / 2) * w
            ax.bar(x, np.mean(vv["proxy_r"]), width=w - 0.02, color=SET_COL[s])
            ax.scatter(np.full(len(vv["proxy_r"]), x), vv["proxy_r"], s=6, color=P.INK, zorder=3)
    ax.set_xticks(range(len(keys)), [k.replace("_", " ").replace("augL", "aug 2–8 s") for k in keys], fontsize=7)
    ax.set_ylabel("proxy r (mean of 4 dims)")
    ax.set_title("(e) Closed-loop proxy (weak metric, #35)", fontsize=9); ax.grid(axis="x", visible=False)

    fig.suptitle("Extension: decoding with one of the two Utah arrays (channel split assumed, register #36)",
                 fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "ext_arrays")
    print("wrote", P.FIG_DIR / "ext_arrays.png")
