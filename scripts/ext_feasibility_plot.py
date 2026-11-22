"""Phase 10 summary + figure (called by ``ext_feasibility.py --stage plot``).

Reads results/ext_feas/{trials_w*.jsonl, mute.json, selection.json, timing.json}, the Phase 8
192-channel LSTM models (``ext_lstm.seed_records("lstm", "augL")``) and the Phase 9 one-array LSTM
models (results/ext_arrays/trials.jsonl). Writes results/ext_feas/summary.json and
results/figures/ext_feasibility.{png,svg}.

Every number is test-block r (block 7 of day 1869 unless stated), mean of x and y, per hand.
"""

from __future__ import annotations

import json

import numpy as np

import ext_arrays as XA
import ext_feasibility as E
import ext_lstm as XL
import fig4_rnn as F
from bimanual import plotting as P
from bimanual.cache import atomic_write_json, read_jsonl

ARR_COL = {"A": P.SERIES[0], "B": P.SERIES[1]}
METHOD_STYLE = {"greedy_swap": ("*", "greedy + swap"), "group_lasso": ("D", "group lasso"),
                "ablation": ("v", "LSTM ablation")}


def _hand(recs, key="test"):
    return (np.array([r[key]["r_R"] for r in recs]), np.array([r[key]["r_L"] for r in recs]))


def participation_ratio(z: np.ndarray) -> float:
    """(sum of eigenvalues)^2 / sum of squared eigenvalues of the channel covariance: an
    'effective number of channels' (Gao et al. 2017 use it as a dimensionality measure)."""
    ev = np.clip(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def summarize():
    recs = [r for r in E.QUEUE.records() if r.get("tag") != "failed"]
    failed = [r["run_id"] for r in E.QUEUE.records() if r.get("tag") == "failed"]
    ids = [r["run_id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate run_ids across worker files"
    by = {r["run_id"]: r for r in recs}
    trained = [r for r in recs if "test" in r]
    ref192 = XL.seed_records("lstm", "augL")
    one = [r for r in read_jsonl(XA.TRIALS) if r.get("cell") == "lstm" and r.get("kind_ext") == "augL"]
    out = dict(failed=failed)
    R192, L192 = _hand(ref192)
    out["ref192"] = dict(r_R=float(R192.mean()), r_L=float(L192.mean()), run_ids=[r["run_id"] for r in ref192])
    out["ref96"] = {a: dict(zip(("r_R", "r_L"), map(lambda x: float(x.mean()), _hand(
        [r for r in one if r["channel_set"] == a])))) for a in ("A", "B")}

    # channel count: random-n and best-n per array
    cc = {}
    for arr in ("A", "B"):
        row = {}
        for n in E.NS:
            rr = [r for r in trained if r.get("group") == "random_n" and r["channel_spec"]["array"] == arr
                  and r["channel_spec"]["n"] == n]
            ent = {}
            if rr:
                R_, L_ = _hand(rr)
                ent["random"] = dict(r_R=R_.tolist(), r_L=L_.tolist(), mean_R=float(R_.mean()), mean_L=float(L_.mean()))
            for meth in METHOD_STYLE:
                pref = {"greedy_swap": "greedy", "group_lasso": "glasso", "ablation": "ablat"}[meth]
                mr = [r for r in recs if r["run_id"].startswith(f"{pref}{n}{arr}_")]
                mr = [by[r["run_id"]] if "test" in r else _dup_target(r, by, n, arr) for r in mr]
                mr = [r for r in mr if r is not None]
                if mr:
                    R_, L_ = _hand(mr)
                    ent[meth] = dict(r_R=R_.tolist(), r_L=L_.tolist(), mean_R=float(R_.mean()), mean_L=float(L_.mean()),
                                     dup_of_greedy=any(r.get("tag") == "dup" for r in recs if r["run_id"].startswith(
                                         f"{pref}{n}{arr}_")))
            row[str(n)] = ent
        row["96"] = dict(one_array=out["ref96"][arr])
        cc[arr] = row
    out["channel_count"] = cc

    # selection (surrogate) summary
    sel = json.loads(E.SELECTION.read_text()) if E.SELECTION.exists() else None
    if sel:
        out["selection"] = {arr: dict(full_val_r=sel[arr].get("full_val_r"), percentiles=sel[arr].get("percentiles"),
                                      swap={n: {k: v for k, v in d.items() if k != "channels"}
                                            for n, d in sel[arr].get("greedy_swap", {}).items()})
                            for arr in ("A", "B") if arr in sel}
        out["selection_overlap"] = {}
        for arr in ("A", "B"):
            for n in ("8", "16"):
                sets = {m: set(sel["sets"].get(m, {}).get(arr, {}).get(n, [])) for m in METHOD_STYLE}
                if all(sets.values()):
                    out["selection_overlap"][f"{arr}{n}"] = {
                        f"{a}|{b}": len(sets[a] & sets[b]) for i, a in enumerate(METHOD_STYLE)
                        for b in list(METHOD_STYLE)[i + 1:]}
    # surrogate vs LSTM across the random-n runs
    try:
        from bimanual import select as SEL
        data = F.Data("clip")
        pts = []
        for r in trained:
            if r.get("group") == "random_n":
                pts.append((SEL.surrogate_score(data, r["channels"]), r["val"]["r_mean"], r["test"]["r_mean"]))
        if len(pts) >= 4:
            p = np.array(pts)
            out["surrogate_vs_lstm"] = dict(points=p.tolist(), r_val=float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]),
                                            r_test=float(np.corrcoef(p[:, 0], p[:, 2])[0, 1]))
        # participation ratio (training blocks 1-5)
        ztr = np.concatenate([data.z[a:b] for a, b in data.seg["train"]])
        pr = {"192": participation_ratio(ztr), "A": participation_ratio(ztr[:, :96]),
              "B": participation_ratio(ztr[:, 96:])}
        for (n, arr, i), ch in E.random_subsets().items():
            if i == 0:
                pr[f"rand{n}{arr}"] = participation_ratio(ztr[:, ch])
        if sel:
            for arr in ("A", "B"):
                for n, ch in sel["sets"]["greedy_swap"][arr].items():
                    pr[f"greedy{n}{arr}"] = participation_ratio(ztr[:, ch])
        out["participation_ratio"] = pr
    except (ImportError, AttributeError, KeyError) as e:  # selection module or file not ready
        out["surrogate_vs_lstm_error"] = repr(e)

    # training-data amount
    da = {}
    for k in (1, 2, 3, 4):
        rr = [r for r in trained if r.get("group") == "data_amount" and r["k_blocks"] == k]
        if rr:
            R_, L_ = _hand(rr)
            da[str(k)] = dict(r_R=R_.tolist(), r_L=L_.tolist(), mean_R=float(R_.mean()), mean_L=float(L_.mean()))
    da["5"] = dict(mean_R=out["ref192"]["r_R"], mean_L=out["ref192"]["r_L"])
    try:
        d0 = F.Data("clip")
        blk_bins = {b: int((d0.block == b).sum()) for b in F.TRAIN_BLOCKS}
        for k in range(1, 6):
            da.setdefault(str(k), {})["minutes"] = sum(blk_bins[b] for b in F.TRAIN_BLOCKS[-k:]) * 0.02 / 60
    except Exception:  # noqa: BLE001
        pass
    out["data_amount"] = da

    # other days
    days = {}
    for day in (1750, 1867):
        for which in ("all", "A", "B"):
            rr = [r for r in trained if r["run_id"].startswith(f"d{day}_{which}_")]
            if rr:
                R_, L_ = _hand(rr)
                ent = dict(r_R=R_.tolist(), r_L=L_.tolist(), mean_R=float(R_.mean()), mean_L=float(L_.mean()))
                if "r_bi_R" in rr[0]["test"]:
                    ent.update(bi_R=float(np.mean([r["test"]["r_bi_R"] for r in rr])),
                               bi_L=float(np.mean([r["test"]["r_bi_L"] for r in rr])))
                days[f"{day}_{which}"] = ent
    out["days"] = days

    # dropout and muting
    drop = [r for r in trained if r.get("group") == "dropout"]
    if drop:
        R_, L_ = _hand(drop)
        out["dropout_unmuted"] = dict(r_R=R_.tolist(), r_L=L_.tolist(), mean_R=float(R_.mean()), mean_L=float(L_.mean()))
    mpath = E.OUT / "mute.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        agg = {}
        for kind in ("normal", "dropout"):
            rows = [v for v in m["models"].values() if v["kind"] == kind]
            if not rows:
                continue
            agg[kind] = {mode: {f: dict(r_R=float(np.mean([v["f"][f][mode]["r_R"] if mode in v["f"][f] else
                                                          v["f"][f]["zero"]["r_R"] for v in rows])),
                                       r_L=float(np.mean([v["f"][f][mode]["r_L"] if mode in v["f"][f] else
                                                          v["f"][f]["zero"]["r_L"] for v in rows])))
                                for f in rows[0]["f"]} for mode in m["modes"]}
            agg[kind]["proxy"] = {f: float(np.mean([np.mean(v["proxy"][f]) for v in rows])) for f in rows[0]["proxy"]}
        out["mute"] = agg
    t = E.OUT / "timing.json"
    out["timing"] = json.loads(t.read_text()) if t.exists() else {}
    atomic_write_json(E.OUT / "summary.json", out)
    _print(out)
    return out


def _dup_target(rec, by, n, arr):
    """A 'dup' record (set identical to greedy + swap): use the greedy run of the same seed if any."""
    if rec.get("tag") != "dup":
        return None
    sd = rec["run_id"].split("_seed")[-1]
    return by.get(f"greedy{n}{arr}_seed{sd}")


def _print(out):
    print(f"192 ch: {out['ref192']['r_R']:.3f}/{out['ref192']['r_L']:.3f}; one array A "
          f"{out['ref96']['A']['r_R']:.3f}/{out['ref96']['A']['r_L']:.3f}, B {out['ref96']['B']['r_R']:.3f}/"
          f"{out['ref96']['B']['r_L']:.3f}")
    for arr, row in out["channel_count"].items():
        for n in map(str, E.NS):
            ent = row.get(n, {})
            s = "  ".join(f"{k}: {v['mean_R']:.3f}/{v['mean_L']:.3f}" for k, v in ent.items())
            if s:
                print(f"  array {arr} n={n:>2}: {s}")
    for k, v in out["data_amount"].items():
        if "mean_R" in v:
            print(f"  train blocks {k} ({v.get('minutes', float('nan')):.1f} min): {v['mean_R']:.3f}/{v['mean_L']:.3f}")
    for k, v in out["days"].items():
        print(f"  day {k}: {v['mean_R']:.3f}/{v['mean_L']:.3f}" + (f"  two-hand {v['bi_R']:.3f}/{v['bi_L']:.3f}"
                                                                    if "bi_R" in v else ""))
    if "dropout_unmuted" in out:
        print(f"  dropout-trained, nothing muted: {out['dropout_unmuted']['mean_R']:.3f}/{out['dropout_unmuted']['mean_L']:.3f}")
    if "mute" in out:
        for kind, modes in out["mute"].items():
            z = modes["zero"]
            print(f"  muted ({kind}, zero): " + "  ".join(f"f={f}: {v['r_R']:.2f}/{v['r_L']:.2f}" for f, v in z.items()))
    if "surrogate_vs_lstm" in out:
        print(f"  surrogate vs LSTM (random-n): r = {out['surrogate_vs_lstm']['r_val']:.2f} (val), "
              f"{out['surrogate_vs_lstm']['r_test']:.2f} (test)")


def summarize_and_plot():
    import matplotlib.pyplot as plt

    s = summarize()
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3)
    hands = (("R", "-", "RH"), ("L", "--", "LH"))

    # (a, b) muting curves, RH and LH
    if "mute" in s:
        for col, (h, _, lab) in enumerate(hands):
            ax = fig.add_subplot(gs[0, col])
            for kind, color in (("normal", P.INK_MUTED), ("dropout", P.SERIES[3])):
                if kind not in s["mute"]:
                    continue
                for mode, ls in (("zero", "-"), ("dead", "--"), ("noisy", ":")):
                    cur = s["mute"][kind][mode]
                    fs = sorted(cur, key=float)
                    ax.plot([float(f) * 100 for f in fs], [cur[f][f"r_{h}"] for f in fs], ls=ls, color=color, lw=1.6,
                            marker="o", ms=3, label=f"{'dropout-trained' if kind == 'dropout' else 'normal'}, {mode}")
            ax.set_xlabel("% of the 192 channels muted at random (test only)"); ax.set_ylabel(f"{lab} test r")
            ax.set_title(f"({'ab'[col]}) Muted channels, {lab} (3 models each, 20 masks)", fontsize=9)
            ax.legend(fontsize=6.5, loc="lower left")
    # (c) channel count per array
    ax = fig.add_subplot(gs[0, 2])
    for arr in ("A", "B"):
        row = s["channel_count"][arr]
        for h, ls, lab in hands:
            xs, ys = [], []
            for n in E.NS:
                ent = row.get(str(n), {}).get("random")
                if ent:
                    xs.append(n); ys.append(ent[f"mean_{h}"])
                    ax.scatter([n] * len(ent[f"r_{h}"]), ent[f"r_{h}"], s=6, color=ARR_COL[arr], alpha=0.5)
            xs.append(96); ys.append(row["96"]["one_array"][f"r_{h}"])
            ax.plot(xs, ys, ls=ls, color=ARR_COL[arr], lw=1.4, label=f"array {arr} random, {lab}")
            for meth, (mk, mlab) in METHOD_STYLE.items():
                pts = [(n, row[str(n)][meth][f"mean_{h}"]) for n in E.NS if meth in row.get(str(n), {})]
                if pts:
                    ax.scatter(*zip(*pts), marker=mk, s=40 if mk == "*" else 22, color=ARR_COL[arr],
                               edgecolor=P.INK, lw=0.5, zorder=4,
                               label=f"{mlab}" if (arr == "A" and h == "R") else None)
        for h, ls, lab in hands:
            ax.axhline(s["ref192"][f"r_{h}"], color=P.INK_MUTED, ls=ls, lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 64, 96], ["8", "16", "32", "64", "96"])
    ax.set_xlabel("channels used (within one array)"); ax.set_ylabel("test r")
    ax.set_title("(c) Channel count: random vs best-chosen (grey: 192 ch)", fontsize=9)
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    # (d) surrogate: search methods vs 1000 random subsets
    ax = fig.add_subplot(gs[1, 0])
    sel = json.loads(E.SELECTION.read_text()) if E.SELECTION.exists() else None
    if sel:
        for arr, dx in (("A", -0.12), ("B", 0.12)):
            for i, n in enumerate(E.NS):
                rnd = sel[arr].get("random", {}).get(str(n), {}).get("val_r")
                if rnd:
                    q = np.percentile(rnd, [5, 50, 95])
                    ax.plot([i + dx] * 2, [q[0], q[2]], color=ARR_COL[arr], lw=3, alpha=0.35)
                    ax.scatter(i + dx, q[1], color=ARR_COL[arr], s=10)
                for meth, (mk, mlab) in METHOD_STYLE.items():
                    d = sel[arr].get(meth, {})
                    v = d.get(str(n), {}).get("val_r") if meth != "ablation" else None
                    if meth == "ablation":
                        continue
                    if v is not None:
                        ax.scatter(i + dx, v, marker=mk, s=40 if mk == "*" else 20, color=ARR_COL[arr], edgecolor=P.INK,
                                   lw=0.5, zorder=4, label=mlab if (arr == "A" and i == 0) else None)
                g = sel[arr].get("greedy", {}).get("val_r")
                if g and len(g) >= n:
                    ax.scatter(i + dx, g[n - 1], marker="o", facecolor="none", edgecolor=ARR_COL[arr], s=24,
                               label="greedy (before swaps)" if (arr == "A" and i == 0) else None)
        ax.set_xticks(range(len(E.NS)), [str(n) for n in E.NS])
        ax.set_xlabel("channels (blue A, orange B)"); ax.set_ylabel("surrogate validation r (lagged ridge)")
        ax.set_title("(d) Channel selection on the surrogate (bars: 5-95% of 1000 random sets)", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
    # (e) surrogate vs LSTM
    ax = fig.add_subplot(gs[1, 1])
    if "surrogate_vs_lstm" in s:
        p = np.array(s["surrogate_vs_lstm"]["points"])
        ax.scatter(p[:, 0], p[:, 2], s=14, color=P.SERIES[0])
        ax.set_xlabel("surrogate validation r"); ax.set_ylabel("LSTM test r (mean of 4 dims)")
        ax.set_title(f"(e) Does the surrogate track the LSTM? (random-n sets; r = {s['surrogate_vs_lstm']['r_test']:.2f})",
                     fontsize=9)
    # (f) training data amount
    ax = fig.add_subplot(gs[1, 2])
    da = s["data_amount"]
    ks = sorted((k for k in da if "mean_R" in da[k]), key=int)
    for h, ls, lab in hands:
        x = [da[k].get("minutes", int(k)) for k in ks]
        ax.plot(x, [da[k][f"mean_{h}"] for k in ks], ls=ls, marker="o", color=P.SERIES[0] if h == "R" else P.SERIES[2],
                label=lab)
        for k in ks:
            if f"r_{h}" in da[k]:
                ax.scatter([da[k].get("minutes", int(k))] * len(da[k][f"r_{h}"]), da[k][f"r_{h}"], s=6, color=P.INK)
    ax.set_xlabel("training data (minutes; the last k of blocks 1-5)"); ax.set_ylabel("test r")
    ax.set_title("(f) How much training data?", fontsize=9); ax.legend(fontsize=7)
    # (g) other days
    ax = fig.add_subplot(gs[2, 0:2])
    keys = [k for k in ("1750_all", "1750_A", "1750_B", "1867_all", "1867_A", "1867_B") if k in s["days"]]
    for i, k in enumerate(keys):
        v = s["days"][k]
        for j, (h, _, lab) in enumerate(hands):
            x = i + (j - 0.5) * 0.3
            ax.bar(x, v[f"mean_{h}"], width=0.28, color=P.SERIES[0] if h == "R" else P.SERIES[2],
                   label=lab if i == 0 else None)
            ax.scatter([x] * len(v[f"r_{h}"]), v[f"r_{h}"], s=6, color=P.INK, zorder=3)
            if "bi_R" in v:
                ax.scatter(x, v[f"bi_{h}"], marker="_", s=200, color=P.INK, zorder=4,
                           label="two-hand trials only" if (i == 0 and j == 0) else None)
    ax.set_xticks(range(len(keys)), [k.replace("_", " · ").replace("all", "both arrays").replace(" A", " array A")
                                     .replace(" B", " array B") for k in keys], fontsize=7.5)
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("test r (last 2 OL blocks)")
    ax.set_title("(g) Other days: day 1750 (one- and two-hand trials), day 1867 (one-hand)", fontsize=9)
    # upper left: the day-1750 bars top out at about 0.75, so the legend sits clear of every bar
    ax.legend(fontsize=7, loc="upper left", ncol=3, framealpha=1); ax.grid(axis="x", visible=False)
    # (h) proxy under muting
    ax = fig.add_subplot(gs[2, 2])
    if "mute" in s:
        for kind, color in (("normal", P.INK_MUTED), ("dropout", P.SERIES[3])):
            if kind in s["mute"]:
                px = s["mute"][kind]["proxy"]
                fs = sorted(px, key=float)
                ax.plot([float(f) * 100 for f in fs], [px[f] for f in fs], marker="o", color=color,
                        label="dropout-trained" if kind == "dropout" else "normal")
        ax.set_xlabel("% channels muted (zeroed)"); ax.set_ylabel("closed-loop proxy r")
        ax.set_title("(h) Closed-loop proxy under muting (weak metric, #35)", fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("Extension: feasibility for a smaller implant (T5 threshold crossings, LSTM, 2-8-s clips; day 1869 "
                 "unless stated)", fontsize=10, color=P.INK)
    fig.tight_layout()
    P.save(fig, "ext_feasibility")
    print("wrote", P.FIG_DIR / "ext_feasibility.png")
