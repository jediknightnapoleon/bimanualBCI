"""Phase 11 figure: threshold crossings (TX) vs spike-band power (SBP) in T5 (results/ext_sbp/summary.json).

Panels
------
(a) velocity r by feature and decoder        (b) 8-target accuracy (movement / delay period)
(c-e) channel count, velocity r vs n: all 48 channels / array A / array B; random-subset median
      with 5-95% band, greedy best-n dashed
(f, g) velocity r vs muted fraction, normal / dropout-trained ridge; mode by line style
(h) per-channel overlap: held-out R^2 of logSBP from the same electrode's TX vs TX rate

Colours (fixed by entity, register of plotting.SERIES): TX slot 1, SBP slot 2, both slot 3,
residual slot 4; raw-SBP points use the SBP colour with an open marker.
Run: pixi run python scripts/ext_sbp_plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bimanual import plotting as P  # noqa: E402
from bimanual.sbp import BIN_MS, OUT_DIR  # noqa: E402

plt = P.plt
COL = dict(TX=P.SERIES[0], logSBP=P.SERIES[1], SBP=P.SERIES[1], both=P.SERIES[2], resid=P.SERIES[3])
LABEL = dict(TX="TX", logSBP="SBP (log)", SBP="SBP (raw)", both="TX + SBP", resid="SBP residual")
FEATS = ["TX", "SBP", "logSBP", "both", "resid"]
MODE_LS = dict(zero="-", dead="--", noisy=":")


def _feature_dots(ax, groups, data, ylabel, chance=None):
    """Dots per feature within each group; groups = [(name, label)], data[name][feat]['score']."""
    width = 0.75
    for gi, (g, glabel) in enumerate(groups):
        for fi, f in enumerate(FEATS):
            x = gi + (fi - (len(FEATS) - 1) / 2) * width / len(FEATS)
            v = data[g][f]["score"]
            ax.plot([x], [v], "o", ms=7, color=COL[f], mfc="white" if f == "SBP" else COL[f], mew=1.6,
                    label=LABEL[f] if gi == 0 else None, zorder=3)
        dd = data[g]["logSBP"]["vs_TX"]
        ax.annotate(f"SBP−TX {dd['diff']:+.3f}\n[{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}]",
                    (gi, 0), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=P.INK_MUTED)
    if chance is not None:
        ax.axhline(chance, color=P.INK_MUTED, lw=1, ls="--")
        ax.text(len(groups) - 0.5, chance, "chance", fontsize=7, color=P.INK_MUTED, ha="right", va="bottom")
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel)


def main():
    s = json.loads((OUT_DIR / "summary.json").read_text())
    m = s["meta"]
    i1 = s["item1_features"]
    fig = plt.figure(figsize=(13, 11.5))
    gs = fig.add_gridspec(3, 6, hspace=0.55, wspace=0.9, top=0.93)

    # (a) velocity r
    ax = fig.add_subplot(gs[0, 0:3])
    _feature_dots(ax, [("velocity_ridge", "ridge"), ("velocity_ridge_lags", "ridge + lags"), ("velocity_ffn", "FFN")],
                  i1, "cursor velocity r (held-out blocks)")
    lo = min(i1[g][f]["score"] for g in ("velocity_ridge", "velocity_ridge_lags", "velocity_ffn") for f in FEATS)
    ax.set_ylim(lo - 0.15, None)
    ax.set_title("(a) Velocity decoding, per 20-ms bin", loc="left")

    # (b) classification
    ax = fig.add_subplot(gs[0, 3:6])
    _feature_dots(ax, [("class8_ridge", f"movement\n{m['class8_window_ms'][0]}–{m['class8_window_ms'][1]} ms"),
                       ("delay8_ridge", f"delay period\n{m['delay8_window_ms'][0]}–{m['delay8_window_ms'][1]} ms")],
                  i1, "8-target accuracy", chance=m["chance_class8"])
    ax.set_ylim(-0.12, 1.3)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("(b) Target classification (ridge)", loc="left")
    ax.legend(loc="upper right", ncol=3, fontsize=7.5, title="feature (legend for a and b)", title_fontsize=7.5)

    # (c-e) channel count
    if "item2_channel_count" in s:
        cc = s["item2_channel_count"]["velocity"]
        for k, (pool, title) in enumerate([("all", "all 48 channels"), ("A", f"array A ({m['n_chan']['A']} ch)"),
                                           ("B", f"array B ({m['n_chan']['B']} ch)")]):
            ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
            ns = [int(n) for n in cc[pool]]
            for f in ("TX", "logSBP"):
                med = [cc[pool][str(n)][f]["median"] for n in ns]
                p5 = [cc[pool][str(n)][f]["p5"] for n in ns]
                p95 = [cc[pool][str(n)][f]["p95"] for n in ns]
                ax.fill_between(ns, p5, p95, color=COL[f], alpha=0.15, lw=0)
                ax.plot(ns, med, "o-", color=COL[f], ms=5, label=f"{LABEL[f]} random (median, 5–95%)")
                g = cc["greedy"][pool][f]["score"]
                ax.plot(ns, [g[str(n)] for n in ns], "s--", color=COL[f], ms=4, mfc="white", lw=1.5,
                        label=f"{LABEL[f]} greedy best-n")
            ax.set_xscale("log", base=2)
            ax.set_xticks(ns, [str(n) for n in ns])
            ax.set_xlabel("channels")
            if k == 0:
                ax.set_ylabel("velocity r (ridge)")
                ax.legend(fontsize=7, loc="lower right")
            ax.set_title(f"({'cde'[k]}) Channel count: {title}", loc="left")

    # (f, g) muting
    if "item3_muting" in s:
        mu = s["item3_muting"]["velocity"]
        for k, kind in enumerate(("normal", "dropout")):
            ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
            for f in ("TX", "logSBP"):
                for mode, ls in MODE_LS.items():
                    c = mu[f][kind][mode]
                    ax.plot([e["f"] for e in c], [e["mean"] for e in c], ls=ls, color=COL[f], marker="o", ms=3.5,
                            label=f"{LABEL[f]}, {mode}" if k == 0 else None)
            ax.set_xlabel("fraction of channels muted at test")
            ax.set_ylabel("velocity r (ridge)") if k == 0 else None
            ax.set_title(f"({'fg'[k]}) Muting, {'normal' if kind == 'normal' else 'dropout-trained'} ridge", loc="left")
            if k == 0:
                ax.legend(fontsize=6.5, ncol=2, loc="lower left")

    # (h) overlap
    if "item4_overlap" in s:
        o = s["item4_overlap"]["per_channel"]
        ax = fig.add_subplot(gs[2, 4:6])
        rate = np.array(o["tx_mean_count_per_bin"]) * 1000 / BIN_MS
        arr = np.array(o["array"])
        for a, mk in (("A", "o"), ("B", "^")):
            sel = arr == a
            ax.plot(rate[sel], np.array(o["heldout_r2_own_tx"])[sel], mk, color=P.INK_MUTED, ms=5, mfc="none",
                    label=f"array {a}")
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("TX rate (Hz, smoothed counts)")
        ax.set_ylabel("held-out R² of SBP from own TX")
        ax.set_ylim(-0.05, 1.1)
        ov = s["item4_overlap"]
        ax.set_title("(h) How much of SBP is TX", loc="left")
        ax.text(0.02, 0.97, f"median R² {ov['heldout_r2_own_tx']['median']:.2f} (own electrode), "
                f"{ov['heldout_r2_all_tx']['median']:.2f} (all 48 TX)\nmedian corr(TX, log SBP) "
                f"{ov['corr_tx_logsbp']['median']:.2f}", transform=ax.transAxes, va="top", fontsize=7, color=P.INK)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.86))

    fig.suptitle("Phase 11: TX vs SBP in T5 (Rigotti-Thompson 2026 data; open-loop radial-8 wrist task, 48 channels, "
                 "acausally smoothed; leave-one-block-out)", fontsize=10, x=0.02, ha="left")
    print(P.save(fig, "ext_sbp"))


if __name__ == "__main__":
    main()
