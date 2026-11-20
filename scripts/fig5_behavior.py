"""Phase 5: Fig 5a-c and Fig 4e, online two-cursor performance re-scored from the logs.

Purpose
-------
Closed-loop (CL) control can't be re-run without the participant, but the
logged cursor and target traces let us re-score what happened:

* Fig 5a: RNN two-cursor control on the evaluation days. The paper reports
  that bimanual time to acquisition (TTA) is only slightly longer than
  unimanual, and that bimanual angular errors are higher.
* Fig 5b (days 1881, 1883): the SEQUENTIAL strategy (one cursor at a time,
  with an RNN recalibrated on unimanual data) is faster than SIMULTANEOUS
  bimanual movement.
* Fig 5c (days 1853, 1855): the RNN beats a linear decoder at every gain.
  Linear decoders mostly fail when the hands have to be separated, but work
  on unimanual-only blocks where the idle cursor is locked.
* Fig 4e (days 1867, 1869): the RNN trained on augmented data gives
  continuous movements and acquires targets; the raw-trained RNN makes
  pulse-like movements.

Metrics (register #7, #8)
-------------------------
* Success = durationBinnedMove < 499 bins (499 = the 10-s timeout).
* TTA = durationBinnedMove x 20 ms, the task's own record, which includes
  the 500-ms dwell. Reported (i) over successful trials and (ii) with
  failures counted as 10 s.
* Angular error = mean |angle(cursor displacement) - angle(target - cursor
  at go)| over 300-500 ms after go (behavior.angular_error), per moving hand.
  Bimanual trials use the mean of the two hands.
* 95% CIs from 10,000 bootstrap resamples of trials (Methods).
* Only "CL" evaluation blocks; assisted recalibration (CLR) blocks are excluded.
* Fig 5a day set (register #29): MAIN = 1776, 1778, 1790, 1792, the days
  whose CL blocks are standard RNN evaluation. Days 1752/1771 have no CL
  data; they were RNN calibration days. The 1881/1883 "simultaneous" blocks
  are shown as an extra panel (Supp Table 2 lists them under Fig 5A).

Inputs : Data/ (CL days)
Outputs: results/fig5/trials.csv (one row per CL trial); results/fig5/summary.json
         results/figures/fig5_online.{png,svg}, fig4e_online_speed.{png,svg}
Run    : pixi run fig5-behavior        (about 1 min; --force to redo)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bimanual import behavior as B
from bimanual import plotting as P
from bimanual.cache import RESULTS_DIR, atomic_write_json, should_skip
from bimanual.io import load_session
from bimanual.stats import bootstrap_ci

CL_DAYS = (1776, 1778, 1790, 1792, 1853, 1855, 1867, 1869, 1881, 1883)
FIG5A_DAYS = (1776, 1778, 1790, 1792)
TIMEOUT_BINS = 499
OUT_DIR = RESULTS_DIR / "fig5"
MT_NAME = {1: "uniR", 2: "uniL", 3: "bi"}
MT_COL = {1: P.SERIES[0], 2: P.SERIES[2], 3: P.SERIES[1]}  # same entity colours as the Fig 2 PCA


# ----------------------------------------------------------------------------- trial table
def trial_table() -> pd.DataFrame:
    rows = []
    for day in CL_DAYS:
        s = load_session(day)
        t = s.trials
        for r in t[t.kind == "CL"].itertuples():
            ok = r.move_bins < TIMEOUT_BINS
            angs = []
            if r.move_type in (1, 3):
                angs.append(B.angular_error(s.Cp, s.Tp, r.go, "R"))
            if r.move_type in (2, 3):
                angs.append(B.angular_error(s.Cp, s.Tp, r.go, "L"))
            rows.append(dict(
                day=day, block=r.block, decoder=r.decoder, gain=r.gain, variant=r.variant, uni_only=r.uni_only,
                move_type=MT_NAME[r.move_type], success=bool(ok),
                tta_s=r.move_bins * 0.02 if ok else np.nan,
                tta_fail10_s=min(r.move_bins * 0.02, 10.0) if ok else 10.0,
                ang_err=float(np.nanmean(angs)) if np.any(~np.isnan(angs)) else np.nan,
                go=r.go,
            ))
    return pd.DataFrame(rows)


def ci(x):
    """Mean with a 95% bootstrap CI (10k), as a [mean, lo, hi] list."""
    return list(bootstrap_ci(np.asarray(x, float)))


def summarize(df: pd.DataFrame, by: list[str]) -> list[dict]:
    out = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        out.append(dict(**{k: (None if pd.isna(v) else v) for k, v in zip(by, key)}, n=len(g),
                        success=ci(g.success.astype(float)), tta=ci(g.tta_s), tta_fail10=ci(g.tta_fail10_s),
                        ang_err=ci(g.ang_err)))
    return out


# ----------------------------------------------------------------------------- sanity check (register #7)
def tta_sanity(day=1776, radius=60.0):
    """How often the TTA recomputed from positions (fixed radius) agrees with the logged value."""
    s = load_session(day)
    t = s.trials[(s.trials.kind == "CL") & (s.trials.move_bins < TIMEOUT_BINS)]
    diffs = []
    for r in t.itertuples():
        tta = B.time_to_acquisition(s.Cp, s.Tp, r.go, radius, end_bin=r.move_end + 2)
        diffs.append(tta - r.move_bins * 0.02)
    d = np.array(diffs)
    return dict(day=day, radius=radius, n=len(d), frac_within_2_bins=float(np.mean(np.abs(d) <= 0.04 + 1e-9)),
                median_diff_s=float(np.nanmedian(d)))


# ----------------------------------------------------------------------------- Fig 4e speed
def speed_profiles(df):
    """Mean speed of the active cursor, 0-4 s after go, for raw vs augmented RNN blocks (1867, 1869)."""
    out = {}
    n_bins = 200
    for day in (1867, 1869):
        s = load_session(day)
        sub = df[(df.day == day) & df.variant.isin(["raw", "augmented"])]
        for var in ("raw", "augmented"):
            prof = []
            for r in sub[sub.variant == var].itertuples():
                hand = "R" if r.move_type == "uniR" else "L"
                sp = B.speed_profile(s.Cp, int(r.go), hand, n_bins)
                end = min(n_bins, int(round(r.tta_fail10_s / 0.02)))  # only until the trial ended
                sp[end:] = np.nan
                prof.append(sp)
            out[(day, var)] = np.array(prof)
    return out


# ----------------------------------------------------------------------------- figures
def plot(df, summ, speeds):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    order = ("uniR", "uniL", "bi")

    # (a) TTA and angular error by movement type, per evaluation day
    for ax, metric, ylab in ((axes[0, 0], "tta", "TTA, successful trials (s)"),
                             (axes[0, 1], "ang_err", "angular error 300–500 ms (deg)")):
        for i, day in enumerate(FIG5A_DAYS):
            for j, mt in enumerate(order):
                g = df[(df.day == day) & (df.decoder == "RNN") & df.variant.isna() & (df.move_type == mt)]
                m, lo, hi = ci(g.tta_s if metric == "tta" else g.ang_err)
                x = i + (j - 1) * 0.25
                ax.bar(x, m, width=0.23, color=MT_COL[j + 1], label=mt if i == 0 else None)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=P.INK, lw=0.8, capsize=2)
        succ = [df[(df.day == d) & (df.decoder == "RNN")].success.mean() for d in FIG5A_DAYS]
        ax.set_xticks(range(len(FIG5A_DAYS)), [f"{d}\n{100 * s_:.0f}% succ." for d, s_ in zip(FIG5A_DAYS, succ)], fontsize=7)
        ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    axes[0, 0].set_title("(a) RNN two-cursor control: time to acquisition", fontsize=9)
    axes[0, 1].set_title("(a) RNN: initial angular error", fontsize=9)
    axes[0, 0].legend(fontsize=7, ncol=3)

    # (b) sequential vs simultaneous strategy (1881 + 1883 pooled)
    ax = axes[0, 2]
    sub = df[df.day.isin([1881, 1883]) & df.variant.isin(["sequential", "simultaneous"])]
    for k, (var, hatch) in enumerate((("sequential", None), ("simultaneous", "//"))):
        for j, mt in enumerate(order):
            g = sub[(sub.variant == var) & (sub.move_type == mt)]
            m, lo, hi = ci(g.tta_fail10_s)
            x = j + (k - 0.5) * 0.36
            ax.bar(x, m, width=0.34, color=MT_COL[j + 1], hatch=hatch, edgecolor="white" if hatch else None,
                   label=var if j == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=P.INK, lw=0.8, capsize=2)
    ax.set_xticks(range(3), order); ax.set_ylabel("TTA, failures = 10 s (s)")
    ax.set_title("(b) Sequential vs simultaneous strategy (1881+1883)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="x", visible=False)

    # (c) RNN vs linear across gains (1853, 1855), mixed blocks and uni-only blocks
    for ax, (uo, title) in zip((axes[1, 0], axes[1, 1]),
                               ((False, "(c) RNN vs linear: blocks with bimanual + unimanual trials"),
                                (True, "(c) RNN vs linear: unimanual-only blocks (1855)"))):
        sub = df[df.day.isin([1853, 1855]) & (df.uni_only == uo) & df.gain.notna()]
        for dec, col, mk in (("RNN", P.SERIES[1], "o"), ("linear", P.SERIES[0], "s")):
            for day, ls in ((1853, "--"), (1855, "-")):
                g = sub[(sub.decoder == dec) & (sub.day == day)]
                if g.empty:
                    continue
                pts = [(gn, *ci(gg.tta_fail10_s)) for gn, gg in g.groupby("gain")]
                xs, ms, los, his = map(np.array, zip(*pts))
                ax.plot(xs, ms, ls=ls, marker=mk, color=col, label=f"{dec} ({day})")
                ax.fill_between(xs, los, his, color=col, alpha=0.12, lw=0)
        ax.axhline(10, color=P.INK_MUTED, ls=":", lw=1); ax.text(ax.get_xlim()[0], 9.6, " timeout", fontsize=7, color=P.INK_MUTED)
        ax.set_xlabel("decoder output gain"); ax.set_ylabel("TTA, failures = 10 s (s)"); ax.set_ylim(0, 10.5)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7)

    # Fig 4e summary: success + TTA for raw vs augmented
    ax = axes[1, 2]
    for i, day in enumerate((1867, 1869)):
        for k, (var, col) in enumerate((("raw", P.SERIES[0]), ("augmented", P.SERIES[1]))):
            g = df[(df.day == day) & (df.variant == var)]
            m, lo, hi = ci(g.success.astype(float))
            x = i + (k - 0.5) * 0.36
            ax.bar(x, 100 * m, width=0.34, color=col, label=var if i == 0 else None)
            ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=P.INK, lw=0.8, capsize=2)
            ax.text(x, 100 * hi + 2, f"{100 * m:.0f}%", ha="center", fontsize=7, color=P.INK)
    ax.set_xticks([0, 1], ["1867", "1869"]); ax.set_ylim(0, 110); ax.set_ylabel("trials acquired (%)")
    ax.set_title("(Fig 4e) Online success: RNN raw vs augmented", fontsize=9); ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Fig 5 / Fig 4e replication: online performance re-scored from closed-loop logs "
                 "(95% bootstrap CI over trials)", fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "fig5_online")

    # Fig 4e speed traces
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    t = np.arange(200) * 0.02
    for ax, day in zip(axes, (1867, 1869)):
        for var, col in (("raw", P.SERIES[0]), ("augmented", P.SERIES[1])):
            pr = speeds[(day, var)]
            m = np.nanmean(pr, 0)
            # 95% bootstrap CI of the mean speed at each time bin (1000 resamples of trials)
            rng = np.random.default_rng(0)
            boots = np.array([np.nanmean(pr[rng.integers(0, len(pr), len(pr))], 0) for _ in range(1000)])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(t, m, color=col, label=f"{var} (n={len(pr)})")
        ax.set_title(f"Day {day}: mean active-cursor speed after go (95% CI; trials end at acquisition)", fontsize=8)
        ax.set_xlabel("time after go (s)")
    axes[0].set_ylabel("cursor speed (units/s)"); axes[0].legend(fontsize=7)
    fig.suptitle("Fig 4e replication: raw-trained RNN gives pulse-like movement, augmented RNN gives continuous movement",
                 fontsize=9, color=P.INK)
    fig.tight_layout()
    P.save(fig, "fig4e_online_speed")
    print("wrote", P.FIG_DIR / "fig5_online.png", "and fig4e_online_speed.png")


def main(force=False):
    out_json = OUT_DIR / "summary.json"
    if should_skip([out_json], force):
        print("outputs exist; --force to redo")
        return
    df = trial_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["go"]).to_csv(OUT_DIR / "trials.csv", index=False)
    rnn_std = df[df.day.isin(FIG5A_DAYS) & (df.decoder == "RNN") & df.variant.isna()]
    summ = dict(
        fig5a_by_day_type=summarize(rnn_std, ["day", "move_type"]),
        fig5a_pooled_by_type=summarize(rnn_std, ["move_type"]),
        fig5a_extra_simultaneous_1881_1883=summarize(df[df.variant == "simultaneous"], ["day", "move_type"]),
        fig5b=summarize(df[df.variant.isin(["sequential", "simultaneous"])], ["variant", "move_type"]),
        fig5c=summarize(df[df.day.isin([1853, 1855]) & df.gain.notna()], ["day", "uni_only", "decoder", "gain"]),
        fig4e=summarize(df[df.variant.isin(["raw", "augmented"])], ["day", "variant"]),
        tta_sanity=tta_sanity(),
    )
    atomic_write_json(out_json, summ)
    for row in summ["fig5a_pooled_by_type"]:
        print(f"5a {row['move_type']:5s}: success {100 * row['success'][0]:.0f}%  TTA {row['tta'][0]:.2f} s  "
              f"ang.err {row['ang_err'][0]:.1f} deg  (n={row['n']})")
    for row in summ["fig5b"]:
        print(f"5b {row['variant']:12s} {row['move_type']:5s}: TTA(fail=10) {row['tta_fail10'][0]:.2f} s  success {100 * row['success'][0]:.0f}%")
    print("TTA sanity (recomputed vs logged, radius 60, day 1776):", summ["tta_sanity"])
    plot(df, summ, speed_profiles(df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
