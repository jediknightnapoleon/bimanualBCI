"""Step 0.3: data inventory of the Dryad release -> docs/data_inventory.md

Purpose
-------
Describe what is actually inside each session file, and check it against
what the paper and readme say:

* variables, shapes, recording length, number of trials
* per block: readme label (OL / CLR / CL, decoder, gain, variant), trial
  counts per movement type, closed-loop success rate
* blocks present in the data but missing from the readme, and the reverse
* target layout (direction spread, distance), which decides the quadrant
  labels (open item 1)
* how ``durationBinnedMove`` behaves on closed-loop trials (open items 2-3)

Inputs : Data/t5_*.mat, Data/readme.txt; block labels from bimanual/sessions.py
Outputs: docs/data_inventory.md (committed); results/inventory.json (numbers)
Run    : pixi run inventory          (about 1 min; add --force to redo)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio

from bimanual import sessions
from bimanual.cache import REPO_ROOT, RESULTS_DIR, atomic_write_json, should_skip

OUT_MD = REPO_ROOT / "docs" / "data_inventory.md"
OUT_JSON = RESULTS_DIR / "inventory.json"
TIMEOUT_BINS = 499  # max durationBinnedMove seen on closed-loop trials = 10-s timeout
MT = {1: "uniR", 2: "uniL", 3: "bi"}


def block_label(b) -> str:
    """Short human-readable label for a sessions.Block (or a missing one)."""
    if b is None:
        return "**not in readme**"
    parts = [b.kind]
    if b.decoder:
        parts.append(b.decoder)
    if b.gain is not None:
        parts.append(f"gain {b.gain:g}")
    if b.variant:
        parts.append(b.variant)
    if b.uni_only:
        parts.append("uni only")
    return " · ".join(parts)


def inventory_session(s: sessions.Session) -> tuple[dict, list[str]]:
    """Collect stats for one session; returns (json dict, markdown lines)."""
    d = sio.loadmat(s.path, squeeze_me=True)
    go = d["goCue"].astype(int) - 1  # 1-based in file -> 0-based
    mt = d["moveType"].astype(int)
    mv = d["durationBinnedMove"].astype(int)
    dl = d["durationBinnedDelay"].astype(int)
    bn = d["blockNum"].astype(int)
    Tp, Cp = d["Tp"], d["Cp"]
    n_bins = d["tx"].shape[1]
    trial_block = bn[go]

    md = [f"## {s.date} · trial day {s.trial_day} · session {s.session_num}", ""]
    md.append(f"*Paper use:* {s.figures}  ")
    md.append(f"*Notes field:* {str(d['Notes']).strip()}  ")
    md.append(
        f"*Size:* `tx` {d['tx'].shape} ({d['tx'].dtype}), {n_bins} bins = "
        f"{n_bins * 0.02 / 3600:.2f} h recorded; {go.size} trials"
    )
    md.append("")
    md += [
        "| Block | Label (readme) | Trials | uniR / uniL / bi | Move dur. median (bins) | CL success |",
        "|---|---|---|---|---|---|",
    ]
    blocks_json = {}
    for b in np.unique(bn):
        m = trial_block == b
        lab = s.blocks.get(int(b))
        counts = [int(np.sum(mt[m] == k)) for k in (1, 2, 3)]
        succ = ""
        if lab is not None and lab.kind == "CL" and m.any():
            rate = float(np.mean(mv[m] < TIMEOUT_BINS))
            succ = f"{100 * rate:.0f}%"
        md.append(
            f"| {b} | {block_label(lab)} | {int(m.sum())} | {counts[0]} / {counts[1]} / {counts[2]} | "
            f"{np.median(mv[m]) if m.any() else '-'} | {succ} |"
        )
        blocks_json[int(b)] = dict(
            label=block_label(lab), n_trials=int(m.sum()), counts=counts,
            n_bins=int(np.sum(bn == b)),
        )
    missing = sorted(set(s.blocks) - set(np.unique(bn).tolist()))
    extra = sorted(set(np.unique(bn).tolist()) - set(s.blocks))
    md.append("")
    md.append(f"Readme blocks missing from the data: {missing or 'none'}. "
              f"Data blocks missing from the readme: {extra or 'none'}.")
    md.append("")

    # Target layout: direction and distance from cursor to target at the go cue, moving hands only.
    err = Tp[:, go] - Cp[:, go]
    layout = {}
    for hand, rows, movers in (("R", slice(0, 2), (1, 3)), ("L", slice(2, 4), (2, 3))):
        m = np.isin(mt, movers)
        e = err[rows][:, m]
        ang = np.degrees(np.arctan2(e[1], e[0])) % 360
        layout[hand] = dict(
            n=int(m.sum()),
            dist_min=float(np.hypot(*e).min()), dist_med=float(np.median(np.hypot(*e))),
            dist_max=float(np.hypot(*e).max()),
            quadrant_counts=np.histogram(ang, bins=[0, 90, 180, 270, 360])[0].tolist(),
            n_unique_angles=int(np.unique(np.round(ang, 1)).size),
        )

    # Closed-loop move durations: is 499 the timeout?
    cl = np.array([(s.blocks.get(int(b)) is not None and s.blocks[int(b)].kind == "CL") for b in trial_block])
    cl_dur = Counter(mv[cl].tolist()) if cl.any() else Counter()
    out = dict(
        file=s.path.name, trial_day=s.trial_day, n_bins=n_bins, n_trials=int(go.size),
        n_channels=int(d["tx"].shape[0]), tx_max=int(d["tx"].max()),
        movetype_counts={MT[k]: int(np.sum(mt == k)) for k in (1, 2, 3)},
        delay_bins=[int(dl.min()), int(dl.max())],
        blocks=blocks_json, readme_blocks_missing=missing, data_blocks_unlabeled=extra,
        target_layout=layout,
        cl_move_max=int(mv[cl].max()) if cl.any() else None,
        cl_n_at_max=int(cl_dur.get(int(mv[cl].max()), 0)) if cl.any() else 0,
        workspace=[float(Tp.min()), float(Tp.max())],
    )
    return out, md


def main(force: bool = False) -> None:
    if should_skip([OUT_MD, OUT_JSON], force):
        print(f"{OUT_MD} exists; use --force to redo")
        return
    all_json, body = {}, []
    for s in sessions.SESSIONS:
        print("inventory", s.path.name)
        j, md = inventory_session(s)
        all_json[s.date] = j
        body += md

    # Summary of the open items, worked out across all sessions.
    def cat(hand, key):
        return np.array([all_json[k]["target_layout"][hand][key] for k in all_json])

    qR = np.sum([all_json[k]["target_layout"]["R"]["quadrant_counts"] for k in all_json], axis=0)
    qL = np.sum([all_json[k]["target_layout"]["L"]["quadrant_counts"] for k in all_json], axis=0)
    cl_max = sorted({j["cl_move_max"] for j in all_json.values() if j["cl_move_max"] is not None})

    head = [
        "# Data inventory (Dryad doi:10.5061/dryad.sn02v6xbb)",
        "",
        "Generated by `scripts/01_inventory.py` (`pixi run inventory`). Block labels are transcribed from",
        "`Data/readme.txt` into `bimanual/sessions.py` and checked against the blocks actually in the data.",
        "",
        "## File format (all 13 sessions)",
        "",
        "| Variable | Shape | Meaning |",
        "|---|---|---|",
        "| `tx` | 192 × T, uint8 | Threshold-crossing counts per 20-ms bin (−3.5×RMS) |",
        "| `blockNum` | T | Block number of each bin. Gaps between blocks were not recorded |",
        "| `Cp`, `Tp` | 4 × T | Cursor / target positions: [R x, R y, L x, L y] |",
        "| `goCue` | R | **1-based** bin index of each trial's go cue |",
        "| `moveType` | R | 1 = uniR, 2 = uniL, 3 = bimanual (`moveTypeName`) |",
        "| `durationBinnedDelay` | R | Delay length in bins before the go cue |",
        "| `durationBinnedMove` | R | Move-period length in bins after the go cue |",
        "| `Notes` | str | Short session description |",
        "",
        "## Findings on the open items",
        "",
        "1. **Target layout and quadrants.** Targets are not at a few fixed angles: each trial's target is at a",
        "   random position in the workspace, which spans about ±300 units, and the cursor starts where the",
        "   previous target was. Movement directions cover all angles evenly. Summed over all sessions,",
        f"   the axis-aligned quadrant counts [0–90°, 90–180°, 180–270°, 270–360°] are R = {qR.tolist()} and L = {qL.tolist()}.",
        f"   The cursor-to-target distance at the go cue ranges from {cat('R','dist_min').min():.0f} to {cat('R','dist_max').max():.0f} units (R).",
        "   **Decision:** ASSUMPTION, axis-aligned quadrants of the cursor→target vector at the go cue. The",
        "   8-wedge labels of `tuningAnalysis.m` are kept for the Fig 1 tuning curves.",
        "2. **Target radius.** It is not stored in the files. Bounding it from closed-loop dwell periods (day 1776)",
        "   gives contradictory bounds at every alignment: during dwells the cursors are up to about 71 units from",
        "   the target centre, yet just before a dwell starts the distance can be as small as about 50. So",
        "   acquisition is not a simple fixed-radius circle around the target centre (possible causes: the",
        "   cursor's own radius, sub-bin timing, per-trial sizes). **Decision:** the radius is not needed; see item 3.",
        "3. **Time to acquisition.** On closed-loop trials, `durationBinnedMove` is the logged trial length after",
        f"   the go cue. Its maximum over closed-loop blocks is {cl_max} bins, which is the 10-s timeout. With a",
        "   trial radius of about 60, `behavior.time_to_acquisition` agrees with it to within 1–2 bins on most",
        "   trials of day 1776. **Decision:** Fig 5 uses the logged `durationBinnedMove`: TTA = value × 20 ms,",
        f"   and a value ≥ {TIMEOUT_BINS} is a failure. This is the task's own record, so it includes the 500-ms dwell.",
        "   `behavior.time_to_acquisition` stays as a sanity check only.",
        "",
        "## Per-session details",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    atomic_write_json(OUT_JSON, all_json)
    print("wrote", OUT_MD, "and", OUT_JSON)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    main(ap.parse_args().force)
