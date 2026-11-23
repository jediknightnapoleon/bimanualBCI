"""Cursor replays for visual inspection: neural activity, cursors, targets and our decoders' outputs.

Three kinds of scenario, each tied to a reported result:

* **online**  Logged closed-loop blocks (the authors' decoders, run live with T5). The cursor shown is
               the cursor T5 controlled; nothing is re-decoded. Fig 4e (raw vs fractured RNN), Fig 5a
               (uni- vs bimanual), Fig 5b (sequential vs simultaneous), Fig 5c (linear vs RNN).
* **offline** Day 1869, held-out block 7 (open loop: T5 attempted to follow a cued cursor). Our saved
               RNNs (Phases 4 and 8-10) are re-run on the recorded activity. Their decoded velocity is
               integrated from each trial's start and drawn next to the cued cursor.
               # ASSUMPTION: display gain per hand = least-squares fit of the cued cursor's per-bin
               # displacement on the training target velocity (block 7, both dims). It only sets the
               # drawing scale; the same gain is used for every decoder.
               This is an open-loop replay: no feedback, so errors accumulate within a trial. It is a
               view of the decoded velocity, not a simulation of closed-loop control.
               # DEVIATION (display only): by default the viewer puts each decoded cursor back onto the
               # cued cursor at every trial start, so drift doesn't carry over between trials. The jump
               # is drawn as a dashed line and can be switched off in the page. r is computed from the
               # velocity here in Python and does not depend on it.
* **fig3**    Day 1883 single-bin decoders (ridge, FFN, FFN without laterality): their cross-validated
               per-bin direction outputs in the 400-ms window, from results/cache.

Every re-run RNN is checked against the test r logged when it was trained (|diff| < 2e-3).

Outputs: docs/html/replays.html (standalone, data embedded); with --artifact-dir DIR also the
page and one JSON file per recording / scenario for the published artifact.
Run: pixi run replays [--artifact-dir DIR]   (about 1-2 min on a GPU)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_lstm as XL  # noqa: E402
import fig4_rnn as F  # noqa: E402

from bimanual import conditions as C  # noqa: E402
from bimanual import rnn as R  # noqa: E402
from bimanual.cache import RESULTS_DIR, read_jsonl  # noqa: E402
from bimanual.decoders import pearson_per_dim  # noqa: E402
from bimanual.io import load_session, zscore_by_block  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "replay_viewer.html"
STANDALONE = ROOT / "docs" / "html" / "replays.html"
HM_STEP = 2          # heat map: mean of 2 bins = 40 ms per column
HM_CLIP = 3.0        # heat map: z clipped to +-3, quantised to 0..255
VEL_SCALE = 1000.0   # decoded velocity stored as int16 / 1000
OFFLINE_S, ONLINE_S = 150, 120  # seconds kept per recording (keeps the page small)


# ----------------------------------------------------------------------------- encoding
def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def enc_i16(x, scale=1.0) -> str:
    return b64(np.clip(np.round(np.asarray(x, float) * scale), -32767, 32767), "<i2")


def enc_heatmap(z: np.ndarray) -> str:
    """(T, 192) z-scores -> (T // HM_STEP, 192) uint8, row-major by time."""
    n = len(z) // HM_STEP * HM_STEP
    m = np.nan_to_num(z[:n]).reshape(-1, HM_STEP, z.shape[1]).mean(1)
    return b64(np.round((np.clip(m, -HM_CLIP, HM_CLIP) + HM_CLIP) / (2 * HM_CLIP) * 255), "u1")


def trial_rows(s, a: int, b: int):
    """Trials that start inside [a, b), with bins relative to a."""
    t = s.trials[(s.trials.start >= a) & (s.trials.start < b)]
    return [dict(s=int(r.start - a), g=int(r.go - a), e=int(min(r.move_end, b) - a), mt=int(r.move_type),
                 ok=bool(r.move_bins < 499), tta=round(float(r.move_bins) * 0.02, 2)) for r in t.itertuples()]


def recording(s, a: int, b: int, z: np.ndarray, extra=None) -> dict:
    rec = dict(day=int(s.info.trial_day), n=int(b - a), dt=0.02, hm=enc_heatmap(z[a:b]), hm_step=HM_STEP,
               hm_clip=HM_CLIP, cp=enc_i16(s.Cp[:, a:b].T), tp=enc_i16(s.Tp[:, a:b].T), trials=trial_rows(s, a, b))
    rec.update(extra or {})
    return rec


# ----------------------------------------------------------------------------- online (logged)
ONLINE = [  # (scenario id, day, block, group, title, what to look for)
    ("fig4e_raw", 1869, 8, "Fig 4e · online RNN", "Raw-trained RNN, online",
     "The raw-trained RNN gives one burst of speed after the go cue, then the cursor nearly stops short of "
     "the target. 3 of 29 trials in this block were acquired. Compare with the fractured-data RNN."),
    ("fig4e_aug", 1869, 9, "Fig 4e · online RNN", "Fractured-data RNN, online",
     "Same day, same task. The RNN trained on time-warped, shuffled clips keeps the cursor moving until it "
     "reaches the target: 26 of 29 trials acquired."),
    ("fig5a_mixed", 1778, 20, "Fig 5a · two cursors", "One hand vs both hands",
     "Mixed block of right-hand, left-hand and two-hand trials. Two-hand trials take about twice as long "
     "(median 4.44 s vs 2.05 / 2.34 s on this day) and the paths are less direct."),
    ("fig5b_seq", 1881, 22, "Fig 5b · strategy", "Sequential strategy",
     "T5 was told to move one cursor at a time on two-hand trials. One cursor travels while the other "
     "waits: in this block the two never moved at the same time (0% of moving bins, vs 41% with the "
     "simultaneous strategy). Bimanual median 5.3 s over days 1881 and 1883."),
    ("fig5b_sim", 1881, 23, "Fig 5b · strategy", "Simultaneous strategy",
     "Same day, both cursors at once: on two-hand trials both move together in 41% of moving bins. Trials "
     "take longer (bimanual median 7.1 s) and fewer succeed: 38 of 49 vs 47 of 49 sequentially."),
    ("fig5c_lin", 1855, 17, "Fig 5c · decoder", "Linear decoder, mixed block",
     "A linear decoder on a block with one- and two-hand trials can't keep the hands apart. On one-hand "
     "trials the idle cursor travels almost as far as the active one (median 0.94 of its path length, vs "
     "0.01 with the RNN). 0 of 29 trials acquired."),
    ("fig5c_rnn", 1855, 16, "Fig 5c · decoder", "RNN decoder, mixed block",
     "The RNN on the same kind of block: the idle cursor stays put (0.01 of the active path length) and "
     "all 29 trials are acquired. On its two-hand trials the cursors moved one at a time."),
]


def online_scenarios(recs: dict, scens: list):
    cache = {}
    for sid, day, block, group, title, note in ONLINE:
        s = cache.get(day) or cache.setdefault(day, load_session(day))
        if ("z", day) not in cache:
            cache[("z", day)] = zscore_by_block(s.tx, s.block).astype(np.float32)
        idx = np.where(s.block == block)[0]
        a = int(s.trials[s.trials.block == block].start.min())
        b = min(int(idx[-1]) + 1, a + int(ONLINE_S / 0.02))
        rid = f"rec_{day}_b{block}"
        recs[rid] = recording(s, a, b, cache[("z", day)], dict(label=f"Day {day}, block {block}", kind="online"))
        scens.append(dict(id=sid, kind="online", group=group, title=title, note=note, rec=rid, decoders=[],
                          source=f"Logged closed-loop block {block} of day {day}; first {ONLINE_S} s."))
        print(f"online {sid}: day {day} block {block}, {len(recs[rid]['trials'])} trials")


# ----------------------------------------------------------------------------- offline (re-decoded)
def all_records():
    out = {}
    for p in ("rnn_search/trials.jsonl", "ext_lstm/trials.jsonl", "ext_arrays/trials.jsonl", "ext_feas/trials_w0.jsonl"):
        for r in read_jsonl(RESULTS_DIR / p):
            out.setdefault(r["run_id"], r)
    return out


def first(recs, prefix):
    return next(r for k, r in recs.items() if k.startswith(prefix) and "test" in r)


def offline_scenarios(recs_out: dict, scens: list):
    data = F.Data("clip")
    s = data.session
    (A, B), = data.seg["test"]
    n_keep = int(OFFLINE_S / 0.02)
    a, b = A, min(B, A + n_keep)
    Xt = data.z[A:B]
    tr = np.concatenate([s.tx[x:y] for x, y in data.seg["train"]]).astype(np.float64)
    dead_val = (-tr.mean(0) / np.where(tr.std(0) > 0, tr.std(0), 1.0)).astype(np.float32)

    # display gain per hand: cued-cursor displacement per bin ~ g * target velocity
    dC = np.diff(s.Cp[:, A:B], axis=1, append=s.Cp[:, B - 1:B]).T
    V = data.vel[A:B]
    gain = []
    for h in (slice(0, 2), slice(2, 4)):
        v, d = V[:, h].ravel(), dC[:, h].ravel()
        ok = np.isfinite(d) & (np.abs(d) < 60)  # drop the jump back to centre between trials
        gain.append(float(np.dot(v[ok], d[ok]) / np.dot(v[ok], v[ok])))
    rid = "rec_1869_b7"
    recs_out[rid] = recording(s, a, b, data.z_all, dict(
        label="Day 1869, held-out block 7 (open loop)", kind="offline", gain=gain,
        vel=enc_i16(V[: b - a], VEL_SCALE)))

    R_ = all_records()
    normal = XL.seed_records("lstm", "augL")[0]
    drop = first(R_, "drop_seed")
    models = {
        "gru_raw": (XL.seed_records("gru", "raw")[0], "GRU, raw data"),
        "gru_augL": (XL.seed_records("gru", "augL")[0], "GRU, 2-8-s fractured clips"),
        "gru_augS": (XL.seed_records("gru", "augS")[0], "GRU, 200-800-ms fractured clips"),
        "lstm_raw": (XL.seed_records("lstm", "raw")[0], "LSTM, raw data"),
        "lstm_augL": (normal, "LSTM, 2-8-s clips, all 192 channels"),
        "arrA": (first(R_, "arrA_lstm_augL"), "LSTM, array A only (channels 1-96)"),
        "arrB": (first(R_, "arrB_lstm_augL"), "LSTM, array B only (channels 97-192)"),
        "greedy8B": (first(R_, "greedy8B"), "LSTM, 8 selected channels of array B"),
        "rand8B1": (R_["rand8B1"], "LSTM, 8 random channels of array B (set 1)"),
        "rand8B0": (R_["rand8B0"], "LSTM, 8 random channels of array B (set 0)"),
        "drop": (drop, "LSTM trained with channel dropout"),
    }
    # (decoder key, model key, mute mode, fraction): masks as in ext_feasibility.stage_mute (mask 0)
    variants = {k: (k, None, 0.0) for k in models}
    variants.update({"normal_z50": ("lstm_augL", "zero", 0.5), "drop_z50": ("drop", "zero", 0.5),
                     "normal_d30": ("lstm_augL", "dead", 0.3), "drop_d30": ("drop", "dead", 0.3)})
    labels = {"normal_z50": "Normal LSTM, 50% of channels zeroed (detected)",
              "drop_z50": "Dropout-trained LSTM, 50% zeroed (detected)",
              "normal_d30": "Normal LSTM, 30% dead channels (not detected)",
              "drop_d30": "Dropout-trained LSTM, 30% dead (not detected)"}
    decoded, loaded = {}, {}
    for key, (mkey, mode, f) in variants.items():
        rec, lab = models[mkey]
        ch = rec.get("channels")
        if mkey not in loaded:
            loaded[mkey] = R.load_from_record(rec, len(ch) if ch else 192, F.CKPT_DIR)
        X = Xt.copy()
        muted = []
        if mode:
            rng = np.random.default_rng(7000 + int(round(100 * f)))
            muted = sorted(int(c) for c in rng.choice(192, int(round(f * 192)), replace=False))
            X[:, muted] = 0.0 if mode == "zero" else dead_val[muted]
        vh = R.decode(loaded[mkey], X[:, ch] if ch else X)[0]
        r = pearson_per_dim(V, vh)
        if not mode:  # the re-run reproduces the logged test r
            assert abs(r[:2].mean() - rec["test"]["r_R"]) < 2e-3 and abs(r[2:].mean() - rec["test"]["r_L"]) < 2e-3, \
                (rec["run_id"], r, rec["test"])
        decoded[key] = dict(key=key, label=labels.get(key, lab), run_id=rec["run_id"],
                            r=[round(float(r[:2].mean()), 3), round(float(r[2:].mean()), 3)],
                            channels=list(map(int, ch)) if ch else None, muted=muted, mute_mode=mode,
                            vel=enc_i16(vh[: b - a], VEL_SCALE))
        print(f"offline {key:11s} {rec['run_id']:28s} r {r[:2].mean():.3f} / {r[2:].mean():.3f}")

    OFF = [  # (id, group, title, decoders, note)
        ("p4_fracture", "Fig 4d · RNN training", "Raw vs fractured training",
         ["gru_raw", "gru_augL", "gru_augS"],
         "The paper's GRU. Raw training gives the smoothest, most accurate offline output; 2-8-s fractured "
         "clips cost a little and are noisier; 200-800-ms clips cost much more (3-seed means 0.96 / 0.89, "
         "0.92 / 0.86, 0.76 / 0.67). Offline r does not predict online success: compare the Fig 4e replays."),
        ("p8_lstm", "Phase 8 · LSTM vs GRU", "LSTM vs GRU under fracturing",
         ["gru_augL", "lstm_augL", "lstm_raw"],
         "With 2-8-s clips the LSTM tracks the cue more closely and more smoothly than the GRU "
         "(3-seed means r 0.93 / 0.88 vs 0.92 / 0.86; roughness 0.08 vs 0.11). On raw data they tie."),
        ("p9_arrays", "Phase 9 · one array", "One array instead of two",
         ["lstm_augL", "arrA", "arrB"],
         "Array B alone keeps essentially all of the performance; array A alone loses most on the left hand. "
         "The heat map marks the channels each decoder used when you select it in the legend."),
        ("p10_channels", "Phase 10 · channel count", "8 selected vs 8 random channels",
         ["greedy8B", "rand8B1", "rand8B0"],
         "8 channels picked by greedy selection keep 96% / 97% of the 192-channel r. Random sets of 8 from "
         "the same array vary widely (here r 0.59 / 0.35 and 0.33 / 0.15)."),
        ("p10_zeroed", "Phase 10 · lost channels", "Half the channels lost (detected)",
         ["lstm_augL", "normal_z50", "drop_z50"],
         "Zeroing 50% of channels (pale rows on the heat map) shrinks and bends a normal decoder's output; the "
         "decoder trained with channel dropout barely changes. Over 20 random masks it keeps 95% / 94% of its "
         "r vs 67% / 71%; the mask drawn here is milder than average for the normal decoder."),
        ("p10_dead", "Phase 10 · lost channels", "Dead channels nobody detected",
         ["lstm_augL", "normal_d30", "drop_d30"],
         "If 30% of channels go silent without being detected, their z-scores sit far below anything seen in "
         "training and every decoder drifts, dropout-trained or not. Averaged over 20 masks r falls to about "
         "0.5 / 0.3 for both; individual masks vary (see the legend). Detecting bad channels matters."),
    ]
    for sid, group, title, keys, note in OFF:
        scens.append(dict(id=sid, kind="offline", group=group, title=title, note=note, rec=rid,
                          decoders=[decoded[k] for k in keys],
                          source=f"Day 1869 block 7, first {OFFLINE_S} s; models re-run on the recorded activity."))


# ----------------------------------------------------------------------------- Fig 3 (single-bin decoders)
def fig3_scenario(recs_out: dict, scens: list, every: int = 4):
    import fig3_decoding as D3

    s = load_session(D3.DAY)
    z = zscore_by_block(s.tx, s.block)
    ol = (s.trials.kind == "OL").to_numpy()
    mt = s.trials.move_type.to_numpy()
    idx = np.where(ol & np.isin(mt, (1, 2)))[0]
    dirs = C.trial_directions(s.Tp, s.Cp, s.go, mt)
    W = D3.WINDOW[1] - D3.WINDOW[0] + 1
    jit = json.loads((RESULTS_DIR / "fig3" / "decoding_direction.json").read_text())["results"]
    preds = {k: np.load(RESULTS_DIR / "cache" / f"fig3_direction_uni_{k}.npz")["pred"] for k in ("ridge", "ffn", "ffn_nolat")}
    assert all(len(p) == len(idx) * W for p in preds.values())
    keep = np.arange(0, len(idx), every)
    trials, hm = [], []
    for k in keep:
        i = idx[k]
        g0 = s.go[i] + D3.WINDOW[0]
        trials.append(dict(mt=int(mt[i]), dir=[round(float(v), 3) for v in dirs[i]], go_bin=int(s.go[i])))
        hm.append(z[g0:g0 + W])
    rid = "rec_1883_fig3"
    recs_out[rid] = dict(day=D3.DAY, kind="fig3", label="Day 1883, open-loop one-hand trials", W=W, dt=0.02,
                         window_ms=[D3.WINDOW[0] * 20, (D3.WINDOW[1] + 1) * 20], trials=trials,
                         hm=enc_heatmap_full(np.concatenate(hm)), hm_step=1, hm_clip=HM_CLIP)
    names = {"ridge": "Ridge (linear)", "ffn": "FFN (1 layer, 512 units)", "ffn_nolat": "FFN, laterality removed"}
    decs = []
    for kk, p in preds.items():
        sel = np.concatenate([p[k * W:(k + 1) * W] for k in keep])
        j = jit["uni_" + kk]["jitter"]
        decs.append(dict(key=kk, label=names[kk], jitter=[round(j["R"], 2), round(j["L"], 2)],
                         r=[round(float(np.mean(jit["uni_" + kk]["r"][:2])), 3), round(float(np.mean(jit["uni_" + kk]["r"][2:])), 3)],
                         vel=enc_i16(sel, VEL_SCALE)))
    scens.append(dict(
        id="fig3_jitter", kind="fig3", group="Fig 3 · nonlinear decoding", title="Idle-hand jitter: ridge vs FFN",
        rec=rid, decoders=decs,
        source=f"Day 1883, every {every}th one-hand open-loop trial; cross-validated predictions (5 folds).",
        note="Each trial shows the decoded direction, step by step over the 400-ms window, for the moving hand "
             "and the idle hand. The idle hand should not move. Ridge leaves it wandering (idle / active speed "
             "0.84 for the left hand); the FFN keeps it near the centre (0.42). Removing the laterality "
             "dimension makes the FFN as jittery as ridge, which is the paper's argument for nonlinear decoders."))
    print(f"fig3: {len(trials)} trials")


def enc_heatmap_full(z: np.ndarray) -> str:
    return b64(np.round((np.clip(np.nan_to_num(z), -HM_CLIP, HM_CLIP) + HM_CLIP) / (2 * HM_CLIP) * 255), "u1")


# ----------------------------------------------------------------------------- output
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=None)
    a = ap.parse_args()
    recs, scens = {}, []
    online_scenarios(recs, scens)
    offline_scenarios(recs, scens)
    fig3_scenario(recs, scens)
    order = ["fig4e_raw", "fig4e_aug", "fig5a_mixed", "fig5b_seq", "fig5b_sim", "fig5c_lin", "fig5c_rnn",
             "p4_fracture", "fig3_jitter", "p8_lstm", "p9_arrays", "p10_channels", "p10_zeroed", "p10_dead"]
    scens.sort(key=lambda x: order.index(x["id"]))
    index = [{k: v for k, v in sc.items() if k in ("id", "kind", "group", "title")} for sc in scens]
    page = TEMPLATE.read_text(encoding="utf-8")

    data = json.dumps(dict(recordings=recs, scenarios={sc["id"]: sc for sc in scens}), separators=(",", ":"))
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n')
    inject = f"<script>window.REPLAY_INDEX={json.dumps(index)};window.REPLAY_DATA={data};</script>\n"
    STANDALONE.write_text(head + inject + page + "\n</body>\n</html>\n", encoding="utf-8")
    print(f"wrote {STANDALONE.relative_to(ROOT)} ({STANDALONE.stat().st_size / 1e6:.1f} MB)")

    if a.artifact_dir:
        d = a.artifact_dir
        (d / "data").mkdir(parents=True, exist_ok=True)
        for rid, rec in recs.items():
            (d / "data" / f"{rid}.json").write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
        for sc in scens:
            (d / "data" / f"{sc['id']}.json").write_text(json.dumps(sc, separators=(",", ":")), encoding="utf-8")
        (d / "replays.html").write_text(f"<script>window.REPLAY_INDEX={json.dumps(index)};</script>\n" + page,
                                        encoding="utf-8")
        print(f"wrote artifact files to {d}")


if __name__ == "__main__":
    main()
