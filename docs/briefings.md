# Phase briefings

Written at the end of each phase while the user was away (see the 2026-11-21 entry in PROGRESS.md). Newest first.

---

## Phase 8: LSTM vs GRU (day 1869, Fig 4 setup). Done.

- **What:** 53 new models (3.03 GPU-h; skipped 1, `lstm433_augL_seed102`):
  - the LSTM-512 paired search on raw data, 2–8-s clips and 200–800-ms clips
  - the GRU search on 2–8-s clips (#34)
  - 3 seeds of each best setting
  - a parameter-matched LSTM-433
- **Result:** the LSTM is better, mostly once temporal fracturing is used.
  - Paired validation r, LSTM − GRU:
    - raw +0.013 (9/10 settings, p = 0.006)
    - 2–8 s +0.033 (10/10, p = 0.002)
    - 0.2–0.8 s +0.113 (10/10, p = 0.002)
  - Test r (RH / LH): raw ties at 0.96 / 0.89; 2–8 s LSTM 0.93 / 0.88 vs GRU 0.92 / 0.86, with smoother output (roughness 0.083 vs 0.113).
  - LSTM-433 ≈ LSTM-512, so this is not parameter count.
- **Proxy (weak, #35):** the raw LSTM scores about like the augmented GRU (r 0.34 vs 0.36; angle 47° vs 51°; raw GRU 0.29, 56°). It is only a hint about online robustness.
- **Docs:**
  - [figure](../results/figures/ext_lstm_vs_gru.png)
  - the "Extensions" section of `REPLICATION.md`
  - a section in `docs/report.md`
  - rows in `docs/methods_notes.md`
- **Next:** Phase 9 (one array) is already training in the background; then its summary and the final briefing.

---

## Phase 7: Write-up. Done.

- **What:** [docs/report.md](report.md), a short narrative of the replication written for you. It covers a TL;DR, each claim with numbers and figures, where we differ and why, how robust each result is, and the pitfalls. It is linked from the README and the top of `REPLICATION.md`. No new analysis.
- **Next (block 2, high effort):** the LSTM code and the closed-loop proxy metric, the array-split check, and the one-array script. Then one background GPU job runs the LSTM/GRU search (cap 3 h) and the one-array runs (cap 2 h).

---

## Phase 6: Final report. Done; replication complete.

- **`REPLICATION.md`:** has a summary at the top, the full paper-vs-ours table (every figure and supplementary item), the decisions/assumptions register (#1–31) and how to reproduce.
- **`docs/paper_check.md`:** Claude's PMC verification table, plus the items only you can check (values shown only in figures, e.g. the Supp Fig 1 counts and the Fig 5a TTA bars).
- **`docs/methods_notes.md`:** maps every paper method to its code.
- **README:** its "Replication" section lists every `pixi run` command.

**Headline:** every qualitative claim reproduces, and most numbers are close. Read the summary at the top of `REPLICATION.md` first.

**Open for you:**
1. Go through `docs/paper_check.md`, especially the ❓ rows: the Fig 5a TTA bars (#30) and the Supp Fig 1 counts.
2. **Tell Claude the next steps** (e.g. extensions, robustness checks, a write-up, a pull request to the authors).

---

## Phase 5: Online performance re-scored from closed-loop logs (Fig 5a-c, Fig 4e). Done.

### 1. What was done

`scripts/fig5_behavior.py` scores every closed-loop evaluation trial (CL blocks only; assisted recalibration blocks are excluded):
- success and time to acquisition (TTA), taken from the logged `durationBinnedMove` (≥ 499 bins = timeout);
- angular error at 300–500 ms after go;
- 10k bootstrap CIs.

It also plots the active-cursor speed profiles for Fig 4e. The trial table is in `results/fig5/trials.csv`.

### 2. Results ([Fig 5](../results/figures/fig5_online.png), [Fig 4e](../results/figures/fig4e_online_speed.png))

| Claim | Ours | |
|---|---|---|
| 5a: bimanual angular error higher | bi 52° vs uniR 34° / uniL 43°, on every day | ✅ |
| 5a: bimanual TTA "only slightly longer" | bi 5.4 s vs uni 2.8 / 3.0 s (about 2× on every day) | ⚠️ |
| 5a: uniR and uniL TTA similar | 2.8 vs 3.0 s | ✅ |
| 5a: decoders sometimes failed | success 99% / 97% (1776 / 1778) vs 63% / 43% (1790 / 1792) | ✅ |
| 5b: sequential faster than simultaneous | bi 5.3 vs 7.1 s; success 96% vs 67% | ✅ |
| 5c: RNN better than linear at all gains | Linear mostly times out on mixed blocks; works on uni-only locked blocks | ✅ |
| 4e: augmented RNN works online, raw makes pulses | success 59% / 86% vs 7% / 16%; raw makes one speed pulse, then nearly stops | ✅ |

### 3. Decisions to review

- **#29, Fig 5a days.** The paper lists 1752, 1771, 1776, 1778, 1790 and 1792, but the Dryad files for 1752 / 1771 contain no CL blocks, so Fig 5a uses the 4 days that have them.
- **#30, bimanual TTA.** About 2× unimanual, larger than "only slightly longer". It is on your paper-check list: compare it with the Fig 5a bars.
- **#7, TTA source.** The logged value is used. Recomputing TTA from positions with a fixed radius of 60 agrees within 2 bins on only 60% of trials, so the task's own record is more trustworthy.

---

## Phase 4: RNN decoders and temporal fracturing (Fig 4b-d, day 1869). Done.

### 1. What was done

- `bimanual/rnn.py`: a GRU(512) decoder with a day-specific input transform, a velocity output and a context output (which hands are moving). Its checkpoints save the random-number state too, so an interrupted run resumes exactly.
- `scripts/fig4_rnn.py`: the open-loop data split by block (train 1–5, select on 6, test on 7) and a paired random search: 10 settings, each trained on raw and on augmented data.
- Extras: extra seeds, an alternative target reading, and a clip-length check. About 1.6 GPU-hours in total.

### 2. Results ([figure](../results/figures/fig4_rnn.png))

| Training data | Test r (RH / LH) | Velocity roughness |
|---|---|---|
| Raw | **0.96 / 0.89** (paper 0.90 / 0.83) ✅ | 0.04 |
| Augmented, 2–8-s clips (main) | 0.93 / 0.87, "slightly poorer" ✅ | 0.08 (about 2× noisier) ✅ |
| Augmented, 200–800-ms clips | 0.76 / 0.67, much poorer | 0.31 |

Online (Phase 5, logged data): the augmented RNN acquired 59% / 86% of targets on days 1867 / 1869, against 7% / 16% for the raw-trained one. That is the payoff of augmentation that offline r doesn't show.

### 3. Decisions to review

- **#14, clip length.** The paper contradicts itself: Methods say 0.2–0.8 × the 10-s snippet, Results say 200–800 ms (confirmed by the PMC check). Only the 2–8-s reading reproduces "slightly poorer" offline, so it is MAIN.
- **#13, "saturated" targets.** The "hold" reading was worse, so the clip reading is kept.
- **#28, design.** Single-day training on 1869 (the paper used 1867 + 1869). The long-clip models reuse the best short-clip hyperparameters (#31).

---

## Phase 3: Fig 3, Supp Figs 2 and 4 (day 1883). Done.

### 1. What was done

| Code | Purpose |
|---|---|
| `bimanual/decoders.py` | Single-bin ridge and FFN decoders |
| `scripts/fig3_decoding.py` | Fig 3a-c and Supp Fig 2 |
| `bimanual/simulate.py`, `scripts/fig3_simulations.py` | Simulations and Supp Fig 4 |

All tests pass.

### 2. Results

**Real data**, day 1883, 491 open-loop unimanual trials ([figure](../results/figures/fig3_decoding_direction.png)):

| Claim | Result | |
|---|---|---|
| FFN decodes better than ridge, most of all the LH | +0.07 r overall, **+0.12 for the LH** (p < 0.001) | ✅ |
| Removing laterality hurts the FFN, not ridge | FFN −0.08 (LH −0.13); ridge ±0.000 (p = 0.78) | ✅ |
| FFN keeps the idle cursor steadier | LH jitter: FFN 0.42 vs ridge 0.84, back to 0.84 when laterality is removed | ✅ |
| Bimanual: RH decodes better than LH | RH 0.68–0.77 vs LH 0.32–0.49 | ✅ |

**fSNR** ([figure](../results/figures/supp_fig4_fsnr.png)):

| | Paper | Ours |
|---|---|---|
| Real single-bin fSNR | 2.03 | **2.11** |
| Window-average, real vs simulated | Should not match (simulated noise is independent across bins) | 4.60 vs 9.01 |

**Simulations** ([figure](../results/figures/fig3_simulations.png)):
- Ridge degrades as RH-LH tuning correlation rises: 0.59 → 0.50.
- The FFN holds at about 0.69 for every correlation once laterality is at least 2× the real size.

### 3. Decisions to review (register #11, #12, #25–27)

- **#11, decoder targets.** The recorded open-loop cursor velocity was tried first. The cursor only starts moving about 300 ms after the go cue, so in the 300–700 ms window that target is mostly an acceleration ramp, and every r stayed low (0.2–0.5). The MAIN target is the unit direction to the target. With cursor velocity the same effects appear, smaller but significant.
- **#12, trial subset.** 491 trials are used vs the paper's 482.
- **#26, environment.** The process crashed with an OpenMP conflict between conda's MKL and the PyTorch wheel. conda numpy/scipy now use OpenBLAS; the reference cross-check is unchanged.
- **#25 and #27.** FFN training details and simulation details are choices of ours; the paper doesn't give them.
