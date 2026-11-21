# Replication report: Deo et al. 2024, Sci Rep

**Paper:** "Brain control of bimanual movement enabled by recurrent neural networks". DOI: [10.1038/s41598-024-51617-3](https://doi.org/10.1038/s41598-024-51617-3).

**Data:** Dryad, [doi:10.5061/dryad.sn02v6xbb](https://doi.org/10.5061/dryad.sn02v6xbb).

**Short write-up:** [docs/report.md](docs/report.md).

**Status (2026-11-21): complete.** Every result that can be reproduced offline was attempted. Phase-by-phase notes are in [docs/briefings.md](docs/briefings.md); the build log is [PROGRESS.md](PROGRESS.md).

## Summary

**Every qualitative claim of the paper reproduces from the public data.** Most headline numbers land close to the paper's:

- **Neural tuning (Figs 1–2):**
  - The non-dominant LEFT hand's tuning is suppressed (41% vs 18% for the RH; paper 34% / 12%) and decorrelated (uni/bi r 0.49 / 0.60 vs RH 0.86 / 0.86; paper 0.42 / 0.45 vs 0.87 / 0.84) during bimanual movement.
  - Bimanual activity sits on top of unimanual-RH activity.
  - Laterality explains more variance than direction (cross-validated dPCA 39% vs 29%; paper 39% / 30%).
- **Decoders (Fig 3):**
  - A one-layer nonlinear decoder beats ridge regression, mostly on the LH (+0.12 r).
  - It does so by using the laterality dimension: removing it erases the advantage and the steadier idle cursor.
  - Simulations reproduce the mechanism. The real single-bin fSNR is 2.11 (paper 2.03).
- **RNNs (Fig 4):**
  - Raw-trained RNN offline r = 0.96 / 0.89 (paper 0.90 / 0.83).
  - Time-warped, shuffled ("fractured") training data costs a little offline (0.93 / 0.87, noisier).
  - Online, the logged sessions show the fractured-data RNN acquiring 59–86% of targets vs 7–16% for the raw-trained one.
- **Online control (Fig 5):** RNN ≫ linear decoder; sequential strategy faster than simultaneous; bimanual angular errors higher.

**Where we differ:**

1. Bimanual time to acquisition is about 2× unimanual in the logs, where the paper says "only slightly longer" (#30).
2. RH suppression comes out slightly larger (18.7% vs 12%).
3. Two methods had to deviate to work at all:
   - split-half instead of 5-fold cvOLS, because 5-fold explodes on small sets (#23);
   - unit-direction regressors instead of raw position differences in Eq. 2 (#21).
4. The paper contradicts itself on the augmentation clip length (#14). Only the 2–8 s reading reproduces its offline result.

**Not reproducible offline:** live closed-loop control, so Figs 4e and 5 are re-scored from the logs. Days 1752 / 1771 have no closed-loop data on Dryad.

## Results: paper vs replication

The paper's numbers were re-checked against the full text on PMC in step 6.2 (see [docs/paper_check.md](docs/paper_check.md)). Values given only in figures are marked "figure only".

| Figure | Quantity | Paper | Ours (95% CI) | Status |
|---|---|---|---|---|
| Fig 1b-c | Electrode 97 / 13 / 23 tuning, uni vs bi | 97 keeps both hands; 13 suppressed for both hands; 23 LH suppressed | 97: RH identical, LH kept but weaker. 13: both hands weaker. 23: LH abolished, RH kept. ([figure](results/figures/fig1_tuning_curves.png)) | ✅ qualitative match |
| Supp Fig 1c | Tuned electrodes (p < 1e-5), uniR / uniL / biR / biL | RH-preferring in both contexts (counts only in the figure) | 83 / 50 / 87 / 28 ([figure](results/figures/supp_fig1_tuning_counts.png)) | ✅ qualitative match (exact counts are figure-only; see paper_check) |
| Supp Fig 1e | Electrodes tuned to 0 / 1 / 2 / 3 / 4 contexts | Pie chart (figure only) | 88 / 25 / 29 / 35 / 15 | Figure-only; see paper_check |
| Fig 2b | RH tuning suppression, bimanual vs unimanual | 12% | 18.7% (CI 15.5–22.5%); all schemes 18–19% ([figure](results/figures/fig2bc_tuning_change_splithalf.png)) | ✅ direction and size similar; somewhat larger |
| Fig 2b | LH tuning suppression, bimanual vs unimanual | 34% | 41.3% (CI 34–49%); all schemes 36–41% | ✅ LH ≫ RH; t-test p = 6e-6 (n = 22) |
| Fig 2c | RH uni/bi tuning correlation, x / y | 0.87 / 0.84 | 0.86 / 0.86 | ✅ |
| Fig 2c | LH uni/bi tuning correlation, x / y | 0.42 / 0.45 | 0.49 / 0.60 | ✅ qualitatively (LH decorrelated); y somewhat higher |
| Fig 2c | UniR-UniL correlation, x / y | anti-correlated in x, correlated in y | −0.34 / +0.51 | ✅ |
| Supp Table 1 | Pairwise t-tests between correlation types (12 p-values) | 11 significant, 1 n.s. (y: UniL-BiL vs UniR-UniL, 0.76) | Same significance pattern at 0.01 in 11/12; the exception is x UniL-BiL vs BiR-BiL (paper 0.05, ours 0.001) | ✅ |
| Supp Fig 1d | RH vs LH unimanual tuning strength (day 1776) | about 2× (LH 48% weaker) | 1.70× (LH 41% weaker) | ✅ close |
| Fig 2 PCA | Bimanual activity is closest to unimanual RH (day 1881) | Yes (qualitative) | Centroid distance in PC1-2: bi–uniR 0.28 vs bi–uniL 3.21; full space 2.21 vs 3.53 ([figure](results/figures/fig2_pca.png)) | ✅ |
| Fig 2 dPCA | Share of marginalized variance: laterality / direction | 39% / 30% | 0–1000 ms: **39% / 29%**; 300–700 ms: 41% / 33% (5 days, cross-validated) ([figure](results/figures/fig2_dpca.png)) | ✅ laterality > direction on every day. Non-CV estimates flip the order. |
| Fig 3b | Unimanual single-bin decoding: FFN vs ridge (day 1883) | FFN better on all dims, largest gain for the LH (values in the figure only) | r ridge R 0.72/0.79, L 0.45/0.56; FFN R 0.75/0.80, L 0.56/0.69. FFN − ridge: +0.07 overall, +0.12 LH (p < 0.001) ([figure](results/figures/fig3_decoding_direction.png)) | ✅ |
| Fig 3b | Laterality removed | Hurts the FFN, not ridge | FFN −0.08 (LH −0.13), p < 0.001; ridge ±0.000 (p = 0.78) | ✅ |
| Fig 3c | Idle-cursor jitter | FFN steadier; laterality removal increases FFN jitter | L jitter: ridge 0.84, FFN 0.42, FFN − lat 0.84 | ✅ |
| Supp Fig 2c | Bimanual decoding | RH better than LH | RH 0.68–0.77 vs LH 0.32–0.49 | ✅ |
| Supp Fig 4 | Single-bin fSNR, unimanual RH (day 1883) | 2.03 | **2.11**; window-average 4.60 ([figure](results/figures/supp_fig4_fsnr.png)) | ✅ |
| Supp Fig 4 | i.i.d. noise matches single-bin but not window-average fSNR | Yes | Simulated: 2.12 single-bin (matched), 9.01 window-average (vs real 4.60) | ✅ |
| Fig 3d-e | Simulation: ridge degrades as RH-LH tuning correlation rises; the FFN holds when laterality is large | Qualitative | Ridge r 0.59 → 0.50 (LH 0.45 → 0.32) for ρ 0→1, flat across laterality. FFN ≈ 0.69 for every ρ at laterality ≥ 2× real; FFN jitter is far below ridge once laterality > 0 ([figure](results/figures/fig3_simulations.png)) | ✅ |
| Fig 4d | Raw-trained RNN offline r, RH / LH (day 1869, held-out block) | 0.90 / 0.83 | **0.96 / 0.89** (best setting, 3 seeds) ([figure](results/figures/fig4_rnn.png)) | ✅ |
| Fig 4d | Augmented RNN offline: "slightly poorer ... noisier output velocities" | Qualitative | 2–8-s clips (main): 0.93 / 0.87, roughness 0.08 vs raw 0.04. 200–800-ms clips: 0.76 / 0.67, roughness 0.31 | ✅ with 2–8-s clips; the 200–800-ms reading drops much more than "slightly" |
| Fig 4e | Online: augmented RNN gives continuous movement and acquires more targets than raw | Qualitative | Success: 1867 59% vs 7%; 1869 86% vs 16%. Raw: one speed pulse, then nearly stops; augmented: sustained speed ([figure](results/figures/fig4e_online_speed.png)) | ✅ |
| Fig 5a | Bimanual TTA vs unimanual (RNN, 1776/1778/1790/1792) | Bimanual "only slightly longer" | bi 5.4 s vs uniR 2.8 s / uniL 3.0 s (successful trials; about 2× on every day) ([figure](results/figures/fig5_online.png)) | ⚠️ larger gap than described (register #30) |
| Fig 5a | Bimanual angular error higher | Yes | bi 52° vs uniR 34° / uniL 43°; higher on every day | ✅ |
| Fig 5a | Decoders sometimes failed | Yes | Success 99% / 97% (1776 / 1778) vs 63% / 43% (1790 / 1792) | ✅ |
| Fig 5b | Sequential faster than simultaneous (1881 + 1883) | Yes | bi 5.3 vs 7.1 s; uniR 2.6 vs 3.4 s; uniL 2.5 vs 4.4 s (failures = 10 s) | ✅ |
| Fig 5c | RNN beats linear at all gains; linear fails at separating the hands | Yes | Mixed blocks: linear mostly times out (0–47% success) while the RNN succeeds on 59–100%. Uni-only locked blocks: linear works (TTA about 3.4–4.2 s) | ✅ |

## Decisions, deviations and assumptions
This is the full register. Each entry is also marked in the code with `# ASSUMPTION:` or `# DEVIATION:`, and [docs/methods_notes.md](docs/methods_notes.md) maps each to its code. **Type:** D = decision (made from evidence in the data), A = assumption (the paper is silent or ambiguous), V = deviation (we do something differently from the paper).

| # | Type | Topic | What we do | Why / evidence | Status / check |
|---|---|---|---|---|---|
| 1 | A | Dryad data versions | Use the current download. Assume the 3 Dryad versions (Dec 28 2023, Jan 2 2024, Jan 18 2024) contain identical `.mat` data and differ only in the README. | The zip is 138.35 MB in all three versions; only the README size changes (2.16 → 2.17 KB). | Accepted (user, 2026-11-20) |
| 2 | D | Corrupted file | Day 1867 (`t5_09_27_2021.mat`) was re-extracted by the user after its `tx` failed to decompress. | All 13 files now load fully. | Resolved |
| 3 | D | Session/block labels | Block conditions are transcribed by hand from `Data/readme.txt` into `bimanual/sessions.py`. | `pixi run inventory` checks every block both ways: none missing, none unlabeled. | Verified |
| 4 | V | Language | The MATLAB analysis is ported to Python. | User choice. The port matches the authors' `tuningAnalysis.m` run in Octave on day 1750 (max diff 3e-14). | Verified. The Mac MATLAB check was dropped by the user. |
| 5 | D | Port fidelity | MATLAB quirks are kept: angles are floored to whole degrees, a target exactly on an axis gets NaN, and windows are inclusive ((15, 35) = 21 bins). | Needed for an exact match with the authors' code. | Verified |
| 6 | A | Direction quadrants (Supp Fig 1 ANOVA; augmentation balancing) | Axis-aligned quadrants of the cursor→target vector at the go cue. | The paper doesn't define them. Targets are at random positions, and directions are evenly spread over 0–360°. | Open: affects Supp Fig 1 counts |
| 7 | D | Fig 5 time to acquisition | Use the logged `durationBinnedMove` × 20 ms; ≥ 499 bins = 10-s timeout = failure. This includes the 500-ms dwell. | The target radius isn't stored, and no fixed radius fits all dwells (step 0.3). The logged value is the task's own record. | Decided; `behavior.time_to_acquisition` is a sanity check |
| 8 | A | Fig 5 angular error | The target direction is fixed at the go cue; movement direction comes from the per-bin cursor displacement over 300–500 ms; bins with no movement are skipped. | The paper says only "average difference ... 300–500 ms". | Check in Phase 5 |
| 9 | V | cvOLS / unbiased statistics | Reimplemented from the description in Willett et al. 2020, not from their code. | No code was released. Synthetic tests show it is unbiased. | Check against the Fig 2 numbers in Phase 2 |
| 10 | A | Fig 2 block grouping | MAIN: consecutive pairs of all OL blocks per day, including uni-only blocks: exactly 25 pairs, 22 usable (3 uni-only pairs have about 2 bimanual trials). Robustness: all pairs of mixed blocks (38) and adjacent mixed pairs (17). | Only `adjacent_all` reproduces n = 25 (user decision 2026-11-21: matches the figure's description). | Done; PMC (step 6.2) CONFIRMS "consecutive blocks together in pairs", n = 25 |
| 11 | A | Fig 3 decoder targets | MAIN: unit cursor→target direction at the go cue (0 for the idle hand), constant over the 400-ms window. ALT: recorded OL cursor velocity (`--target cursor`). | In OL trials the cursor starts moving about 300 ms after go, so cursor velocity in the window is mostly the onset ramp scaled by distance, which caps every r low. Unit direction equals the saturated targets the paper's online decoders used. Every effect holds with both targets. | Done; PMC CONFIRMS the unit cursor→target vector (the paper updates it per time step; we fix it at go) |
| 12 | A | Fig 3 trial subset | All 491 open-loop unimanual trials of day 1883 (254 uniR, 237 uniL, from all 8 OL blocks). | The closest natural subset to the paper's 482 (the file has 951 trials in total). | Done; PMC: "482 total trials", unimanual |
| 13 | A | RNN "saturated" targets | Min-jerk speed bell clipped at 0.5 of its peak and rescaled (alternative: rise then hold to the end). | The wording is ambiguous. | Both tested; "hold" is worse offline for raw and aug, so the clip reading stays MAIN |
| 14 | A | Augmentation clip length | 200–800 ms (10–40 bins). | The paper's wording is ambiguous ("0.2–0.8 ×" vs "200–800 ms"). | DECIDED: MAIN = 2–8 s (Methods reading). It reproduces "slightly poorer, noisier" offline (r -0.03, 2x roughness), while 200–800 ms gives r -0.2. Online behaviour of the 2–8 s version can't be tested |
| 15 | A | Augmentation velocity scaling | Velocity amplitude is not rescaled when a clip is stretched. | The targets encode intent, not physical speed. | PMC CONFIRMS ("compress or stretch both the input and output clips") |
| 16 | V | RNN hyperparameter search | 10 random settings, each trained on raw AND augmented data (20 models, paired), plus 2 extra seeds per best setting; selection on the val block. Space: lr 3e-4–3e-3, batch {16, 32, 48}, steps {2000, 3000, 4000}, L2 1e-6–1e-4, noise 0.1–0.6, drift 0–0.3. | GPU budget (RTX 3050, 4 GB; 69 ms/step). User approved a default of about 20 models with a ≤ 3 h cap (2026-11-21). | Done: 20 search + 4 seed + 2 hold + 3 long-clip models, about 1.6 GPU-h |
| 17 | D | Paper target numbers | Taken from summaries of the paper text. | The paper was read through a summarizer. | Done; PMC CONFIRMS all headline values (see docs/paper_check.md) |
| 18 | A | Supp Fig 1 ANOVA window | Same 300–700 ms window (bins 15–35) as the encoding analyses. | The paper doesn't state the window for the ANOVA. | Applied in Phase 1 |
| 19 | V | Supp Fig 1b layout | FVAF is shown as an electrode × context heat map, not on the array anatomy. | The data has no electrode positions. | Phase 1 |
| 21 | A | Encoding-model regressors (Eq. 2) | Unit cursor→target direction at the go cue, held constant over 300–700 ms; 0 for the idle hand. | Eq. 2 says target − cursor, but target distances vary from 30 to 750 units, and a direction-tuning model should not encode distance. Ratios and correlations don't depend on scale. | PMC: Eq. 2 is the raw per-step difference, so this is a confirmed DEVIATION (scale-free statistics only) |
| 22 | A | Minimum trials per context in a set | ≥ 10 trials of each type. | The cross-validated fits need enough trials. | Excludes exactly the 3 uni-only pairs |
| 23 | V | cvOLS estimator | MAIN: split-half cross-validation (50 random halves of trials). The paper's literal 5-fold is kept as `--method kfold`. | 5-fold fits each fold alone; with about 25 bimanual trials that is about 5 trials for 5 parameters, so the estimates explode (main-scheme k-fold LH suppression 62%, correlation 1.20). Synthetic test: split-half is unbiased with far lower variance. The mixed-only schemes give similar values with either method. | Decided 2026-11-21 |
| 24 | A | dPCA window and scaling | Unimanual OL trials; z-scored per block. Windows: 300–700 ms (pre-registered main) and 0–1000 ms (robustness). 100 random two-way splits per day; mean over 5 days. | The paper gives no window. The 0–1000 ms window reproduces 39% / 30% almost exactly, so the paper probably used a longer window. | Both reported |
| 25 | V | FFN training details | ReLU, Adam (lr 1e-3, weight decay 1e-4), MSE on standardized targets, batch 512, early stopping on 10% of the training trials; PyTorch on the GPU. Ridge: closed form, alpha grid 0.1–1e5, best mean CV r. | The paper gives only "1 dense layer, 512 units, TensorFlow v1" and "RidgeCV swept to maximize Pearson correlation". | Phase 3 |
| 26 | D | Environment: BLAS | conda-forge numpy/scipy use OpenBLAS instead of MKL. | The MKL build loads LLVM `libomp.dll`, which aborts against the Intel `libiomp5md.dll` in the PyTorch wheel (OMP Error #15). The reference cross-check is unchanged (3e-14). | Done |
| 27 | A | Simulation details | 20 noisy bins per trial; 8 wedge directions; one population draw per grid cell; ρ ∈ {0, .25, .5, .75, .9, 1}, laterality ×{0, .5, 1, 2, 4} of the real size; noise σ calibrated (bisection) to the real single-bin fSNR at ρ = 0, lat ×1; ridge alpha chosen on 10% of the training trials. | The paper gives no per-trial bin count, grid values or seeds. | Phase 3 |
| 28 | V | RNN implementation | PyTorch GRU(512), day-affine input (one day), Adam + gradient clipping (1.0), MSE + CE + L2; white noise + per-snippet mean drift. Day 1869 OL only, split by block (train 1–5 / val 6 / test 7). Raw data = random contiguous 10-s windows; augmented = 2000 fixed snippets. | The paper trained online decoders on pooled multi-day data (40% newest day) in TF v1. For the offline Fig 4d comparison, a single-day, block-split design isolates the effect of augmentation. | Phase 4 |
| 29 | D | Fig 5a day set | MAIN: 1776, 1778, 1790, 1792 (standard RNN CL evaluation blocks). 1752 / 1771 have no CL data (calibration days); 1881 / 1883 "simultaneous" blocks are reported separately. | Resolves the Supp Table 2 vs Methods discrepancy using the data. | Done; PMC lists 1752, 1771, 1776, 1778, 1790, 1792, but the Dryad data for 1752 / 1771 has no CL blocks |
| 30 | D | Fig 5a bimanual TTA | Reported as logged: bimanual about 2× unimanual on every day, larger than the paper's "only slightly longer". Possible causes: a different TTA definition in the paper (e.g. excluding the dwell, or time to first target), or a different trial subset. | Our TTA is the task's own record (`durationBinnedMove`). | Flagged for the paper check |
| 31 | D | Long-clip augmentation run | The 2–8-s clip models reuse the best hyperparameters found with 200–800-ms clips (no separate search). | Budget. A dedicated search could do slightly better. | Phase 4 |
| 20 | D | Fig 1 curve centring and CIs | Curves are centred on their 8-direction mean, as in `tuningAnalysis.m`. CIs come from a 10k bootstrap over trials. | Follows the authors' script; the paper's CI method is 10k bootstrap. | Phase 1 |

### Data quirks noted (no action yet)
- "Unimanual only" open-loop blocks each contain about 1 bimanual trial.
- One moving-hand trial in the whole dataset (day 1869, closed-loop) has the target at the cursor at the go cue. It gets a NaN quadrant (`min_dist` = 1). Checked in Phase 1: day 1750 is clean.

## How to reproduce
Install pixi, run `pixi install`, put the Dryad `.mat` files in `Data/`, then run the `pixi run …` commands in [README.md](README.md) (Replication section) in figure order. GPU steps (`fig3-decoding`, `fig3-sim`, `fig4-rnn`) are resumable.

## Open items for the user
- [ ] **Check the paper against [docs/paper_check.md](docs/paper_check.md)** and report anything that contradicts our assumptions.
- [x] **After Phase 6, give Claude the next steps.** Given 2026-11-21: a write-up (Phase 7), LSTM vs GRU (Phase 8) and one-array decoding (Phase 9).
