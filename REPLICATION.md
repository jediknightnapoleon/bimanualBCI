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

1. Bimanual time to acquisition is about 2× unimanual in the logs, where the paper's text says "only slightly longer" (#30). The paper's own Fig 5a shows the same 2×, so only the wording differs ([figure comparison](docs/figure_comparison.md)).
2. RH suppression comes out slightly larger (18.7% vs 12%).
3. Two methods had to deviate to work at all:
   - split-half instead of 5-fold cvOLS, because 5-fold explodes on small sets (#23);
   - unit-direction regressors instead of raw position differences in Eq. 2 (#21).
4. The paper contradicts itself on the augmentation clip length (#14). Only the 2–8 s reading reproduces its offline result.

**Not reproducible offline:** live closed-loop control, so Figs 4e and 5 are re-scored from the logs. Days 1752 / 1771 have no closed-loop data on Dryad.

## Results: paper vs replication

The paper's numbers were re-checked against the full text on PMC in step 6.2 (see [docs/paper_check.md](docs/paper_check.md)). Values given only in figures were read off the published figures on 2026-11-23. A side-by-side, panel-by-panel comparison is in [docs/figure_comparison.md](docs/figure_comparison.md).

| Figure | Quantity | Paper | Ours (95% CI) | Status |
|---|---|---|---|---|
| Fig 1b-c | Electrode 97 / 13 / 23 tuning, uni vs bi | 97 keeps both hands; 13 suppressed for both hands; 23 LH suppressed | 97: RH identical, LH kept but weaker. 13: both hands weaker. 23: LH abolished, RH kept. ([figure](results/figures/fig1_tuning_curves.png)) | ✅ qualitative match |
| Supp Fig 1c | Tuned electrodes (p < 1e-5), uniR / uniL / biR / biL | 90 / 60 / 97 / 52 (figure) | 83 / 50 / 87 / 28 ([figure](results/figures/supp_fig1_tuning_counts.png)) | ⚠️ RH preference matches; we find fewer tuned electrodes, most for biL (28 vs 52); threshold-sensitive: at p < 1e-3 with 8 directions 90 / 62 / 100 / 40 ([figure comparison](docs/figure_comparison.md#major-differences-and-their-causes)) |
| Supp Fig 1e | Electrodes tuned to 0 / 1 / 2 / 3 / 4 contexts | 82 / 16 / 33 / 27 / 34 (figure) | 88 / 25 / 29 / 35 / 15 | ⚠️ fewer electrodes tuned to all 4 (15 vs 34), following from biL |
| Fig 2b | RH tuning suppression, bimanual vs unimanual | 12% | 18.7% (CI 15.5–22.5%); all schemes 18–19% ([figure](results/figures/fig2bc_tuning_change_splithalf.png)) | ✅ direction and size similar; somewhat larger |
| Fig 2b | LH tuning suppression, bimanual vs unimanual | 34% | 41.3% (CI 34–49%); all schemes 36–41% | ✅ LH ≫ RH; t-test p = 6e-6 (n = 22) |
| Fig 2c | RH uni/bi tuning correlation, x / y | 0.87 / 0.84 | 0.86 / 0.86 | ✅ |
| Fig 2c | LH uni/bi tuning correlation, x / y | 0.42 / 0.45 | 0.49 / 0.60 (split-half, unit direction); **0.41 / 0.44** with the paper's literal 5-fold cvOLS and Eq. 2 per-step regressors | ✅ the gap is the estimator / regressor choice ([figure comparison](docs/figure_comparison.md#major-differences-and-their-causes)) |
| Fig 2c | UniR-UniL correlation, x / y | anti-correlated in x, correlated in y | −0.34 / +0.51 | ✅ |
| Supp Table 1 | Pairwise t-tests between correlation types (12 p-values) | 11 significant, 1 n.s. (y: UniL-BiL vs UniR-UniL, 0.76) | Same significance pattern at 0.01 in 11/12; the exception is x UniL-BiL vs BiR-BiL (paper 0.05, ours 0.001) | ✅ |
| Supp Fig 1d | RH vs LH unimanual tuning strength | 1.95× (figure; mean of per-set dots) | **1.92×** as the mean over 22 block sets (1.70× from one pooled fit on day 1776) | ✅ match once aggregated like the paper ([figure comparison](docs/figure_comparison.md#major-differences-and-their-causes)) |
| Fig 2 PCA | Bimanual activity is closest to unimanual RH (day 1881) | Yes (qualitative) | Centroid distance in PC1-2: bi–uniR 0.28 vs bi–uniL 3.21; full space 2.21 vs 3.53 ([figure](results/figures/fig2_pca.png)) | ✅ |
| Fig 2 dPCA | Share of marginalized variance: laterality / direction | 39% / 30% | 0–1000 ms: **39% / 29%**; 300–700 ms: 41% / 33% (5 days, cross-validated) ([figure](results/figures/fig2_dpca.png)) | ✅ laterality > direction on every day. Non-CV estimates flip the order. |
| Fig 3b | Unimanual single-bin decoding: FFN vs ridge (day 1883) | Ridge R 0.69/0.77, L 0.45/0.55; FFN R 0.77/0.82, L 0.59/0.70 (figure) | r ridge R 0.72/0.79, L 0.45/0.56; FFN R 0.75/0.80, L 0.56/0.69. FFN − ridge: +0.07 overall, +0.12 LH (p < 0.001) ([figure](results/figures/fig3_decoding_direction.png)) | ✅ |
| Fig 3b | Laterality removed | Hurts the FFN, not ridge | FFN −0.08 (LH −0.13), p < 0.001; ridge ±0.000 (p = 0.78) | ✅ |
| Fig 3c | Idle-cursor jitter | FFN steadier; laterality removal increases FFN jitter | L jitter: ridge 0.84, FFN 0.42, FFN − lat 0.84 | ✅ |
| Supp Fig 2c | Bimanual decoding | RH better than LH: ridge R 0.69/0.69, L 0.40/0.31; FFN R 0.76/0.75, L 0.42/0.32 (figure) | RH 0.68–0.77 vs LH 0.32–0.49 | ✅ RH ≫ LH; but our ridge ≥ FFN here, while the paper has FFN slightly above ridge |
| Supp Fig 4 | Single-bin fSNR, unimanual RH (day 1883) | 2.03 | **2.11**; window-average 4.60 ([figure](results/figures/supp_fig4_fsnr.png)) | ✅ |
| Supp Fig 4 | i.i.d. noise matches single-bin but not window-average fSNR | Yes | Simulated: 2.12 single-bin (matched), 9.01 window-average (vs real 4.60) | ✅ |
| Fig 3d-e | Simulation: ridge degrades as RH-LH tuning correlation rises; the FFN holds when laterality is large | Qualitative | Ridge r 0.59 → 0.50 (LH 0.45 → 0.32) for ρ 0→1, flat across laterality. FFN ≈ 0.69 for every ρ at laterality ≥ 2× real; FFN jitter is far below ridge once laterality > 0 ([figure](results/figures/fig3_simulations.png)) | ✅ |
| Fig 4d | Raw-trained RNN offline r, RH / LH (day 1869, held-out block) | 0.90 / 0.83 | **0.96 / 0.89** (best setting, 3 seeds) ([figure](results/figures/fig4_rnn.png)) | ✅ |
| Fig 4d | Augmented RNN offline: "slightly poorer ... noisier output velocities" | 0.82 / 0.82 (figure) | 2–8-s clips (main): 0.93 / 0.87, roughness 0.08 vs raw 0.04. 200–800-ms clips: 0.76 / 0.67, roughness 0.31 | ✅ with 2–8-s clips; the 200–800-ms reading drops much more than "slightly" |
| Fig 4e | Online: augmented RNN gives continuous movement and acquires more targets than raw | Qualitative | Success: 1867 59% vs 7%; 1869 86% vs 16%. Raw: one speed pulse, then nearly stops; augmented: sustained speed ([figure](results/figures/fig4e_online_speed.png)) | ✅ |
| Fig 5a | Bimanual TTA vs unimanual (RNN; 6 sessions: 1776, 1778, 1790, 1792 + simultaneous blocks of 1881, 1883) | Text: bimanual "only slightly longer". Figure (median, failures = 10 s): bi ≈ 4.4–5.9 s vs uni ≈ 2–3 s on good sessions, timeouts on 3 | **Median with failures = 10 s: all 18 points match the paper's figure** (e.g. 1778: 2.05 / 2.34 / 4.44 s vs ≈ 2.1 / 2.3 / 4.4). Mean over successful trials (our earlier summary): bi 5.4 s vs uni 2.8 / 3.0 s | ✅ matches the figure; only the text understates the gap (#30; [figure comparison](docs/figure_comparison.md#major-differences-and-their-causes)) |
| Fig 5a | Bimanual angular error higher | Yes; ≈ uniR 11–25°, biR 20–40°, uniL 13–49°, biL 30–65° (figure) | Paper-consistent definition (net displacement 300–500 ms, median): 11–23° / 20–40° / 12–39° / 32–63°. Our per-bin mean (#8): 26–50° / 36–63° / 30–65° / 54–77° | ✅ ([figure comparison](docs/figure_comparison.md#major-differences-and-their-causes)) |
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
| 8 | A | Fig 5 angular error | The target direction is fixed at the go cue; movement direction comes from the per-bin cursor displacement over 300–500 ms; bins with no movement are skipped. | The paper says only "average difference ... 300–500 ms". | RESOLVED 2026-11-23: the paper's values match the angle of the NET displacement over 300–500 ms with the median over trials (diagnostic D2); our per-bin mean is about 10–25° higher. Main pipeline unchanged; the paper-style figure uses the paper-consistent definition |
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
| 29 | D | Fig 5a day set | MAIN: 1776, 1778, 1790, 1792 (standard RNN CL evaluation blocks). 1752 / 1771 have no CL data (calibration days); 1881 / 1883 "simultaneous" blocks are reported separately. | Resolves the Supp Table 2 vs Methods discrepancy using the data. | RESOLVED 2026-11-23: the paper's 6 Fig 5a sessions are 1776, 1778, 1790, 1792 + the simultaneous blocks of 1881 and 1883 (their medians match the paper's two remaining sessions); 1752 / 1771 are not in Fig 5a |
| 30 | D | Fig 5a bimanual TTA | Reported as logged: bimanual about 2× unimanual on every day, larger than the paper's "only slightly longer". Possible causes: a different TTA definition in the paper (e.g. excluding the dwell, or time to first target), or a different trial subset. | Our TTA is the task's own record (`durationBinnedMove`). | RESOLVED 2026-11-23: the paper's own Fig 5a shows bimanual about 2× unimanual on its good sessions, so our numbers match the figure; only the text's wording understates it ([figure comparison](docs/figure_comparison.md#fig-5-online-two-cursor-control)) |
| 31 | D | Long-clip augmentation run | The 2–8-s clip models reuse the best hyperparameters found with 200–800-ms clips (no separate search). | Budget. A dedicated search could do slightly better. | Phase 4 |
| 32 | D | LSTM initialization (Phase 8) | The LSTM forget-gate bias starts at 1 (0.5 in each of PyTorch's two bias vectors); everything else is PyTorch's default. | The standard LSTM choice (Gers 2000; Jozefowicz 2015). | Extension |
| 33 | D | GRU vs LSTM comparison design | A paired search: the same 10 hyperparameter draws (`default_rng(2021)`), per-setting seeds, fixed snippet sets, splits and targets as the GRU search; only the cell differs. Width 512 (main) plus a parameter-matched LSTM of 433 units (1.086M vs 1.084M recurrent parameters) at the best LSTM-512 settings. | Isolates the effect of the cell type. Retraining s07_raw through the new code reproduces its test r exactly. | Extension |
| 34 | D | GRU long-clip search | The GRU is also searched with 2–8-s clips (10 settings), so each kind gets its own best GRU setting. | Resolves #31. | Extension |
| 35 | A | Closed-loop proxy target | On day 1869's 12 CL blocks: the unit cursor→target vector of the active hand from go + 200 ms until the logged end of pursuit (the last 500 ms of successful trials are the dwell, so 0 there); 0 elsewhere. Metrics: r, decoded angular error at 300–500 ms, "sustain" (1–3 s vs 0.3–1 s projected speed), idle-hand jitter. | ReFIT-style intention estimate; the target radius isn't stored (#7). **Validity check on the existing GRU models:** augmented beats raw on r (about 0.33–0.35 vs 0.29) and angular error (about 43–52° vs 55–58°), matching the online direction, but the gap is far smaller than online, and the raw RNN's "pulse, then stop" doesn't appear offline (sustain about 0.8 for both). So it is a WEAK, secondary metric. | Extension |
| 36 | A | Which channels belong to which array | Channels 1–96 = array A, 97–192 = array B (contiguous; usual BrainGate layout). Which one is medial and which lateral is unknown. | Data check (`ext-arrays --stage check`, days 1869 / 1750 / 1883): within-half correlation exceeds across-half beyond every one of 200 random 96/96 splits, and about 92% of channels correlate more with their own half. Among the 96 contiguous shifted splits, 1–96 ranks 2 / 1 / 40, so the boundary location is supported only moderately. Clustering is uninformative (r ≈ 0.01), and there is no firing-rate step at 96. Verdict: consistent with the split, not proof. | Extension; on the paper-check list |
| 37 | V | One-array hyperparameters | The one-array and random-96 models reuse the best 192-channel setting of their (cell, kind); there is no separate search. | Budget. This may slightly favour the 192-channel models; the random-96 control uses the same settings, so the array-vs-random comparison is fair. | Extension |
| 38 | D | Zeroing one array at test | Set that array's z-scored inputs to 0, i.e. its block mean (a silent / disconnected array after re-centring). | The simplest failure model; no retraining. | Extension |
| 39 | D | Channel-dropout training (Phase 10) | `HParams.chan_dropout` = 0.5: each training sample draws p ~ U(0, 0.5) and sets that fraction of its channels to 0 (z-units) after the noise and drift augmentation; no rescaling. At 0, no extra RNG draws are made, so old runs reproduce bit for bit. | Simulates channels lost to bad signal, which cannot be predicted. | Extension; tested (`tests/test_rnn.py`) |
| 40 | D | Training-data amount (Phase 10) | Train on the last k of blocks 1–5 (k = 1–4; k = 5 is the Phase 8 model); validation 6 / test 7 unchanged; 3 seeds. | The newest blocks are closest to the test block, as in a real recalibration. | Extension |
| 41 | D | Splits for days 1750 and 1867 (Phase 10) | The last 2 open-loop blocks are test, the 2 before them validation, the rest training. 1750: train 12–24, val 25–26, test 27–28 (the plan said 28–29, but the readme's range ends at 28). 1867: train 1–4, 12–17, 22–23; val 24–25; test 26–27. Two-hand r is scored on bins of two-hand trials (> 500 bins). | Same rule as day 1869 (train early, test late). | Extension; the 1750 test blocks contain two-hand trials (asserted) |
| 42 | A | Muting modes (Phase 10) | At test only, f ∈ {0, 0.1, …, 0.6} of the 192 channels, 20 masks per f (`default_rng(7000 + 100f)`). Modes: **zero** (detected and re-centred: 0 in z-units); **dead** (undetected silent electrode: the z-value of 0 counts under training-block statistics, −μ/σ); **noisy** (undetected noise: N(0, 1) per bin). Proxy r at f = 0, 0.3, 0.5. | Three plausible failure types for a chronic implant. | Extension; f = 0 reproduces the logged test r (asserted) |
| 43 | V | Best-n channel selection (Phase 10) | Greedy forward selection with swap refinement on a fast surrogate: ridge from each channel's raw bin plus causal exponential smoothings (τ = 2, 5, 12, 30, 75 bins) to the 4 velocity targets; train 1–5, score validation block 6; α chosen once per array (1e-3 A, 1e-4 B). Joint benchmark: group lasso (one group per channel, FISTA, λ path, ridge refit). Baseline: 1,000 random subsets per n. Cross-check: LSTM ablation ranking on block 6. Block 7 is never used (asserted). | DEVIATION from the plan's 10 raw lags, which decoded poorly (all-96 val r 0.35 vs 0.61 / 0.72 with smoothings). The surrogate tracks the LSTM across random subsets (r = 0.94 with LSTM test r). | Extension |
| 44 | V | Phase 11 data | The Rigotti-Thompson 2026 T5 files (Dryad doi:10.5061/dryad.brv15dvr9, `fig1_T5_{move,delay}.h5`): TX and SBP from the same 48 electrodes (TX rate > 2 Hz), acausal Gaussian smoothing σ = 100 ms, 20-ms bins, trial-aligned, open-loop radial-8 wrist task. Replaces the planned T11 (MINDFUL) comparison and the TX→SBP stand-in for the bimanual sessions. Ridge and FFN only, no LSTM. | The MINDFUL download was blocked and the user could fetch only the T5 files. A raw-bin stand-in cannot be fitted from smoothed, trial-aligned data. | Extension; caveats in [docs/ext_sbp_notes.md](docs/ext_sbp_notes.md) |
| 45 | D | Phase 11 splits and windows | Leave-one-block-out over blocks 7 / 8 / 9; ridge α by an inner leave-one-block-out on the training blocks; z-scoring with training-block statistics. Velocity: cursor velocity −200 to 500 ms after go. Classification: 0–400 ms (movement); 200–700 ms of the delay period for trials with delay ≥ 1 s (183 trials). Paired bootstrap (2,000 trial resamples). | Only 3 blocks / 337 usable trials. | Extension |
| 46 | A | Phase 11 feature and muting details | log SBP is the main SBP feature (chosen in advance; raw SBP, slightly better, reported alongside). Dead SBP channel = its training 1st percentile; dead TX channel = 0 counts; noisy = N(0, 1) per bin (unsmoothed). Masks as #42. SBP residual = log SBP minus a ridge prediction from the same electrode's TX at shifts −4, −2, 0, 2, 4 bins. | The dead-SBP value is a guess; real failed SBP channels might sit at amplifier noise or rail high. | Extension |
| 47 | D | R² rescoring | Coefficient of determination R² = 1 − SSE/SST per velocity dimension, pooled over the same held-out bins as r, averaged over x and y per hand (`decoders.r2_per_dim`, `scripts/rescore_r2.py`, outputs in `results/r2/`). Only the reported metric changes: every model, setting, ridge α and channel set is the one chosen with r. Each re-run must reproduce its logged r (RNNs within 0.002, cached Fig 3 predictions to 10⁻⁹). Fig 3 ridge is also reported with α chosen by R² (same α for unimanual ridge). | Many groups report R², which also penalises gain and offset errors; it is not r² and needs the predictions. | Extension (metric only) |
| 20 | D | Fig 1 curve centring and CIs | Curves are centred on their 8-direction mean, as in `tuningAnalysis.m`. CIs come from a 10k bootstrap over trials. | Follows the authors' script; the paper's CI method is 10k bootstrap. | Phase 1 |

### Data quirks noted (no action yet)
- "Unimanual only" open-loop blocks each contain about 1 bimanual trial.
- One moving-hand trial in the whole dataset (day 1869, closed-loop) has the target at the cursor at the go cue. It gets a NaN quadrant (`min_dist` = 1). Checked in Phase 1: day 1750 is clean.

## Extensions (Phases 8–11, beyond the paper)
Phases 8–10: all offline, day 1869 unless stated, Fig 4 setup (train blocks 1–5 / select on 6 / test on 7). A closed-loop proxy (#35) scores each model on the 12 recorded closed-loop blocks. It is a weak metric: it ranks augmented above raw for the GRU, but by far less than online success does.

### Phase 8: LSTM vs GRU ([figure](results/figures/ext_lstm_vs_gru.png), `results/ext_lstm/summary.json`)
**Paired search** (same 10 settings and seeds, #33): LSTM − GRU validation r.

| Data | Mean diff [95% CI] | Wilcoxon p | LSTM better |
|---|---|---|---|
| Raw | +0.013 [+0.008, +0.017] | 0.006 | 9/10 |
| 2–8-s clips | +0.033 [+0.019, +0.049] | 0.002 | 10/10 |
| 200–800-ms clips | +0.113 [+0.087, +0.138] | 0.002 | 10/10 |

**Best setting × 3 seeds**, test block:

| Model | r RH / LH | Roughness | Proxy r | Proxy angle |
|---|---|---|---|---|
| GRU raw | 0.959 / 0.887 | 0.038 | 0.29 | 56° |
| GRU 2–8 s | 0.918 / 0.860 | 0.113 | 0.36 | 51° |
| GRU 0.2–0.8 s | 0.764 / 0.667 | 0.313 | 0.33 | 48° |
| LSTM raw | 0.959 / 0.885 | 0.054 | 0.34 | 47° |
| LSTM 2–8 s | 0.931 / 0.876 | 0.083 | 0.37 | 44° |
| LSTM 0.2–0.8 s | 0.808 / 0.744 | 0.288 | 0.36 | 46° |
| LSTM-433 raw (parameter-matched) | 0.964 / 0.885 | 0.052 | 0.33 | 46° |
| LSTM-433 2–8 s (2 seeds) | 0.930 / 0.887 | 0.083 | 0.37 | 41° |

**Findings:**
- On raw data the LSTM is no better on the test block. It wins the paired search by a small, consistent margin across settings.
- With augmentation the LSTM loses less of the raw-data r than the GRU does. The gap grows as clips get shorter (short clips are where the GRU struggled, #14). Its decoded velocity is also smoother (2–8 s: roughness 0.083 vs 0.113).
- The parameter-matched LSTM-433 does as well as LSTM-512, so the advantage isn't parameter count.
- On the proxy, every LSTM scores at or above the best GRU, and the raw LSTM is close to the augmented GRU. That hints the LSTM might be less fragile in closed loop. It can only be tested online, and the proxy is weak (#35).
- **Budget:** 3.03 GPU-h. One run was skipped (`lstm433_augL_seed102`), so that row has 2 seeds.

### Phase 9: one array only ([figure](results/figures/ext_arrays.png), `results/ext_arrays/summary.json`)
**Array split.** We assume channels 1–96 are array A and 97–192 are array B (#36). The data's correlation structure supports this split at moderate strength. Which array is medial and which is lateral is not known.

**Setup.** Each model uses the Phase 8 best hyperparameters for its (cell, data kind) (#37). Each one-array model is trained with 3 seeds. The random-96 controls are 3 subsets drawn across both arrays, 1 seed each, GRU only.

**Test r, RH / LH (% of the 192-channel r retained):**

| Model | Both arrays | Array A only | Array B only | Random 96 |
|---|---|---|---|---|
| GRU raw | 0.959 / 0.887 | 0.859 / 0.765 (90% / 86%) | 0.948 / 0.887 (99% / 100%) | 0.937 / 0.854 (98% / 96%) |
| GRU 2–8 s | 0.918 / 0.860 | 0.861 / 0.773 (94% / 90%) | 0.904 / 0.844 (99% / 98%) | 0.901 / 0.823 (98% / 96%) |
| LSTM raw | 0.959 / 0.885 | 0.869 / 0.835 (91% / 94%) | 0.959 / 0.908 (100% / 103%) | not run |
| LSTM 2–8 s | 0.931 / 0.876 | 0.877 / 0.811 (94% / 93%) | 0.924 / 0.878 (99% / 100%) | not run |

**One array zeroed at test, no retraining (#38).** These are the 192-channel models, mean of 3 seeds.

| Model | Nothing zeroed | Array A zeroed | Array B zeroed |
|---|---|---|---|
| GRU raw | 0.96 / 0.89 | 0.79 / 0.76 | 0.69 / 0.62 |
| GRU 2–8 s | 0.92 / 0.86 | 0.75 / 0.60 | 0.44 / 0.49 |
| LSTM raw | 0.96 / 0.89 | 0.82 / 0.77 | 0.65 / 0.54 |
| LSTM 2–8 s | 0.93 / 0.88 | 0.76 / 0.77 | 0.58 / 0.55 |

**Findings:**
- **Array B alone keeps essentially all of the offline performance**: 98–103% of the 192-channel r, for both hands and both cells.
- **Array A alone loses 6–14%.** The left hand loses more with the GRU (86–90% retained), less with the LSTM (93–94%).
- **Random 96 channels from both arrays** retain 96–98%, between A and B. So the result depends on which array, not only on channel count. On this day, adding array A to B buys almost nothing offline.
- **Retraining is essential.** A 192-channel decoder with one array silenced degrades badly (r 0.44–0.82), worst when B is lost and for the augmented models.
- **The closed-loop proxy ranks the sets the same way**: A < B ≤ 192.
- **Caveats:**
  - one day (1869), offline only
  - array identity is assumed (#36)
  - no per-array hyperparameter search (#37), which may slightly favor the 192-channel models
  - random-96 was GRU only
- **Budget:** 1.82 GPU-h, nothing skipped.

### Phase 10: feasibility for a smaller implant ([figure](results/figures/ext_feasibility.png), `results/ext_feas/summary.json`)
**Why.** The user is considering a different implant: 8 flexible shanks × 32 electrodes, about one Utah array's area in one hemisphere, SBP only. These tests ask how far the paper's decoder survives fewer channels, bad channels, less data and other days.

**Setup.** LSTM-512 on 2–8-s clips with the Phase 8 best setting; day 1869 unless stated; test block 7, mean of the runs shown; RH / LH test r. Reference: 192 channels 0.931 / 0.876. 85 runs, 5.57 GPU-h, none failed.

**Channel count within one array** (random: 3 subsets × 1 seed; greedy + swap: 2 seeds; group lasso and LSTM ablation: 1 seed):

| Array, n | Random (mean) | Greedy + swap | Group lasso | LSTM ablation |
|---|---|---|---|---|
| A, 8 | 0.42 / 0.30 | 0.80 / 0.80 | 0.79 / 0.69 | 0.83 / 0.70 |
| A, 16 | 0.64 / 0.57 | 0.88 / 0.82 | 0.86 / 0.83 | 0.90 / 0.76 |
| A, 32 | 0.73 / 0.59 | 0.87 / 0.83 | 0.90 / 0.83 | — |
| A, 64 | 0.87 / 0.77 | 0.89 / 0.83 | 0.88 / 0.84 | — |
| A, 96 (Phase 9) | 0.88 / 0.81 | | | |
| B, 8 | 0.45 / 0.29 | 0.89 / 0.85 | 0.87 / 0.80 | 0.91 / 0.83 |
| B, 16 | 0.78 / 0.63 | 0.92 / 0.87 | 0.92 / 0.87 | 0.92 / 0.89 |
| B, 32 | 0.85 / 0.74 | 0.93 / 0.88 | 0.94 / 0.89 | — |
| B, 64 | 0.91 / 0.85 | 0.93 / 0.90 | 0.92 / 0.90 | — |
| B, 96 (Phase 9) | 0.92 / 0.88 | | | |

On the surrogate (validation block 6), every selection method scores at or above the 99.7th percentile of 1,000 random sets. Swaps help only at n = 8 (+0.01), and group lasso is below greedy + swap at every n. The sets differ: at n = 8, greedy and group lasso share 4–5 channels, greedy and ablation 4–5.

**Muted channels at test** (192 channels, mean of 3 models × 20 masks):

| Model | Mode | 30% muted | 50% muted |
|---|---|---|---|
| Normal | zero (detected) | 0.87 / 0.84 | 0.62 / 0.62 |
| Normal | noisy | 0.85 / 0.80 | 0.60 / 0.53 |
| Normal | dead (undetected) | 0.47 / 0.32 | 0.13 / −0.01 |
| Dropout-trained | zero (detected) | 0.91 / 0.87 | 0.87 / 0.83 |
| Dropout-trained | noisy | 0.89 / 0.85 | 0.78 / 0.72 |
| Dropout-trained | dead (undetected) | 0.54 / 0.32 | 0.11 / −0.03 |

Dropout-trained, nothing muted: 0.917 / 0.883 (normal 0.931 / 0.876). Proxy r (#35): dropout 0.43 / 0.37 / 0.29 at 0 / 30 / 50% vs normal 0.37 / 0.25 / 0.11.

**Training data** (last k blocks, 3 seeds): 1 block (4.2 min) 0.884 / 0.840; 2 (8.4 min) 0.909 / 0.863; 3 (12.8 min) 0.921 / 0.867; 4 (17.1 min) 0.924 / 0.867; 5 (21.4 min) 0.931 / 0.876.

**Other days** (3 seeds; all trials, then two-hand trials only):

| Day | Both arrays | Array A | Array B |
|---|---|---|---|
| 1750 (one- and two-hand trials) | 0.736 / 0.618; two-hand 0.734 / 0.600 | 0.664 / 0.514; two-hand 0.670 / 0.549 | 0.733 / 0.475; two-hand 0.736 / 0.457 |
| 1867 (one-hand) | 0.906 / 0.891 | 0.893 / 0.844 | 0.896 / 0.862 |

**Findings:**
- **A few well-chosen channels go a long way.** 8 chosen channels of array B keep 96% / 97% of the 192-channel r; 16 keep 98–100%. On array A, 8 chosen channels keep 86% / 92%. Random small subsets are far worse and highly variable (8 random: r 0.11–0.61). So the implant's *best* channels matter far more than the channel count.
- **Selection method:** greedy + swap is best or tied for both hands. LSTM ablation matches it for the RH but is weaker for the LH, and group lasso is slightly weaker at n = 8. Selection used block 6 only, so the test-block numbers are fair, but the chosen sets fit day 1869; their value on other days is untested.
- **Channel dropout training makes the decoder robust to detected or noisy channel loss** at almost no cost: with 50% of channels zeroed it keeps 95% / 94% of its r, vs 67% / 71% for the normal decoder. It also scores higher on the closed-loop proxy at every f.
- **Undetected dead channels are catastrophic for every decoder**, dropout-trained or not (r 0.47 / 0.32 at 30%). A silent channel's z-value (−μ/σ) is far outside the training range. Bad-channel detection and re-zeroing is essential.
- **Little data suffices offline.** One block (4.2 min) keeps 95% / 96% of the r of 21 min.
- **"One array is enough" depends on the day.** On 1867 either array keeps 95–99%. On 1750 (harder; both arrays only 0.74 / 0.62) array B keeps the RH (100%) but loses the LH (77%), and array A keeps 90% / 83%. Two-hand trials score about like all trials.

### Phase 11: crossings vs SBP in T5 ([figure](results/figures/ext_sbp.png), `results/ext_sbp/summary.json`, **caveats: [docs/ext_sbp_notes.md](docs/ext_sbp_notes.md)**)
**Setup.** Public T5 data with both features on the same 48 electrodes (#44): one session, 337 open-loop wrist trials, acausally smoothed. Ridge and FFN, leave-one-block-out (#45). CPU only, about 8 min.

| Analysis (all 48 channels) | TX | log SBP | SBP − TX [95% CI] | TX + SBP |
|---|---|---|---|---|
| Velocity r, ridge | 0.662 | 0.613 | −0.049 [−0.070, −0.031] | 0.681 |
| Velocity r, FFN | 0.700 | 0.673 | −0.027 [−0.047, −0.009] | 0.713 |
| 8-target accuracy, movement | 0.810 | 0.736 | −0.074 [−0.128, −0.018] | 0.843 |
| 8-target accuracy, delay period | 0.617 | 0.568 | −0.049 [−0.131, +0.033] | 0.639 |

**Findings:**
- On all 48 channels SBP is 4–9% behind TX; raw SBP is slightly better than log SBP (velocity 0.625 / 0.676, accuracy 0.766).
- With 4–16 random channels SBP equals or beats TX; within array B it is equal or better at every n, and much better for classification (n = 28: 0.78 vs 0.66). Chosen (greedy) channels favour TX slightly.
- Under muting SBP keeps about the same fraction as TX (50% zeroed: 77% vs 80%). A flat, undetected SBP channel hurts classification badly (the floor value is an assumption, #46).
- An electrode's TX explains a median 35% of its SBP variance; TX + SBP beats either.
- **The main caveats:** the 48 electrodes were kept for their TX rate (favours TX); acausal smoothing; open-loop one-hand task; one session; a different SBP band from the target system's; no LSTM; no bimanual data.

## How to reproduce
Install pixi, run `pixi install`, put the Dryad `.mat` files in `Data/`, then run the `pixi run …` commands in [README.md](README.md) (Replication section) in figure order. GPU steps (`fig3-decoding`, `fig3-sim`, `fig4-rnn`) are resumable.

## Open items for the user
- [ ] **Check the paper against [docs/paper_check.md](docs/paper_check.md)** and report anything that contradicts our assumptions.
- [x] **After Phase 6, give Claude the next steps.** Given 2026-11-21: a write-up (Phase 7), LSTM vs GRU (Phase 8) and one-array decoding (Phase 9).
