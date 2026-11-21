# Replication progress

Live status of the replication of Deo et al. 2024. The replication plan has phases 0–6. Extensions (approved 2026-11-21) are Phases 7–9: the write-up, LSTM vs GRU, and one-array decoding. The plan file is `C:\Users\10910\.claude\plans\i-have-just-cloned-modular-giraffe.md`.
**To resume after an interruption:** find the first step that isn't `done` below and read the latest log entry.

## Status

| Step | Description | Effort | Status |
|---|---|---|---|
| 0.0 | Fork, remotes, `replication` branch | low | done |
| 0.1 | pixi environment, scaffold, memory note | low | done |
| 0.2 | Download Dryad data into `Data/` | low | done (all 13 files OK) |
| 0.3 | Data inventory → `docs/data_inventory.md` | medium | done |
| 0.4 | Port loader / conditions / epochs (+tests) | medium | done |
| 0.5 | Octave cross-check vs `tuningAnalysis.m` | low–medium | done (PASS, max diff 3e-14) |
| 1.1–1.2 | Fig 1, Supp Fig 1b,c,e | medium | done |
| 2.1–2.4 | Fig 2, Supp Fig 1d, Supp Table 1 | high/medium | done |
| 3.1–3.3 | Fig 3, Supp Figs 2, 4 | medium/high | done |
| 4.1–4.4 | RNN + augmentation, Fig 4 | high/low/medium | done |
| 5.1–5.2 | Fig 5 re-scored from closed-loop logs | medium | done |
| 6.1–6.2 | Final report | medium | done |
| 7.1 | Write-up `docs/report.md` | medium | done |
| 8.1 | LSTM code (`rnn.py` cell option, `ext_lstm.py`, tests) | high | todo |
| 8.2 | LSTM / GRU matched search (background GPU) | background | todo |
| 8.3 | Closed-loop proxy metric (`proxy.py`) + validity check | high | todo |
| 8.4 | LSTM vs GRU summary, figure, docs | medium | todo |
| 9.1 | Array-split check (correlation structure) | medium | todo |
| 9.2 | `ext_arrays.py` (one-array, random-96, zero-out) | medium | todo |
| 9.3 | One-array runs (background GPU, chained after 8.2) | background | todo |
| 9.4 | One-array summary, figure, docs | medium | todo |

## Log

### 2026-11-20
- **0.0:**
  - The user forked the repo; `origin` is `jediknightnapoleon/bimanualBCI` and `upstream` is `d-r-deo/bimanualBCI`.
  - The `replication` branch is pushed.
- **0.1:**
  - Added `pyproject.toml` (pixi workspace), `.gitignore`, `bimanual/cache.py` with tests, this file, `REPLICATION.md`, and `docs/`.
  - The GPU driver supports CUDA 12.7; torch uses the cu126 wheels.
  - Resume: run `pixi install`, then `pixi run check-gpu`, then `pixi run test`.
- **0.1 done:**
  - `pixi install` finished: torch 2.14.1+cu126 detects the CUDA GPU (RTX 3050), the dPCA package imports, and 3/3 tests pass.
  - Next: step 0.2, the data download.
- **0.2:**
  - The scripted Dryad download failed: the API returned 401 and requires a login token.
  - The user is downloading `bimanualData.zip` and `README.md` into `Data/` in the browser; the browser estimates hours.
- **0.5, done early while waiting for the data:**
  - Added `matlab_reference/`: `export_tuning_reference.m`, plus the helpers copied verbatim from `tuningAnalysis.m` and an Octave `nanmean` shim.
  - Started the Octave install with `winget install GNU.Octave`.
  - Still to do once the data arrives: run the wrapper on day 1750, then write and run `compare_reference.py`.
- **0.4, partly done early:**
  - Added `bimanual/conditions.py`, a faithful port of the wedge labeling in `tuningAnalysis.m`, and `tests/test_conditions.py`. Total tests: 33/33 pass.
  - **Open:** the 4-quadrant labels the paper uses for the ANOVA and augmentation aren't defined anywhere. Decide them after seeing the target layout in the data.
  - Resume: wait for `Data/bimanualData.zip`, then extract it and continue with step 0.3.
- **5.1, done early while waiting for the data:**
  - Added `bimanual/behavior.py` (TTA, angular error, speed profile) and `bimanual/stats.py` (`bootstrap_ci`), with tests in `tests/test_behavior.py`. Total tests: 41/41 pass.
  - **Open:** the target radius and how trial ends are logged are unknown until the data inventory (step 0.3). Assumptions are listed in the module docstring.
- **0.5:**
  - Octave 11.3.0 is installed at `%LOCALAPPDATA%\Programs\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe`.
  - Smoke test: the helpers run with the `nanmean` shim, and on the test vectors the results agree with the Python port (wedges 6, 8, NaN, 5).
- **2.1, done early:**
  - Added `bimanual/encoding.py`: cvOLS with folds split by trial, unbiased squared norm and magnitude, tuning strength, cross-validated Pearson r, laterality axis and its removal.
  - The tests in `tests/test_encoding.py` use synthetic neurons. Untuned channels come out unbiased at about 0, true magnitudes and correlations are recovered, and the naive estimates are shown to be biased.
- **4.1, done early:**
  - Added `bimanual/targets.py` (saturated min-jerk velocity targets, 4 context classes, balancing strata) and `bimanual/augment.py` (time warp, balanced `SnippetSampler`, train/val split before augmentation, raw-snippet baseline).
  - Tests in `tests/test_augment.py`.
  - Fixed a bug the tests caught: clip centers were barred from bins near segment edges, so short labels like "rest" were never sampled.
  - Total tests: 60/60 pass.
  - Resume: wait for the data, then do step 0.2 (extract) and step 0.3 (inventory).
- **0.2 done:**
  - `Data/` has `readme.txt` and all 13 session files (`t5_MM_DD_YYYY.mat`, MATLAB v5, readable with `scipy.io.loadmat`).
  - 12 of the 13 files load fully.
  - **`t5_09_27_2021.mat` (day 1867) is corrupted in its `tx` variable** (zlib "invalid distance code", probably from the extraction error the user saw). Its `Cp`, `Tp`, `blockNum`, `goCue`, `moveType` and the duration variables all read fine.
  - Impact: day 1867 is used only for Fig 4E (cursor speed traces), which needs only `Cp`, so nothing is blocked. Its neural data isn't used by any figure.
  - To repair it, re-extract from a fresh copy of the zip; the loader must read the file variable by variable until then.
  - The readme gives per-day block labels (OL / CLR / CL, decoder type, gains, augmented vs raw, sequential vs simultaneous), so Fig 5 can be fully re-scored.
- **0.2 update:** the user replaced `t5_09_27_2021.mat` (16.0 MB, read-only attribute). All 13 files now load fully, each `tx` is 192 × T, and the corruption note above no longer applies.
- **0.3 done:**
  - Added `bimanual/sessions.py`, the session catalogue with per-block labels transcribed from the readme, and `scripts/01_inventory.py` (`pixi run inventory`).
  - Outputs: `docs/data_inventory.md` and `results/inventory.json`.
  - All readme blocks are in the data, and every data block has a label.
  - Open items resolved:
    1. Quadrants: targets are at random positions and all directions are evenly covered, so I'm using axis-aligned quadrants of the cursor→target vector at the go cue (ASSUMPTION).
    2. Target radius: it isn't recoverable as a fixed circle, and it isn't needed.
    3. TTA: use the logged `durationBinnedMove` (≥ 499 = timeout failure).
  - Side notes:
    - "Uni only" OL blocks contain about 1 stray bimanual trial each.
    - Some moving-hand cursor→target distances at the go cue are about 0; check this in Phase 1.
    - Day 1883 has 951 trials, while the paper's Fig 3 uses 482; work out the subset in Phase 3.
    - The day-1853 CL blocks already show the RNN ≫ linear success pattern of Fig 5c.
  - Next: step 0.4 (`io.py`, `epochs.py`).
- **0.4 done:**
  - Added `bimanual/io.py` (`load_session`, block-mean removal, z-scoring, and the `trials` DataFrame with readme labels; 1-based→0-based conversion happens only here).
  - Added `bimanual/epochs.py` (`extract_windows`, plus `feature_tensor`, a faithful port of `computeFeatureMatrices`) and `bimanual/tuning.py`, a faithful port of the `tuningAnalysis.m` curves.
  - Total tests: 64/64 pass.
- **0.5 done:**
  - Octave ran `export_tuning_reference.m` on day 1750 in 10 s, and `pixi run compare-reference` passes.
  - Trial labels are identical; the uniR / uniL / biR / biL curves have a max absolute diff ≤ 2.8e-14.
  - The Mac MATLAB check remains an optional extra confirmation.
- **Phase 0 complete:** committed and pushed (see the phase 0 commit).
- **Between Phase 0 and Phase 1, at the user's request:**
  - Turned the deviations section of `REPLICATION.md` into a numbered register of decisions, deviations and assumptions (17 entries).
  - Added assumption #1: the 3 Dryad versions hold identical data and differ only in the README.
  - Dropped the Mac MATLAB check (the user's decision; the Octave cross-check passed).
- **1.1 done:**
  - Added `scripts/fig1_tuning.py` (`pixi run fig1-tuning`), writing `results/figures/fig1_tuning_curves.png` and `results/fig1/tuning_curves.json`.
  - The curves equal the verified port (diff 2.5e-14).
  - Qualitatively they match the paper:
    - electrode 97: RH preserved, LH kept but weaker;
    - electrode 13: both hands weaker in bimanual;
    - electrode 23: LH tuning abolished in bimanual, RH preserved.
  - Also added `bimanual/plotting.py`, `conditions.quadrant`/`trial_quadrants` and `stats.anova_fvaf`, with tests; 66/66 pass.
  - The near-zero-distance quirk affects only 1 trial (day 1869, CL); day 1750 is clean.
- **1.2 done:**
  - Added `scripts/fig1_supp_anova.py` (`pixi run fig1-anova`), writing `results/figures/supp_fig1_tuning_counts.png` and `results/fig1/supp_fig1_anova.json`.
  - Tuned electrodes at p < 1e-5: uniR 83, uniL 50, biR 87, biL 28.
  - Electrodes tuned to 0/1/2/3/4 contexts: 88/25/29/35/15.
  - This matches the paper's claims: the RH is preferred in both contexts, LH tuning drops in bimanual, and tuning is mixed across contexts.
- **Phase 1 complete:** committed and pushed.
- **2.2 done (2026-11-21):**
  - Added `bimanual/blocksets.py` with 3 block-set schemes. The main one, `adjacent_all`, gives exactly 25 sets (22 usable); `all_pairs_mixed` gives 38 and `adjacent_mixed` 17.
  - Added `conditions.trial_directions` and `scripts/fig2_tuning_change.py` (`pixi run fig2-tuning [--method splithalf|kfold]`).
  - **Finding:** the paper's literal 5-fold cvOLS explodes on sets with about 25 bimanual trials (each fold has about 5 trials for a 5-parameter model; in synthetic tests the SD is about 44,000 against a true value of 442).
    - I added a split-half cvOLS (`method="splithalf"`, 50 random trial halves). It is unbiased, stable, and has lower variance than k-fold even with 50 trials.
    - Split-half is now the main estimator; k-fold results are kept for comparison (register #23).
  - Results (split-half, main scheme):
    - Suppression: RH 18.7% (paper 12%), LH 41.3% (paper 34%); t-test p = 6e-6.
    - Uni-vs-bi correlation: RH 0.86 / 0.86 (paper 0.87 / 0.84); LH 0.49 / 0.60 (paper 0.42 / 0.45).
    - UniR-UniL: x −0.34, y +0.51 (anti-correlated in x, correlated in y, as in the paper).
    - Supp Table 1: 11/12 p-values agree with the paper at 0.01.
    - Supp Fig 1d: RH/LH = 1.70 (LH 41% weaker; paper about 2×, 48%).
  - All 3 schemes agree to within a few percentage points.
- **2.3 done:**
  - Added `scripts/fig2_pca.py` (`pixi run fig2-pca`), writing `results/figures/fig2_pca.png` and `results/fig2/pca.json`.
  - Day 1881 OL trials: 277 uniR, 237 uniL, 138 bi. PC1 explains 17.6% of variance and PC2 10.0%.
  - Centroid distances in PC1-2: bi–uniR 0.28, bi–uniL 3.21. Full space: 2.21 vs 3.53.
  - So bimanual is closest to uniR, as in the paper.
  - Next: step 2.4, dPCA (high effort).
- **2.4 done:**
  - Added `bimanual/dpca_analysis.py` (Kobak-style marginalization, two-split cross-validated variance) with `tests/test_dpca.py`, and `scripts/fig2_dpca.py` (`pixi run fig2-dpca`).
  - Cross-validated fractions, mean over the 5 days:
    - 300–700 ms: laterality 41%, direction 33%, interaction 23%, time 3%.
    - 0–1000 ms: laterality 39%, direction 29%, interaction 20%, time 12%.
    - The paper reports 39% / 30%, so the 0–1000 ms window matches almost exactly.
  - Laterality > direction on every day.
  - Without cross-validation the order flips (direction 35% > laterality 28%), so cross-validation is essential to the claim.
- **Phase 2 complete:** committed and pushed.
- **Between Phase 2 and Phase 3:** added `docs/paper_check.md`, a check-list of paper details for the user to verify (linked to register numbers), plus plain-language explanations of the paradigm and of dPCA/CV. It goes into the Phase 3 commit.
- **3.1–3.2 done:**
  - Added `bimanual/decoders.py` (stratified trial folds, ridge with an alpha sweep by Pearson r, and a 1×512 ReLU FFN with early stopping, in PyTorch on the GPU) with `tests/test_decoders.py`.
  - Added `scripts/fig3_decoding.py` (`pixi run fig3-decoding [--target direction|cursor]`). Predictions are cached in `results/cache/` so a rerun can resume.
  - **Env fix:** the conda-forge MKL (LLVM libomp) clashed with torch's Intel libiomp5md (OMP Error #15). The conda BLAS is now OpenBLAS (`libblas *openblas` in pyproject). Tests and the Octave cross-check still pass.
  - Day 1883: 491 OL unimanual trials (the paper says 482; register #12).
  - Targets: the recorded OL cursor velocity was tried first. It is mostly the onset ramp (the cursor starts moving about 300 ms after go), so every r is capped low. MAIN = unit direction (register #11); cursor velocity is kept as an alternative.
  - Direction results:
    - r: ridge R 0.72/0.79, L 0.45/0.56; FFN R 0.75/0.80, L 0.56/0.69.
    - Paired bootstrap: FFN − ridge +0.072 overall and +0.120 for the LH (p < 0.001). Removing laterality costs the FFN 0.080 (LH 0.131) and ridge 0.000 (p = 0.78).
    - Jitter (L): ridge 0.84, FFN 0.42, FFN − lat 0.84.
    - Bimanual: RH > LH; the FFN is slightly below ridge (only 131 trials).
  - Cursor targets: the same effects, smaller but significant.
  - Next: step 3.3, simulations (high effort).
- **2026-11-21, user instruction (going to sleep):** run on through Phases 3 and 4 WITHOUT pausing, and stop before Phase 5.
  - Phase-end briefings go to `docs/briefings.md`; phase-end commits and pushes go ahead as approved.
  - The RNN search budget defaults to about 20 models, capped at about 3 GPU-hours.
  - Effort stays high.
  - Ambiguities get the documented default plus a register entry, flagged for review.
- **3.3 done:**
  - Added `bimanual/simulate.py` (Eq. 4 population, i.i.d. Gaussian bin noise, clipping, fSNR) with `tests/test_simulate.py`, and `scripts/fig3_simulations.py` (`pixi run fig3-sim`, resumable through `results/fig3/sim_grid.jsonl`).
  - Real day-1883 magnitudes (Hz): Rx 120, Ry 149, Lx 48, Ly 81, lat 86. RH-LH correlation: x −0.38, y +0.53.
  - **fSNR:** real single-bin 2.11 (paper 2.03), window-average 4.60. Simulated (σ = 56.6 Hz): 2.12 single-bin and 9.01 window-average, the same mismatch the paper reports.
  - **Sweep** (6 ρ × 5 laterality scales, 2000 + 2000 trials × 20 bins, 20% held out):
    - Ridge falls from 0.59 to 0.50 as ρ goes 0→1 (LH 0.45 → 0.32), independent of laterality.
    - The FFN rises with laterality and stays at about 0.69 for every ρ when laterality is ≥ 2× real.
    - At laterality 0 the FFN is slightly below ridge.
- **Phase 3 complete:** committed and pushed. `bimanual/rnn.py` and `tests/test_rnn.py` (step 4.2, written while the sweep ran; 3 tests including exact checkpoint/resume) are in this commit too.
- **4.2 done:**
  - Added `scripts/fig4_rnn.py` (`pixi run fig4-rnn [--stage all|smoke|search|extras|plot]`).
  - Day 1869 OL split by block: train 1–5, val 6, test 7.
  - 2000 augmented snippets take 6 s to generate (17 strata); training runs at 69 ms/step (batch 32).
  - 400-step sanity check on the val block: raw r 0.94 / 0.88 (RH / LH), augmented 0.67 / 0.57 and rougher.
- **4.3 started:** background search with 10 settings × {raw, aug} = 20 models, then extras (2 extra seeds per best setting and the "hold" target reading). Estimated about 1.5 h.
  - Log: `results/fig4_rnn.log`; per-model results: `results/rnn_search/trials.jsonl`; checkpoints: `results/checkpoints/`.
  - **Resume after an interruption:** rerun `pixi run fig4-rnn`. Finished models are skipped, and an unfinished one resumes from its last checkpoint (every 250 steps).
- **2026-11-21, user instruction:** if usage remains after Phase 4, continue straight into Phase 5 without pausing. Its briefing also goes to `docs/briefings.md`.
  - Phase 5 (CPU only) was started while the Phase 4 GPU search ran.
- **5.2 done (run while the Phase 4 search was on the GPU):**
  - Added `scripts/fig5_behavior.py` (`pixi run fig5-behavior`), writing `results/fig5/trials.csv`, `summary.json`, `results/figures/fig5_online.png` and `fig4e_online_speed.png`.
  - **5a** (1776/1778/1790/1792 RNN CL, pooled):
    - success: uniR 83%, uniL 86%, bi 71%;
    - TTA (successful trials): uniR 2.82 s, uniL 3.02 s, bi 5.37 s;
    - angular error: uniR 34°, uniL 43°, bi 52°.
    - Bimanual angular error is higher, as in the paper. Bimanual TTA is about 2× unimanual, NOT "only slightly longer" (register #30; flagged in paper_check).
  - **5b:** sequential is faster than simultaneous for every type. bi 5.3 s vs 7.1 s; success 96% vs 67%.
  - **5c:** the RNN is better than linear at every gain on mixed blocks (linear mostly times out). On uni-only locked blocks (1855) linear works (TTA about 3.4–4.2 s), as the paper says.
  - **4e:**
    - success augmented vs raw: 1867 59% vs 7%; 1869 86% vs 16%.
    - Speed: raw gives a single pulse at 0.4–1.0 s, then about 40 units/s; augmented gives sustained 150–200 units/s.
- **4.3 interim finding:**
  - Raw-trained models match the paper: test r about 0.92–0.95 RH and 0.83–0.89 LH (paper 0.90 / 0.83).
  - Augmented models with 200–800-ms clips are MUCH lower (about 0.56–0.63 RH, 0.43–0.49 LH), while the paper says "slightly lower".
  - Added `--stage clips`: the best augmented setting retrained with 2–8-s clips (the "0.2–0.8 × snippet length" reading, register #14), 3 seeds.
  - **Resume:** after the search finishes, run `pixi run fig4-rnn --stage clips`, then the Phase 4 briefing, commit and push. Phase 5 is already analysed; commit it after Phase 4.
- **2026-11-21, user instruction:** after Phase 4, finish everything (Phases 5 and 6) without pausing.
  - Phase 6 adds no new analysis.
  - At the end, give the final briefing and ASK the user for next steps.
- **6.2 done early:** a web-fetch agent re-read PMC10796685. The results are in `docs/paper_check.md` (a new table) and the register statuses. Key points: all headline numbers are confirmed; the clip length is contradictory within the paper; Eq. 2 is the raw per-step difference (deviation #21); Fig 5a lists days 1752 / 1771, which have no CL data in Dryad.
- **4.3–4.4 done:**
  - 20 search models, plus 2 extra seeds × {raw, aug}, plus the "hold" target × {raw, aug}, plus 3 long-clip models. Each model takes about 100–250 s, about 1.6 GPU-h in total.
  - Test block 7, mean of 3 seeds:
    - raw: RH 0.96, LH 0.89, roughness 0.04 (paper 0.90 / 0.83);
    - augmented with 200–800-ms clips: RH 0.76, LH 0.67, roughness 0.31;
    - **augmented with 2–8-s clips: RH 0.93, LH 0.87, roughness 0.08**.
  - The 2–8-s reading reproduces the paper's "slightly poorer ... noisier": it is now the MAIN reading (register #14).
  - The "hold" target reading is worse for both (raw RH 0.94/LH 0.86; aug RH 0.69/LH 0.60), so "clip" stays the main reading (#13).
  - Figure: `results/figures/fig4_rnn.png`.
- **Phase 4 complete:** committed and pushed.
- **Phase 5 complete:** briefing written; committed and pushed.
- **Phase 6 complete (2026-11-21):** `REPLICATION.md` has its summary; the briefings are written. REPLICATION DONE. Waiting for the user's next steps.

### 2026-11-21: extensions (Phases 7–9)
- **New plan approved.** Pause only when the effort level must change (two pauses: medium → high after Phase 7, and high → medium once the GPU driver is running). Briefings go into `docs/briefings.md`.
- **7.1 done:** `docs/report.md`, a short narrative written for the user, is linked from the README and `REPLICATION.md`.
  - Next: block 2 [high], steps 8.1, 8.3, 9.1 and 9.2; then start the GPU driver.
- **Block 2 (high) done:**
  - **8.1:** `rnn.RNNDecoder(cell="gru"|"lstm")`; `GRUDecoder` is kept as an alias and old checkpoints load. `fig4_rnn.py` now takes `cell`, `channels`, a trials path and a proxy hook. Retraining s07_raw through the new code reproduces its test r exactly, and `fig4-rnn --stage plot` reproduces `summary.json` byte for byte.
  - **8.3:** `bimanual/proxy.py` with tests. Validity check on the existing GRU models: augmented beats raw on r (0.33–0.35 vs 0.29) and angular error (43–52° vs 55–58°), but far less than online, and "sustain" doesn't separate them. A weak, secondary metric (#35).
  - **9.1:** array check PASSES at moderate strength (#36). Within-half correlation > across-half beyond the random null on 3 days; 92% of channels prefer their own half; the 1–96 split ranks 2 / 1 / 40 of 96 contiguous shifts.
  - **9.2:** `scripts/ext_arrays.py` (check / run / zero / plot). Tests: 93 pass.
  - A dry run with a stub trainer confirmed the run order: 54 Phase 8 models, then 30 Phase 9 models.
  - **GPU driver started:** `pixi run ext-driver`, logging to `results/ext_driver.log`. It runs the LSTM/GRU search (cap 3 h), then the one-array runs (cap 2 h), then the zero-out evaluation.
  - **Resume:** rerun `pixi run ext-driver`; finished models are skipped.
  - **Next (block 3, medium):** write `scripts/ext_lstm_plot.py` and `scripts/ext_arrays_plot.py`, then the summaries and docs.
