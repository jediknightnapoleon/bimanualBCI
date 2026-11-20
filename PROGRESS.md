# Replication progress

Live status of the replication of Deo et al. 2024. The full plan has phases 0–6 with numbered steps.
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
| 1.1–1.2 | Fig 1, Supp Fig 1b,c,e | medium | todo |
| 2.1–2.4 | Fig 2, Supp Fig 1d, Supp Table 1 | high/medium | 2.1 done early (`encoding.py`); 2.2–2.4 todo |
| 3.1–3.3 | Fig 3, Supp Figs 2, 4 | medium/high | todo |
| 4.1–4.4 | RNN + augmentation, Fig 4 | high/low/medium | 4.1 done early (`targets.py`, `augment.py`); 4.2–4.4 todo |
| 5.1–5.2 | Fig 5 re-scored from closed-loop logs | medium | 5.1 done early (`behavior.py`); 5.2 todo |
| 6.1–6.2 | Final report | medium | todo |

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
