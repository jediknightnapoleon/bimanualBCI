# Brain control of bimanual movement enabled by recurrent neural networks

![gitbimanual](https://github.com/d-r-deo/bimanualBCI/assets/153785714/379b8a81-07f6-4cd1-a7c5-7ade4a6018c3)

# Overview
This repo is associated with this [paper](https://www.nature.com/articles/s41598-024-51617-3) and [dataset](https://doi.org/10.5061/dryad.sn02v6xbb). The code contains a simple Matlab analysis script that demonstrates how to load the datasets and plot cosine tuning curves relating brain activity to directional unimanual/bimanual hand movement. 

To execute the analysis script (tuningAnalysis.m), change the dataPath variable to point at the [data](https://doi.org/10.5061/dryad.sn02v6xbb) directory and hit 'run'.

# Replication (fork)
This branch (`replication`) adds an independent Python replication of the paper's offline analyses, plus four extensions that go beyond the paper. Everything runs from the public Dryad data. The online experiments can't be re-run, so Fig 5 is re-scored from the logged closed-loop trials.

**Start here:** [docs/report.md](docs/report.md) is a short narrative of all findings.

## Findings in brief
- **The replication holds.** Every qualitative claim of the paper replicates, and most headline numbers fall within a few points:
  - The left hand's tuning is suppressed about 2× more than the right hand's during bimanual movement, and laterality dominates the neural variance.
  - Nonlinear decoders win because they use the laterality signal.
  - Time-warp augmentation costs a little offline accuracy.
- **Main disagreement:** in the logs, bimanual time to target is about 2× unimanual, not "only slightly longer".
- **Extensions:**
  1. **LSTM vs GRU** (Phase 8): the LSTM ties the GRU on raw data and loses less under augmentation.
  2. **One array** (Phase 9): on day 1869, array B alone (channels 97–192) keeps 98–103% of the two-array performance.
  3. **Feasibility for a smaller implant** (Phase 10): 8 well-chosen channels of one array keep 86–97% of the performance. Training with channel dropout makes the decoder robust to detected channel loss. Undetected dead channels break every decoder. One 4-min block of training data gives about 95%.
  4. **Threshold crossings vs spike-band power** (Phase 11; public T5 data, 48 electrodes): SBP is 4–9% behind with all electrodes and equal or ahead with 4–16. See the caveats in [docs/ext_sbp_notes.md](docs/ext_sbp_notes.md).

## Documents
| File | Contents |
|---|---|
| [docs/report.md](docs/report.md) | Narrative write-up, claim by claim, plus the extensions |
| [REPLICATION.md](REPLICATION.md) | Results tables with CIs, and the numbered register of every decision, deviation and assumption (#1–46) |
| [docs/paper_check.md](docs/paper_check.md) | A reader's guide (paradigm, dPCA) and a list of paper details to verify |
| [docs/ext_sbp_notes.md](docs/ext_sbp_notes.md) | Crossings vs SBP: results and caveats |
| [docs/methods_notes.md](docs/methods_notes.md) | Paper method → code map |
| [docs/briefings.md](docs/briefings.md) | Per-phase briefings |
| [PROGRESS.md](PROGRESS.md) | Status and log |
| `results/figures/` | All figures (PNG and SVG) |

## Setup
1. Install [pixi](https://pixi.sh), then run `pixi install`.
2. Download the [Dryad data](https://doi.org/10.5061/dryad.sn02v6xbb) and extract the `.mat` files and `readme.txt` into `Data/`.
3. Phase 11 only: download `fig1_T5_delay.h5` and `fig1_T5_move.h5` from [Rigotti-Thompson et al. 2026](https://doi.org/10.5061/dryad.brv15dvr9) into `Data/ext/`.

The GPU steps were run on an RTX 3050 laptop GPU (4 GB). The times below refer to it.

| Command | What it does |
|---|---|
| `pixi run test` | Unit tests |
| `pixi run inventory` | Writes `docs/data_inventory.md` (contents of each session file) |
| `pixi run compare-reference` | Checks the Python port against `tuningAnalysis.m`. First run `octave-cli matlab_reference/export_tuning_reference.m` or the MATLAB equivalent. |
| `pixi run fig1-tuning` | Fig 1b-c tuning curves (day 1750) |
| `pixi run fig1-anova` | Supp Fig 1b,c,e single-electrode tuning |
| `pixi run fig2-tuning` | Fig 2b-c, Supp Table 1, Supp Fig 1d (add `--method kfold` for the paper's literal 5-fold) |
| `pixi run fig2-pca` | Fig 2 PCA (day 1881) |
| `pixi run fig2-dpca` | Fig 2 dPCA variance split |
| `pixi run fig3-decoding` | Fig 3 / Supp Fig 2 decoders (add `--target cursor` for the alternative target) |
| `pixi run fig3-sim` | Fig 3 simulations + Supp Fig 4 fSNR (about 20 min on the GPU, resumable) |
| `pixi run fig4-rnn` | Fig 4 RNN search, raw vs augmented (about 1.5 h on the GPU, resumable) |
| `pixi run fig5-behavior` | Fig 5a-c and Fig 4e online re-scoring |
| `pixi run ext-driver` | Phases 8–9: LSTM vs GRU search, then the one-array runs and the zero-out test (about 5 h on the GPU, resumable). Plots: `pixi run ext-lstm --stage plot`, `pixi run ext-arrays --stage plot` |
| `pixi run ext-select` | Phase 10 best-n channel selection: greedy + swap, group lasso, random baseline, ablation (CPU, about 45 min) |
| `pixi run ext-feas-driver` | Phase 10 feasibility runs: a job queue of 85 LSTM runs, then the muting test (about 5.6 h on the GPU, resumable; best-n jobs wait for `ext-select`). Plot: `pixi run ext-feas --stage plot` |
| `pixi run ext-sbp` | Phase 11 crossings vs SBP (CPU only, about 8 min). Plot: `pixi run ext-sbp-plot` |
| `pixi run check-gpu` | Confirms PyTorch sees the GPU |
