# Brain control of bimanual movement enabled by recurrent neural networks

![gitbimanual](https://github.com/d-r-deo/bimanualBCI/assets/153785714/379b8a81-07f6-4cd1-a7c5-7ade4a6018c3)

# Overview
This repo is associated with this [paper](https://www.nature.com/articles/s41598-024-51617-3) and [dataset](https://doi.org/10.5061/dryad.sn02v6xbb). The code contains a simple Matlab analysis script that demonstrates how to load the datasets and plot cosine tuning curves relating brain activity to directional unimanual/bimanual hand movement. 

To execute the analysis script (tuningAnalysis.m), change the dataPath variable to point at the [data](https://doi.org/10.5061/dryad.sn02v6xbb) directory and hit 'run'.

# Replication (fork)
This branch (`replication`) adds a Python replication of the paper's offline analyses. For a short narrative of the findings, read [docs/report.md](docs/report.md). See [REPLICATION.md](REPLICATION.md) for results, [docs/paper_check.md](docs/paper_check.md) for a reader's guide (paradigm, dPCA) and a list of paper details to verify, [PROGRESS.md](PROGRESS.md) for status, and [docs/](docs/) for the data inventory and method notes.

**Setup:** install [pixi](https://pixi.sh), then run `pixi install`. Download the Dryad data and extract the `.mat` files and `readme.txt` into `Data/`.

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
| `pixi run check-gpu` | Confirms PyTorch sees the GPU |
