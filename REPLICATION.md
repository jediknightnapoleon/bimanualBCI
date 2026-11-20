# Replication report: Deo et al. 2024, Sci Rep

**Paper:** "Brain control of bimanual movement enabled by recurrent neural networks". DOI: [10.1038/s41598-024-51617-3](https://doi.org/10.1038/s41598-024-51617-3).

**Data:** Dryad, [doi:10.5061/dryad.sn02v6xbb](https://doi.org/10.5061/dryad.sn02v6xbb).

This report is filled in as each phase finishes. Progress is tracked in [PROGRESS.md](PROGRESS.md).

## Results: paper vs replication

The target numbers come from summaries of the paper text and are re-checked against the PDF in step 6.2.

| Figure | Quantity | Paper | Ours (95% CI) | Status |
|---|---|---|---|---|
| Fig 2b | RH tuning suppression, bimanual vs unimanual | 12% | – | pending |
| Fig 2b | LH tuning suppression, bimanual vs unimanual | 34% | – | pending |
| Fig 2c | RH uni/bi tuning correlation, x / y | 0.87 / 0.84 | – | pending |
| Fig 2c | LH uni/bi tuning correlation, x / y | 0.42 / 0.45 | – | pending |
| Fig 2 dPCA | Variance explained: laterality / direction | 39% / 30% | – | pending |
| Supp Fig 1d | RH vs LH tuning strength | about 2× | – | pending |
| Supp Fig 4 | Single-bin fSNR, unimanual RH | 2.03 | – | pending |
| Fig 4d | Raw-trained RNN offline r, RH / LH | 0.90 / 0.83 | – | pending |

## Deviations and assumptions
Each one is also marked in the code with `# ASSUMPTION:` or `# DEVIATION:`. Details are in [docs/methods_notes.md](docs/methods_notes.md).
- **Quadrants:** axis-aligned, from the cursor→target vector at the go cue. The paper doesn't define them, and its targets are at random positions.
- **Fig 5 time to acquisition:** taken from the logged `durationBinnedMove` (≥ 499 bins = failure). The target radius isn't stored and can't be recovered as a fixed circle.
- **cvOLS:** reimplemented from the description in Willett et al. 2020.
- **RNN targets and augmentation:** the meaning of "saturated", the clip length (200–800 ms), and no velocity rescaling are all to be checked in Phase 4 and step 6.2.

## Open items for the user
- [ ] **MATLAB cross-check on the Mac (optional).** The Octave cross-check PASSED on 2026-11-20 (Python vs the authors' code on day 1750, max diff 3e-14). Running it in real MATLAB would confirm that Octave behaves like MATLAB here.
  1. On the Mac, clone the fork, check out `replication`, and download the day-1750 `.mat` file from Dryad.
  2. Run `matlab -batch "run('matlab_reference/export_tuning_reference.m')"`.
  3. Push `results/reference/tuning_ref_matlab.mat`.
  4. On Windows, run `pixi run compare-reference`.
- [ ] **After Phase 6, give Claude the next steps.**
