# Methods notes: paper methods → code

This file maps each method in the paper to the code that implements it. Each section is filled in when the corresponding code is written.

| Paper method | Code | Notes |
|---|---|---|
| Re-run safety (not in the paper) | `bimanual/cache.py` | Skip a step if its outputs exist, write files atomically, keep append-only JSONL logs |
| Trial labels for Fig 1b-c: movement type and 8 direction wedges | `bimanual/conditions.py` (`trial_conditions`, `which_target`) | Faithful port of `tuningAnalysis.m:30-53, 154-202`. Keeps the MATLAB quirks: angles are floored to whole degrees, and a vector exactly on an axis gets NaN. `goCue` is 1-based in the `.mat` files. |
| MATLAB/Octave reference oracle | `matlab_reference/export_tuning_reference.m` | Runs `tuningAnalysis.m` lines 13–118 verbatim, without plotting, and saves the tuning curves for comparison with the Python port |
| Fig 5 TTA (500-ms simultaneous dwell, 10-s timeout) | `bimanual/behavior.py::time_to_acquisition` | ASSUMPTION: TTA is measured to the end of the dwell (`include_dwell=True`). The target radius is a parameter, to be read from the data. |
| Fig 5 angular error (300–500 ms after go) | `bimanual/behavior.py::angular_error` | ASSUMPTION: the target direction is fixed at the go cue; movement direction comes from the per-bin cursor displacement; bins with no movement are skipped. |
| Fig 4e speed traces | `bimanual/behavior.py::speed_profile` | Speed per bin |
| 95% bootstrap CIs (10k) | `bimanual/stats.py::bootstrap_ci` | Percentile bootstrap; NaNs (failed trials) are dropped |
| Eq. 1–2 encoding model, cvOLS (5-fold) | `bimanual/encoding.py::cv_ols` | Folds split by trial (`groups`). Each fold yields a train fit and a test-fold-only fit. |
| Tuning strength (Fig 2b) | `encoding.py::cv_magnitude`, `tuning_strength` | Unbiased squared norm = mean of cross-fold products <train_k, test_k>; magnitude = signed square root; a hand's strength = mean of its x and y magnitudes. DEVIATION: reimplemented from the Willett et al. 2020 description, not their code. |
| Tuning correlation (Fig 2c) | `encoding.py::cv_correlation` | Channel-centered (Pearson). Models from different trials use full-fit cross products; within one model, cross-fold products. Can slightly exceed ±1. |
| Eq. 4 laterality axis and removal (Figs 2, 3) | `encoding.py::laterality_axis`, `remove_axis` | Removal projects out the unit b_lat; a refit then gives b_lat ≈ 0 |
| RNN targets (Fig 4b; Supp Fig 3) | `bimanual/targets.py::build_targets` | ASSUMPTION: "saturated" = min-jerk speed bell clipped at `sat`=0.5 of its peak and rescaled (alternative: `hold_to_end=True`). 200-ms reaction time. 4 context classes. Balancing strata: delay×3 + rest + movement type × quadrant. |
| Temporal fracturing (Fig 4c-d) | `bimanual/augment.py::SnippetSampler` | 10-s snippets (500 bins); ASSUMPTION: clips are 10–40 bins (200–800 ms); dilation U[0.5, 2] by linear interpolation (labels use the nearest bin); strata sampled uniformly; train/val split by trial before augmentation. ASSUMPTION: velocity amplitude is not rescaled by the dilation. |
| Non-augmented baseline (Fig 4d) | `augment.py::raw_snippets` | Consecutive 500-bin windows in the original order |
| Session catalogue and block labels | `bimanual/sessions.py` | Supp Table 2 plus the readme's block conditions, checked against the data by `scripts/01_inventory.py` |
| Loading and preprocessing (block-mean removal, ×50 Hz, z-score) | `bimanual/io.py` | Time-major arrays; the only place where 1-based `goCue` becomes 0-based |
| Trial windows; `computeFeatureMatrices` port | `bimanual/epochs.py` | Windows are inclusive, like MATLAB: (15, 35) = 21 bins |
| Fig 1b-c tuning curves (port of `tuningAnalysis.m`) | `bimanual/tuning.py::tuning_curves` | Matches Octave running the authors' code to 3e-14 (`pixi run compare-reference`) |
| Direction quadrants (Supp Fig 1 ANOVA, augmentation balancing) | to be added in Phase 1 | ASSUMPTION: axis-aligned quadrants of the cursor→target vector at the go cue. Targets are at random positions; directions are evenly spread. |
| Fig 5 TTA source | `durationBinnedMove` (logged) | Decision from step 0.3: use the task's logged trial length (≥ 499 bins = timeout failure); `behavior.time_to_acquisition` is a sanity check only |
