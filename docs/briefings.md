# Phase briefings

Written at the end of each phase while the user was away (see the 2026-11-21 entry in PROGRESS.md). Newest first.

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
