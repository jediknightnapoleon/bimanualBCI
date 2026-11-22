# Phase 11: threshold crossings vs spike-band power in T5

**Question.** The 8 × 32 flexible-shank system records spike-band power (SBP) only. The bimanual T5 data used in Phases 0–10 has threshold crossings (TX) only. Would an SBP-only system lose decoding performance, or degrade differently with fewer or bad channels?

**Short answer.** With the same participant and arrays, SBP decoded slightly worse than TX on all 48 channels:
- velocity r 0.61–0.63 vs 0.66 with ridge, 0.67 vs 0.70 with the FFN;
- 8-target accuracy 0.74–0.77 vs 0.81.

The gap shrinks or reverses with few channels:
- With 4–8 random channels, SBP was equal or slightly better.
- On array B alone, SBP matched TX for velocity and beat it for target classification (0.78–0.81 vs 0.66).

Under random channel muting, SBP keeps about the same fraction of its performance as TX. SBP is not just a copy of TX:
- on average, an electrode's own TX explains only 35% of its SBP variance;
- SBP added to TX improves decoding slightly (+0.02 velocity r).

Read every number with the caveats below. Several of them probably favour TX.

Figure: `results/figures/ext_sbp.png`. Numbers: `results/ext_sbp/summary.json`. Code: `scripts/ext_sbp.py`, `scripts/ext_sbp_plot.py`, `bimanual/sbp.py`.

---

## Caveats (read first)

| Caveat | Why it matters |
|---|---|
| **Only 48 of 192 electrodes, kept by a TX firing-rate cut (> 2 Hz)** (20 on array A, 28 on array B). | Electrodes with few threshold crossings were removed before we got the data. Those are exactly the electrodes where SBP can still carry signal (small, distant units below the −3.5 × RMS threshold). The selection therefore **favours TX**, and the SBP-only advantage on a full array is likely underestimated. It also caps the channel-count curves at 20 / 28 per array, not 64 / 96. |
| **Acausal Gaussian smoothing (σ = 100 ms) applied by the data authors** | The features use future samples, so nothing here is a real-time decoder. Absolute r and accuracy values are inflated relative to a causal decoder. The TX–SBP *comparison* is fair because both features had the same smoothing. Smoothing also helps TX more than it would online: raw 20-ms TX counts are noisier than raw SBP, so causal decoding would likely narrow the gap further. That is an expectation, not something tested here. |
| **Open-loop, one-hand, radial-8 attempted wrist movements** | The cursor followed a cued trajectory, so "velocity" is the instructed movement, not a decoded one. There is no bimanual, closed-loop or two-cursor condition. Whether SBP preserves the *laterality* signal that the paper's FFN/RNN exploit is **not tested**. |
| **One session, 3 blocks (7, 8, 9), 337 usable trials** (5 dropped for NaNs) | Leave-one-block-out has 3 folds and about 225 training trials. Classification accuracies have wide CIs (±5–8 points), and day-to-day drift (a known difference between TX and SBP) is not tested. |
| **The SBP band differs from the target system's** | Here: 250–5000 Hz band-pass, acausal 4th-order Butterworth, mean power per bin. Target system: 300 Hz high-pass, with the rest of the processing unknown. Band edges, power vs amplitude, rectification and decimation all change SBP's noise level and how much unit activity it carries. |
| **No LSTM and no bimanual pipeline** | Only ridge and a 1-layer FFN were used. The smoothed, trial-aligned data cannot feed the Phase 8 LSTM. We also could not build an SBP stand-in for the bimanual sessions: a raw-bin TX→SBP model cannot be fitted from smoothed, trial-aligned data, so that planned step was dropped (register #44). |
| **T5 only** | One participant, one pair of Utah arrays, years after implant. Flexible shanks have different electrode sizes, impedances and tissue contact, so their SBP-to-TX relation may differ. |
| **Log vs raw SBP** | log(SBP) was chosen a priori as the main SBP feature. Raw SBP turned out slightly better (by 0.01–0.03), so the log-SBP numbers in items 2–3 are mildly pessimistic. Raw SBP is in `summary.json` throughout. |
| **The "dead channel" muting value for SBP is an assumption** | A disconnected SBP channel was set to its training 1st percentile, i.e. its floor. Real failed SBP channels may instead sit at amplifier noise or rail high. The dead-mode classification results (below) depend on this choice. |
| **The "noisy" muting mode is unsmoothed N(0, 1)** | Real channels in this data set are smoothed, so the injected noise is rougher than a real noisy smoothed channel would be. It is the same for TX and SBP. |

---

## What was tested

Everything is CPU only and leave-one-block-out (3 folds). Ridge α is chosen by an inner leave-one-block-out on the training blocks. Features are z-scored with training-block statistics.

| # | Analysis | Details |
|---|---|---|
| 1 | Feature comparison | Features:<br>• TX<br>• SBP (raw)<br>• log SBP<br>• TX + log SBP<br>• **SBP residual**: log SBP minus the part predicted from the same electrode's TX (5 time shifts, fitted on training blocks)<br><br>Targets:<br>• per-bin cursor velocity, −200 to 500 ms after the go cue: the cursor's outward movement, excluding its jump back at about 600 ms<br>• 8-target classification from mean features over 0–400 ms<br>• preparatory classification from the delay period, 200–700 ms after delay start, for trials with delay ≥ 1 s (183 trials)<br><br>Decoders: ridge, ridge with causal lags (0/40/80 ms), FFN with 3 seeds.<br>Paired bootstrap over trials (2000 resamples) of SBP − TX. |
| 2 | Channel count | 200 random subsets per n, **the same subsets for TX and SBP**:<br>• all 48: n = 4, 8, 16, 32, 48<br>• array A: n = 4, 8, 16, 20<br>• array B: n = 4, 8, 16, 28<br><br>Plus greedy best-n (forward selection on the inner-fold velocity r, separately for TX and SBP). |
| 3 | Muting at test time | Fraction f = 0–0.6, 20 masks each (seed 7000 + 100f, as in Phase 10).<br><br>Modes:<br>• zero: detected and removed<br>• dead: flat channel<br>• noisy: N(0, 1)<br><br>Ridge trained normally and with channel-dropout augmentation. |
| 4 | Overlap | Per-electrode correlation of TX and SBP; held-out R² of SBP predicted from the same electrode's TX and from all 48 TX channels. |

---

## Results

### 1. Feature comparison, all 48 channels (SBP = log SBP unless marked)

| Analysis | TX | SBP | SBP − TX [95% CI] | raw SBP | TX + SBP | SBP residual |
|---|---|---|---|---|---|---|
| Velocity r, ridge | 0.662 | 0.613 | −0.049 [−0.070, −0.031] | 0.625 | 0.681 | 0.546 |
| Velocity r, ridge + lags | 0.663 | 0.619 | −0.044 [−0.064, −0.027] | 0.634 | 0.685 | 0.520 |
| Velocity r, FFN (3 seeds) | 0.700 | 0.673 | −0.027 [−0.047, −0.009] | 0.676 | 0.713 | 0.547 |
| 8-target accuracy, movement | 0.810 | 0.736 | −0.074 [−0.128, −0.018] | 0.766 (−0.045 [−0.098, +0.009]) | 0.843 | 0.576 |
| 8-target accuracy, delay period (preparatory) | 0.617 | 0.568 | −0.049 [−0.131, +0.033] | 0.590 | 0.639 | 0.525 |

Chance for 8-target accuracy is 0.125.

- SBP reaches about 93–96% of TX velocity r and about 91–95% of TX target accuracy.
- The gap is smallest for the nonlinear decoder (FFN) and is not significant for preparatory activity.
- TX + SBP beats TX alone for velocity: +0.019 [+0.010, +0.028] for ridge and +0.013 [+0.002, +0.025] for the FFN. So SBP carries some information TX does not.
- The SBP residual (SBP with its own electrode's TX removed) still decodes velocity at r = 0.55 and targets at 0.58. Part of that is information shared with *other* electrodes' TX, which is why adding SBP to TX helps only slightly.
- Causal lags barely help either feature (+0.001 to +0.009), as expected for already-smoothed data.

### 2. Channel count (ridge, median of 200 random subsets; velocity r / 8-target accuracy)

| Pool | n | TX | log SBP | raw SBP | SBP better in (log / raw) |
|---|---|---|---|---|---|
| all 48 | 4 | 0.256 / 0.31 | 0.273 / 0.32 | 0.283 / 0.32 | 61% / 67% of subsets (velocity) |
| all 48 | 8 | 0.380 / 0.44 | 0.396 / 0.43 | 0.409 / 0.44 | 56% / 59% |
| all 48 | 16 | 0.510 / 0.59 | 0.507 / 0.59 | 0.518 / 0.59 | 53% / 57% |
| all 48 | 32 | 0.622 / 0.73 | 0.581 / 0.70 | 0.594 / 0.71 | 20% / 26% |
| all 48 | 48 | 0.662 / 0.81 | 0.613 / 0.74 | 0.625 / 0.77 | — |
| array A | 8 | 0.396 / 0.47 | 0.405 / 0.43 | 0.410 / 0.43 | 53% / 59% |
| array A | 20 | 0.543 / 0.69 | 0.524 / 0.56 | 0.526 / 0.58 | — |
| array B | 8 | 0.369 / 0.42 | 0.411 / 0.49 | 0.423 / 0.50 | 70% / 74% |
| array B | 16 | 0.520 / 0.56 | 0.537 / 0.65 | 0.551 / 0.66 | 69% / 82% |
| array B | 28 | 0.618 / 0.66 | 0.624 / 0.78 | 0.641 / 0.81 | — |

**Greedy best-n, velocity r** (TX / log SBP / raw SBP):

| Pool | n = 4 | n = 8 | n = 16 |
|---|---|---|---|
| all 48 | 0.48 / 0.49 / 0.50 | 0.61 / 0.54 / 0.58 | 0.64 / 0.60 / 0.61 |
| array B | 0.52 / 0.46 / 0.47 | 0.59 / 0.54 / 0.51 | 0.61 / 0.58 / 0.60 |

- **With few random channels (4–16), SBP is as good as TX or slightly better.** Its disadvantage appears only when most channels are used.
- On array B, SBP is equal or better at every n, and much better for target classification.
- On array A, SBP is worse for classification (0.56 vs 0.69 at n = 20).
- With *chosen* channels, TX has a small edge (about 0.03–0.07 at n = 8), presumably because the best TX electrodes have well-isolated units.
- The per-subset spread is large (5–95% band, about ±0.1 at n = 8). Which channels you get matters more than which feature you use.
- Oddity: SBP target accuracy on all 48 channels (0.74) is *below* array B alone (0.78). With about 225 training trials and 8 classes, adding array A's weaker SBP channels costs more than it adds. This looks like a small-data effect, not a property of SBP.

### 3. Muting (ridge on all 48; velocity r; mean over 20 masks)

| Mode | f | TX | log SBP | Retained (TX / SBP) |
|---|---|---|---|---|
| zero (detected) | 0.3 | 0.600 | 0.546 | 91% / 89% |
| zero (detected) | 0.5 | 0.529 | 0.470 | 80% / 77% |
| dead (undetected) | 0.3 | 0.590 | 0.526 | 89% / 86% |
| dead (undetected) | 0.5 | 0.507 | 0.434 | 77% / 71% |
| noisy (undetected) | 0.3 | 0.543 | 0.481 | 82% / 78% |
| noisy (undetected) | 0.5 | 0.417 | 0.350 | 63% / 57% |

- For velocity, SBP loses about the same *fraction* of performance as TX under muting, a few points more.
- Undetected noisy channels hurt both most, so **bad-channel detection matters more than the choice of feature**.
- **8-target classification with undetected dead channels** (assumed floor value; see caveats): at f = 0.3, SBP drops to 0.37–0.40 vs TX 0.54, from 0.74–0.77 and 0.81 at f = 0. A flat SBP channel at its floor pushes the trial-mean input far from the training distribution. This is a warning to detect flat SBP channels online, e.g. by variance, not to assume they are harmless.
- **Channel-dropout augmentation does not help a ridge decoder** (±0.01 at every f). A linear model can't learn to ignore a channel conditionally. Phase 10 tests dropout on the LSTM, which is where it can matter.

### 4. How much of SBP is TX?

- **Correlation** between an electrode's TX and log SBP: median 0.62 (IQR 0.45–0.72); array A 0.59, array B 0.63.
- **Held-out R² of SBP from TX:**
  - from the same electrode's TX: median 0.35 (IQR 0.14–0.47);
  - from all 48 TX channels: median 0.56 (IQR 0.35–0.67).
- R² rises with the electrode's TX rate (panel h). On low-rate electrodes, SBP is mostly *not* TX.

---

## What this means for the 8 × 32 SBP system

1. **SBP alone is very likely sufficient for decoding.** On the same electrodes, SBP is at most about 5–7% behind TX with all channels. It is level or ahead with few channels, and ahead on one of the two arrays. Two things work against TX in a real system: the TX-favouring channel cut here, and raw, causal 20-ms bins, where TX counts are noisier. Neither is tested here. So the gap here is *probably* an overestimate of what an SBP-only system would see, if its SBP is comparable. That is a reasoned expectation, not a measured bound.
2. **The bimanual-specific question is not answered.** Whether SBP keeps the laterality signal and the hand-specific tuning the paper relies on can only be tested with bimanual SBP data. This is the main reason to ask for SBP from the 8 × 32 system with *both* hands' attempted movements, even one-hand-at-a-time blocks.
3. **Channel quality handling matters more than the feature.** Plan for:
   - online detection of flat or noisy SBP channels: a flat channel hurt SBP classification badly here;
   - re-zeroing detected channels;
   - for the LSTM, the channel-dropout training tested in Phase 10.
4. **Match the SBP processing before comparing.** Ask the system's team for:
   - the exact SBP definition (filter, band, rectification, bin);
   - whether they also log TX;
   - raw-rate SBP, i.e. not acausally smoothed.
   If both features are available, TX + SBP was the best combination here (+0.01–0.03).
