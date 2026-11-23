# Reader's guide and paper-check list

**Who this is for:** you, reading the paper (Deo et al. 2024, *Sci Rep* 14:1598) alongside this replication.

There are three parts:

1. **Things to check in the paper yourself.** Claude read the paper through a summarizer, not the PDF, so some details are uncertain. Each entry says what to look for, where it probably is, what we assumed, and why it matters.
2. **The experiment paradigm**, in plain terms.
3. **dPCA**, the time window and cross-validation, in plain terms.

The numbered register in [REPLICATION.md](../REPLICATION.md) is the full list of decisions and assumptions; register numbers (#) refer to it.

---

## 1. Things to check in the paper yourself

Tick each box once checked, and note what the paper actually says. Anything that contradicts our choice should be raised with Claude so the analysis can be redone.

### High priority: could change headline numbers

- [ ] **Target numbers (#17).** All "paper" values in `REPLICATION.md` came from summaries. Check them against the text and figures:
  - Fig 2b suppression: RH 12%, LH 34%.
  - Fig 2c correlations: RH 0.87 (x) / 0.84 (y); LH 0.42 (x) / 0.45 (y).
  - dPCA: laterality 39%, direction 30%.
  - Supp Fig 1d: LH 48% weaker.
  - Fig 4d: raw-trained RNN r = 0.90 (RH) / 0.83 (LH).
  - Supp Fig 4: single-bin fSNR 2.03.

  *Where:* Results sections for Figs 2 and 4, the figure legends, and the Supplementary PDF.
- [ ] **Block grouping for Fig 2b-c (#10).** The paper says the data were "grouped in pairs of blocks to ~40 repetitions per trial type", with n = 25.
  - **Our main rule:** consecutive pairs of all open-loop blocks, including the unimanual-only ones. This gives exactly 25 sets, but 3 of them have almost no bimanual trials, so 22 are usable.
  - **What to look for:** any sentence on how blocks were paired, whether unimanual-only blocks were used, and whether n = 25 counts sets or something else.
  - *Where:* Methods, "Tuning strength and correlation"; the Fig 2 legend.
- [ ] **Five-fold cvOLS (#23).** We found that the literal five-fold estimate becomes unstable when a set has about 25 bimanual trials, so we switched to split-half.
  - **What to look for:** exactly how the paper's cvOLS works. Does each fold's "test" coefficient come from that fold alone? Is a reference (e.g. Willett et al. 2020) followed?
  - *Where:* Methods, "Encoding model"; the cited references.
- [ ] **Encoding-model regressors (#21).** Eq. 2 defines d as target position minus cursor position.
  - **Our choice:** the unit direction at the go cue, held constant over 300–700 ms, because target distances vary from 30 to 750 units.
  - **What to look for:** is d per time bin, at the go cue, or normalized?
  - *Where:* Methods, Eq. 2.
- [ ] **dPCA time window (#24).** We report both windows:
  - 300–700 ms gives laterality 41% / direction 33%;
  - 0–1000 ms gives 39% / 29%, which matches the paper.

  **What to look for:** the window used for dPCA, and whether rates were z-scored.
  - *Where:* Methods, "Demixed PCA"; the Fig 2d legend.

### Medium priority: needed for upcoming phases

- [ ] **Fig 3 trial subset (#12).** *(We now use the 491 OL unimanual trials; check what the paper's 482 are.)* The paper uses 482 trials from day 1883, while the file has 951 (open-loop, recalibration and closed-loop). Which blocks or trial types make up the 482?
  - *Where:* Methods, "Offline single-bin decoding"; the Fig 3 legend.
- [ ] **Fig 3 decoder targets (#11).** *(We now use the unit direction; recorded cursor velocity gives much lower r. Check which the paper used, and what its r values look like in Fig 3b.)* What were ridge and the FFN trained to output: cursor velocity, the direction to the target, or something else? What did the idle hand's target look like?
  - *Where:* Methods, "Single-bin decoders".
- [ ] **"Saturated" velocity targets for the RNN (#13).**
  - **Our reading:** the minimum-jerk speed bell clipped at half its peak.
  - **Alternative:** rise, then hold until the trial ends.

  Look for the exact definition, a formula, or the description of Fig 4b.
  - *Where:* Methods, "RNN decoder" / "Velocity targets"; the Fig 4b legend.
- [ ] **Augmentation clip length (#14).** Is it "0.2–0.8 × snippet length" or "200–800 ms"? We now use 2–8 s (the Methods reading). Fig 4d supports it: the paper's augmented r is 0.82 / 0.82 vs raw 0.90 / 0.83, a drop like our 2–8-s models' and far smaller than with 200–800-ms clips ([figure comparison](figure_comparison.md#fig-4d-rnn-offline-decoding)).
  - *Where:* Methods, "Data augmentation"; the Fig 4c legend.
- [ ] **Augmentation velocity scaling (#15).** When a clip is stretched or compressed in time, is the velocity target rescaled? We assume not.
- [ ] **RNN hyperparameters.** Any values for learning rate, batch size, number of steps, input noise, mean drift or L2 weight, even in a table or supplement.

### Extensions (Phases 8–11)

- [ ] **Did the online decoder use SBP as well as threshold crossings? NEW (Phase 11).** The Dryad data has threshold crossings only. Check the Methods ("Neural signal processing" or similar) for spike-band power, and if it was used, how it was computed (band, bin). If the online RNN used SBP too, our offline crossings-only results may understate it. Phase 11's T5 comparison suggests SBP alone is roughly as informative ([ext_sbp_notes.md](ext_sbp_notes.md)).

- [ ] **Which channels belong to which array (#36). NEW.** The one-array analysis assumes channels 1–96 are one array and 97–192 the other. The data supports this only moderately ([check figure](../results/figures/ext_arrays_check.png)). Check the paper (Supp Fig 1b array layout, Methods) or other T5 datasets for:
  - the channel → array mapping;
  - which array is medial and which lateral.

### Low priority: details and sanity checks

- [x] **Supp Fig 1 counts.** Read on 2026-11-23 ([figure comparison](figure_comparison.md#supp-fig-1-single-electrode-tuning)). Paper: tuned uniR 90 / biR 97 / uniL 60 / biL 52; contexts 0–4: 82 / 16 / 33 / 27 / 34. We find fewer, most for biL. Still open: the paper's ANOVA threshold, window and session. Ours:
  - tuned electrodes: uniR 83, uniL 50, biR 87, biL 28;
  - electrodes tuned to 0/1/2/3/4 contexts: 88 / 25 / 29 / 35 / 15.
- [ ] **Supp Fig 1 ANOVA window and quadrant definition (#6, #18).** We use 300–700 ms and axis-aligned quadrants of the cursor→target direction.
- [x] **Fig 5 angular error (#8).** Answered by the data (2026-11-23): the paper's values match the angle of the net cursor displacement over 300–500 ms, median over trials ([figure comparison](figure_comparison.md#fig-5-online-two-cursor-control)). Original question: How is the "movement direction" computed: from cursor displacement or from decoded velocity? Is the target direction fixed at the go cue?
- [x] **Fig 5a bimanual vs unimanual TTA (#30).** Read on 2026-11-23: the paper's Fig 5a itself shows bimanual ≈ 4.4–5.9 s vs unimanual ≈ 2–3 s on good sessions (about 2×), so it agrees with our logs; the text understates it. In our re-scoring, bimanual TTA is about 2× unimanual on every day (e.g. 1776: 5.7 s vs 2.2–2.6 s). The paper calls bimanual "only slightly longer". Read the Fig 5a values off the plot and check how TTA is defined (with or without the dwell? time to the first target?).
- [ ] **Fig 5 time to acquisition (#7).** Does it include the 500-ms dwell? We use the task's logged trial length, which does.
- [x] **Fig 5a day list.** Answered by the data (2026-11-23): 1776, 1778, 1790, 1792 plus the simultaneous blocks of 1881 and 1883 reproduce all six of the paper's sessions; Supp Table 2 was right.
  - Supplementary Table 2 lists 1881 and 1883 under Fig 5A.
  - The Methods summary listed only 1752, 1771, 1776, 1778, 1790 and 1792.

  Which is right?

---

## Claude's PMC check (step 6.2, 2026-11-21)

Claude re-read the full text on PMC (PMC10796685): Results, figure legends and Methods. Items still marked ❓ need your eyes, mostly on the figures themselves.

| # | Item | What the paper says | Verdict |
|---|---|---|---|
| 17 | Target numbers | Fig 2b "12% suppression" (RH), "suppressed by 34%" (LH); Fig 2c RH "0.87 and 0.84", LH "0.42 and 0.45"; dPCA laterality "39%", direction "30%"; raw RNN "0.9 and 0.83" | ✅ confirmed |
| 10 | Fig 2 block sets | "grouped consecutive blocks together in pairs to reach around 40 repetitions, at least, per trial type"; "sample size of 25" | ✅ matches our main rule (consecutive pairs, n = 25) |
| 9/23 | cvOLS | "five-fold cross-validated ordinary least-squares regression"; no detail on how the folds are used | ❓ our split-half deviation stands (k-fold is unstable on small sets) |
| 21 | Eq. 2 regressors | per time step, d = target position − cursor position, not normalized | ⚠️ confirmed DEVIATION: we use the unit direction at the go cue (ratios and correlations are scale-free) |
| 24 | dPCA window / scaling | not stated; dPCA "in unimanual movement data only" | ❓ not stated; our 0–1000 ms window reproduces 39 / 30 |
| 12 | Fig 3 trials | "482 total trials", unimanual, day 1883 (open-loop not stated) | ✅ about our 491 OL unimanual trials |
| 11 | Fig 3 targets | "velocity target for each time step as the unit vector pointing from the cursor to the target" | ✅ our main choice (we fix the vector at the go cue; the paper updates it per step) |
| 13 | "Saturated" targets | "moving with maximum velocity from movement onset up until they are very near the target"; 200-ms reaction time; no formula | ❓ wording leans toward a long plateau. Both of our readings were tested; "clip" did better offline |
| 14 | Clip length | Methods: "ranging between 0.2 and 0.8 times the 10 s total snippet length" (2–8 s). Results: "ranging between 200 and 800 ms". **The paper contradicts itself.** | ⚠️ both readings tested in Phase 4 |
| 15 | Velocity scaling under dilation | "Linear interpolation was used to either compress or stretch both the input and output clips"; no amplitude rescaling | ✅ matches |
| 4d | Augmented offline r | not given; only "slightly poorer decoding performance on offline held-out test data and noisier output velocities" | ✅ qualitative only |
| 7/30 | TTA | "the amount of time after the go cue in which the targets were successfully acquired"; dwell inclusion not stated; "bimanual ... only slightly longer than ... unimanual on average"; "average TTA for right- and left-hand trials was similar" | ❓ our bimanual TTA is about 2× unimanual; check the Fig 5a bars yourself |
| 8 | Angular error | "average difference between movement direction within the 300–500 ms window after the go cue"; reference direction not stated | ❓ |
| 29 | Fig 5a days | "6 days (trial days 1752, 1771, 1776, 1778, 1790, 1792)" | ⚠️ the Dryad files for 1752 / 1771 contain only open-loop blocks, so those days can't be re-scored |
| — | RNN hyperparameters | random search over 100 RNNs; no values | ✅ not stated (our search is documented) |
| — | Fig 4 training days | RNNs for Fig 4 trained on days 1867 and 1869 only | ⚠️ we trained on 1869 OL only (block split) |

---

## 2. The experiment paradigm

**Participant and recording.**
- Participant T5 has paralysis from a cervical spinal cord injury.
- He has two 96-electrode Utah arrays (192 electrodes in total) in the hand area of the LEFT motor cortex, the dominant hemisphere.
- On each electrode, voltage dips below −3.5 × that electrode's RMS noise are counted as "spikes" (threshold crossings) in 20-ms bins. The files contain these counts (`tx`) for every bin.

**The screen.** Two cursors: a white "R" for the right hand and a purple "L" for the left. Each has a target circle in its own colour. T5 imagines gripping a joystick in each hand and pushing it.

**Trial types.** Each trial is one of three, chosen at random:
- **Bimanual:** both cursors move.
- **Unimanual right:** only the right cursor moves.
- **Unimanual left:** only the left cursor moves.

**Open-loop (OL) blocks, the "instructed delay" task.**

1. **Delay (1–2 s, random).** Lines connect each cursor to its target, and T5 prepares the movement.
2. **Go cue (a beep).** The lines disappear.
3. **Movement (1–2 s).** The cursors glide to their targets on their own, along smooth minimum-jerk paths, and arrive together. T5 attempts the movement in sync.
4. **End beep.** T5 stops and prepares for the next trial.

The computer moves the cursors, so the neural activity is a clean response to known intended movements. That makes this the data used to study tuning (Figs 1–2) and to train decoders.

Targets are at random positions in a ±300-unit workspace, and each trial starts where the previous target was. Movement directions therefore cover all angles.

**Closed-loop (CL) blocks, real brain control.**
- The cursors are driven by a decoder (the RNN or a linear decoder) that reads the neural activity in real time.
- **Success:** both cursors sit inside their targets at the same time for 500 ms without a break. Leaving a target restarts the 500-ms count.
- **Timeout:** 10 s, which counts as a failure.

**Closed-loop recalibration (CLR) blocks.** Assisted closed-loop blocks used to retune the decoder before evaluation:
- **"Lock mode"** freezes the idle cursor on unimanual trials.
- **Push assistance** nudges the cursor toward the target.
- **Error assistance** damps sideways motion.

**Days and what they test** (see `bimanual/sessions.py` and `docs/data_inventory.md`):

| Days | Purpose |
|---|---|
| 1750 | Open-loop only; tuning examples (Fig 1, Supp Fig 1) |
| 1752, 1771 | Open-loop pilot data used to first train the RNN |
| 1776, 1778, 1790, 1792 | Open-loop + RNN closed-loop evaluation (Fig 2, Fig 5a) |
| 1853, 1855 | RNN vs linear decoder at several gains (Fig 5c) |
| 1867, 1869 | RNN trained with vs without augmentation (Fig 4) |
| 1881, 1883 | Sequential (one hand at a time) vs simultaneous strategy (Fig 5b); also Figs 2 and 3 |

**What `durationBinnedMove` means.**
- On open-loop trials: the length of the move period.
- On closed-loop trials: the time to acquisition. The maximum is 499 bins, the 10-s timeout.

---

## 3. dPCA, the time window, and cross-validation

### The question

Each of the 192 electrodes' firing rates varies with **which hand** moves (laterality), **where** it moves (4 direction quadrants), and **time** within the movement.

The paper asks: how much of the population activity is explained by laterality versus direction?
- If laterality dominates, the cortex carries a strong "which hand" signal.
- A decoder can use that signal to tell the hands apart.
- This is the paper's explanation for why nonlinear decoders work (Fig 3).

### Splitting the variance (marginalization)

Average the trials into 8 conditions (2 hands × 4 directions), each a time course per electrode. Then split that array into four non-overlapping parts that add back up to the data:

| Part | Captures | Computed as |
|---|---|---|
| Time | What all conditions share, e.g. the ramp-up after go | Average over all conditions |
| Laterality | How right-hand trials differ from left-hand trials | Average over directions, minus time |
| Direction | How the 4 directions differ | Average over hands, minus time |
| Interaction | Direction effects that differ between hands | Whatever is left |

The parts don't overlap, so their variances add up. "Laterality 39%" means 39% of the summed variance is in the laterality part.

Full dPCA also finds a few neural directions that best capture each part, for plotting. The percentages come from the split alone, so we compute them directly (`bimanual/dpca_analysis.py`).

### Why the time window matters

The time part only captures change within the window.

| Window | Time explains | Laterality / direction | |
|---|---|---|---|
| 300–700 ms after go | 3% (activity is roughly steady mid-movement) | 41% / 33% | |
| 0–1000 ms | 12% (includes the rise from rest after the go cue) | 39% / 29% | ≈ paper |

The conclusion (laterality > direction, every day) holds in both windows; only the exact percentages shift. The paper doesn't state its window, which is why it's on the check list.

### Why cross-validation decides the answer

Condition averages still contain leftover noise. Noise always ADDS variance, since squared wobbles are positive, so a naive estimate inflates every part.

The inflation is uneven:
- Laterality comes from just 2 big averaged groups, so its noise largely averages out.
- Direction and interaction come from smaller groups, so they are inflated more.

**The paper's fix (two-split cross-validation):**

1. Split the trials into two random halves.
2. Do the full split on each half separately.
3. Multiply the two halves' parts together element by element, and sum.

The true signal is the same in both halves, so it survives. The noise is independent between halves, so its products cancel to 0 on average. We repeat this over 100 random splits.

| | Laterality | Direction |
|---|---|---|
| Without CV | 28% | 35% (wrong order) |
| With CV | 41% | 33% |

Without cross-validation, the paper's main claim would come out backwards purely because of noise. Our simulated tests confirm it: on pure noise the CV estimate is 0 on average, while the naive estimate shows a large fake laterality effect.

**The same idea in Fig 2b-c.** Tuning strengths and correlations are also computed from independent halves of the data (cvOLS):
- a noisy tuning vector looks stronger than it really is;
- noisy correlations shrink toward 0.

Comparing estimates from independent data removes both biases.
