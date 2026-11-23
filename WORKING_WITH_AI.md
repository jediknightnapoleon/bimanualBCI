# Working with AI on a science project

Most of the code in this branch was written by an AI coding assistant (Claude Code) under the direction of one researcher. Every commit it co-wrote carries a `Co-Authored-By` line.

This page collects what kept the science understandable and defensible while the AI did most of the typing. Each tip comes with an example from this project, so you can check it against the repository.

The short version:
- **Decide what the result should be before the code runs.**
- **Read the code that produces numbers.**
- **Make every claim point to evidence you can open.**
- **Look at the data.**
- **Explain each result in your own words before you present it.**

## Before anything runs

**Write down the claim, the expected result and what would count as failure.**
- The numbered register in [REPLICATION.md](REPLICATION.md) (#1–46) records every decision, deviation and assumption, with the reason, before or as the results came in.
- The dPCA window (#24) was fixed at 300–700 ms in advance. A longer window turned out to match the paper better, but because both are reported, nobody can say the window was chosen to fit.

**Pin down the facts about the data first, before any analysis.**
- [docs/data_inventory.md](docs/data_inventory.md) and `bimanual/sessions.py` list every session and block label, checked against the files.
- The go-cue index in the `.mat` files is 1-based, and it is converted to 0-based in exactly one place (`bimanual/io.py`). Off-by-one errors in trial alignment silently shift every result.

## While the AI writes code

**Read closely the code that decides a number; skim the rest.**

| Read closely | Skim |
|---|---|
| Data loading and trial alignment | Plotting |
| Train / validation / test splits (leakage) | File handling, caching, job queues |
| Targets and labels | Command-line options |
| Metrics and statistics (what is averaged over what, and what is resampled) | Formatting of tables and documents |

**Work in small steps, with a checkpoint at each.**
- This project ran in phases. Each phase ended with a briefing ([docs/briefings.md](docs/briefings.md)), a commit, and a log entry in [PROGRESS.md](PROGRESS.md).
- Large changes nobody has reviewed are where misunderstandings pile up.

**Make assumptions visible in the code.**
- Comments marked `# ASSUMPTION:` and `# DEVIATION:` sit at the exact line where a choice was made, and each links to a register entry.
- `grep -rn "ASSUMPTION\|DEVIATION" bimanual scripts` lists every one of them.

## Checking the results

**Use checks with a known answer.** Whenever possible, compare against something whose answer you already know.
- **The authors' own code:** our port of their tuning analysis matches their MATLAB code run in Octave to 3 × 10⁻¹⁴ (`pixi run compare-reference`).
- **Synthetic data with a planted answer:** for example, channel selection must recover the planted channels and skip a duplicate (`tests/test_select.py`).
- **Re-running saved models:** a re-run must reproduce the logged test r within 0.002. The muting analysis and the replays both assert this.
- **Assertions on what must never happen:** for example, channel selection never sees the test block.
- **Controls:** shuffled labels or random channel sets should give chance-level or baseline results.

**Ask for evidence, not reassurance.**
- Every claim should point to a file and line, or a saved number in `results/`. "The tests pass" is not evidence about the science.
- Useful questions to ask:
  - "How could this be wrong?"
  - "What else would explain this result?"
  - "Which number would change if this assumption were false?"
- A fresh AI session, given only the code and the claim, makes a good critic because it doesn't share the first session's assumptions.

**Redraw your results in the paper's exact format.** This caught two definition mismatches that summary tables had hidden:
- **Time to target (#29, #30):** the paper plots the *median*, with failures counted as 10 s. We had used the *mean of successful trials*. With the paper's statistic, all 18 points match.
- **Angular error (#8):** the paper uses the angle of the net displacement. We had averaged noisy per-bin angles, which inflated the error by 10–25°.

Both are in [docs/figure_comparison.md](docs/figure_comparison.md).

**Look at the data, not only at summary numbers.** [docs/html/replays.html](docs/html/replays.html) replays the logged cursors and our decoders' outputs next to the neural activity. A result you have watched happen is much easier to explain and defend.

**Reproduce one key number yourself.** In a blank notebook, recompute one headline value from the raw data without the project's helper functions. For example, take the ridge decoding r on day 1883. If you can't, you don't yet understand the pipeline.

**Check the paper yourself.** AI summaries of papers can be wrong or partial.
- In this project, all of the paper's numbers were re-checked against the full text ([docs/paper_check.md](docs/paper_check.md)).
- Details read from a web summary, such as some figures for the Willsey et al. paper, are marked as unverified until read in the original.

## Understanding it well enough to present

**Teach it back after every phase.** Explain the result aloud in two minutes, without notes: what was computed, from which data, compared with what, and what result would have contradicted it. Wherever you hesitate, there is a question to ask before someone else does.

**Keep a one-page sheet of claims, evidence and limits.**

| Claim | Evidence (file, number) | What would contradict it | Limits |
|---|---|---|---|
| The left hand's tuning is suppressed more than the right hand's in bimanual movement | Fig 2b; 41% vs 19%, p = 6 × 10⁻⁶; holds in every estimator variant | RH ≥ LH in some block-pairing scheme | The magnitudes depend on the estimator (#23) |
| … | … | … | … |

**Know where you differ from the paper, and why.**
- [docs/figure_comparison.md](docs/figure_comparison.md) lists 12 differences, each with its tested cause or marked as open.
- Open items, such as our FFN being jitterier at rest than the paper's, are stated plainly rather than explained away.

**Know what the data cannot show.**
- Nothing offline can test live closed-loop control.
- Offline r does not predict online success. In the logs, raw and fractured RNNs are within 0.03 r offline but more than 50 percentage points apart online.

**Know the statistics you are using.**
- Resample trials, not time bins: neighbouring bins are not independent.
- Report confidence intervals, and say when several tests were run.
- Keep separate the numbers that come from cross-validated data and those that don't. Without cross-validation, the dPCA variance shares even reverse order.

## Questions to expect, and where the answers are

| Question | Answer |
|---|---|
| Did you just run the authors' code? | No. The pipeline is an independent Python implementation written from the Methods. The authors' MATLAB tuning code was used only as a cross-check (3 × 10⁻¹⁴). |
| How do you know there is no train/test leakage? | Splits are by trial or block, and laterality removal is fitted on training folds only ([methods_notes.md](docs/methods_notes.md)). Channel selection never sees the test block (asserted). |
| Why do your numbers differ from the paper's? | [figure_comparison.md](docs/figure_comparison.md): 12 differences, each with its cause or marked as open. |
| How much did the AI do, and how did you check it? | This page, the tests (`pixi run test`), the register, and the known-answer checks above. |
| Would this work online, or on another implant? | It can't be tested offline. See the feasibility phases and their caveats in [REPLICATION.md](REPLICATION.md) and [docs/ext_sbp_notes.md](docs/ext_sbp_notes.md). |

## Warning signs

- **A result much better than the paper's.** Look for leakage or a different metric before celebrating. Our raw RNN scores 0.06 higher than the paper's, which is probably because we train and test on the same day (#28); that is not tested.
- **A number nobody can trace to a file.**
- **A test changed so that it passes, rather than the code fixed.**
- **Code that silently falls back to something else.** For example, a missing file replaced by a default, or a failed fit replaced by zeros.
- **An explanation that sounds right but was never tested.** Mark it "likely, untested" until it is tested.
- **Problems found only by running the code.** These were found by testing, not by reading:
  - The augmentation sampler never placed a clip at the edge of a segment, so the decoder never saw rest periods. It is now fixed and covered by a test.
  - 5-fold cross-validated regression blew up on small sets (#23).
