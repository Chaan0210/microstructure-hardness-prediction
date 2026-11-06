# Re-test of 082's residual-CNN correction — result (2026-10-17 01:50 KST)

**Pre-registered verdict: SUPPORTS 082. Keep 082 as the final file, with 076 as the rollback.**
- The supporting evidence comes from new fold assignments of the same 400 dev images.
- On the 100 holdout images, which were never used to develop this correction, it did not help (35 gated rows; low power).
- Read the result as "the correction is real on dev-like rows, smaller than first measured, and unproven on fresh rows". Expected private effect: small.

## What was run (protocol in PLAN.md, frozen before any result; amendment A1 changed only the TabPFN device, CPU → MPS, for runtime)
- **Teachers.** 34 fresh nested teachers with round4's `full_branch.fit`, imported read-only with the output directory redirected; recipe unchanged.
  - 3 per outer fold, for the 10 folds of partitions 20261031 and 20261101.
  - 4 for the holdout: 400 dev rows → 100 holdout rows.
- **CNNs.** 22 aux residual CNNs with round4's `cnn_data` / `fit_one` unchanged: two seeds per fold, 40 epochs, context inputs as round4.
- **Correction.** Exactly as deployed in 082: 0.5468 × mean of the two seeds, on rows with noise ≥ 12 and not failed. No coefficient was refit.
- **Runtime.** About 1 h 50 min, within the 3.5 h budget.

## Results
Gated rows; dMSE = (076 + c) minus 076; negative = better; 10,000-draw image bootstrap; `results.json`.

| evaluation set | gated dMSE ± SE | P(better) | 95% CI | RMSE 076 → 076+c (all rows) | folds better | without top 10 helped |
|---|---:|---:|---|---|---|---:|
| original partition (reference; the correction was developed here) | −12.7 ± 3.8 | 1.00 | [−20.1, −5.6] | 9.508 → 9.299 | 5/5 | −3.7 |
| **20261031** | **−7.1 ± 3.1** | **0.99** | [−13.2, −1.2] | 9.537 → 9.421 | 5/5 | −0.5 |
| **20261101** | **−4.3 ± 3.3** | **0.91** | [−10.8, +2.0] | 9.593 → 9.523 | 4/5 | +2.7 |
| **two new partitions, combined** (per-image average) | **−5.7 ± 3.0** | **0.97** | [−11.7, +0.1] | — | — | +0.8 |
| holdout-100 (teacher 400→100 as base) | +6.2 ± 7.0 | 0.18 | [−7.2, +19.9] | 9.813 → 9.923 | — | +22.6 |

- **Not a stretch effect.** The stretch coefficient is about 0.02 in every partition; the non-stretch share of the gain is 0.10 (20261031) and 0.05 (20261101).
- **Concentration.** The top 10 images carry 94% (20261031) and 158% (20261101) of the net gain.
- **Alignment of the correction with the base error** (gated rows): +0.26 and +0.18 on the new partitions, **−0.02** on the holdout (14 improved, 21 worse). The two seeds agree well on the holdout (r 0.93), so it is not seed noise.
- **Teacher component.** The grain-set component of the teacher sometimes blows up (RMSE up to 26 here), exactly as in round4's own teachers (up to 77). This is part of the frozen recipe.

## Decision rule applied
- Combined new-partition dMSE < 0 with P = 0.97 ≥ 0.80, and neither partition worse with P ≥ 0.80, so the rule gives **SUPPORTS 082**.
- The holdout clause only applies when the combined P(better) is below 0.80; it is 0.97 here. The holdout's P(worse) of 0.82 therefore does not change the verdict.

## Honest reading (not part of the rule)
- The gain shrinks as the evaluation moves away from the data the method was developed on:
  - +0.21 RMSE on the original partition, where round3/round4 chose the architecture, arm, noise gate and weight
  - +0.12 and +0.07 on re-partitions of the same images
  - about −0.11 on 100 fresh images (35 gated rows; weak evidence)
- This is the signature of selection bias plus a modest real effect on dev-like rows.
- Fresh-row evidence (Public 300 rows: −0.030 RMSE; holdout: +0.11 RMSE) roughly cancels.
- So the private effect of 082 vs 076 is expected to be small, probably between −0.03 and +0.02 RMSE. Neither file is clearly better. Keeping 082 follows the pre-registered rule; switching to 076 would rest on a post-hoc reading of 35 rows.

## Reproduction (from the project root)
```
.venv/bin/python -B experiments/04-residual-cnn/20261017-rescnn-retest/teachers.py graph
env TAB_DEVICE=mps experiments/frameworks/tabpfn-v2/.venv/bin/python -B -u experiments/04-residual-cnn/20261017-rescnn-retest/teachers.py run <w> 3     # w = 0,1,2
experiments/frameworks/tabpfn-v2/.venv/bin/python -B -u experiments/04-residual-cnn/20261017-rescnn-retest/cnn_retest.py 0 1
.venv/bin/python experiments/04-residual-cnn/20261017-rescnn-retest/retest_eval.py
```

## Integrity
- All writes are in this directory. Round4 was imported with bytecode writing disabled; its `__pycache__` and `teachers/` are unchanged.
- After the run:
  - the 076 chain rebuilds byte-identically;
  - 082 rebuilds byte-identically (`0fa34cec…`);
  - 17,246 protected files unchanged; no file modified or deleted.
- No submission and no final-selection change.
