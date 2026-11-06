# Re-test of 082's residual-CNN correction (pre-registered 2026-10-17 ~00:15 KST, before any new teacher, CNN or score exists)

## Question
- 082 = 076 + c on gated test rows (noise ≥ 12 and not failed), where c = 0.5468 × mean(aux-CNN seed 2026101501, seed 2026201501).
- Its dev evidence comes from ONE official partition (original, 5 folds), with about 81% of the gain from 10 images.
- **Does the same frozen recipe, rebuilt from scratch with fresh nested teachers, improve 076 on the other two official partitions (20261031, 20261101)?**
- Secondary: does it improve on the holdout-100 rows when trained on all 400 dev rows?
- Limitation stated up front: the two partitions re-use the same 400 images.
  - They test robustness to fold assignment and retraining, not generalization to new images.
  - The holdout-100 rows are fresher for this correction: never used to develop it, though inspected by other earlier studies.

## Frozen recipe (from round4 / FINAL500, unchanged)
- **Teacher T.** The 073 non-021 branch exactly as `round4-full073/full_branch.py`:
  - TabPFN v3.5, 3 slots × 4 U-Net variants × 3 seeds (100·s + fold); nb columns on noise < 12, full columns on noise ≥ 12.
  - RidgeCV; logRidge α 2, four-variant geometric mean; source-restricted degraded copies only in the noise ≥ 12 branch.
  - GMM-Ridge; grain-set NN with seeds 731503/731509/731527.
  - Weights /0.84; segmentation-failed rows get the teacher-training mean.
  - The code is copied unchanged except the output directory.
- **Teacher graph per outer fold** (round4 structure). Inside the outer training set T:
  - B = 1/3 of T (seeded); A = the rest; A1/A2 = seeded halves of A.
  - outer0: A → B; outer1: T − A1 → A1; outer2: T − A2 → A2. Every training row's residual target comes from a teacher that never saw its label.
  - full: T → V supplies the 6 context inputs on V (round4 used the strict-073 outer predictions; those exist only for the original partition).
  - Seeds: `RandomState(20261017 + 100·p + k)`, with p = 1 (20261031), 2 (20261101), 3 (holdout).
- **CNN.** round4 `cnn_fitter` "aux" arm, unchanged.
  - 68,788-parameter native-256 multiscale moment model; residual target y − T normalised by training label SD; 6 context inputs.
  - Auxiliary hardness head with weight 0.2; 40 epochs, AdamW 0.002, weight decay 0.01, cosine, batch 32, D4, final-five-state average, four-view inference.
  - Two seeds 2026101501 + 100·k + {0, 100000}; correction = equal mean of the two seeds.
- **Application.** Weights frozen from 082: clean regime 0, noisy regime 0.5468, on noise ≥ 12 and not failed rows. Failed rows get no correction.
  - No coefficient is refit anywhere in this study.
- **Baselines.**
  - New partitions: the reproduced 076 dev OOF (`stepchange-20261016/repro/components/c14b-orig.npz`, rows 1 and 2).
  - Holdout: the full T(400→100) teacher prediction; there is no 076 holdout prediction. Earlier AI analysis found the correction's gain on 073 and on 076 nearly identical (+0.231 vs +0.240).

## Statistics
- Primary: dMSE = mean over gated rows of [(076 + c − y)² − (076 − y)²].
  - Per-image values are averaged over the two new partitions first (the partitions are correlated, not independent).
  - Image bootstrap with 10,000 draws.
- Also reported: each partition separately, fold-wise dMSE, removal of the top-5/top-10 helped images, the stretch decomposition, and the original partition recomputed with the same fixed 0.5468 for reference.
- The holdout-100 is reported separately (about 35 gated rows; low power).

## Decision rule (fixed now)
- **Supports 082** if both hold:
  - combined new-partition dMSE < 0 with P(better) ≥ 0.80;
  - neither partition has dMSE > 0 with P(worse) ≥ 0.80.
- **Contradicts 082** if either holds:
  - combined dMSE > 0 with P(worse) ≥ 0.80;
  - combined P(better) < 0.80 AND holdout-100 dMSE > 0 with P(worse) ≥ 0.80.
  - Recommendation: revert the final file to 076. That is the user's action; this study changes nothing.
- **Inconclusive** otherwise. Recommendation: keep 082, because the original partition and the Public reading favour it and there would be no contrary evidence. The uncertainty is stated.

## Budget
- 44 teacher fits (10 new folds × 4, plus holdout × 4), run in 3 parallel CPU processes with 2 threads each, plus 22 CNN fits.
- Stop rule: about 3.5 h of wall clock. If it is reached, report only the completed partitions and label the result partial.
- No submissions, no final-selection change, no writes outside this directory.

## Amendment A1 (2026-10-17 00:40, before any teacher was complete or any CNN/score existed)
- With 3 parallel CPU workers, TabPFN ran at about 17 s per fit (about 22 min per teacher, more than 4 h in total, over budget).
- TabPFN in the teachers now runs on MPS (`TAB_DEVICE=mps`). This is an execution-device change only: same checkpoint, seeds, data and recipe.
  - Earlier check in this project: the same seed on CPU vs MPS gave 8.025 vs 8.024.
- Members already finished on CPU (their cached `tab-*.npz` files) are kept, and their file times record which ones they are.
- Nothing else changes.
