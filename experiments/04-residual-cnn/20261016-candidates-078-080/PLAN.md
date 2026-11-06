# 078 Public update and bounded residual-learning experiments

Started 2026-10-16 after the user reported 078 Public RMSE 9.0298958718.
All outputs stay in this new directory. Preserve 061, 073, every numbered
submission, other-agent code, and all previous experiments. One CPU training
process, two threads; no new packages, paid resources, external data, test
fitting, uploads, or Public-derived numerical coefficient selection.

## Attribution and fixed composition

Compare the frozen 076, 078, and 080 CSVs and actual generation sources.
078 modifies only low-noise partners, whereas 080 adds the existing CNN only
on high-noise rows. Verify their supports are disjoint and create the fixed
ablation 076 + (080 - 078). Keep the original CNN coefficient unchanged.
Validate IDs, fallback rows, finite output, source hashes, and matched OOF
transport onto recorded076. Distinguish historical076 OOF from strict073.
The new 078 score supersedes the earlier uncertainty about CNN's Public
marginal effect; preserve that earlier audit as a dated record.

## Two pre-locked new training hypotheses

H1 `high_loss`: the deployed correction only acts on noise>=12, but the
current CNN spends most training examples predicting low-noise residuals.
Replace the main loss by high-noise residual squared error, normalized by
the *training* high-noise fraction. Keep auxiliary original-hardness MSE
on every image, weighted .2, so low-noise images still train the encoder.
No oversampling, discarded labels, robust loss, new encoder or longer fit.

H2 `no_teacher_context`: supervised teacher predictions change with training
sample size (the preceding audit measured this). Zero exactly the first two
standardized context columns (T and Tab-minus-Ridge), retaining the four
physical/image context fields, all-image residual loss and auxiliary loss.
This isolates teacher conditioning from the auxiliary objective; it was not
isolated by the earlier image-only residual arm.

Reuse immutable full073 nested teachers and original400 five outer folds.
For every arm, train A213 and select the coefficient on B107; refit on T320
using honest inner OOF residual targets; evaluate on V80. Use seed
2026101501+100*fold and exactly the existing 40 epochs/AdamW/D4/averaging.
Fit both hypotheses on all five folds before reading comparative scores.
No outer-label early stopping or checkpoint choice.

Deployment gate is pre-fixed at noise>=12; low-noise correction is zero.
For an interpretable matched control, calibrate the old cached auxiliary
CNN with the same single bounded OLS coefficient on eligible high-noise B
rows: clip(dot(r,y-P073)/dot(r,r),0,1). No intercept or hyperparameter grid.
Report its original two-regime calibration separately to separate weighting
effects from new training. All experiments preserve failed-row fallback.

## Advancement, uncertainty and limits

Complete 20 new fits (two arms, five folds, selector plus refit), costing
roughly 10-20 CPU minutes. Stop an individual fit only for failure or grossly
excessive runtime. Repeat a promising arm with seed offset100000 only if
its gain over the matched old single seed is >=.02 RMSE and it improves at
least three folds. Predefined equal averaging combines the two seeds; refit
its coefficient inside B. A pilot-selected arm's confirmation remains
conditional because the same 400 labels have already been explored.

Report all arms, pooled/fold RMSE, calibration coefficients, paired ID
bootstrap intervals, simultaneous bands across these new comparisons,
residual correlation, high/low subsets, and top-error contribution. Retain
unfavorable results. Do not promote a new neural recipe merely for beating
073 if it loses to the existing CNN. Full500 training/inference is justified
only by a reproducible incremental gain; the fixed076+oldCNN candidate is
independent of whether these new neural experiments succeed.

No claim of independent confirmation: all500 training labels have prior
exposure, and the old100-image holdout is no longer independent. Public is
used for the observed component contrast, never an inverse fitted weight.
