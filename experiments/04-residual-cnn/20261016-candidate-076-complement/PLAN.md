# Further improvement of076 plus the frozen residual CNN

2026-10-16. Preserve the recommended CSV,061/073,all previous experiments and
other agents' files. New writes only here. CPU2,one fitting/inference process,
no new packages/external data/test training/remote inference/Public weight
optimization/submission. No subagents. Reuse verified existing code and caches.

## Start with a valid076 baseline

The preceding076 comparison transported CNN predictions onto historical OOF.
Build a source-restricted076 on the original400 five folds: keep strict073's
G021/grain-set/GMM and low-noise linear components; use the existing12-member
H9 TabPFN OOF caches after checking IDs,folds,labels,feature hashes and generation
code. These Tab models train on the actual320 development rows without DTA.
Refit the high-noise Ridge and four logRidges with the076 source-grouped inner
alpha-selection rule, but allow augmented copies ONLY of the actual outer
training sources. No global500 alpha choice inside CV. Record all selected
alphas and compare fixed-rule reconstruction with saved076 production recipe.

## Locked cheap inference probes before expensive retargeted training

The native residual CNN uses D4 augmentation in training but only four of the
eight views at inference. A full eight-view mean may reduce view-dependent
error. Rotation invariance is not guaranteed by the synthetic target; this is
an empirical test, not a physical assertion. Old TinyCNN D4-training ablations
do not settle full-D4 inference for this new residual branch.

The deployed CNN receives073's T and Tab-minus-Ridge context although it is
added to076. Refresh exactly those two query context columns to strict076.
Keep normalization and all trained weights unchanged. This tests a fixed
inference transformation, NOT a CNN trained on076 residuals.

Evaluate one2x2 comparison on all five original outer folds and both already
trained seeds: four/eight views times old/refreshed context. Primary comparison
uses the original fold-specific coefficients, fitted on inner B only; zero
low-noise correction and failed-row fallback stay fixed. No outer-fitted
weights, view selection, threshold search, or training/checkpoint selection.
Replay old four-view outputs as an integrity check. Report seed-level and
equal-mean effects, all folds, high-noise subsets and paired-ID intervals.

Advance a probe only if gain>=.005, at least3folds improve, and the conditional
95% interval versus the current correction excludes0. No independent-label
claim: all400/500 training labels were previously explored. Include all three
new contrasts in simultaneous conditional intervals. A passing probe needs
another frozen-anchor role test or actual076 internal calibration before any
promotion. An unsuccessful cheap probe does not by itself disprove training
the CNN on honest076 residuals; use its effect sizes to prioritize that cost.

Before new expensive fits, inspect the candidate's residual and correction
differences quantitatively. Do not repeat the previously failed high-only
primary loss, removal of teacher context, or stronger OLS coefficient.
