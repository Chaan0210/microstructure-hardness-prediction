# 080 follow-up: at most three minimal diagnostic experiments

Frozen 2026-10-16 before new experiment outcomes. Local existing CPU only,
two threads, one fitting process. No installation, paid compute, external
data, test fitting, Public-implied coefficient, new submission, Git action,
other-thread mutation, or interruption/modification of another process.
All new outputs remain in this directory. Existing files are read-only.

Known record: Public080=8.999772193; Public076=8.9646534604. There is no078
submission score in the current submission ledger; its non-submission is
also explicit in push-1300 NOTES. Hence080 is worse than076 by.0351187326,
but its marginal CNN effect relative to078 is not identified. Do not use
Public inversion, full-test squared-change proxy, or inferred
scaling coefficient as evidence for selecting/tuning a model.

## E1 — assembly/version and baseline transport

Verify CSV hashes, all1000 IDs/orders, actual080=078+(round4-addition−073),
the318 changed rows, the existing failure/noise mask, and coefficient/model
provenance. Compare the exact frozen OOF correction on the as-recorded073
and078 predictors; no weight refit. Explicitly distinguish historical078
OOF from source-corrected073. Quantify overlap with078−073 and the disjoint
supports of078−076 versus080−078. Recompute concentration measures and
identify what spread-removal and no-image controls do NOT establish.
Stop interpretation if ID/hash/composition checks fail.

## E2 — actual refit/teacher/input-distribution transport

Use original anchors0 and1 only, selected by index, not by a new outcome.
Existing A213/B107/T320/V80 definitions remain fixed. Extend each complete
non021 A-only teacher to V80 plus eight B overlap checks, using the exact
old full073 recipe. Reuse original G_A, auxiliary/context selector CNNs and
final CNNs. No neural retraining. Check fresh B predictions against the
stored reference (max difference <1e-4; otherwise stop the dependent tests).

On the SAME V80 rows compare: A-base/A-CNN with native A context;
T-base/A-CNN with T context; T-base/T-CNN with native T context. Use the
original per-fold B-only coefficients unchanged. This separates a base/
context transition from the combined CNN-refit transition; it does NOT
individually identify teacher-size versus sample-composition versus optimizer
or normalization causes. Profile paired142/213/333 teacher targets on common
IDs and original A/T/full500 normalization and update counts. Never score a
500-trained CNN on its training labels as validation. Never apply the pooled
production coefficient to old OOF and call it an honest validation score.

At most two fresh full teacher fits (144 TabPFN contexts, six grain-set seeds
plus their stopping fits); existing local checkpoints only. Stop after60min
of new fitting, a nonfinite result, overlap mismatch, or broken source scope.
Keep partial work if stopped; do not silently weaken the recipe.

## E3 — frozen-model calibration split sensitivity

For each A-only model, its187 non-training rows are already available as
B107 plus V80. Freeze the CNN/base predictors and make five new107/80
ShuffleSplit calibration/evaluation splits, seed2026101608. Within each
calibration set select noise12 coefficients and pooling with the unchanged
leave-one-out algorithm. Compare auxiliary CNN with baseline and matched
two-seed context-only control. No reusing the old B coefficient on a newly
selected evaluation set; no per-split model/threshold selection.

Run64 fixed-seed within-noise permutations of the CNN direction, refitting
calibration only. This is an alignment-null stress test, not a conditional
causal-information test and not a correction for the historical80+ searches.
Aggregate squared errors by unique image ID before uncertainty calculations;
overlapping anchors/splits do not create independent observations.

Report original and reshuffled split results even if worse. Do not remove
the largest-error/improvement IDs for the primary metric. Tail deletion is
descriptive only. Audit all100 non-dev IDs' exposure in prior experiments
and full500 fitting: no subset is labelled an independent holdout merely
because it was excluded from the original400 benchmark.

## Interpretation and stopping

RMSE on original target units is primary. Report paired MSE/RMSE changes,
fold/split values, coefficient variability and prediction RMS. No parameter
is selected from Public or test predictions. No promotion or further fitting
is triggered automatically by a favorable pilot. These images and hypotheses
were already explored; even a positive resplit result is conditional
robustness evidence, not a previously untouched validation dataset. Separate
confirmed implementation facts, plausible mechanisms and unresolved causes.
