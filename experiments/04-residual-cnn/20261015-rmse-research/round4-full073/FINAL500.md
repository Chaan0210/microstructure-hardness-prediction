# Full500 experimental candidate — frozen before test inference

Use the auxiliary-objective two-seed residual CNN, with the existing noise12
gate. Matched mean2 noise-calibrated controls have addition RMSE9.48940
(image mean/SD) and9.50134(context only), versus9.31965 for the auxiliary CNN.
Replacement reaches9.25879 on the original400-image5-fold benchmark.
The noise-calibration hypothesis was formed after inspecting outer residual
structure. Therefore these remain exploratory local results, not an unbiased
new holdout or a claim about private leaderboard performance.

Produce BOTH addition and replacement CSVs in `final500/`, without uploading,
renaming a shared candidate or changing061/073. Keep the existing073 full500
prediction and all original component predictions frozen for inference.

1. Fit full non021 T teachers on a new, fixed shuffled3-fold partition of
   all500 training IDs, KFold seed2026101607. Same full12 feature banks ×3
   TabPFN seeds ×2 gates, strict-source degradation, Ridge/logRidge, GMM and
   three grain-set seeds. Each label is excluded from its own teacher fit.
2. Fit the unchanged auxiliary CNN on all500 images with these cross-fitted
   residual targets and six context inputs. Two seeds2026101501 and2026201501,
   40 epochs, final-five-state average, identical augmentation and four-view
   prediction. No test inputs in teacher or CNN fitting.
3. Freeze production coefficients as the arithmetic mean of the five
   internally fitted noise-regime coefficient vectors, separately for addition
   and replacement. This aggregation is fixed now, without test inputs or
   any global outer-OOF coefficient optimization. Do not report a local score
   obtained by applying these pooled production coefficients to the old OOF:
   that would not be an honest nested comparison.
4. Only after fitting, read test images/features and the unchanged073/021
   predictions. Average the two CNN corrections, apply the frozen coefficient
   for each image's noise regime, and preserve the existing segmentation-fail
   fallback. Test features are transformed using training-only normalization.

The full500 model uses more training data than the validated outer320 fits;
the local scores describe the nested validation procedure, not a newly scored
500-trained model. Full500 coefficient aggregation likewise has no separate
OOF score. Both limitations must accompany the candidate files.

The CNN fitter is derived from the preserved round4 fitter by adding only an
empty-query-safe maximum in reload diagnostics. All learning operations remain
unchanged. No validation/test dataset is used for epoch or checkpoint selection.
Independently verify saved model predictions on training probes for serialization
only; do not treat those probes as a validation score.
