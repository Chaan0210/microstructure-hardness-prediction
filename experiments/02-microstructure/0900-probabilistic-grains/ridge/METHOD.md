# Fixed Ridge confirmation protocol

Frozen before the first new outer-fold scores, 2026-10-12.

Only new files in this directory are written. Existing data and code are read-only;
no legacy project module is imported and no test images or predictions are used.
Run with `.venv/bin/python -B` and numerical-library thread counts set to one.

The 400 unique development image IDs and three saved five-fold partitions
(`original`, `20261031`, `20261101`) are retained. All twelve legacy OOF archives
(four variants each for 038, 039, 044) must agree after explicit ID alignment on
labels, folds, 021 predictions, and failure flags. The exact comparator is
`0.2*p021 + 0.8*mean(g038,g039,g044)` with failed masks replaced by p021.

Use the 98-column average-variant feature matrix in `push-800/eda/resid045.npz`,
verified against the original 500-row feature archive. In each outer fold fit
median imputation and standardization on its 320 training rows only, followed by
RidgeCV with fixed alphas `[0.1, 1, 10, 100, 1000]` and its default analytic
leave-one-out criterion. No hyperparameter is selected on an outer validation
score. The imputation and scaling are outer-training-only, but they are not
refitted within the analytic leave-one-out criterion.

Predetermined comparisons, always retaining the p021 fallback:

1. Primary: `0.8*baseline045 + 0.2*ridge`.
2. Same 021-share: `0.2*p021 + 0.8*(0.8*grain045 + 0.2*ridge)`.
3. Weight-only control: `0.16*p021 + 0.84*grain045`.
4. Standalone Ridge, for diagnostic context.

The fixed secondary FGLS Ridge obtains training-row squared errors with a
four-fold, shuffled seed-900 cross-fit contained entirely in the outer training
fold (and its preprocessing is fitted inside each of those inner folds). Fit a
nonnegative variance model with columns `1`, `1/Nproxy`, and `noise_gray**2`.
Nproxy is `65536/exp(m_la_mean)`, a proxy, not a true grain count. Inverse predicted
variances are normalized to mean one and clipped to `[0.25, 4]`; fit a new RidgeCV
with these weights. This is a fixed secondary method, not a search over weighting
schemes or noise thresholds. Ridge's preprocessing remains unweighted.

Report pooled and per-partition RMSE, all 15 folds, image-ID-grouped paired
bootstrap intervals (10,000 replicates, seed 901), and descriptive high-noise
tercile/Nproxy<300 and worst-five-baseline-error exclusion diagnostics. The
bootstrap resamples 400 image IDs, preserving their three partition scores.
Diagnostics do not select models or weights. Report uncertainty both against
baseline045 and the matched weight-only control where applicable.

Save fold models, exact train/validation/inner-fold IDs, predictions, metrics,
input hashes, and validation results. Check exact model reload predictions,
training/validation ID exclusion, feature order invariance after explicit
alignment, and unchanged fitting after poisoning held-out labels. No deployment
or test candidate is generated.

This is conditional validation of a previously developed representation and
reused folds. Legacy feature extractors were not retrained excluding each fold;
they and the development history may leak representation or selection information.
These results do not constitute fresh independent validation or establish any
public/private leaderboard improvement.
