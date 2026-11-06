# Follow-up to the completed, unsuccessful new-CNN screen

Both pre-locked new training arms failed their advancement criterion.
Do not train second seeds for them or promote them over the existing CNN.

The matched control unexpectedly improved with a high-noise-only bounded
OLS coefficient: old two-seed CNN 9.319650 ->9.228277. However, its additional
gain is almost entirely outer fold4 (0.453824); the conditional gain interval
is [-.034096,.237405]. This is not sufficient evidence to increase the
production coefficient from its original internally fitted value.

Test precisely two calibration rules on frozen A-only predictors, using
the prior audit's 5 ShuffleSplit107/80 assignments, seed2026101608:

1. Original noise-regime partial pooling/LOO rule, followed by the pre-fixed
   high-only deployment gate (zero correction for low rows in both methods).
2. Bounded OLS on high-noise calibration rows only, without a new grid.

Use prior frozen query predictions for anchors0 and1. Extend exactly one
more existing full073 A213 teacher (anchor4) to its excluded V80, plus eight
overlap B rows as a replay control. Use the original A-trained CNNs and
complete021 predictions: no CNN training on the187 query rows. Anchor4 was
selected after seeing its dominant gain, so it is a targeted stress test,
not independent confirmation. All three anchors are reported separately.

Compare original versus new calibration, and no-image context controls,
aggregating squared losses by original ID before computing RMSE; repeated
roles are not independent observations. Bootstrap IDs conditionally and
report tail concentration. A stronger production coefficient is not
adopted unless its incremental gain is positive in every anchor, its pooled
95% conditional interval excludes zero, and it remains positive after
removing its ten largest-benefit IDs. These thresholds are fixed before
reading the new stress-test results. Even passing is not new-data evidence.

Budget: one72-TabPFN/three-grain-set teacher bundle, CPU2, at most15min.
No Public inverse calculation or Public coefficient fit. The separate
fixed076+(080-078) composition keeps the original coefficient regardless
of this result.
