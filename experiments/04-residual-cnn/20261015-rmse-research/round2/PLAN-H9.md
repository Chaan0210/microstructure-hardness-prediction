# H9: nonlinear use of a small physical measurement block

Registered after H7's weak full-block linear result and before any H9 fit.
H7's 18-column GMM-Ridge extension improved historical strict OOF by 0.00861,
but its random-block control also improved; the first two fresh-500 partitions
give pooled +0.00346 with an interval spanning zero. No claim of established
physical information is made.

The scale-1 third/fourth-cumulant phase estimate is correlated 0.789 with u_ph,
with only 1.2% of estimates at the inverse-model boundary. This label-free
measurement diagnostic is a reason to test nonlinear conditional use rather
than more linear model complexity. Preselect exactly scale-1 p34 and its
contrast estimate (feature indices 1 and 4); not chosen by target correlations.
No other scale, subset, smoothing or noise threshold will be searched.

On the exact same all-500 splits and fixed CPU baseline as H8, refit each
TabPFN head with these two extra features. Keep original full/nb features,
train/query IDs, seed, n_estimators=4, weights and common failure fallback.
Negative control: two independent fixed N(0,1) columns (seed 2026101509), with
no feature selection. Replace only TabPFN at its frozen normalized weight.
Primary applies on every non-failed image. Noise-band effects are diagnostics,
not a license to choose a post-hoc gate. Evaluate all 15 folds and all three
partitions before deciding; compare paired effects to baseline and random.

If promising in both actual and placebo comparison, confirm against original
073 component architecture with its original CPU seeds/banks before constructing
a new submission. If not, retain the measurement as a failed/inconclusive study.
All inputs remain train-only; originals 061/073, other-agent CNN and all prior
results remain untouched.
