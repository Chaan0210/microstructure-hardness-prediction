# Unchanged confirmation after the first complete five-fold screen

The image-only residual CNN improved all five folds: pooled 9.885309 ->9.853156,
gain .032153, conditional CI [-.000220,.063931]. The auxiliary/context model
also improved pooled RMSE .032143 and four folds, but was much less stable.
Both beat the context-only pooled result 9.883851. Gains exceed the respective
conditional bootstrap SD estimated by the interval width; this only allocates
confirmation compute, not a claim of independent significance.

Repeat exactly these two candidates and the context-only control on preregistered
partitions 1 and 2 (seeds2026101502/2026101503), all five folds, seed+fold. Reuse
the unmodified complement_cnn.py training/model/prediction functions. No change
to epochs40, image256, D4, learning rate, normalization, shrinkage.25, auxiliary
weight.2, model initialization or feature/context definitions. The direct-y and
context-without-aux arms failed the screen and are not rerun.

All first-stage five-arm files remain intact. Confirmation gets its own summary.
Require positive pooled effect, nonnegative effect on every partition and a
majority of folds improved, and examine the context-only contrast and paired
ID bootstrap. The same500 images are reused, so even confirmation remains
conditional evidence. Full073 replacement and a Private improvement are not
established by this tabular-anchor prototype.
