# Model audit and bounded repairs

Frozen before new candidate fitting/scoring, 17 October 2026 KST.
082 public8.9343098075 is incumbent;0768.9646534604 rollback.083 submission
was interrupted before attachment and is not part of this work. No uploads,
paid compute, installations, external training data or new test predictions.
All new writes here; previous154570-file inventory checked before research.

Budget:90 minutes of new compute, one fitting process at a time, CPU2 threads
or one MPS process, own combined RSS/device allocation <5GiB. Reserve10min
verification/reproduction. If a projected arm exceeds remaining budget, stop
that arm before main scoring and report incomplete. No outer-score early stop,
seed search or tuning. Existing work/cache is reused only after checking IDs,
source labels/features, model seed and original recipe. Budget excludes reading,
code authoring and preservation hashing, which are separately timed.

## Ranked hypotheses

1. **Wrong residual teacher.**082 trains y−T073 and applies on076. Its earlier
   T076 pilot passed its own gate (+.025 matched over2folds) but stopped during
   the remaining teachers. Finish the original frozen recipe on all5folds.
   This is completion, not a new independent replication. Expected upside is
   uncertain and probably modest; decisive test is the remaining3folds and
   complete paired OOF. Reuse27 complete teacher nodes and36 valid members in
   the incomplete node. Fit only180 missing Tab contexts plus missing linear
   fits, then12 missing CNNs. Preserve the8 completed CNNs. New Tab contexts
   use MPS explicitfloat32,8 estimators,3seeds100*j+fold; compare one completed
   CPU context at identical source size first. Tolerance1e-3HV. CNN remains
   exact old CPU40epoch/D4/2seed/final5mean recipe.
   Primary: new vs old CNN with BOTH calibrated on complete076 A→B using the
   existing noise-regime LOO policy. Also compare against fixed082 and report
   frozen old coefficients to isolate training from coefficient changes.

2. **Retransformation bias.**The logarithmic partner minimizes log error but is
   scored in hardness units. Exponentiation/geometric averaging can shift
   conditional means down. Test a single train-derived smearing correction,
   with matched source-grouped logRidge and fixed0.128 deployed share. No
   parameter grid for the new correction. Fit each variant's alpha using its
   original grouped inner procedure; derive smearing from source-out-of-fold
   residuals on ORIGINAL real noisy training sources only, never copies or
   query labels. Use a common cross-fitted factor for the geometric ensemble:
   mean(y/exp(mean(logpred_variant))) on noisy train rows. Fixed same5folds.
   Clean/failure rows unchanged. A nested global offset using those same
   residuals is a calibration control; improvement must exceed that control.
   Existing alpha search is nested inside each inner fitting split, including
   its preprocessing and source derivatives. This is a statistical hypothesis,
   not a claimed software error or new image signal.

3. **Early sampling discards structure.**The residual CNN first uses7x7/stride4,
   then only spatial means/SDs. Audit subpixel-lattice sensitivity with four
   1px translation phases and fixed contexts, on16 preset high-noise fold0
   validation sources (ascendingID). No labels in this diagnostic; reflect
   shifts are diagnostic views only, not training examples.
   Compare two SAME-parameter, SAME-compute dense-first-block models:7x7stride1,
   GroupNorm/SiLU, then (a) every fourth pixel, (b) fixed4x4 average pooling.
   Later blocks unchanged. This separates dense computation/normalization
   from anti-alias averaging. Original input256 remains unchanged; only D4
   training views. No weak masks or intensity reconstruction objective.
   Use original073 targets/context to isolate sampling fromH1; all5outer folds,
   A213→B107 and T320→V80, existing2seeds2026101501+100fold+100000*repeat,
   40epochs, AdamW.002/.01, batch32, cosine to.0001, final5weightmean.
   Both arms CPU unless a label-free resource probe requires MPS for both;
   device choice is recorded before fitting. Coefficients useB only, same076
   base/noise LOO rule asH1. Primary: average arm vs subsample control AND082.
   No crop, blur-strength, shift-grid, epoch or channel search.

## Comparison/decision

Original400 reused development sources,5outerfolds320/80, historicalA213/B107
roles.082 conditional OOF9.256738746625928, high-noise12411.508128300723715.
0769.492439956633985. All500 labels have previous exposure. New fits are
nested; frozen segmentation representations and historical model selection
are inherited, so this is conditional evaluation, not fully fresh end-to-end
validation or a private-score forecast. Repartitions do not create fresh labels.

Report paired overall/high-noise RMSE,5fold gains,2seed results where relevant,
10000image bootstrap(seed2026101801),95% descriptive and98.33% family intervals
for3primary hypotheses, largest5/10 contribution removals, affine component
diagnostic and failure IDs. Use original source as resampling unit. Bootstrap
does not undo past selection or dependencies among fitted folds.

Advancement requires overall gain>=.02 and high-noise gain>0 vs082, >=3/5folds
positive, family lower bound>0, positive after top10 removal, and beating the
specified matched/calibration control. No winner chosen from controls. If
multiple pass use fixed priority H1,H3,H2. Production candidate construction
requires a separately frozen full500 recipe and successful original artifact
replay; never change official best or submit without explicit approval.

Verify source disjointness, complete derivative grouping, outside-label poison
invariance for new fits/calibration, independent prediction/metric math and
saved-model replay. End by reproducing076 cached assembly and082 saved-weight
inference into this directory, distinguishing that from full training replay.
All old artifacts, GitHEAD and pre-existing tracked diff must remain unchanged.
