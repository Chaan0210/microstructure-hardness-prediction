# Full 073 residual-CNN confirmation — locked before new outer scores

Date: 2026-10-16. User explicitly requests full073, its original folds,
inner-only weight selection, preservation of 061/073 and other agents' work.
All new writes stay in this directory. CPU, two threads, one training process.
No test labels, test fitting, external data, leaderboard selection or upload.

## Question and fixed arms

Let G be the complete 021 ensemble (not merely its TinyCNN).
Let T = (.384 TabPFN + .128 Ridge + .128 logRidge + .05 grain-set
+ .15 GMM-Ridge) / .84. Then P073 = .16 G + .84 T.
The corrected baseline uses source-restricted augmentation and actual GMM
components, as established in rounds 1–3. Its outer predictions are frozen.

Train the previously selected round3 image-only residual CNN on honest
cross-fitted targets y − T. This tests complementarity to the non021 branch;
it does not pretend that the training objective is y − full073.

1. Full073 baseline.
2. Addition: P073 + lambda_add * rCNN.
3. Deletion control: T.
4. Replacement: T + lambda_replace * rCNN.

Each scalar lambda is constrained least squares on one pre-existing internal
validation set only: clip(dot(r, y − base)/dot(r,r), 0, 1).
No intercept, band-specific weights, outer-label weight fit or outer-score
choice. Zero is a permissible result. Report the deletion control to avoid
crediting the CNN for benefits from dropping G.

## Label graph and cached 021

Use all 400 development images and the original five outer folds of073:
outer T320 / V80. For each fold, reuse the pre-existing 219 contract's
inner A213 / B107 and its *complete*021 anchor trained only on A. Verify
training IDs, all nested label membership, artifact hash, and reproduction
of its saved V predictions before using its B predictions. It contains the
same KRR, three TinyCNNs, S/B/L heads, short-B, affine and residual blocks.
Its prediction on V is used solely as a cache integrity check, never as
the outer full073 predictor (which retains the original320-trained G).

Fit T on A and predict B, retaining every non021 component of073. Train
the selection CNN only on A, with three further folds C142/D71 for its
residual targets. Evaluate that CNN on B. Select both lambdas from B.
Then refit the same CNN on all outer T320 using three honest T teachers
(one is A→B; the other two complete coverage). The two other inner
validation sets are a seeded random division of A, independent of labels.
Evaluate only once on V, using original full073 outer predictions and
the already locked lambdas. Every teacher's augmentation source must
belong to that teacher's actual training IDs.

## Exact non021 branch and training recipe

TabPFN v3.5 local checkpoint, default estimator count, all 038/039/044
slots × four U-Net variants × three seeds (100*s + outer_fold). On
noise<12 remove precisely the original het/intensity columns; otherwise
retain full columns. RidgeCV and logRidge(alpha2, four-variant geometric
mean) preserve the production recipe, with source-restricted degraded
copies only in the high-noise branch. GMM-Ridge retains its original
three-fold alpha selection. Grain-set retains all three seeds731503,
731509,731527, original architecture and its own inner early stopping.

CNN: round3's 68,788-parameter native256 multiscale moment model,
image-only residual arm, 40 epochs, AdamW .002, weight decay .01,
cosine schedule, batch32, D4, final-five-epoch state average, four-view
inference. Change only the fitter's hardcoded400 training-size assertion
to allow the legitimate213/320 inner/outer sizes. Source stays unchanged;
derive the adapted fitter in this directory and lock both hashes.
Seed = 2026101501 + 100*outer_fold, same for selector and final refit.
Use original-target training SD to normalize residuals, as in round3.
CNN sees no validation targets, including for epoch/checkpoint selection.

Retain full073's original p021 fallback on failed segmentation rows in
all outer arms. Omit failed rows from the lambda loss. For residual
teacher targets only, use the teacher-training mean if segmentation fails.

## Evaluation and limitations

Complete all five outer folds before choosing a winner. Report pooled
RMSE, each fold, both inner lambdas, error correlation, tail contributions,
and paired ID bootstrap intervals. This is conditional confirmation on
previously explored development data, not a fresh unbiased benchmark.
One internal weight split per outer fold has only107 rows; report its
uncertainty and any weight hitting a boundary. Do not claim replacement
improvement without comparing against deletion-only T.

If the primary CNN improves, run a second fixed CNN seed through the same
inner-selection procedure and a statistics-only control on the same folds;
reuse all expensive teacher fits. Further full-stack partitions require
new complete021 inner anchors and are justified only by the results.
No model promotion based solely on a marginal pooled improvement.

Verify train/validation membership, component completeness, checkpoint
reload, inner weight recomputation, held-out-label poison independence,
and immutable061/073/source hashes. Save reusable teachers, CNN checkpoints,
exact configuration and commands, OOF predictions, results and limitations.
