# Locked research protocol — 2026-10-16, 20:35 KST

Incumbent: recorded public best 076 (8.9646534604), CSV SHA256
`b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0`.
No official replacement, upload, submission, installation, external data, test
fitting, public-score inversion, or new generated training images. Everything
new stays here. Original source and environments are read-only. A concurrent
076-teacher job belongs to other work; do not stop, modify, or duplicate it.

## Ranked hypotheses and decisive screens

1. **Correct a measurement probability mismatch.** U-Net classification uses
   inverse-square-root class weights, but feature extraction treats softmax
   outputs as class probabilities. At the weighted cross-entropy population
   optimum q(c|x) is proportional to w(c)p(c|x); undoing w before spatial
   aggregation may recover phase/pore fractions. This is a concrete potential
   calibration defect, not proof of a prediction defect: regression may already
   compensate, weak labels may be biased, and averaging TTA probabilities before
   correction is only approximate. First reconstruct exact z1 training class
   weights from its 219 original training images, without hardness. On the
   same 30 images excluded from z1 training compare original and deweighted maps
   to the original mask-derived targets. Require >=5% lower phase-fraction MSE
   and <=5% deterioration in each pore/boundary Brier score. Stop if either fails.
   If passed, test replacement of existing phase summary columns with corrected
   z1 summaries versus matched uncorrected-z1 replacement, keeping head recipes
   and weights fixed. Cost: minutes, existing maps; upside: systematic phase
   bias under uncertainty. No neural retraining warranted if this gate fails.

2. **Learn from the joint distribution of local physical states.** Existing
   summaries discard the coupling between local size, darkness, elongation and
   phase probability. Use z1 maps as a frozen measuring device; encode their
   grain-probability-weighted quantiles along four physical axes and 16 fixed
   random directions (seed 2026101601, 17 quantiles .025 to .975). A positive
   kernel on these projected distributions provides a small-data, permutation-
   invariant learner. It differs from spatial map PCA, DINO tokens and CNNs:
   no spatial alignment or image filters are learned from hardness; the geometry
   explicitly compares physical distributions. Fit an additive linear+RBF
   kernel on current GMM-partner inputs and the distribution signature. Compare
   joint versus marginal-only kernels and a linear-only same-grid control.
   Inner three-fold CV selects alpha [.001,.01,.1,1,10] and extra-kernel scale
   [0,.1,1]; RBF bandwidth is median squared training distance. Keep the GMM
   partner's .15 ensemble share. Cost: minutes; upside: missing local coupling.
   Reject if the joint model does not beat both production and marginal control.

3. **Learn nonlinear similarity in corruption-invariant directions.** Prior
   raw-feature RBF and feature-denoising trials failed. Their geometry still
   treats corruption-sensitive directions as equally reliable. Estimate the
   full second moment C of standardized original-to-permitted-degradation
   differences using only current training sources, then compare the RBF metric
   (I+C)^(-1) with identity and a fixed random rotation of C's eigenvectors.
   This changes nonlinear neighborhoods, unlike the earlier linear consistency
   penalty or point prediction of clean features. All arms use exactly the same
   source-grouped original+deg1..4 rows and targets. Inner three-fold grouped CV
   selects RBF gamma multipliers [.03,.3,3] / feature count and alpha [.01,.1,1,10].
   Replace only high-noise Ridge at its existing .128 weight. No weight sweep.
   Reject unless true C beats both controls and production. Cost: minutes,
   <1 GiB target RSS, one CPU thread; upside: more appropriate nonlinear learning
   under unreliable measurements without forcing lost information to reappear.

## Evaluation and selection

Run all five original outer folds before reading comparisons. Their 400 IDs
are already heavily inspected; none is an untouched holdout. Use the existing
source-restricted 076 reconstruction as the production reference, verify its
components and IDs, and keep the old 076+CNN correction fixed as a sensitivity
comparison. Changing the reconstruction is not credited as model improvement.

Every *new* imputer, scaler, covariance, kernel bandwidth and hyperparameter
selection is fit inside its training partition. Every degradation stays with
its original source at inner and outer levels. No training labels outside the
fit are available to that fit. Existing U-Net/Cellpose and inherited feature
choices are frozen shared representations, not freshly nested image training;
the head screen is conditional on them. This limitation prevents claiming a
fully independent generalization estimate. New full-pipeline promotion would
require separately justified representation-crossfit confirmation.

Use seed 2026101635 for three-fold inner partitions, +100*outer_fold; no
label-stratified choices. Bootstrap 10,000 times with seed 2026101636, image-ID
unit, paired squared errors; report pointwise and simultaneous intervals across
the three prespecified primary comparisons. Report each outer fold, clean/mid/
noisy bands, prediction-defined quintiles, top 5/10/20 contributors removed,
paired failure cases, and change orthogonal to affine spread (diagnostic only).
Do not select a model or deploy a spread adjustment from these diagnostics.

Advance only if improvement >=.03 RMSE against production, >=3/5 folds improve,
the conditional 95% gain interval excludes zero, the gain remains positive after
the ten largest contributors are removed, and the mechanism-specific control
passes. A successful screen requires an independently executed nested policy
comparison on all500, with training sizes explicitly matched. Same-label
resplits are sensitivity checks, never fresh confirmations. If nothing passes,
stop these hypotheses, retain076, and document the negative evidence.

Save every prediction, inner selection, membership and source hash. Test outside-
label poisoning, prediction reload, and relevant algebra. Protect the incumbent
before any experiment, reassemble its predictions afterwards from frozen
upstream outputs into this directory, and check exact CSV identity. Also verify
saved-weight inference on a bounded raw-image sample; explicitly distinguish
component reassembly, saved-weight inference and full training reproduction.
Do not claim the last one unless actually completed. If a historical learner
does not serialize its trained state, record the gap instead of calling its
prediction cache a saved-weight model.
