# Plateau investigation — 16 October 2026

**Retain candidate 076. None of the three new hypotheses earned advancement.**
The recorded public best remains **8.9646534604**. No official best, submission,
model, dependency, or historical source was replaced. No submission, upload,
paid compute, installation, external data, or test-label fitting occurred.

The three investigations were implemented and evaluated, including matched
controls, nested parameter selection, saved predictions and model replay. Their
failure is evidence against these particular remedies, not proof that the
competition has reached an irreducible error floor. No leaderboard score is
predicted for a new combination.

## Verified starting point

The root has no `NOTES.md`; recent history is in `experiments/push-*/NOTES.md`
and the dated research reports. The score records, numbered submission hashes,
and builders identify 076 as the incumbent. Files 082/083 exist, but the inspected
records do not establish a new measured public best.

The normal-row formula is

`0.16 G021 + 0.384 H9-TabPFN + 0.128 Ridge + 0.128 logRidge + 0.05 grain-set + 0.15 GMM`.

G021 itself contains the older kernel/CNN/DINO ensemble. H9-TabPFN uses three
feature banks, four U-Net variants and three seeds, with different feature
columns below/above noise 12. The high-noise linear partners use permitted
train-derived degradation copies and source-grouped alpha selection. Failed
segmentation rows fall back to G021. See `incumbent.json` and the protected
historical reproduction documents for the complete assembly chain.

The public observations support a **0.0301236788** reduction for 080 versus 078.
They also establish that 080 remained **0.0351187326** worse than 076. This does
not establish that the CNN contributed nothing. The 076+existing-CNN combination
was already present before this task; it is not a new discovery here. Public
scores and test predictions were not used to choose new numerical parameters.

For experiments, I verified the existing source-restricted 076 reconstruction:
**9.4924399566 RMSE**, original 400 images / five outer folds /320 training images
per fold. Rebuilding its six components agrees within 1.09e-11 HV. This differs
from the historical 076 OOF because its training-source scope was corrected;
the difference is not credited as a model improvement. The same reconstructed
base plus the existing fold-selected CNN correction is 9.2567387466.

## Diagnosis: established facts and remaining hypotheses

| Region | Images | Incumbent conditional OOF RMSE | Share of squared error | Mean prediction−target |
|---|---:|---:|---:|---:|
| noise<9.5 |205|8.0082|36.48%|−0.1549|
| 9.5≤noise<12 |71|8.1412|13.06%|+0.8866|
| noise≥12 |124|12.1117|50.47%|+0.0771|

**Measurement loss remains the leading structural bottleneck.** The earlier
paired original/degraded-image studies isolate feature deterioration while
keeping source and hardness fixed. Their evidence points to missing boundaries,
blur and grain contrast, rather than additional white noise alone. This study
found an additional concrete probability interpretation mismatch:
`push-400/phase_unet4.py:113` trains class-weighted cross-entropy, while
`unet4_feats.py:37` applies unadjusted softmax and subsequently aggregates it as
class probability. The weighted-loss population optimum is proportional to
class weight times posterior probability. Inverting that weighting is justified
as a hypothesis, but neither model misspecification nor pseudo-label bias is
removed by the algebra.

**Correcting that probability interpretation did not fix hardness prediction.**
On 30 images excluded from z1 training, phase-fraction MSE declined 8.7%, and all
four Brier scores improved. Phase-fraction RMSE was 0.035452→0.033867; its paired
95% gain interval was[−0.007244,+0.008823], and only 15/30 individual images
improved. These are mask-derived auxiliary targets, not independently measured
physical truth. The downstream hardness comparison subsequently failed.

**Calibration and validation instability are real, but no stable remedy was
demonstrated.** High-noise OOF has a descriptive target-on-prediction slope 1.2333
(conditional 95% interval 1.0836–1.3702), despite almost zero mean bias. This does
not license a deployed spread expansion: all labels have been inspected, sample
composition differs with training size, and historical rescaling experiments
already failed transfer. The new methods' changes also fail after removing their
affine spread component; none conceals a useful non-calibration gain.

The existing residual CNN learns `y−T073`, where T073 is the non-G021 branch
normalized by 0.84, while it is combined with the complete ensemble. Its teacher
and context sizes differ across selector/refit/production, and the coefficient
selection has only 107 internal validation images. Prior coefficients span 0–1;
prior role reassignment reversed apparently promising improvements. Existing
142/213/333-teacher comparisons change **both size and composition**, so they
cannot identify a pure training-size effect. A separate 076-retarget job was
already running when this task started; it was left untouched and not counted
as a completed experiment here.

**Shared residuals do not prove irreducibility.** In the noisy group,076's
residual correlations with GMM/G021/grain-set are 0.935/0.864/0.773. The worst 10
images contribute 16.54% of total SSE. Agreement can reflect a shared missing
representation; there are no repeated independent hardness measurements here
that would identify a noise floor. The earlier CNN's improvement is positive
evidence that at least some complementary image signal exists.

## Predeclared experiments and actual results

`PLAN.md` and `pre-score-lock.json` precede the new hardness scores. Rank and
stopping rules were fixed: physical probability correction first; joint local
physical distributions second; corruption-aware nonlinear similarity third.
Each used the same outer partitions, baseline weights, gate and fallback.

| Method | RMSE | Gain vs 076¹ | Paired conditional 95% interval | Improved folds | Gain after removing top 10 contributors |
|---|---:|---:|---|---:|---:|
| Source-restricted 076 |9.492440|—|—|—|—|
| H1 corrected probabilities |9.496240|−0.003800|[−0.035862,+0.028703]|3/5|−0.034358|
| H2 joint-distribution kernel, tie repair |9.507017|−0.014577|[−0.048611,+0.020618]|2/5|−0.046956|
| H3 corruption-invariant kernel |9.547817|−0.055377|[−0.095200,−0.020657]|0/5|−0.076475|

¹ Gain means baseline RMSE minus candidate RMSE. Negative is worse. These are
conditional head comparisons, not estimates of public/private leaderboard gains.

**H1:** Reconstruct the exact 219-source class frequencies and undo the original
classification weighting. The label-blind auxiliary gate passed, so both raw-z1
and corrected-z1 versions replaced the same three existing phase columns in the
GMM partner, retaining its 0.15 weight. The matched raw version scored 9.492589;
correction itself lost 0.003651. Twenty fitted head pipelines were saved. Better
agreement with weak segmentation targets did not yield a better hardness model.

**H2:** Encode the grain-weighted distribution of local log area, darkness,
elongation and phase probability along four physical axes and 16 fixed random
directions. A linear-plus-RBF kernel compares those projected distributions.
This learns no hardness-supervised image filters, token embeddings or spatial
PCA basis. Marginal-only and same-grid linear controls scored 9.509490 and 9.496219.
Joint structure beat marginals by only 0.002474 and lost to both production and
the linear control. Nested selection disabled the distribution kernel in 7/10
fold/regime fits. No expansion or additional random-direction search was run.
An additional verification fixture exposed a tied-value ordering defect in the
first interpolated weighted quantile. The repaired empirical-CDF quantile is
exactly invariant to pixel reordering, including a real map with ties. All five
folds were rerun with unchanged directions, splits, grids and selection rules;
the linear control remained identical and the conclusion did not change. The
first run is preserved; `h2-tie-repair/` is authoritative for this hypothesis.

**H3:** Learn the nuisance second-moment matrix from paired original/permitted
degradation features inside each training fold. Use `(I+C)^−1` as the RBF metric,
and compare identity and randomly rotated covariance with identical sources,
targets, grids and preprocessing. This changes nonlinear neighborhoods rather
than predicting a clean feature vector or imposing the old linear consistency
penalty. Identity and rotated controls scored 9.548429 and 9.553976. True covariance
gained only 0.000612 over identity; both lost every outer fold to 076. Four of five
folds chose the broadest kernel. This geometry change did not recover useful
signal from the existing imperfect augmentation/measurement system.

All three methods also lost when the existing CNN correction was retained:
incremental gains−0.005991,−0.015058 and−0.051867, respectively. Three-comparison
Bonferroni-adjusted bootstrap intervals, all controls, folds, subgroup scores,
affine decompositions and influential-image removals are in `final-results.json`.
The advancement rule required≥0.03 RMSE gain,≥3 improved folds, a positive lower
95% bound, a positive top 10-removed gain and successful mechanism controls.
No method passed. The all 500 confirmation stage was therefore not triggered.

## Concrete failure cases

These are the five largest baseline squared errors, selected for explanation
after evaluation. They are not an independent subset. The new methods generally
make small or adverse changes on these large errors. Full ID-aligned targets,
predictions and paired SSE differences are in `final-per-image.csv`.

## A separate implementation mismatch, measured at matched training sizes

The incumbent replay exposed automatic reduced precision on MPS. On one frozen
full 500 model, MPS automatic versus CPU differed by up to 0.099956 HV. Explicit
float32 reduced that to 0.000145 HV; query grouping made no difference under
float32. Failed replay attempts and their outputs are retained.

I then locked and completed a **train-only** diagnostic: one existing H9 physical
bank, all five original folds, three seeds, both feature gates and three backend
policies, totaling 90 inference contexts. Every fit used exactly 320 training
images. CPU and MPS used eight estimators and identical inputs and seeds.

| Numerical policy | Single-bank OOF RMSE | Prediction RMS difference from CPU | Maximum difference |
|---|---:|---:|---:|
| CPU automatic, autocast off |10.075895|0|0|
| MPS automatic, autocast on |10.074993|0.019414|0.061585|
| MPS explicit float32, autocast off |10.075897|0.00002988|0.00009450|

MPS automatic reproduces the saved historical development predictions within
7.14e-7 HV. Therefore the historical development/production precision mismatch
is established, rather than inferred from a changed sample size. Its apparent
single-bank RMSE difference is only 0.000902, conditional 95% interval
[−0.001385,+0.003093]. **It does not explain the competition gap or establish a
better predictor.** The new reproduction code explicitly matches CPU precision;
the protected incumbent code remains unchanged. This diagnostic is not a full
12-bank performance comparison or a candidate-selection exercise. Details and
its pre-score protocol are in `precision-train-results.json` and
`PRECISION-DIAGNOSTIC.md`.

## Validation scope and reproducibility

Every new imputer, scaler, covariance estimate, bandwidth and hyperparameter
selection was fit within its relevant training partition. Original/degradation
source groups were checked at both levels. H2/H3 nested replay after changing
all labels outside the training set produced identical predictions. Fifteen
final saved kernel bundles independently reproduce the evaluated predictions within
1e-8 HV; H1's 20 saved pipelines replay exactly. An independent scalar RMSE
calculation agrees with every reported score. See `verification.json` and
`h2-tie-repair/verification.json`; the original ten H2 bundles are also retained.

**No genuinely untouched labels remain.** The 400 development labels and the old
100 holdout labels have prior exposure. Existing auxiliary U-Net/Cellpose models,
feature definitions, degradation recipes and ensemble decisions remain frozen
historical representations. They were not rebuilt inside every new outer fold.
Thus these are properly nested *new-head selection* tests conditional on that
representation, not fully nested end-to-end image-pipeline validation. The
bootstrap resamples 400 IDs, not 1200 repeated-fold observations, and does not
remove past model-selection bias or dependence from overlapping training sets.
This is sufficient to reject these changes; it would not suffice to claim a
private-leaderboard breakthrough.

The initial inventory hashes 149,460 files /37.51 GiB, including environments,
code, weights, caches, predictions and records. Additional olderG021 ancestors
are inventoried separately and checked against their historical model manifests.
Runtime write guards prevent writes outside this new directory. The starting
Git commit is `f8f2b3ad`; full version, pre-existing uncommitted state and package
versions are in `repository.json` and `environments.json`. No dependency changed.

Incumbent cached assembly is byte-identical, SHA256
`b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0`.
The full 500 Ridge/logRidge/GMM fitting recipes replay exactly from frozen
features; all 1000 grain-set predictions replay exactly from saved weights.
G021's CPU replay differs from its canonical prediction by at most 4.88e-6 HV.
All 72 TabPFN members were replayed on their deployed rows from the frozen
checkpoint, original 500-image training context and frozen features. Every
deployed bank agrees within 0.000128 HV. Recomputing all final components gives
maximum difference **0.0000306983 HV** and RMS difference **0.0000088617 HV** from
the incumbent across all 1,000 rows. This meets the declared 0.002-HV aggregate
CPU/MPS tolerance; the actual discrepancy is much smaller. The per-bank
0.001-HV bound alone limits the Tab contribution to RMSE drift to 0.000384 HV.
No parameters were fitted to test predictions to obtain this agreement.

Saved z1 weights also passed eight-view raw-pixel replay on the first three train
and first three test images. The map comparison allows one stored float16 ULP
plus 2e-5 for float32 CPU/MPS drift. It is a bounded extraction check, not a claim
to have regenerated every historical image feature. Cached assembly, saved-weight
inference, feature extraction and a full training-run reproduction are distinct
scopes; the entire historical training run was not repeated.

Final preservation status: **all 150,102 protected files (39.61 GiB) are unchanged**. Git HEAD, tracked diff and staged diff also match the recorded starting state.

`RUN.md` gives exact rerun commands, seeds, output locations and provenance.
The rollback remains the original protected 076 pipeline and submission.
No new official best or replacement submission is designated.

The remaining unresolved question is how much hardness-relevant structure is
lost in the actual noisy images versus discarded by the shared auxiliary
representations. The existing degradation pairs and weak segmentation targets
do not identify that decomposition on real noisy images. This work rejects the
tested z1/GMM probability correction, distribution readout and invariant kernel;
it does not rule out every possible probability correction or unmeasured image
signal. Another small gain on these same labels would not resolve that question.
