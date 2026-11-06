# Raw-image relational learning for high-noise hardness

Registered before new candidate training or scores, 16 October2026.

## Incumbents and scope

082 is now the recorded measured public best,8.9343098075 (submission56).
076,8.9646534604, remains the rollback. Preserve both complete pipelines and all
prior experiments. No submission, upload, install, paid compute, external data,
test training, pseudo-labeling or modification of shared artifacts. All new
files stay in this directory. Other work is processing information-matched
degradation copies; do not duplicate, stop or modify it. Use one CPU training
process, two threads, streaming uint8 source images; no accelerator contention.

## New learning problem and distinction from failed work

Train a native256 image encoder to represent **soft hardness neighborhoods in a
32-dimensional image embedding**, alongside an absolute-hardness scalar head.
For each minibatch, two independently sampled D4 views of each original image
give embeddings z1,z2. Center each view's embeddings within the training batch,
then L2-normalize. Student probabilities are row-softmax(z1 z2^T /0.2).
Target probabilities are row-softmax(−0.5*((y_i−y_j)/(0.5*s_train))²), including
the paired same-source view. Average the forward and reverse KL divergences.

Loss = mean squared error of both scalar views in train-standardized hardness
units +1.0*neighborhood KL. Width0.5 training SD, temperature0.2 and weight1 are
fixed, not searched. Nearby labels receive soft weights rather than hard bins;
two views share the original hardness. No pixels are generated, no intensity
or boundary changes are introduced, and all D4 derivatives stay with their source.

Why this is different:
- Existing082 learns scalar residuals against cross-fitted073 teachers, with
  context and an auxiliary original-y head. This new encoder consumes no teacher
  prediction, residual target, physical-feature vector or test image.
- Study120 used a0.1 scalar pairwise logistic ranking penalty on an existing
  frozen ViT-B feature head. It did not learn pixel filters or vector-space
  neighborhood geometry. That fixed scalar-ranking experiment failed and is
  not being repeated here.
- Token/spatial PCA studies altered representation inputs/readouts. This changes
  the supervised image-learning objective, with all learned filters fit inside
  each training partition.

Mechanism hypothesis: cross-image label neighborhoods may discourage encoding
rendering/noise differences that separate otherwise similarly hard images.
They provide no extra independent labels, and noisy hardness labels can make the
constraint harmful. The matched controls below must determine whether the
relationship targets add useful signal.

## Three matched arms and training

1. MSE: scalar loss only, neighborhood weight0.
2. REL: scalar loss plus the stated true-label neighborhood objective.
3. SHUFFLED: same asREL, but the neighborhood labels are permuted once over
   training sources (seed+7919). Scalar targets remain correct and paired views
   retain their source identity. This controls for generic view/embedding
   regularization and additional computation without correct cross-image labels.

All arms share initialization, image/target normalization, example order, D4
draws, dropout stream, optimizer, epochs and evaluation views. Existing compact
encoder: stride4 7x7 convolution then stride2 3x3 blocks with16/32/48/64 channels,
GroupNorm4/SiLU, concatenated input and all-stage spatial means/SDs (322 values).
Readout: Linear322→64, SiLU, dropout.25, Linear64→32 embedding, scalar Linear32→1.
Scalar last layer initially zero. Both views are forwarded together for every
arm, even MSE. No additional context or pretrained encoder is added.

40 epochs, batch32 sources/two views, AdamW lr.002, weight_decay.01, cosine to
.0001, gradient norm cap5, arithmetic mean of final5 epoch states. Four-view
inference identical across arms. No outer-label early stopping/checkpoint choice.
Seed2026101651+100*outer_fold; identical seed for selector/refit. CPU2 only.
One discarded two-epoch real-training-only resource/gradient probe precedes the
main run. It is never evaluated against validation labels. Complete all30 main
fits (3arms ×5outer folds ×selector/refit) before comparative scoring. Stop only
for invalid numerics, memory>1.5GiB or a projected main cost>90minutes.

## Honest nested evaluation and exact082 comparator

Original400-image five-fold split, outer T320/V80; existing internal A213/B107
roles. New pixel encoder, normalization and every supervised parameter are
trained on A only for calibration, then freshly refit on T for outer evaluation.
No query labels are accepted by the trainer. No residual teacher is used for
new image training, eliminating that particular teacher-size mismatch.

For each fold, reconstruct the old082 coefficient from its original internal
B-only noise-regime selection using strict073 A→B and old CNN A→B predictions.
Require agreement with its saved selection. The old correction is applied only
to noise≥12, with the original fallback, exactly as082. Construct076 A→B from
the verified complete G021 A-anchor and complete076 non021 A-teacher. All their
hardness-training sources must equal A and exclude B,V. Their cached frozen
representations retain the limitations below.

For each new image arm, calibrate one additional coefficient on eligible noisy B:
v = clip(sum((a_B−P082_B)*(y_B−P082_B))/sum((a_B−P082_B)²),0,.25).
Use v unchanged after refitting the image model on T; evaluate
P082_V +v*(a_V−P082_V) on eligible noisy V only. Other rows remain082 exactly.
The old coefficient and v both use B as calibration-training data; B's fitted
score is not claimed as validation of their whole composition. The entire
calibration procedure is evaluated on V. No old full-OOF residual re-splitting,
outer coefficient fitting, seed choice, posthoc routing or spread correction.

Save selector predictions on B and V as well as refit predictions on V. These
permit a fixed-A student diagnostic without pretending comparisons of teachers
with different training counts/compositions isolate training-size causality.

Primary comparisons: REL vs082 and REL vs matchedMSE. SHUFFLED is a required
mechanism control. Report076 separately and retain082's exact conditional
reference9.2567387466 before new changes. Source-restricted076 is9.4924399566;
these are inherited conditional OOF values, not public scores.

## Evidence and stopping

Report all arms, image-only RMSE, full-ensemble/high-noise/fold RMSE, calibration
coefficients, paired10000-draw image-ID bootstrap (seed2026101652), Bonferroni
intervals for the two primary comparisons, top5/10/20 contributor removals,
paired failure cases, and affine spread decomposition (diagnostic only).
Also inspect embedding variance/effective rank and gradient contribution to
verify that the objective is active and does not collapse. Pairs and views are
not independent observations.

Advance REL only if it gains≥.03 overall RMSE vs082, improves≥3/5 folds, both
primary conditional95% lower bounds exceed0, improvement vs082 remains positive
after removing its top10 contributors, and it beatsSHUFFLED. If passed, repeat
all three arms at seed+100000, use equal seed means and recompute coefficients
on B. No seed winner is selected. If failed, stop this objective; no temperature,
bandwidth, weight, epoch or architecture search follows.

A passing repetition is still not independent data confirmation. Further
full500 training/candidate construction requires the evidence to survive these
controls and a separately locked confirmation protocol; do not designate a new
official best or automatically submit. If no arm passes, preserve082/076 and
report the negative result.

## Limits and verification

All500 hardness labels have prior exposure; no untouched holdout exists. New
image-learning and calibration are nested, but frozen082's auxiliary image
representations and historical model choices were not rebuilt from scratch
inside every fold. Thus the primary comparison is honest about new fitting and
conditional on the inherited baseline, not a fresh full-pipeline generalization
estimate. CPU/MPS precision is explicit; new training/inference use CPU float32.

Verify raw train IDs/dimensions, source grouping, exact calibration reconstruction,
same-arm initial states/exposures, checkpoint reload, independently implemented
KL math, active encoder gradients, an actual outer-label-poison training replay,
and final protected hashes. Reassemble082/076 into this new directory; verify
the full500 saved CNN correction without modifying or retraining incumbents.
