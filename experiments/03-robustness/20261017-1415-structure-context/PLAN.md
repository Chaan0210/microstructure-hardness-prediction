# Masked structural context learning — prospective protocol,17 October2026

Goal: obtain useful high-noise structural information that complements082.082 remains the recorded public best8.9343098075 and076 the rollback8.9646534604. Preserve both complete pipelines and every prior study. All new code, weights, logs and predictions live here. No installs, external training data, generated microstructures, test training, pseudo-labels, submission, paid resources or Git mutations.

## Audit and ranked mechanisms

1. **Learn raw-image spatial context before hardness fitting.** Highest priority: a scratch global-attention encoder predicts hidden image structure, then an existing nonlinear physical regressor tests its conditional value. It can retain relationships across the entire256px field, beyond the preceding small CNN's63px receptive field and global moment pooling. Dense within-image supervision uses supplied pixels, not new independent samples. Cost:40 encoder fits plus paired tabular readouts. Falsification: no useful masked-structure prediction, no advantage over shuffled topology/random features, or no robust nested082 gain.
2. **Learned grain graphs/segmentation-affinity supervision.** Potentially physical, but lower priority: old K6 grain-graph message passing over measured objects was highly correlated with the existing grain-set model and failed; uncertain grain masks could become false labels. Do not repeat or treat those masks as truth. A future affinity objective would first need fold-local reliable targets.
3. **Masked-pixel denoising or stronger degradation invariance.** Lower priority: Noise2Self102 failed its signal/noise preflight, U-Net boundary-erasure and degradation-copy studies failed transfer. This run does not assume iid noise or that destructive transformations preserve useful evidence. No new corruption generator is fitted.

Distinct from prior token work: push1000 used frozen U-Net maps or frozen DINO token projections with pointwise/conditioned set readers; spatial161 learned local convolutions on frozen DINO grids. Here the native pixel mapping AND global interactions are trained within each partition. Distinct from the previous hardness-neighborhood loss: no hardness-based cross-image neighbor graph. Distinct from the recent VICReg run: predict observed missing structure rather than force nearly identical views into matching low-rank vectors; do not collapse the encoder through scalar-hardness fine-tuning.

Conceptual source: He et al., https://arxiv.org/abs/2111.06377 . This custom dense mask-token model is inspired by masked prediction, not a reproduction of MAE's asymmetric/visible-only architecture or its external-data results. No paper weights/data are used.

## Encoder and objective, fixed before scores

Native256 grayscale -> nonoverlapping8x8 pixel vectors -> shared Linear64->64/GELU -> neighboring2x2 token concatenation -> Linear256->96. This yields16x16 macro tokens covering16px each. Four global Transformer blocks, width96, four heads, FFN192, pre-LayerNorm, dropout.1, fixed2D sine/cosine position codes, learned CLS and mask tokens. No CNN feature extractor, U-Net features, DINO tokens or external encoder weights.

Mask40% of the8x8 coarse block grid, expanding each to2x2 macro tokens (32x32 raw pixels). Replace hidden token embeddings internally; retain original images/targets. Training D4 is shared across matched arms. The decoder predicts each hidden macro patch's4x4 grid of4x4 pixel means. Loss = DC MSE +4*within-patch centered-field MSE +2*within-patch horizontal/vertical finite-difference MSE. Input normalization uses only the current encoder training images. Target DC/AC are observed image quantities, not claimed noise-free or ground-truth grain labels. Fine boundaries can be diluted in this target; native pixels remain encoder inputs. No hardness-invariance claim is made for masked inputs, and hardness inference uses complete original images.

AdamW lr.001, weight_decay.02,100 epochs,5-epoch linear warmup then cosine to.00005, gradient norm cap5, batch16, final5 epoch state mean. Two predeclared seeds2026101771 and2026111771, plus100*outer_fold. Same seed for selector/refit, fresh models. Encoder fits see no hardness values. D4/mask/order/dropout streams match between spatial and shuffled arms. All model math float32, no autocast.

Arms: S=trained true spatial layout; P=same architecture/objective with a fixed source-specific macro-token permutation (seed271828+source index), retaining every local16x16 patch but destroying its arrangement; R=the matched initial random encoder without pretraining. P permutes targets identically; within-patch DC/AC/gradient target distributions are retained. It is a topology control, not proof that every form of positional information disappears. No arm can be tuned after outer errors.

## Conditional downstream learning, not a weak-image-model replacement

Use the current physical bank u1dmavgH2Z98 columns plus the incumbent H9 cumulants (columns1,4 of cumul-train.npz),100 columns total. Encoder descriptors are CLS96 + mean96 + populationSD96 over contextualized tokens, averaged over four D4 views. For each encoder/partition, StandardScaler and PCA16 fit ONLY its training-source descriptors. No hardness-based PCA or feature selection.

Existing offline TabPFNv3.5, explicitfloat32, memory_saving_mode, seeds100*j+outer_fold for j=0,1,2. Fit physical-only and physical+16 learned coordinates on exactly the same A or T rows. Average the three Tab seeds; average predictions from both encoder seeds. The correction direction is **augmented Tab minus matched physical-only Tab**. This preserves082 and avoids replacing it with a weaker standalone image regressor. Both sides of each difference have identical training source counts and original hardness targets; no inconsistent residual teacher is introduced. R and P receive the same PCA dimension and downstream procedure.

## Honest new-branch nesting

Use the existing400-source original five-fold graph: A213/B107 inside T320, outer V80. All new image normalization, encoder fitting, PCA/scaling and Tab models use A only for B prediction, then fresh T-only refits for V. Every source and derivative stays together. The existing source-restricted082 B anchors/outer OOF are immutable; no production coefficient is applied to OOF. Fit a single new high-noise coefficient on eligible B rows only:

`beta = clip(sum(delta*(y-p082)) / (sum(delta^2)+n_high*1.0^2), 0, .5)`.

Apply `p082 + beta*delta` to eligible high-noise V rows; retain082 elsewhere. This fixed ridge penalty shrinks uncertain corrections without an outer-tuned gate. Same rule separately for S/P/R. Query labels are not available to encoders/readout fitting. Save selector predictions on B and V and refit predictions on V for training-size diagnostics; both physical-only and augmented readouts have matched source counts at each stage.

Primary candidate is S. R/P are controls, not alternative winners selected on outer scores. Report image-derived structure reconstruction on B/V against the zero-AC/per-image-DC reference, including high-noise cases, as diagnostic only. Do not add a label-based epoch/architecture search. Primary uncertainty unit is source ID, not patch, seed or view.

## Budget, verification and decision

First run a discarded40-step implementation/resource probe using only fold0 A images and no hardness. Freeze main code after runtime fixes and before outer scoring. Local MPS, one GPU process at a time, CPU2 threads. Cap RSS+MPS driver allocation5GiB and MPS driver3.5GiB; no unrelated jobs are stopped. Planned compute cap90 minutes for encoder training/readouts, plus validation/provenance; at most150 minutes active work. If timing projects beyond cap, stop before the matrix rather than silently shorten epochs. No extra seeds/width/strength sweeps after scores. Any invalid loss, source-isolation failure or protected change stops work.

Complete all five folds before comparing hardness accuracy. Report overall400/high-noise124 RMSE, paired10000-source-bootstrap95% intervals, both individual encoder seeds, fold gains, beta uncertainty, top5/top10 contributor removal, frozen random/permuted ablations, affine/spread decomposition and matched-size delta transfer. The inherited082 auxiliary representations and all500 labels have historical exposure; this is honest nesting of the new pipeline conditional on that incumbent, not an untouched whole-pipeline/private confirmation.

S earns a candidate recommendation only if overall gain>=.05, high-noise gain>0, improvement in>=3/5 folds,95% lower bounds positive versus082 and BOTH controls, and gain remains positive after top10 helping-source removal. Both encoder seeds must give nonnegative overall gain under their own B-only calibration. Failure closes this recipe; no posthoc branch selection or public-score fitting.

If all criteria pass, fit S encoders/PCA/paired Tab readouts on all500 supplied sources with the same settings, production beta=mean of the five inner-selected S betas fixed before any test inference, and produce a local reviewable candidate file. Do not submit, replace082/076 or designate an official best. Any final inference is separate from training and cannot alter fitted settings.

Verify training/source hashes, all saved checkpoints/features, independent loss/selection math, target-poison isolation and controls. MPS repeatability is measured rather than assumed; compare CPU/MPS inference with a justified1e-3 hardness-unit tolerance, and report measured differences. Preserve the incumbent artifact inventory and previous exact inference evidence; reproduce saved incumbent predictions in a separate reproduction directory if final inference is performed. Hash every protected file again at closure.
