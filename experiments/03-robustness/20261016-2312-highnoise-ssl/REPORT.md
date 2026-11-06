# High-noise local-view SSL — 16–17 October 2026

**Retain082; keep076 as rollback. This bounded self-supervised recipe did not produce a defensible improvement.** The complete fixed five-fold comparison ran. No outer result selected a new strength, objective weight, seed, checkpoint, head, gate or blend rule. No new test candidate was fitted, submitted, uploaded or promoted.

082 remains the verified best recorded public submission at **8.9343098075**, versus076 at8.9646534604. The following scores are conditional development OOF scores on the same400 sources, not public/private scores or forecasts.

## What was tested, and why it was different

The failed hardness-neighborhood study pulled embeddings of **different images** together using their hardness similarities while simultaneously optimizing hardness MSE. This experiment instead pretrained from **two views of the same source**, without hardness, teacher or residual targets in the SSL loss. Its VICReg-style loss combined embedding agreement, a per-dimension variance floor and an off-diagonal covariance penalty. The mathematical template is [Bardes et al., VICReg](https://arxiv.org/abs/2105.04906); the exact, fixed coefficients and implementation are documented in PLAN.md and study.py. This is a small custom implementation, not a claim to reproduce the paper's large-scale recipe or results.

The encoder retained the preceding compact architecture to isolate the objective: native256 grayscale, four convolutional stages16/32/48/64, multiscale spatial means/SDs322->64->32, scalar hardness head, and a32->64->32 SSL projector. It received original pixels, no baseline measurements or pretrained weights. Inference used four fixed D4 views of original images. SSL pretraining and supervised fine-tuning each lasted24 epochs, with final-four-state averaging.

Prior-work checks found no earlier run of this same-source variance/covariance encoder objective. The audit included the failed hardness-neighborhood implementation, scalar ranking120 on frozen ViT features, input-noise110, Noise2Self preflight102, U-Net pseudo-mask/measurement training including boundary-erasure u1b, and the active degradation-copy pilot. Noise2Self was not repeated because its earlier signal/noise assumptions failed screening. Pseudo-mask reconstruction and degradation-copy supervised training were not repeated. These are distinct mechanisms, not three independent successful validations.

## Transformations: permission did not imply information preservation

All new views came from existing training images. Local image analysis selected interior proxies using smoothed gradients, intensity quantiles and4x4 spatial regions. A random subset received local contrast contraction and a small brightness offset around the source region's mean. High-gradient and extreme-intensity proxies were excluded and the mask was feathered. **These are heuristic proxies, not verified grain or pore annotations.** Actual model inputs remained float-valued; no new microstructures, external images, global noise augmentation or test images entered research fitting.

The predeclared screen independently operated inside each A213/T320 training partition. It compared fixed45-dimensional image descriptors and source-grouped3fold Ridge(alpha100) hardness decoding before/after transformation. The noninferiority bounds, descriptor correlation/drift limits and edge-membership threshold were fixed before scoring. The ten partitions overlap heavily; ten passes are not ten independent confirmations.

| Transform | Partitions passing all checks | Worst overall probe-harm upper90% bound | Worst high-noise bound | Minimum partition edge-Jaccard5th percentile |
|---|---:|---:|---:|---:|
| Local contrast strength.03 |10/10|0.0581|0.0897|0.9971|
| Local contrast strength.10 |10/10|0.1077|0.1601|0.9906|
| Local contrast strength.30 |10/10|0.1052|0.1623|0.9731|
| Boundary blend.05 |10/10|0.0329|0.0405|0.9804|
| Boundary blend.15 |0/10|0.0518|0.0810|0.9413|
| Boundary blend.40 |0/10|0.0308|0.0882|0.8477|

Positive probe harm means worse RMSE. Boundary pixels were blended only with surrounding pixels from the **same source image**. The stronger boundary variants failed the predeclared0.95 edge threshold even though the cheap label probe would have accepted them. A gradient edge can also be noise, so this does not prove actual hardness information was erased. It demonstrates why the label probe alone is inadequate to certify safety.

Only the predeclared weakest photometric setting(.03) entered training. All stronger and boundary variants were diagnostic; no boundary invariance was forced. Actual one-view perturbation RMS across400 sources had median **0.0759 gray levels on the0–255 scale**, 5th–95th percentile0.0462–0.1157. High-noise median was0.0848. Thus this was a deliberately weak nuisance perturbation. The screen neither establishes exact hardness invariance nor tests every stochastic view. There is no D4-only SSL arm, so even a positive SSL result could not uniquely attribute benefit to the local photometric component.

Perturbation strength was measured on the original low/high-noise source groups without using hardness labels or prediction errors.

## Nested fitting and matched controls

For each of the five original outer folds, the selector encoder, all image/target normalization, strength screen, supervised head and frozen-feature Ridge used **A213 only**. Predictions on B107 fitted one coefficient on eligible noise>=12 rows:

`w = clip(sum((pImage-p082)*(y-p082))/sum((pImage-p082)^2), 0, .25)`.

Fresh models and screens fitted on T320=A+B then predicted V80. That frozen B coefficient was applied only to eligible noisy V rows; the other rows retained082. All views stayed with their source. Frozen-feature standardization and Ridge were also trained only on A or T. The existing source-restricted082 B anchors and outer OOF predictions were reused; production ensemble coefficients were never retrofitted to OOF.

| Arm | First24-epoch phase | Second24-epoch phase |
|---|---|---|
| N: no pretraining |—|D4 supervised MSE|
| A: ordinary augmentation |—|D4+local-photo supervised MSE|
| S: SSL primary |Same-source label-free SSL|Identical supervised phase to A|
| L: equal-update control |D4+local-photo supervised MSE|Same supervised phase/reset schedule as S|

All shared initializations, image orders, augmentation streams and source/view exposures were verified exactly where matched. The L/S first phases had identical views and optimizer schedules; only their objectives differed. Frozen SSL and random-encoder Ridge heads were secondary ablations, not an outer-selected replacement candidate.

**Validation limit:** all500 labels were previously explored, and the400 evaluated labels are reused development data. New encoder/safety/head/blend fitting is nested; the inherited082 representations and its historical selection process were not trained afresh within these folds. This is conditional validation of the new branch against a frozen incumbent. Paired source-ID bootstrap intervals do not include encoder-refitting, seed, historical search or private-set uncertainty.

## Complete measured results

400 outer predictions; high-noise subset124. Positive gain means lower RMSE than082. Intervals are10,000 paired source bootstrap draws, conditional on the fitted predictors.

| Method combined by the fixed inner rule | Overall RMSE | High-noise RMSE | Gain vs082 | Conditional95% gain interval |
|---|---:|---:|---:|---|
|076 rollback reference|9.492440|12.111669|—|—|
|**082 reference**|**9.256739**|**11.508128**|—|—|
|N: no pretraining|9.391437|11.854951|−0.134698|[−0.249597,−0.034286]|
|A: augmentation only|9.391366|11.854769|−0.134627|[−0.249653,−0.034434]|
|**S: SSL primary**|**9.305011**|**11.633030**|**−0.048272**|**[−0.090386,−0.012245]**|
|L: equal updates|9.256739|11.508128|0|[0,0]|
|Frozen SSL+Ridge|9.314503|11.657510|−0.057765|[−0.117397,−0.008506]|
|Frozen random+Ridge|9.349516|11.747575|−0.092777|[−0.174701,−0.024744]|

L tied082 because every inner coefficient was zero, **not** because its image predictions equaled082. Its standalone image RMSE was14.7323 (high17.5702), versus SSL15.2189 (high17.7804) and augmentation15.0704 (high17.7332). No new image branch earned a robust positive contribution.

SSL's high-noise gain was **−0.124902**, conditional95% interval[−0.229137,−0.034620]. Against082 its gain after removing the ten largest helping samples was **−0.060164**. SSL beat A by0.086355, interval[0.019123,0.162639], but the ten largest contributors accounted for110.2% of that net SSE advantage; removing them reversed it to **−0.009432**. That is reduced harm relative to a poor control, not an improvement over the incumbent.

In fold4, A hit the blend cap0.25 while S received0.0913, despite S's worse standalone image RMSE. Thus the ensemble comparison against A cannot be presented as proof that SSL learned better image features. The full matched coefficient and image-error records are retained in results.json.

## Diagnosis, with causal limits

**The representation provided no useful additional hardness signal with the tested heads.** Frozen SSL's standalone Ridge RMSE was **17.8717**, worse than matched random features at15.9337. Its mean effective rank was1.64, versus random2.71; it was nonconstant but concentrated in a small number of directions. Fine-tuning raised neither standalone performance above the equal-update control nor a stable contribution over082. The training objective decreased substantially, so this was not simply an inactive loss or an inference wiring failure. Low rank alone is not proof of information loss or a universal SSL failure; the poorer held-out decoding is the relevant accompanying evidence. A different readout or regularization could expose other information, but was not searched after these results.

**The tested noise-shortcut explanation was not supported.** A training-only fixed Ridge readout decoded the image-derived noise proxy from frozen random features with outer RMSE2.2943/correlation0.8709, but from SSL features with RMSE4.5233/correlation0.0804. Noise became much less linearly decodable, alongside poorer hardness decoding. This does not prove all noise information vanished, nor that nuisance suppression caused the hardness loss. It rules out presenting improved noise invariance as sufficient evidence of a useful representation.

**Small calibration samples remained unstable.** SSL weights by fold were[0.010944,0,0,0,0.091313]. Overall-fold gains were[+0.001173,0,0,0,−0.232015]. Fold4 had24 noisy calibration sources and26 noisy validation sources; its conditional coefficient interval covered[0,.25]. Fold0's interval also covered[0,.25]. These coefficient intervals hold the already-selected082 weight fixed and exclude its uncertainty. The longer supervised control was rejected in every fold. A more conservative route would avoid this harm, but no new gate was tuned after seeing it.

**Training-size transfer shifted predictions, but did not explain away the failure.** On the same V80, the fold4 new image model changed by8.46 RMS units between A213 and T320 fits. Nevertheless, retaining the A-trained new branch still yielded overall9.299321, gain−0.042582, interval[−0.087237,−0.005099]. This diagnostic changes only the new branch while082 remains refitted; it is not a matched-size causal study of the entire historical pipeline.

**The loss was not solely a global spread effect.** A posthoc prediction-only affine decomposition gave an affine-only overall gain−0.006920 and orthogonal-only gain−0.041383. These are diagnostic counterfactual components, not fitted repairs; RMSE component effects are not additive. The worsening persists beyond a global offset/spread change.

These are all fold4 failures. At both high and low target extremes, the added image predictions moved an already-biased082 prediction farther toward the center. Full per-image evidence is in per-image.csv, including all improvements as well as failures.

The remaining bottleneck is **unproven recovery and retention of hardness-relevant structure from noisy images**, coupled with uncertain blend selection on small noisy calibration sets. This objective/encoder/strength/budget combination did not solve it. The very weak perturbation and short, single-seed pretraining limit the rejection's scope. Neither shared error nor this failed decoder establishes irreducible image ambiguity. Stronger transformations passed these cheap photometric screens, but their task-relevant information preservation and a different objective would require separate prospective evidence; they were not searched here.

## Budget, coordination and conservation

The fixed matrix completed40 final image models, ten SSL checkpoints,20 frozen-feature heads and210 cheap safety-probe Ridge fits. The60 neural phases contained24 epochs each, totaling383,760 source visits and767,520 view visits; views were never counted as independent validation samples. Main neural training took2701.69 seconds(45.03 minutes), with peak RSS916.55MiB. The initial45-minute estimate was amended once to55 minutes including replay, **before opening outer scores**, based solely on the larger refits' timing. No extra seed/variant was added; no resumption was needed. The original plan's phrase“1440 source-epochs” denotes summed epoch counters, not1440 independent sources.

The detached degradation-copy pilot's handover and running jobs were inspected. This experiment used one sequential CPU training process/two threads, no MPS, no pilot copies or feature outputs, and no writes or interventions in that pilot. No corresponding active AI pilot chat existed, so there was no bidirectional message; COORDINATION.md records the actual scope. Timings are observations on a shared machine, not controlled throughput benchmarks.

All40 final models and20 frozen-feature heads reloaded exactly. Training-source normalizations, initializations, matched view/order streams,30 independently recomputed blend weights, outer-label-poison coefficient checks and independent scalar RMSE calculations passed. A full48-epoch SSL/fine-tuning replay with all287 outside-subset labels poisoned reproduced both training states and predictions exactly, while dataset reads were blocked during the fit. Total neural time including replay was2774.21 seconds(**46.24 minutes**), below the amended55-minute cap.

**All151,006 protected files, Git head and tracked diffs remained unchanged. Both incumbent CSVs were reproduced byte-for-byte, maximum numerical difference0.**076 was reassembled from its frozen component caches;082's two saved CNNs were freshly inferred on its318 deployed high-noise test images for verification only. No fitting or new-candidate selection used test pixels/predictions. This verifies saved-artifact inference/assembly, not a complete rerun of historical training. Full closure evidence is in COMPLETION.md, preservation.json, training-replay.json and reproduction/082-inference-verification.json.

Exact commands and clean-rerun instructions: **RUN.md**. Protocol: **PLAN.md**, **BUDGET-AMENDMENT.md** and code locks. Baseline provenance: **incumbent-provenance/**, **starting-state.json**, **protected.json**. Predictions and statistics: **oof.npz**, **results.json**, **comparison.csv**, **per-image.csv**, **diagnostics.json**. No package installation, external training data, paid compute or submission was used.
