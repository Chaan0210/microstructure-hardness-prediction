# CNN as a complement to the current tabular representation

User asks whether 021/all components are optimal and whether a new CNN should
learn what tabular models miss. No global optimum has been established.

021 is .9*018 + .1*DINOv2-L MLP; 018 already contains CNN, DINO-S/B and classical
calibration/residual components. Its weight in 061/073 is .16, not the weight of
a single tiny CNN. The other agent's EfficientNetV2-S raw-y experiments reached
14.40 vs 021 12.81 on original400 and worsened same-weight replacement ~.139.
These bounded tests do not prove all backbones/training methods inferior.

Prior residual-CNN219 really ran and failed: an A213-trained 021 anchor and
107 residual labels, 80 epochs, full-strength correction. RMSE 13.371 ->14.326,
correction variance 36.59 vs aligned MSE benefit 10.14. This is NOT an untried
generic residual-CNN idea. The new test changes the anchor to modern tabular,
uses honest cross-fitted residuals on all 400 outer-train rows, conservative
fixed correction strength, and explicit context/auxiliary-task controls.

## Isolation and validation

Write only this new directory. No modifications to 061/073, other-agent scripts,
environments or downloaded weights. CPU two threads only, no MPS/GPU conflict,
no new external data/weights/downloads and no test reads or Public tuning.
Reuse round2's exact all500 partition seed2026101501 and its 5 outer/15 inner
tabular models/predictions. Each outer training row has an inner-OOF prediction
from a model trained WITHOUT that row and WITHOUT any outer validation source.
Target = y - that inner prediction. Outer validation prediction uses the matched
tabular model trained on outer train400. The inner anchor used ~267 originals,
so the training-size mismatch remains a limitation, not concealed leakage.
No old full-OOF residual re-splitting, and no in-sample tabular predictions.

This economical benchmark's tabular branch omits 021 and the grain-set model;
it is not a new complete 073 score. If promising, confirm on the other two new
partitions/seeds and the actual current ensemble before producing a candidate.

## Fixed architecture and matched arms

Whole native 256px grayscale images, no crop/scale/intensity augmentation.
Batch D4 augmentation and 4-view inference. First learned 7x7/stride4 convolution
retains pixel texture before reducing spatial size; subsequent 3x3/stride2 stages
have channels16/32/48/64 with GroupNorm/SiLU. Concatenate mean/std at every stage
and input mean/std. MLP64/dropout.25 readout; correction output zero-initialized.
This is a small new image branch, not an attempted unverified larger backbone.

Context = tabular OOF prediction, tabular-minus-Ridge disagreement, noise,
u_ph, log1p(n_obj), coverage. First two must come from honest inner predictions
at training and outer predictions at validation. Scaling is outer-train only.

Arms share initial state, image normalization, target scale, seed, batch/D4 order,
optimizer and update count:
A direct: predict original y, image only; evaluate .84*T + .16*A.
B residual: image-only correction of T.
C residual-context: image plus six context values.
D residual-context-aux: C plus an image-only original-y auxiliary head, loss
  residual MSE + .2*auxiliary original-y MSE. This tests whether strong original
  label signal helps learn features when residual labels are mostly noise.
E context-only: same correction head with image features fixed at zero; no image
  encoder or auxiliary updates. Controls whether any benefit actually needs images.

For B-E, prediction = T + .25*correction. Fixed shrinkage comes from the prior
219 variance failure, not new outer scores; full-strength effects are diagnostic
only. No early stopping or outer tuning. 40 epochs, batch32, AdamW lr.002,
weight_decay .01, cosine to .0001, clip norm5, last5 epoch weight average.
Use original-y training SD for all arms (not amplified residual SD). Same per-fold
seed2026101501+k. Failures use the common tabular fallback. Save all models,
normalization, exact train/validation IDs, curves and predictions.

First run a discarded CPU speed/finite-gradient probe. If projected cost is
reasonable, complete all five folds. Advance only if candidate beats T and the
context-only control in pooled RMSE, improves >=3 folds, and its improvement
exceeds conditional bootstrap SD. Otherwise report the failure and scope, without
calling 021 or the overall architecture optimal. A new training strategy has to
earn replacement by final-ensemble RMSE and residual complementarity.
