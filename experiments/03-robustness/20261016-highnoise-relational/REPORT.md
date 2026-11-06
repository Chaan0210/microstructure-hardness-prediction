# High-noise image-learning experiment — 16 October 2026

**Retain 082; keep 076 as rollback. The new image-learning objective did not earn
a candidate.** All 30 planned image-model fits completed, along with matched
controls, checkpoint checks, a complete training replay and incumbent inference.
No second-seed expansion, hyperparameter sweep, full 500 candidate fitting or
submission was triggered after the preregistered gate failed.

082 is now the recorded public best at **8.9343098075**, submitted as 56 on
16 October at 21:25 KST.076's recorded score is 8.9646534604. These are observed
records, not forecasts. No public score or test prediction selected a new
training setting, coefficient, seed or objective in this study.

## What was genuinely different

The new 70,129-parameter CNN learns a **32-dimensional image embedding whose
neighborhoods reflect soft hardness similarities**. It consumes original
256×256 grayscale pixels and original hardness labels. It receives no tabular
context, teacher prediction, residual label, segmentation map or pretrained
image embedding.

Two independent D4 views of each source give embeddings, centered and normalized
within their training batch. The desired neighbor probabilities are
`q(i,j) = softmax_j(−0.5*((y_i−y_j)/(0.5*training_target_SD))²)`.
The image probabilities are `softmax_j(cosine(z_i_view1,z_j_view2)/0.2)`.
Training adds symmetric forward/reverse KL divergence with weight 1 to the
ordinary standardized-hardness MSE of both scalar prediction views.

The soft neighborhood target changes the supervision of the pixel filters.
Study 120's failed ranking experiment instead added scalar pairwise logistic
order constraints to a head on frozen ViT features. Existing 082 learns scalar
residuals against 073 teachers with tabular context and an auxiliary hardness
head. This new experiment repeats neither training problem. It also avoids
introducing another residual-teacher training-size mismatch into the new branch.
Ordinary refit/calibration transfer limitations still remain.

Only D4 rotation/reflection views were used. There were no new degradation
copies, separately generated images, external examples, installations or paid
compute. The label relationships are not additional independent observations.

## Matched controls and nested fitting

All three arms have identical architecture, initialization, two-view exposure,
example/D4/dropout streams, normalization, optimizer,40 epochs and four-view
inference. All learned pixel filters are fresh for each training partition.

| Arm | Scalar targets | Neighborhood targets |
|---|---|---|
| MSE control | Correct hardness | Loss weight 0 |
| REL candidate | Correct hardness | True-label soft neighborhoods |
| SHUFFLED control | Correct hardness | Labels permuted over training sources only |

The shuffled arm keeps paired source identity and the correct scalar MSE while
removing the correct cross-image label relationships. It controls for generic
view/embedding regularization. No architecture or loss setting was chosen from
the outer comparisons.

| Stage within each outer fold | Training sources | Role of excluded data |
|---|---:|---|
| New selector image model | A213 | Predict B107 for calibration; V80 only for later diagnosis |
| Existing 082 coefficient reconstruction | Original A-trained predictors | Select its original coefficient on B only |
| New mixing coefficient | Eligible noisy B rows | One OLS coefficient, bounded to[0,.25] |
| Fresh image-model refit | T320 = A+B | Evaluate on V80, which selects nothing |

The old and new coefficients both use B as **calibration-training data**. Its
fitted score is not advertised as validation of the whole composition. The
complete procedure is evaluated on V. On eligible noisy validation rows,
`candidate = P082 + v*(new_image_prediction − P082)`; all other rows remain 082.

The existing 082 inner selection was recomputed exactly. Complete G021 A-anchors
and 076 non 021 A-teachers were checked for source/query alignment and exclusion.
The inherited baseline's conditional OOF is 9.2567387466, reconstructed exactly;
the matching 076 reference is 9.4924399566. These are not public RMSE values.

Every new mean, SD, image filter, label-neighborhood target and mixing coefficient
uses only its allowed training/calibration partition. Models and source/view
streams are recorded under `models/seed0/`. The fixed seed is 2026101651+100×fold.
Primary training took about 33 minutes, with peak RSS below 650 MiB. There was one
CPU training process with two threads; unrelated jobs were not stopped or edited.

## Results: no added signal usable by 082

Same 400 images, same original five folds. Gain below means 082 RMSE minus candidate
RMSE, so negative is worse. Intervals are paired, image-ID, conditional 95% intervals.

| Method | Overall RMSE | High-noise RMSE | Gain vs 082 | Conditional 95% gain interval |
|---|---:|---:|---:|---|
| Preserved 082 reference |**9.256739**|**11.508128**|—|—|
| MSE image control |9.302842|11.627432|−0.046103|[−0.083529,−0.013373]|
| REL image objective |9.287104|11.586779|−0.030366|[−0.057077,−0.007770]|
| SHUFFLED neighborhoods |9.294085|11.604822|−0.037347|[−0.073460,−0.006633]|

For all three methods, **four folds tied 082 exactly and one worsened**. None
improved any fold. The new mixing weight was zero in folds 0–3. In fold 4 the
weights were 0.084600 for MSE,0.058459 for REL and 0.069915 for SHUFFLED. That fold had
only 24 noisy calibration images and 26 noisy validation images. Each conditional
coefficient bootstrap interval covered the entire permitted[0,.25] range.

REL beat the inferior MSE control by 0.015737 RMSE, conditional 95% interval
[0.001717,0.031377]. This is not evidence of an improvement over 082: all of that
contrast comes from fold 4, where REL simply did less harm. The five largest
contributors explain 99.48% of that relative advantage; removing the largest 10
changes REL-versus-MSE gain to−0.003893. Removing REL's largest 10 contributors
against 082 leaves an even worse gain of−0.037671.

The frozen advancement rule required≥.03 overall gain versus 082, improvement
in≥3 folds, positive lower 95% bounds versus 082 and MSE, a positive top 10-removed
gain versus 082, and superiority toSHUFFLED. It failed. No settings were retuned
and no favorable seed was sought. Full results, simultaneous intervals and
fold-level records are in `results-0.json` and `comparison-0.csv`.

## What the objective learned, and why that did not help RMSE

The loss was active and its embeddings were not constant. Mean effective ranks
were 1.04 for MSE, 1.79 for REL and 2.80 for SHUFFLED, so these remained narrow
representations; the nominal 32 dimensions are not 32 independent physical factors.
On the same excluded
folds, mean neighborhood KL improved from 3.641 for MSE to 2.904 for REL; SHUFFLED
scored 3.065. Mean correlation between embedding distance and label distance was
0.165/0.181/0.075, respectively. These are descriptive geometry checks, not
independent pairwise observations or significance tests.

That geometry improvement came with **worse absolute-hardness prediction**:

| Image learner alone | Overall held-out RMSE | Noisy held-out RMSE | Mean final training RMSE across outer fits |
|---|---:|---:|---:|
| MSE |14.625801|17.584689|13.166472|
| REL |15.565091|18.694876|15.021992|
| SHUFFLED |17.660939|19.881437|17.566605|

REL's noisy errors remained strongly correlated with 082's errors(0.8084).
The measured combination of larger image-model error and inadequate residual
complementarity explains why the inner fits usually selected zero weight.
The neighborhood constraint changed the representation, but did not produce
a better conditional-mean predictor in this fixed recipe. Higher neighborhood
agreement is therefore not a proxy for improved hardness RMSE.

The early training-only gradient probe also found a much larger encoder gradient
from the relational term than from scalar MSE. This is consistent with a loss-
allocation/absolute-accuracy tradeoff, but one early probe does not establish
the gradient balance throughout training or prove that another weight would help.
The predeclared no-sweep rule was honored.

The failure is not just global spread: REL's total loss of 0.56310 MSE units
decomposes into 0.28117 from the high-noise affine change and 0.28193 from the
orthogonal remainder. The latter still worsens RMSE by 0.015216. These are
diagnostic projections of OOF changes, not deployable corrections.

Keeping the new student fixed at its A213 fit also failed: REL's gain remained
−0.031437, compared with−0.030366 after T320 refitting. Thus new-student refitting
alone is not a sufficient explanation. The inherited baseline still follows
its own refit procedure, so this does not isolate the training-size effect of
the entire ensemble.

## Failure cases

The largest REL harms occurred in its sole active fold. The predictions move
further toward the center even where 082 already underpredicts a hard sample
or overpredicts a soft one. Selection here is post-evaluation explanation only.

`per-image-0.csv` contains every ID, target, prediction and paired SSE change.
The separate `baseline-error-panel.json` fixes baseline-only error examples
before the new candidate scores were inspected.

## Verification, preservation and limits

- All 30 saved checkpoints reproduce predictions and embeddings exactly.
- Matched arms have identical initial states, source orders, D4 draws, updates
  and view counts. A separate NumPy KL implementation agrees within 4.43e-7.
- A fresh 40-epoch REL selector was trained after changing all 287 labels outside
  its 213-source training set. Dataset/model-file reads were blocked during that
  fit. Every tensor state and all query predictions remained exactly identical.
- An independent scalar RMSE calculation and coefficient recomputation reproduce
  every reported ensemble result. Query labels never enter the new trainer.
- 076 was reassembled from frozen component caches, then 082's two saved CNNs
  were inferred on the 318 deployed original test images. Both 1000-row CSVs are
  byte-identical to the protected incumbents; prediction difference is 0.
- Final protection verification: **all 150,824 protected files are unchanged**; Git HEAD, tracked diff and staged diff also remain unchanged.

082 SHA256:`0fa34cec1626b0887d3d7971197207c135a447a8f4805096b258e77b912f0e9f`.
076 SHA256:`b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0`.
This is saved-artifact inference/assembly reproduction, not a rerun of the
entire historical training pipeline. `RUN.md` provides exact new experiment
commands; `starting-state.json` and `protected.json` record provenance.

**No genuinely untouched labels remain.** All 500 labels have historical exposure.
The new pixel-learning and calibration stages are nested and exclude V, while
082's old auxiliary representations and historical model choices are frozen
shared inputs. Bootstrap intervals condition on the fitted models and do not
remove past selection bias or overlap between training folds. These results
do not estimate a new private-leaderboard score.

The remaining high-noise error is real:124 of 400 images still account for 47.9%
of 082's squared error. This experiment shows that the tested relational objective
does not recover useful complementary signal. It does not prove that the images
contain no further learnable information, or that every relational objective,
encoder or training strategy would fail. No new official best is designated.
