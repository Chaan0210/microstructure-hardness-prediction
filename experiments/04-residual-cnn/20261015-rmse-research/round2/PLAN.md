# Continued research: strict source validation, 2026-10-15

Requested continuation after the first four hypotheses failed. Preserve all
prior results and 061/073. Write only round2. CPU two threads, no CNN/DINO work,
no installations or new external weights. Do not fit anything on test or Public.

## Corrected reference

Use ../inputs.npz and ../scope.npz: 073 actual component assembly with every
augmented source restricted to that outer fold's actual 320 training IDs.
The strict reference is 9.5816850901, a different evaluation recipe from the
historical 9.5928855251. It is not a new submitted model. No held-out labels or
OOF residuals serve as training labels for any added model.

Keep the fixed production ensemble weights. Compare exact same training rows,
folds, preprocessing, gates and fallback in candidate and control. Three
historical partitions share 400 IDs. Bootstrap unit = image ID, not OOF row.
All hyperparameters must be chosen inside outer training folds. Augmented
sources must also be grouped at every inner split. Pretrained representations
remain inherited: this is still conditional validation, not a fresh independent
estimate of the entire image-pipeline development history.

## New preregistered hypotheses

H5: evidence-based group shrinkage. An isotropic Ridge prior ignores different
redundancies/reliabilities of size, phase, shape and image-quality measurements.
Fit an empirical-Bayes linear regression with one coefficient precision per
preexisting semantic group. No feature is selected from outer residuals. Compare
with an isotropic evidence model and a fixed permutation of group assignments
of the same sizes. These isolate group information from generic extra flexibility.
The group model uses Gaussian posterior exact solves and damped evidence updates
(max 100 iterations, tolerance 1e-4, fixed weak hyperpriors). Replace only Ridge
(weight 0.128). Existing Ridge GCV remains a separate production control.

H6: low-rank nonlinear physical directions. Prior tests used linear PLS, splines
on every noisy feature, or all pair products of handpicked raw features. Instead,
learn four PLS directions within each training fold and append their ten quadratic
terms to the original features. Inner source-grouped 3-fold CV selects alpha
[0.1,1,10,100] and block scale [0,0.1,1]. Scale zero is the matched control. PLS
itself must be refitted inside every inner split; analytic GCV on a label-derived
design would be invalid. Replace only Ridge at the same fixed weight 0.128.

H7: segmentation-independent higher cumulants for dark-phase measurement.
Previous signed three-point and binary occupancy moments did not estimate
contrast/noise-normalized third/fourth intensity cumulants after local shade
removal and pore suppression. Use existing frozen z1 maps only for soft
matrix reference / pore exclusion, never hardness labels. Fixed Gaussian scales
1,2,4 pixels; estimate signal variance after a Gaussian-white-noise correction,
standardized third/fourth cumulants, and the implied binary-mixture fraction.
Do not assert unbiasedness under spatial dependence, blur or non-Gaussian grain
brightness. The measurement model is a hypothesis to test, not generator truth.
Add the fixed small block to the GMM partner (weight 0.15), not to the already
selected OOF residual. Nested CV of its block weight, with zero and label-blind
random-block controls. All other ensemble components unchanged.

Primary screen: existing 3 x 5 folds; positive pooled gain, nonnegative gain on
each partition and majority of folds improved. Report matched-control effect,
ID-cluster bootstrap and tail concentration, including negative evidence. A tiny
gain or a CI crossing zero is inconclusive. A promising candidate must also be
evaluated with new fixed seeds 2026101501/2026101502/2026101503 on all 500 rows.
That additional benchmark must train its base models honestly from the new fold
training rows; never reuse full500 trained predictions as OOF. CNN components
are excluded from that auxiliary benchmark to avoid other-agent interference;
report this scope difference explicitly.

Sources: existing group definitions push-1300/abl/groups.json; Sparse Bayesian
Learning (Tipping 2001, https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf)
for the precision-learning idea, not a claim to reproduce that paper's model;
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstat.html for
cumulant definitions. Spatial weighted plug-in estimates here are not k-statistic
unbiased estimators. No data or pretrained models are imported from these sources.

## Follow-up registered after H5-H7 screens, before new-fold scores

- H5 group evidence beats isotropic and shuffled grouping, and its standalone
  Ridge RMSE is 9.6900 vs 9.8433. But fixed ensemble replacement loses 0.00694.
  Low-noise direction helps (+0.01138 within band), DTA/noisy direction harms.
  Treat a low-noise-only application as a POST HOC discovery needing new-fold
  validation; do not report it as an independently selected dev success.
- H6 quadratic PLS beats its same-grid null (+0.00705) but loses to production;
  do not advance it unless ensemble-aware selection independently selects it.
- H7 cumulants gain +0.00861 (11/15 folds, all partitions); CI crosses zero. A
  random block gains +0.01542 (5/15 folds, mostly one partition), showing the need
  for additional placebo draws and a separate fold benchmark. This is not yet
  evidence of new physical information.
- H5 group precisions often hit the 100-iteration budget, unlike the isotropic
  control. Before drawing conclusions about an evidence optimum, rerun the
  identical update at a 2000-iteration numerical budget; report convergence
  without changing priors, damping, data or model family. Preserve fixed-100.

New all-500 benchmark: fixed 5-fold seeds 2026101501/2026101502/2026101503.
Train only the actual outer train IDs plus their eligible degradation copies.
Use a matched CPU tabular ensemble, with weights (.384,.128,.128,.15)/.79 for
TabPFN, Ridge, logRidge and GMM. One fixed avg feature bank and one TabPFN seed
are used for this economical validation; original 061/073 uses more banks and
neural components. Report the scope difference, never call this full 073 CV.
Evaluate H5(all), H5(low only), H7 and three fixed random-block controls. No
holdout designation for any train rows and no fitting on predictions from a
model that trained on the current validation labels.

H8: ensemble-aware choice inside each outer fold. Prior screens repeatedly
improve a component's standalone loss but harm its ensemble contribution. Build
inner-OOF predictions of the matched tabular ensemble by retraining EVERYTHING
inside each inner fold (including source-filtered DTA and TabPFN). Choose only
between production and group-evidence Ridge at shrinkage [0,.25,.5,1], separately
for the two existing noise bands, minimizing ensemble MSE. This is a proper
nested selection policy, unlike re-splitting the old full OOF predictions.
The larger all-500 folds are the primary evaluation. If a policy improves, repeat
the unchanged procedure on the two other preregistered partitions before any
submission candidate. Keep every comparison, including model-only controls.
