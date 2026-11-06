# RMSE research — 2026-10-15, isolated AI experiment

Scope: read existing artifacts; write only in this directory. Do not change
push-1300, the active cnn_v2 training, the environments, existing model files,
submission CSVs, or the main README. CPU only, two BLAS threads. No new weights,
external data, test fitting, leaderboard fitting, or submission upload.

## Evidence before any new fit

- Competition: https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai
  read 2026-10-15. 256 x 256 synthetic microstructure -> hardness; RMSE in original
  units, Public 30% / Private 70%. Train 500 / test 1000. External data and any
  test learning are forbidden. Train-image augmentation is documented as allowed
  in the existing reproduction notes. New experiments use only existing train
  features and existing train-derived degradation pairs.
- 061 Public 8.9831248984; 073 Public 8.9863906091. These scores do not choose any
  parameter in this study. Historical weights are frozen as inherited choices.
- Existing investigations already reject generic feature expansions, splines,
  target smearing, feature reconstruction, flexible DTA learners, extra ensemble
  seeds, broad reweighting, and generic residual learners. Do not rerun them.
- Another agent is running cnn_v2.py in push-1300. No neural-network experiments.
- Repeated development partitions share the same 400 IDs, not 1200 independent
  observations. The historical 100-row holdout has been inspected repeatedly;
  it cannot be called a fresh independent test.

## Baseline audit

1. Hash protected best CSVs and input artifacts; verify row alignment and labels.
2. Rebuild 061 and 073, retaining the documented 073 delta-transport OOF only as
   a historical number; also evaluate the actual nb GMM component directly.
3. Audit augmentation source exclusion and historical holdout participation.
4. Record fold RMSE, residual covariance, noise/target strata, and tail SSE.

## Preregistered hypotheses

H1: partially pooled domain coefficients. A shared ridge relation plus shrunk
domain deviations can retain useful labels from other imaging regimes without
forcing identical measurement-to-hardness slopes. Unlike previous noise products
and clean-only contexts, use categorical clean/noisy/synthetic domains, a shared
coefficient prior, and source-grouped inner validation scored only on real rows.
First screen: the standard Ridge component, one feature bank, 3 x 5 outer folds.
Compare a same-grid pooled control and the inherited production control. Tune
only alpha [0.1, 1, 10, 100] and deviation scale [0, 0.1, 0.3, 1] inside folds.
For low-noise predictions retain 073's 85 columns; for high noise retain all 98.
Do not change blend weights (Ridge=0.128). Domain scale zero is the null model.

H2: paired consistency regularization. Penalize the squared prediction difference
between a clean source and its degradation, rather than assuming each degradation
provides an independent label. This can suppress measurement-specific directions.
Use source-grouped CV and penalty [0, 0.1, 1, 10]; choose inside training only.
The zero penalty is the control. Same alpha grid and fixed blend weight.
Compare with H1 separately; do not automatically combine them.

Outer folds: existing Study900 400 IDs x original/20261031/20261101 partitions.
For historical comparisons, no augmentation source may belong to the outer
validation set. Also run a strict source-only sensitivity if any improvement
appears: augmented sources must belong to the actual outer training set.
Any promising direction receives additional validation on new folds across all
500 training rows, with identical training procedure for candidate and control;
this reduces split dependence but does not erase accumulated model-selection bias.

## Decision and uncertainty

Primary: original-unit paired MSE/RMSE at the fixed component weight; standalone
head scores are diagnostic. Report all 15 fold scores and all three partitions.
Bootstrap paired per-ID squared errors (10000 draws, seed 20261015), preserving
repeated partitions. Report top-five/top-ten gain concentration and slope-only
contribution without using these diagnostics to tune a prediction.
Advance only if pooled gain is positive, every partition has nonnegative gain,
and most folds improve. An interval spanning zero is inconclusive, not proof.
Stop an unpromising family after its minimal controlled comparison. If both
fail, preserve 061/073 and report the measured limits rather than fabricate gain.

All code, seeds, selected parameters, provenance, predictions, and failures remain
in this directory. Model reload and validation-label poisoning checks are required
for any adopted model. A final submission must use the same verified procedure;
never create or overwrite a numbered candidate in another agent's namespace.

## Adaptive next hypothesis (after H1/H2, before its evaluation)

H1: 9.613929 vs actual-component 073 9.592886, rejected. H2: 9.592526,
but 9.589310 for the matched zero-penalty control; consistency itself worsens all
three partitions. Rejected. The slight alpha-selection gain is below 0.004 and
does not justify another parameter search.

H3: domain-specific nuisance adjustment, only for the low-noise Ridge head.
H1 frees all 85/98 slopes and adds too much variance. 073 removes 13 columns
from every training row although their correlation with hardness is known to
differ by imaging regime. Keep the shared 85-column model and give only noisy
training rows the extra 13 nuisance columns; their columns are zero at clean
inference. This can avoid omitted-variable bias in shared coefficients without
discarding noisy labels or making every physical slope domain-dependent.
Candidate scales [0, 0.3, 1] and alpha [0.1, 1, 10, 100] chosen with real-only
source-grouped inner CV, scoring noise<12. Same 0.128 weight, same 15 folds;
no modifications to other components. Advance by the same criteria.

Also measure (not optimize) the historical augmentation-source discrepancy using
all four log-Ridge banks: matched production settings with/without the 46 sources
outside dev. This is a protocol sensitivity, not a claim that excluding lawful
training data improves the final model.

## H4, justified by the observed split variance (before evaluation)

The three GMM OOF predictions have mean per-ID variance 4.8976 HV², larger than
TabPFN 1.8511 and Ridge 2.3256. Averaging already-stored GMM OOFs would reduce its
RMSE 10.2610 -> 10.0195, but this diagnostic is NOT a deployable gain and is not a
comparison against full-500 training. Test whether marginalizing training-set
influence helps in a real matched outer-fold experiment.

Use exactly 16 fixed 90%-size subsamples without replacement inside each outer
training set, refitting imputation/scaling and Ridge on each. Keep the GMM head's
alpha selected by the original full outer training set's inner CV (no selection
on outer labels; do not search bag count/fraction). Compare the mean with the
exact full-data GMM at unchanged weight 0.15. A separate fixed alpha/0.9 control
tests whether this is merely effective ridge-penalty inflation. Use all 15 outer
folds and the existing failure fallback. Same advancement criteria. No neural
seeds, pretrained weights, new measurements, or test reads are involved.
