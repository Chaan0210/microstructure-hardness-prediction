# Complementary fusion anchored on082

Registered18October2026 before new fusion fitting or scores. User asks to combine
useful parts over082. Performance is an empirical outcome, not a guarantee.
082/076 and their entire reproduction chains are immutable. No submission,
paid compute, installation, external data, test training or public-score fitting.
New code, coefficients, logs and outputs only in this directory.

## Audit and ranked mechanisms

Earlier50:50 duplicate-input averaging and seed-disagreement attenuation failed.
They do not test joint fusion of the now completed076-target, dense-pick,
dense-pool and learned-connectivity branches. All required A213/B107/T320/V80
scopes match exactly. Do not stack the historical400 OOF rows as if they were
independent level-two training features: they can contain the current outer
validation labels in other folds' base training. Use A-only predictions onB.

1. Fixed diversification: retain half of082 and give one eighth each to the
   three newCNN branches and the topology addition branch. CNN replacements
   use the same old082 coefficient; topology beta is the previously fixed
   bounded ridge scalar (denominator sum(g²)+n,range0..0.5), fitted onB only.
   This asks whether different errors cancel without a new ensemble search.
2. Joint constrained fusion: fit all four rawCNN residual predictions and the
   topology physical-readout difference simultaneously, around the082 prior.
   Functional regularization prevents duplicate predictors from artificially
   weakening the penalty. Compare against the same algorithm with only the
   originalCNN (calibration-only), and a duplicate-predictor numerical control.
3. Conservative conditional fusion: allow those coefficients to vary linearly
   with noise and log(1+n_obj), standardized on the relevant meta-training
   high-noise rows and clipped to[-2,2]. Enforce coefficient constraints at all
   four corners of this box. This gives bounded routing by observable quality,
   with a10x extra interaction-output penalty. No target-defined regime,
   per-image exception, threshold search or outer-score routing.

## Frozen fusion specification

ColumnsR=[old residual,076-target residual,dense-pick residual,dense-pool
residual,topology augmentedTab-minus-physicalTab]. Average the two existing
seeds within EACH family; never choose a winning seed. Use original measured
topology weights/readouts, not the later unevaluated deterministic-loss prototype.
Log-smearing is excluded because it is an affine calibration hypothesis with
no independent image content and lacks the required cached A-only calibration
branch. Failed segmentation and noise<12 rows remain082 exactly.

For every fit, reconstruct old082 beta using its original noise-regime LOO
procedure on eligible meta-training rows and the073 A-only base. Base076 A→B
and T→V remain the verified source-restricted frozen versions. Lettheta0 put
beta on the oldCNN and0 on the other four columns. Candidate is
P076 + Rtheta on eligible high-noise rows. Constraints everywhere:
CNN coefficients>=0,oldCNN coefficient>=beta/2,sumCNN<=1,topology coefficient
in[0,.5]. For conditional routing theta(x)=intercept+noise*coef+logcount*coef;
all constraints apply at the fixed quality-box corners. No query-dependent
optimization constraints or preprocessing fits.

Fit convex objective mean((R(theta-theta0)-(y-P082))²)
+ lambda*mean((R(theta-theta0))²)
+10*lambda*mean(interaction_output²) for conditional fits only
+epsilon*||theta-theta0||² for unique solutions.
epsilon=1e-8*max(mean(R²),1), fixed numerical tie breaker.
lambda grid=[1,10,100], no additional grid. Shared-only is1coefficient, global
is5, conditional is15. SLSQP with analytic objective/gradient, max300steps,
ftol1e-10; verify constraints and convergence. The prediction-space penalty
is invariant to duplicated regressors up to the tiny tie breaker.

## Honest nested selection of the whole procedure

Existing400 images/original5folds; no new image training. New weights and every
quality normalization fit insideB. When evaluating a B high-noise source with
LOO, recompute old082 beta, family calibration, scalers and fusion parameters
usingB minus that source. It is NOT enough to LOO only the final weights while
leaving a beta fitted on the held-out label. Low/failure predictions never change.

Per outer fold, choose the configuration with minimum high-noise B-LOO MSE
among082/no change, fixed mixture,shared-only/global/conditional at the3lambda
values. Tie tolerance1e-10; fixed order:082,fixed,shared100/10/1,global100/10/1,
conditional100/10/1. Refit that procedure on allB, then evaluate once onV.
This selected PROCEDURE is the single primary candidate. Do not select the
best separate method after reading outer scores. Also report each predeclared
family with its own B-only lambda selection as an ablation. The shared-only
selected control determines whether improvement is new complementary signal
or merely recalibration; no new-information claim without superiority to it.

## Evidence, stopping and expansion

Primary reference082:overall9.256738746625928,high12411.508128300723715.
All500 labels have prior exposure. New fusion selection is nested conditional
on inherited frozen representations and previously explored family choices.
No resplit or bootstrap is called independent confirmation.

Report overall/high-noise/fold RMSE,paired10000source bootstrap(seed2026101809),
95% primary intervals,top5/10/20 contributor removal,calibration-only ablation,
affine-change decomposition,selected coefficients/routing and source failures.
Report frozen individual components and the MSE diversification identity for
the fixed mixture. V labels never choose weights,lambda,quality variables orseeds.

The primary must gain>=.02 overall, improve high-noise and>=3folds, have a95%
lower bound>0, and remain positive after its10largest contributors are removed
before further expansion. If it fails, stop rather than invent per-image good
parts or widen grids. A qualifying candidate must additionally pass a separately
locked equal-training-size transfer check on available fixedA anchors, with
actual new A→V component inference as needed; reused labels remain a limitation.
Only then freeze a full500 recipe and construct a separate candidate. Never
replace the official082 pipeline or submit without explicit approval.

Budget45min new local compute, one fit process,CPU2,at most5GiB own memory;
reserve10min for source/poison/solver/replay/preservation checks. Screening uses
existing train predictions, not test files. New heavy model fits only if the
frozen expansion gate passes and fit the remaining budget. No unbounded retries.

Verify baseline082 reconstruction, source-scope provenance, duplicate invariance,
independent QP/constraint math, held-out-label poison invariance including full
inner selection, artifact replay and scalar RMSE/assembly. End with protected
file checks and baseline saved-artifact inference/assembly into this directory.
