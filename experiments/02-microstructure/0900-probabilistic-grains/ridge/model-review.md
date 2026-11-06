# Independent model review

Reviewed the current `probabilistic/grain_measurement.py`, `run_stage2.py`,
`run_stage4.py`, `distribution_kernel.py`, and `set_model/run.py` read-only.
Numerical evidence and reviewed source hashes are in `model-review.json`.
No hardness score was used by this audit; no model was retrained or file outside
`ridge/` modified.

**Repair status:** the contrast-adaptive copy now uses a mixture of Gaussian
proposals around distinct modes and requires two successive quadrature-order
agreements. Independent checks on the same frozen old models passed for all 80
first-outer validation images plus the four original difficult cases: 81 cases
finished at order 25 and three at order 41. The repaired TRAIN_000088 posterior
agrees with a separate adaptive Gauss-Kronrod integration to about 1.4e-7 in phase
mean and 8.1e-8 in phase variance. Helper source hashes are included in cache
identity. The initial defect below is repaired in that copy; the original caches
remain unsuitable as the primary evidence. This acceptance concerns numerical
integration, not empirical accuracy or uncertainty calibration.

**Numerical cap extension:** a later TRAIN_000475 guard correctly stopped before
writing its unconverged features. Appending orders 121 and 161, with the same
target, implementation, and tolerances, made it converge at 121; comparison with
161 changes its largest mean/variance by 2.88e-7/1.80e-7. All nine earlier cache
triplets (27 files; 3,040 converged inference records) remained byte-identical.
The hardest existing order-81 case in each cache reproduced its mean and variance
bit for bit. An explicit per-file hash allowlist pins the old and new cache
identities; negative checks reject changed keys, inputs, configuration, indices,
paths, and file hashes. Reusing those already converged caches is numerically
equivalent and does not require repeating their downstream MPS computations.

**Final robust integration fallback:** TRAIN_000339 later needed more than the
Gaussian proposals could reliably resolve. The final implementation instead
integrates prevalence with Beta-weighted Gauss-Jacobi nodes of sufficient degree,
then integrates the scalar offset adaptively. It checks tighter tolerances,
expanded rigorous Gaussian-mixture tail bounds, and twice as many Jacobi nodes.
An independently written implementation in `ridge/reference_beta_quad.py`
agrees with all five production posterior means and variances to about 1e-15;
the final omitted posterior-mass bound is 1.52e-23. The same image under nine
previous cached priors also agrees with the global reference within the declared
engineering tolerance. The updated allowlist preserves 12 cache triplets and
4,080 prior converged inferences; its 36 hashes, mixed provenance keys, and
negative rejection checks pass. Source-level comparison confirms the previously
converged early-return path is unchanged. All numerical audits remained blind
to hardness scores.

## Material defect requiring numerical repair

**P1: Single-mode, seven-node quadrature gives materially wrong moments in
multimodal posteriors, despite a successful optimizer return.**
`grain_measurement.py:302-338` optimizes from one initial prevalence, constructs a
local Gaussian proposal, then applies 7x7 Gauss-Hermite quadrature. The posterior
can have another, higher-density mode outside the first proposal's well-resolved
region. A success flag only establishes a local optimizer condition.

Auditing the lowest quadrature effective-weight counts in existing caches found:

This can affect both the readout's training features and validation features,
including their claimed uncertainty. It is not train/validation leakage.

Use fixed multiple initial modes and integrate a Gaussian-mixture proposal with
the posterior/proposal-mixture density correction, or another numerically reliable
two-dimensional integration scheme. Compare higher quadrature orders without
labels. Simply choosing the best mode can discard genuine secondary-mode mass.
Keep the old caches as the initial approximate implementation, then create a new
hash/cache version for repaired measurements. The numerical defect should be
resolved before treating posterior/kernel scores as evidence for the method.

The first ordinary 80-image outer validation slice looked benign (maximum phase
mean change 0.00055 between orders 7 and 15). This is why an ordinary slice and
optimizer success alone were insufficient; the difficult-case audit matters.

## Valid math and validation boundaries

- The logit-prevalence posterior includes the beta prior's change-of-variable
  Jacobian correctly. Analytic gradients and Hessians agree with central finite
  differences on four actual images: maximum errors about 9.5e-7 and 1.6e-5.
- The Gauss-Hermite importance-weight formula is correct for the selected local
  Gaussian proposal. The problem above is numerical coverage, not a missing
  determinant or sign in the weights.
- Conditional independent Bernoulli grain variance plus between-parameter
  variance is a correct law-of-total-variance calculation under this model.
- The expected RBF expression is correct for independent diagonal-Gaussian input
  distributions. Dividing variance by training feature scale squared is correct.
  On the actual first outer-training cache, symmetry errors were at floating-point
  precision, all tested kernel matrices were positive definite, and zero-variance
  limits matched ordinary RBF within 3.4e-16.
- Kernel self-similarity below one is correct for a distribution mean-embedding
  kernel using independent draws from a distribution. It is not a requirement to
  force the diagonal to one. Such normalization would be a different kernel.
- Stage 2 learns each hierarchy only within the relevant inner/outer training
  subset, with explicit IDs. Its new Ridge preprocessing and alpha selection are
  nested. Stage 4's lengths, regularization, and uncertainty scale are selected
  only on the corresponding inner folds. The cached train/eval indices and input
  hashes are checked.
- The set model's normalization, target scaling, and Ridge skip fit only the
  explicit training subset. Inner stopping uses only the inner validation rows;
  the outer validation labels do not enter the fitting path. Storing all 500 rows
  in tensors does not itself use excluded rows in a loss or fitted statistic.
- No new test training, synthetic training examples, or copied per-grain hardness
  labels were found. Constant component anchors are parameter regularization,
  not generated training images or rows. Per-image posterior inference with a
  frozen learned population model is a transform of that image, not a shared
  test-population refit.
- The exact 045 blend and failed-mask behavior are correctly retained for blend
  metrics. Stage 2/4 standalone metrics score raw predictions, whereas blend
  metrics apply the fallback; describe those separately.

## Material modeling limitations, not implementation leaks

**P2: Spatial shade error is replaced by one scalar shared offset.**
`grain_measurement.py:147` averages per-grain shade variances and adds that scalar
to the prior variance of one offset shared by every grain. Actual smoother error
has a covariance matrix determined by overlapping neighbors and by recentering.
The code instead assumes constant covariance for all grain pairs. It does not
propagate the original spatial shade field or its common-reference uncertainty.
The resulting posterior variance is conditional on this approximation, not a
calibrated total measurement variance. The point-kernel comparison is useful, but
cannot establish correct uncertainty calibration by itself.

**P2: The hierarchy fit is approximate empirical Bayes, not exact EM for the
stated joint likelihood.** The global component updates weight each image by
1/grain-count, while local likelihoods use all grains without that scaling.
Variance updates subtract pixel variance by moments and ignore posterior offset
variance/covariance in component scatter. This is a defensible robustness
heuristic, but no single exact EM likelihood is being monotonically optimized.
It should be reported as a specified approximate model, not exact Bayesian
recovery of the generating process. Global parameter estimation uncertainty and
segmentation uncertainty are not integrated into each image's posterior.

**P2: Stage 4 discards cross-feature posterior covariance.** The five phase
features share grain memberships and offset/prevalence uncertainty, so they are
generally correlated. Only their marginal variances are retained and treated as
independent Gaussian coordinates; the remaining physical features receive zero
measurement variance. The analytic kernel is exact for this diagonal Gaussian
approximation, not for the full grain posterior. Bounded phase fractions can
also have non-Gaussian posteriors. The code is mathematically valid but the
approximation limits the scientific claim and may limit prediction quality.

## Set-model checkpoints and refit assessment

The saved checkpoint is the explicitly refitted outer-training model, not an
accidental final epoch from the inner run. Epoch zero is legitimately eligible
and produces the fixed Ridge skip. Reload, object reordering, zero-weight padding,
and one actual training-step held-out-label exclusion checks are implemented.

The selected inner epoch count transfers from 256 rows (8 batches) to 320 rows
(10 batches), so refitting performs about 25% more parameter updates at the same
epoch count. The normalization, Ridge skip, and random batch order also change.
This is an ordinary approximate epoch-transfer procedure, not exact checkpoint
equivalence; it is a potential performance limitation, not leakage. Likewise,
the neural correction trains against an in-sample fitted Ridge skip; unseen skip
errors can be larger. That resembles ordinary residual fitting and is not a
validation defect, but a weak learned correction would not disprove the whole
class of grain-set models.

## Optional two-feature TabPFN readout

A frozen TabPFN readout over the original 98 features plus the corrected
hierarchical phase mean and standard deviation is a reasonable separate test of
readout capacity. Preserve native outer-fold prior fits, feature alignment,
fixed seed/ensemble/blend settings, and failure fallback. Do not replace its OOF
features with moments from an all-500 hierarchy. Name it as an additional
comparison rather than evidence that the original fixed Ridge implementation was
the best possible realization of the hierarchy proposal.

## Final Ridge candidate output audit

After the root task authorized and ran the prepared finalizer, an independent
read-only check verified the local `ridge/final500/candidate-045-ridge20.csv`:
500 original training IDs, 1,000 test inference rows, selected Ridge alpha 10,
one 021 fallback, exact fixed blend formula and exact CSV round trip. SHA-256 is
`66486dcbba2571e4389d333d74c2c9313cc48235d0c73dbf17d470f17d9ef96e`.
The manifest records `submitted: false`; there is no public/private evaluation
for this candidate.
