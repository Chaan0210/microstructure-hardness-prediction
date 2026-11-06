# Frozen probabilistic measurement and distribution-kernel experiment

**Status:** the original implementation below was stopped and quarantined by a
training-only numerical audit; do not use its `stage2` caches. See
`stage2/INVALID.json`. The corrected and independently reviewed implementation is
in [contrast-adaptive/DESIGN.md](contrast-adaptive/DESIGN.md).

This design was fixed before any new outer-fold scores were inspected. Only files
inside this new `probabilistic` directory are written by these programs.

## Data and validation

- Raw-train grain object bank is produced independently by the parent task using
  existing frozen zero-shot `masks-v3`; no adaptive segmentation model is fitted.
- Although the bank contains 500 train images, the primary experiment uses exactly
  the 400 IDs in `base.npz`. Each existing outer partition has five folds, so global
  measurement parameters and hardness readouts see only 320 images in a fold.
- The three historical partitions are development evidence, not fresh holdouts.
- Three shuffled inner folds (seed `19001 + 101 * partition + outer_fold`) select
  every readout hyperparameter. Measurement priors are refitted separately inside
  each inner training fold. No precomputed full-OOF residual is a learning target.
- Pores (`pore_flag > 0.5`) and invalid/nonpositive grain areas are excluded by the
  same deterministic rule in all three measurement methods. An empty image uses
  a finite fallback representation and retains the baseline in blend evaluation.
- No test bank, synthetic images, extra image data, or public scores are read.

## Stage 2 measurement arms

1. `threshold`: shade-corrected darkness >= 0.5.
2. `image_gmm`: a two-component count-weighted Gaussian mixture fitted separately
   to the observed grains of each image, with fixed variance/contrast floors.
3. `hierarchical`: a train-shared two-component measurement model

   d_ij = mu[z_ij] + b_i + u_ij + epsilon_ij

   where epsilon has known variance intensity_se2 / contrast_span^2, the
   component-specific variance of u is learned separately, and b is a shared
   residual illumination offset. Train images learn the two component means,
   residual spreads, prevalence-prior mean, and residual-offset prior variance.
   Image weights are balanced when updating shared parameters. A weak Beta prior
   contributes two pseudo-grains. Bright/dark means are anchored to disjoint
   plausible intervals; separation and variance floors prevent collapse.

The model infers the offset and prevalence for each image under the fixed learned
prior. A 7-by-7 Gauss-Hermite importance quadrature over their joint posterior
integrates classification uncertainty and covariance induced by a shared offset.
An image's fixed shade_se2 estimate is included as a correlated offset-prior
variance, distinct from per-grain pixel averaging error. This is a scalar residual
illumination approximation, not a full posterior over a spatial shade field.
Uncertainty in global learned parameters and in zero-shot grain geometry is not
included. Phase/illumination identifiability remains a limitation without true
phase masks; these are model-based uncertainties, not calibrated ground truth.

All arms report the *realized* area-weighted fraction and area-weighted phase
moments, not just the inferred Bernoulli prevalence. They share 15 mean features:
phase fraction; phase-weighted centered log area and log aspect ratio; two phase
orientation moments; mean/std log area; mean area^(-1/4); mean log aspect ratio;
two orientation moments; log effective grain count; coverage; relative noise;
and log contrast. The first five posterior standard deviations add five columns.

Six frozen Ridge readouts cross the three arms with either these 20 features or
`X98` augmented with them. Inner CV chooses alpha in [0.1, 1, 10, 100, 1000]. Each
inner fit learns median imputation and scaling from its own training rows.
Standalone predictions and a fixed 0.8*baseline045 + 0.2*new blend are reported.
On baseline segmentation failure or empty object representation, the blend keeps
the baseline prediction. No outer-fold score selects a blending weight.

## Stage 4 kernels (run only after stage 3 is complete)

Use the hierarchical arm's same 15 mean features and diagonal covariance. The
point RBF uses zero covariance. The distribution RBF analytically integrates the
RBF over two independent diagonal Gaussian input distributions:

k(i,j) = product_d (ell^2 / (ell^2 + v_id + v_jd))^(1/2)
         * exp[-0.5 * sum_d (m_id-m_jd)^2/(ell^2+v_id+v_jd)].

This is a positive-semidefinite kernel on distributions. No sampled additional
training observations are generated. Mean/variance scaling is fit inside the
appropriate training fold. Inner CV selects length multiplier [0.5, 1, 2] times
sqrt(feature count) and ridge penalty [0.003, 0.03, 0.3, 3]. The uncertainty arm
also chooses variance multiplier [0.25, 1, 4]. Its features/measurement caches are
the same fold-isolated stage-2 caches; validation images never update priors.
Both arms use the same fixed 20% blend and fallback policy as stage 2.

## Reproducibility

The programs save input hashes, frozen configuration, fold models, inner scores,
OOF predictions, standalone/blended summaries, and measured run time. Cache files
are keyed by input hashes and the frozen code/configuration hash. Computation uses
NumPy/SciPy/sklearn on at most two CPU threads; these small matrix/EM operations
do not benefit from MPS and leave MPS available to the grain-set learner.
