# Contrast-adaptive measurement correction, frozen before outer-score access

The original frozen measurement arm used a dark-component lower anchor of 0.55
in normalized darkness units. Training-only diagnostics from its first outer
fold showed the learned mean pinned to that boundary. The parent task authorized
a separate correction before any original or corrected outer prediction scores
were opened. The original run and its files remain unchanged.

The correction changes only shared-component contrast estimation. There is no
minimum *absolute* dark mean. Means are ordered with a minimum gap of 0.10 and
broad bounds [-0.25, 1.5]. A weak bright-location prior at 0.05 contributes half
an image of precision. The dark mean has no location prior. Pixel-error and
residual-spread separation, image-balanced updates, shared-offset prior,
prevalence prior, variance floors, pore exclusion, and posterior quadrature are
unchanged. Offset/phase ambiguity remains a model limitation.

Two deterministic starts, [0.05, 0.35] and [0.10, 0.75], each receive eight EM
iterations. Image-balanced Laplace approximate marginal likelihood on the
training intensities alone selects the start. Offset/Beta normalization terms
are included. Twelve further EM iterations refine the selected model. This is
28 total iterations of work and 20 iterations along the selected trajectory.
No hardness label participates in initialization or its selection.

Outer/inner folds, all six readouts, alpha grids, feature definitions, fixed
20% blends, same-021-share control and kernel grids match the original design.
Every inner split relearns its own measurement prior and initialization choice.
The 100 bank images outside the 400 baseline IDs remain unused.

An independent training-only numerical audit found that the original single
optimization start and order-7 posterior quadrature could miss distinct modes.
That run was stopped and quarantined with `../stage2/INVALID.json` without opening
outer hardness scores. This corrected implementation uses three deterministic
optimization starts (prior logit, -4, +4), retains distinct modes, and integrates
against a positive mixture of Gaussian proposals with the exact posterior divided
by the full mixture proposal density. Node/grain evaluation is vectorized.

Orders 7, 15, 25, 41, 61, 81, 121, and 161 refine adaptively. Success requires two consecutive
stable comparisons (at least order 25), with all five posterior
mean changes <= 2e-4 + 1e-3*abs(mean), and variance changes <= 2e-6 + .01*variance.
Failure at the final Gaussian-proposal order invokes the boundary-safe exact
prevalence/adaptive offset integration described below. Unconverged features
cannot be scored.
The initialization criterion also sums distinct local-mode Laplace masses.
These are numerical correctness changes selected without hardness labels.

The initial cap of 81 correctly stopped one image (`TRAIN_000475`) after its first
stable comparison at order 81, before its features were saved. Orders 121 and 161
were added solely to provide the second numerical confirmation. Nine already
converged cache triplets (3,040 image inferences) are reused under the explicit
`cache-compatibility-order-extension.json` allowlist. Every NPZ, prior and
diagnostics file retains its original bytes and hash. The loader verifies each
specific file hash, input/dependency identity, unchanged configuration except the
extended order suffix, exact train/eval indices, and two stable comparisons for
every cached inference. Unlisted stale caches remain rejected. The prior source
and failed-run evidence are preserved in `numerical-order-extension`.

One further case (`TRAIN_000339`) converged too slowly even at order 161. Its
fallback integrates phase prevalence with normalized Beta-weighted Gauss-Jacobi
quadrature. For N grains, likelihood and first/second phase-moment numerators
are polynomials of degree at most N, so ceil((N+1)/2) nodes are exact in polynomial
arithmetic. A vector Gauss-Kronrod rule then integrates only the shared offset.
Three evaluations tighten error tolerance, expand integration bounds, and double
the Jacobi order; two consecutive moment/variance agreements are required.

The offset bounds are posterior bounds: conditional on any latent phase
assignment, its offset posterior is Gaussian, with mean a convex combination of
zero and observed `darkness - component_mean`. Every conditional posterior mean
therefore lies in [L,U], the extrema including zero. Every conditional posterior
standard deviation is at most
`smax = (1/prior_variance + sum(1/(pixel_variance + max(component_variance))))^-1/2`.
Bounds [L-k*smax,U+k*smax] omit at most 2*Phi(-k) from every conditional posterior
and hence their posterior-weighted mixture. The final k=10 bound is <1.6e-23.
A global offset grid and all discovered modes guide adaptive subdivisions.

The fallback is entered only after failed Gaussian-proposal refinement; previous
converged return paths are unchanged. The exact allowlist now covers twelve
immutable cache triplets (4,080 image inferences) from both earlier cache keys,
and is bound to the single final implementation key. Historical code, manifest,
and failure evidence are additionally preserved in `numerical-boundary-fallback`.

All model standard deviations are approximate: shade_se2 is a local shade-field
estimation proxy; a shared scalar residual offset approximates correlated
illumination error. Global-parameter, spatial-field-shape, and segmentation
uncertainties are not calibrated or fully integrated.

Commands from repository root:

    .venv/bin/python experiments/02-microstructure/0900-probabilistic-grains/probabilistic/contrast-adaptive/run_stage2.py
    .venv/bin/python experiments/02-microstructure/0900-probabilistic-grains/probabilistic/contrast-adaptive/run_stage4.py --stage3-complete

The second command remains gated until the grain-set stage is complete. All
outputs live under this separate contrast-adaptive folder. CPU threads are
limited to two and no synthetic training observations are generated.
