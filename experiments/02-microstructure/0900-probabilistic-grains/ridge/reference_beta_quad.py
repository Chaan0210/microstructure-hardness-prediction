"""Independent read-only posterior integration reference; never fits a model.

Beta-Jacobi integrates the prevalence boundary tails without logit truncation.
Conditional likelihood times either first or second phase moments is a
polynomial of degree at most the number of grains. The remaining scalar offset
integral uses adaptive Gauss-Kronrod and an assignment-independent tail bound.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import logsumexp, ndtr, roots_jacobi


def reference(model, image, concentration=2.0, jacobi_multiplier=1, epsrel=1e-8, grid_spacing=0.04):
    n = len(image.d)
    order = jacobi_multiplier * ((n + 2) // 2)
    alpha = concentration * model.prior_phase_mean
    beta = concentration * (1-model.prior_phase_mean)
    nodes, weights = roots_jacobi(order, beta-1, alpha-1)
    pi = (nodes+1)/2
    logw = np.log(weights)-np.log(weights.sum())
    logpi, log1pi = np.log(pi), np.log1p(-pi)
    coefficients = image.area_weights[:, None] * image.phase_basis
    v0, v1 = image.pixel_var+model.tau2[0], image.pixel_var+model.tau2[1]
    priorvar = model.offset_var+image.shade_var
    # For every latent assignment, the Gaussian conditional offset mean is a
    # convex combination of 0 and the grain residuals. Its sd cannot exceed this.
    residual_limits = image.d[:, None]-model.mu[None, :]
    lower_mean = min(0.0, float(residual_limits.min()))
    upper_mean = max(0.0, float(residual_limits.max()))
    sdmax = float((1/priorvar + np.sum(1/(image.pixel_var+max(model.tau2)))) ** -0.5)
    lower, upper = lower_mean-9*sdmax, upper_mean+9*sdmax
    grid = np.linspace(lower, upper, max(3, int(np.ceil((upper-lower)/grid_spacing))+1))

    def conditional(offset, moments=False):
        residual = image.d-offset
        log0 = -.5*(np.log(v0)+(residual-model.mu[0])**2/v0)
        log1 = -.5*(np.log(v1)+(residual-model.mu[1])**2/v1)
        component0 = log1pi[:, None]+log0[None, :]
        component1 = logpi[:, None]+log1[None, :]
        mixture = np.logaddexp(component0, component1)
        logmass = logw+mixture.sum(1)-.5*offset*offset/priorvar
        lognormalizer = logsumexp(logmass)
        if not moments:
            return lognormalizer
        q = np.exp(component1-mixture)
        means = q @ coefficients
        variance = (q*(1-q)) @ (coefficients**2)
        posterior_weights = np.exp(logmass-lognormalizer)
        return lognormalizer, posterior_weights@means, posterior_weights@(variance+means**2)

    shift = float(max(conditional(float(b)) for b in grid))

    def integrand(offset):
        lp, means, second = conditional(float(offset), moments=True)
        return np.exp(lp-shift)*np.r_[1.0, means, second]

    integral, error, info = quad_vec(integrand, lower, upper, points=grid[1:-1],
        epsabs=epsrel*0.01, epsrel=epsrel, norm="max", limit=4000, full_output=True)
    if not info.success or not np.isfinite(integral).all() or integral[0] <= 0:
        raise RuntimeError(f"Independent reference failed: {info.message}")
    mean = integral[1:6]/integral[0]
    variance = integral[6:11]/integral[0]-mean**2
    return mean, variance, {
        "jacobi_order": order, "n_grains": n, "offset_bounds": [lower, upper],
        "conditional_sd_upper_bound": sdmax, "posterior_tail_mass_upper_bound": float(2*ndtr(-9.0)),
        "global_grid_spacing_max": float(np.diff(grid).max()), "offset_evaluations": int(info.neval),
        "success": bool(info.success), "reported_scaled_absolute_error": float(error),
        "epsrel": epsrel, "log_normalizer_normalized_beta": float(np.log(integral[0])+shift),
        "likelihood_first_second_moments_degree_upper_bound": n,
        "jacobi_exact_polynomial_degree": int(2*order-1),
    }
