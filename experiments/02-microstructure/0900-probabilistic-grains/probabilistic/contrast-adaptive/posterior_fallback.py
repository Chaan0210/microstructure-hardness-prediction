"""Boundary-safe deterministic integration when Gaussian proposals converge slowly.

Given an offset, likelihood and its first/second phase-moment numerators are
polynomials of degree at most N in the phase prevalence. Beta-weighted Jacobi
quadrature therefore integrates them exactly in polynomial arithmetic using
ceil((N+1)/2) nodes. Only the one-dimensional offset integral remains adaptive.
No synthetic observations or hardness labels are involved.
"""
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_jacobi, betaln, ndtr


def _one_integral(model, image, modes, jacobi_order, tail_sigma, epsrel):
    alpha, beta = 2.*model.prior_phase_mean, 2.*(1.-model.prior_phase_mean)
    nodes, weights = roots_jacobi(jacobi_order, beta-1., alpha-1.)
    prevalence = (nodes+1.)/2.
    logweights = np.log(weights/weights.sum())
    logpi, log1mpi = np.log(prevalence), np.log1p(-prevalence)
    var0, var1 = image.pixel_var+model.tau2[0], image.pixel_var+model.tau2[1]
    priorvar = model.offset_var+image.shade_var
    coefficients = image.area_weights[:, None]*image.phase_basis
    logscale = max(-mode["nlp"] for mode in modes)-betaln(alpha, beta)-.5*np.log(2*np.pi*priorvar)
    lower_mean = float(min(0., np.min(image.d)-np.max(model.mu)))
    upper_mean = float(max(0., np.max(image.d)-np.min(model.mu)))
    largest_sd = float((1./priorvar+np.sum(1./(image.pixel_var+np.max(model.tau2))))**-.5)
    smallest_sd = float((1./priorvar+np.sum(1./(image.pixel_var+np.min(model.tau2))))**-.5)
    lower, upper = lower_mean-tail_sigma*largest_sd, upper_mean+tail_sigma*largest_sd
    # Every conditional-on-assignments Gaussian posterior lies in these bounds
    # up to 2 Phi(-tail_sigma) mass, independently of assignment weights.
    step = min(.05, 4.*smallest_sd)
    grid_count = max(3, int(np.ceil((upper-lower)/step))+1)
    points = list(np.linspace(lower, upper, grid_count)[1:-1])
    for mode in modes:
        b, sd = mode["point"][0], np.sqrt(mode["covariance"][0, 0])
        points.extend([b-2*sd, b, b+2*sd])
    points = sorted(set(float(v) for v in points if lower < v < upper))

    def conditional(offset):
        l0 = -.5*(np.log(var0)+(image.d-offset-model.mu[0])**2/var0)
        l1 = -.5*(np.log(var1)+(image.d-offset-model.mu[1])**2/var1)
        log0, log1 = log1mpi[:, None]+l0, logpi[:, None]+l1
        ll = np.logaddexp(log0, log1)
        q = np.exp(log1-ll)
        logmass = logweights+ll.sum(1)-.5*offset*offset/priorvar-.5*np.log(2*np.pi*priorvar)-logscale
        w = np.exp(logmass)
        means = q@coefficients
        seconds = (q*(1.-q))@(coefficients**2)+means**2
        mass = w.sum()
        return np.r_[mass, w@means, w@seconds, offset*mass, offset*offset*mass, w@prevalence]

    value, error, info = quad_vec(conditional, lower, upper, epsabs=1e-10, epsrel=epsrel,
                                  points=points, quadrature="gk21", full_output=True,
                                  limit=max(1000, grid_count*4), norm="max")
    if not info.success or not np.isfinite(value).all() or value[0] <= 0:
        raise RuntimeError(f"Adaptive offset integral failed for {image.image_id}: {info.message}")
    mean = value[1:6]/value[0]
    variance = np.maximum(value[6:11]/value[0]-mean**2, 0.)
    offset = float(value[11]/value[0])
    diagnostics = {"jacobi_order": jacobi_order, "tail_sigma": tail_sigma,
                   "omitted_posterior_mass_bound": float(2*ndtr(-tail_sigma)),
                   "offset_bounds": [lower, upper], "offset_grid_count": grid_count,
                   "quadrature_error_normalized": float(error/value[0]), "evaluations": int(info.neval),
                   "offset": offset, "offset_sd": float(np.sqrt(max(value[12]/value[0]-offset*offset, 0.))),
                   "prevalence": float(value[13]/value[0]), "log_normalizer": float(np.log(value[0])+logscale)}
    return mean, variance, diagnostics


def infer_fallback(model, image, modes, gh_diagnostics, gh_history):
    order = max(2, (len(image.d)+2)//2)
    settings = [(order, 8., 1e-5), (order, 9., 1e-7), (2*order, 10., 1e-9)]
    previous, stable_count, history = None, 0, list(gh_history)
    for nodes, tail, epsrel in settings:
        mean, variance, check = _one_integral(model, image, modes, nodes, tail, epsrel)
        if previous is not None:
            md, vd = np.abs(mean-previous[0]), np.abs(variance-previous[1])
            stable = bool(np.all(md <= 2e-4+1e-3*np.abs(mean)) and np.all(vd <= 2e-6+.01*variance))
            stable_count = stable_count+1 if stable else 0
            history.append({"order": f"jacobi{nodes}-tail{tail:g}", "max_mean_delta": float(md.max()),
                            "max_variance_delta": float(vd.max()), "log_normalizer_delta": float(abs(check["log_normalizer"]-previous[2])),
                            "stable": stable, **check})
        else:
            history.append({"order": f"jacobi{nodes}-tail{tail:g}", "initial_adaptive_reference": True, **check})
        previous = mean, variance, check["log_normalizer"]
    if stable_count < 2:
        raise RuntimeError(f"Boundary-safe posterior integration did not stabilize for {image.image_id}: {history[-3:]}")
    diagnostics = {**gh_diagnostics, **check, "integration_method": "beta-jacobi-adaptive-gauss-kronrod",
                   "quadrature_converged": True, "convergence_history": history,
                   "gaussian_proposal_fallback": True, "ess": None}
    return mean, variance, diagnostics
