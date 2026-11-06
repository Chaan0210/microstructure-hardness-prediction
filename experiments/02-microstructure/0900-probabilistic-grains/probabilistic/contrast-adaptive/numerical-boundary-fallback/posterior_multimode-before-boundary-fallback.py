"""Label-free numerical integration of a potentially multimodal 2D posterior."""
from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def find_modes(model, image):
    prior_eta = np.log(model.prior_phase_mean/(1.-model.prior_phase_mean))
    modes = []
    for eta in [prior_eta, -4., 4.]:
        result = minimize(lambda x: model._posterior(image, x)[:2], [0., eta], jac=True,
                          method="L-BFGS-B", bounds=[(-.9, .9), (-9., 9.)],
                          options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-7})
        point = result.x
        if any(abs(point[0]-m["point"][0]) < 1e-3 and abs(point[1]-m["point"][1]) < 1e-2 for m in modes):
            continue
        nlp, _, _, hessian = model._posterior(image, point, hessian=True)
        eigen, vec = np.linalg.eigh(hessian)
        covariance = (vec*(1./np.maximum(eigen, 1e-4)))@vec.T
        # Broad numerical proposal, with PSD-preserving rescaling if flat.
        factors = np.minimum(1., np.sqrt(np.array([.25, 16.])/np.maximum(np.diag(covariance), 1e-12)))
        covariance = covariance*factors[:, None]*factors[None, :]*1.5
        eigen, vec = np.linalg.eigh(covariance)
        covariance = (vec*np.maximum(eigen, 1e-8))@vec.T
        chol = np.linalg.cholesky(covariance)
        logdet = float(2*np.log(np.diag(chol)).sum())
        modes.append({"point": point, "nlp": float(nlp), "covariance": covariance,
                      "chol": chol, "logdet": logdet, "inverse": np.linalg.inv(covariance),
                      "success": bool(result.success)})
    mode_mass = np.array([-m["nlp"]+.5*m["logdet"] for m in modes])
    probabilities = np.exp(mode_mass-logsumexp(mode_mass))
    # Positive proposal mass at every mode. Importance correction removes this
    # numerical choice from the target posterior in the converged integral.
    probabilities = .95*probabilities+.05/len(modes)
    return modes, probabilities


def _batch_posterior(model, image, points):
    coefficients = image.area_weights[:, None]*image.phase_basis
    means, variances, log_posteriors = [], [], []
    a = 2.*model.prior_phase_mean
    b = 2.*(1.-model.prior_phase_mean)
    var0, var1 = image.pixel_var+model.tau2[0], image.pixel_var+model.tau2[1]
    for first in range(0, len(points), 256):
        x = points[first:first+256]
        offset, eta = x[:, 0], x[:, 1]
        residual = image.d[None, :]-offset[:, None]
        log0 = -.5*(np.log(var0)[None, :]+(residual-model.mu[0])**2/var0[None, :])-np.logaddexp(0., eta)[:, None]
        log1 = -.5*(np.log(var1)[None, :]+(residual-model.mu[1])**2/var1[None, :])-np.logaddexp(0., -eta)[:, None]
        ll = np.logaddexp(log0, log1)
        q = np.exp(log1-ll)
        logp = ll.sum(1)-.5*offset**2/(model.offset_var+image.shade_var)-a*np.logaddexp(0., -eta)-b*np.logaddexp(0., eta)
        log_posteriors.append(logp)
        means.append(q@coefficients)
        variances.append((q*(1.-q))@(coefficients**2))
    return np.concatenate(log_posteriors), np.concatenate(means), np.concatenate(variances)


def integrate_order(model, image, modes, probabilities, order):
    nodes, weights = hermgauss(order)
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    base_nodes = np.column_stack([u.ravel(), v.ravel()])*np.sqrt(2.)
    log_gh_weights = np.log(np.outer(weights, weights).ravel()/np.pi)
    points, base_log_weights = [], []
    for mode, probability in zip(modes, probabilities):
        points.append(mode["point"]+base_nodes@mode["chol"].T)
        base_log_weights.append(np.log(probability)+log_gh_weights)
    points, base_log_weights = np.concatenate(points), np.concatenate(base_log_weights)
    proposal_log_terms = []
    for mode, probability in zip(modes, probabilities):
        delta = points-mode["point"]
        mahalanobis = np.einsum("ni,ij,nj->n", delta, mode["inverse"], delta)
        proposal_log_terms.append(np.log(probability)-np.log(2*np.pi)-.5*mode["logdet"]-.5*mahalanobis)
    proposal_log_density = logsumexp(proposal_log_terms, axis=0)
    lp, phase_means, phase_vars = _batch_posterior(model, image, points)
    logmass = base_log_weights+lp-proposal_log_density
    w = np.exp(logmass-logsumexp(logmass))
    mean = w@phase_means
    variance = np.maximum(w@(phase_vars+phase_means**2)-mean**2, 0.)
    bmean = float(w@points[:, 0])
    diagnostics = {
        "offset": bmean, "offset_sd": float(np.sqrt(max(w@(points[:, 0]**2)-bmean*bmean, 0.))),
        "prevalence": float(w@expit(points[:, 1])), "ess": float(1./np.dot(w, w)),
        "optimizer_success": bool(all(m["success"] for m in modes)),
        "mode_count": len(modes), "mode_nlp": [m["nlp"] for m in modes],
        "mode_locations": [m["point"].tolist() for m in modes],
        "proposal_probabilities": probabilities.tolist(), "quadrature_order": order,
        "log_normalizer": float(logsumexp(logmass)),
    }
    return mean, variance, diagnostics


def infer_multimode(model, image, config):
    modes, probabilities = find_modes(model, image)
    previous, convergence, stable_count = None, [], 0
    for order in config["quadrature_orders"]:
        mean, var, diagnostics = integrate_order(model, image, modes, probabilities, order)
        if previous is not None:
            mean_delta, var_delta = np.abs(mean-previous[0]), np.abs(var-previous[1])
            converged = (np.all(mean_delta <= 2e-4+1e-3*np.abs(mean)) and
                         np.all(var_delta <= 2e-6+.01*var))
            stable_count = stable_count+1 if converged else 0
            convergence.append({"order": order, "max_mean_delta": float(mean_delta.max()),
                                "max_variance_delta": float(var_delta.max()),
                                "log_normalizer_delta": float(abs(diagnostics["log_normalizer"]-previous[2])),
                                "stable": bool(converged)})
            if stable_count >= 2:
                diagnostics["quadrature_converged"] = True
                diagnostics["convergence_history"] = convergence
                return mean, var, diagnostics
        previous = mean, var, diagnostics["log_normalizer"]
    diagnostics["quadrature_converged"] = False
    diagnostics["convergence_history"] = convergence
    raise RuntimeError(f"Posterior quadrature failed prescribed tolerance for {image.image_id}: {convergence}")
