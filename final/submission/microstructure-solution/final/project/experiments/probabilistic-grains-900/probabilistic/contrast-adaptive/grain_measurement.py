"""Per-image GMM measurements used by the final model."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
MEAN_KEYS = [
    "phase_fraction", "phase_logarea_centered", "phase_logar_centered",
    "phase_cos2", "phase_sin2", "mean_logarea", "std_logarea",
    "mean_area_inverse_quarter", "mean_logar", "mean_cos2", "mean_sin2",
    "log_effective_grain_count", "coverage", "relative_noise", "log_contrast",
]
ARMS = ('threshold', 'image_gmm', 'hierarchical')
CONFIG = {
    "version": 2, "adaptive_initial_means": [[0.05, 0.35], [0.10, 0.75]],
    "adaptive_warm_iterations": 8, "adaptive_refine_iterations": 12,
    "adaptive_min_gap": 0.10, "adaptive_bright_prior_pseudoimages": 0.5,
    "adaptive_initialization_selection": "balanced_train_laplace_evidence", "scope": "train400_only", "arms": list(ARMS),
    "inner_splits": 3, "alpha_ridge": [0.1, 1., 10., 100., 1000.],
    "blend_new_weight": 0.2, "threshold": 0.5, "pore_exclusion": True,
    "global_em_iterations": 20, "local_em_iterations": 16,
    "quadrature_order": 7, "quadrature_orders": [7, 15, 25, 41, 61, 81, 121, 161], "posterior_starts_eta": ["prior", -4., 4.], "beta_concentration": 2.,
    "ridge_inputs": ["physical", "x98_augmented"],
    "kernel_lengths": [0.5, 1., 2.], "kernel_alpha": [0.003, 0.03, 0.3, 3.],
    "kernel_uncertainty_scale": [0.25, 1., 4.],
}

@dataclass
class GrainImage:
    image_id: str
    d: np.ndarray
    pixel_var: np.ndarray
    area_weights: np.ndarray
    phase_basis: np.ndarray
    physical: np.ndarray
    shade_var: float
    empty: bool


def _responsibility(d, v, mu, tau2, offset, pi):
    var = v[:, None] + tau2
    logp = -0.5 * (np.log(var) + (d[:, None]-offset-mu)**2/var)
    logp += np.log([1.-pi, pi])
    return np.exp(logp[:, 1]-logsumexp(logp, axis=1))


def phase_moments(image, q):
    coefficients = image.area_weights[:, None] * image.phase_basis
    mean = q @ coefficients
    var = (q*(1.-q)) @ (coefficients**2)
    return mean, var


def threshold(image):
    q = (image.d >= CONFIG["threshold"]).astype(float)
    return (*phase_moments(image, q), {"offset": 0., "offset_sd": 0., "prevalence": float(q.mean()), "ess": 1.})


def image_gmm(image):
    d = image.d
    mu = np.quantile(d, [.2, .8])
    if mu[1]-mu[0] < .1:
        mu = np.array([min(mu[0], .3), max(mu[1], .7)])
    tau2 = np.array([.12**2, .12**2])
    pi = .35
    for _ in range(30):
        q = _responsibility(d, np.zeros_like(d), mu, tau2, 0., pi)
        resp = np.column_stack([1.-q, q])
        count = resp.sum(0) + 1e-8
        mu = (resp*d[:, None]).sum(0)/count
        tau2 = np.maximum((resp*(d[:, None]-mu)**2).sum(0)/count, .04**2)
        pi = float(np.clip(q.mean(), .01, .99))
        if mu[0] > mu[1]:
            mu, tau2, pi = mu[::-1], tau2[::-1], 1.-pi
    q = _responsibility(d, np.zeros_like(d), mu, tau2, 0., pi)
    return (*phase_moments(image, q), {"offset": 0., "offset_sd": 0., "prevalence": pi, "ess": 1.})


def transform(images, method, model=None):
    means, variances, diagnostics = [], [], []
    for image in images:
        if method == "threshold":
            m, v, diag = threshold(image)
        elif method == "image_gmm":
            m, v, diag = image_gmm(image)
        else:
            raise ValueError(method)
        means.append(np.r_[m, image.physical])
        variances.append(np.r_[v, np.zeros(10)])
        diagnostics.append(diag)
    means, variances = np.asarray(means), np.asarray(variances)
    if means.shape[1] != len(MEAN_KEYS) or not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise ValueError("Invalid measurement features")
    return means, variances, diagnostics


def physical_readout(means, variances):
    return np.column_stack([means, np.sqrt(np.maximum(variances[:, :5], 0.))])

