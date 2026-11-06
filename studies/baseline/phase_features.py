"""Image-local two-population intensity proxies after morphology masking.

These are descriptive mixture estimates, not identified physical phases. There
is no label access, global fitting, file access, or imposed hardness direction.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import warnings

import numpy as np
from scipy import ndimage as ndi
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

BOUNDARY_MULTIPLIERS = (0.8, 1.0, 1.2)
DESCRIPTORS = (
    "dark_fraction", "dark_mean", "light_mean", "separation",
    "overlap", "ambiguous_fraction", "interior_fraction",
    "boundary_fraction", "pore_fraction", "fit_converged",
)
FEATURE_NAMES = tuple(
    f"phase_interior_t{factor:g}_{name}"
    for factor in BOUNDARY_MULTIPLIERS for name in DESCRIPTORS
)
MAX_SAMPLES = 4096


def _fit_populations(values):
    """Return seven mixture diagnostics with deterministic finite fallback."""
    values = np.asarray(values, dtype=np.float64)
    center = float(values.mean()) if values.size else 0.0
    # No evidence for two populations: fractions are deliberately neutral;
    # overlap/ambiguity are maximal, and fit_converged is explicitly zero.
    fallback = (0.5, center, center, 0.0, 1.0, 1.0, 0.0)
    if values.size < 64 or float(np.std(values)) < 1e-4:
        return fallback
    ordered = np.sort(values)
    # Uniform empirical-quantile sampling preserves image-local population
    # proportions, caps work, and avoids spatial-order dependence under D4.
    if ordered.size > MAX_SAMPLES:
        ix = np.linspace(0, ordered.size - 1, MAX_SAMPLES).astype(np.int64)
        ordered = ordered[ix]
    sample = ordered[:, None]
    q25, q75 = np.percentile(ordered, (25, 75))
    if q75 - q25 < 1e-5:
        q25, q75 = np.percentile(ordered, (10, 90))
    if q75 - q25 < 1e-5:
        return fallback
    model = GaussianMixture(
        n_components=2, covariance_type="full", reg_covar=1e-5,
        max_iter=50, tol=1e-3, n_init=1, random_state=20261003,
        init_params="random", means_init=np.array([[q25], [q75]]),
        weights_init=np.array([0.5, 0.5]),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(sample)
        order = np.argsort(model.means_[:, 0])
        means = model.means_[order, 0]
        variances = model.covariances_[order, 0, 0]
        weights = model.weights_[order]
        probabilities = model.predict_proba(sample)[:, order[0]]
        delta = float(means[1] - means[0])
        separation = delta / np.sqrt(float(variances.mean()))
        # Bhattacharyya coefficient of the two unweighted Gaussian densities;
        # 1 means identical, approaching 0 means well-separated populations.
        variance_sum = float(variances.sum())
        overlap = (np.sqrt(2.0 * np.sqrt(float(np.prod(variances))) / variance_sum)
                   * np.exp(-(delta * delta) / (4.0 * variance_sum)))
        result = (
            float(weights[0]), float(means[0]), float(means[1]),
            float(separation), float(np.clip(overlap, 0.0, 1.0)),
            float(np.mean((probabilities >= 0.25) & (probabilities <= 0.75))),
            float(model.converged_),
        )
        return result if np.isfinite(result).all() else fallback
    except (ValueError, np.linalg.LinAlgError):
        return fallback


def extract_phase(image):
    """Return 30 float64 descriptors and names for a 256x256 uint8 image.

    Intensities remain on the absolute [0,1] scale. Sigma=1 Gaussian smoothing
    reduces noise. Dark local ridges (7x7 blackhat, Otsu times .8/1/1.2) are
    dilated one pixel and excluded. Compact deep-dark components with area
    2..128, eccentricity<.9 and solidity>.8 are excluded as pore proxies.
    Their cutoff is p2 - .1*(p98-p2), below the broad dark population.

    Each boundary setting gets an independent fixed-seed two-Gaussian fit on
    at most 4096 interior intensity quantiles. Dark/light labels follow sorted
    means. ``dark_fraction`` is conditional on the retained interior, not the
    whole-image phase fraction. Means are raw intensities; their difference
    is phase-contrast. ``separation`` divides that difference by pooled std.
    ``fit_converged`` reports numerical convergence, not physical validity.
    """
    image = np.asarray(image)
    if image.shape != (256, 256) or image.dtype != np.uint8:
        raise ValueError("Expected a 256x256 uint8 grayscale image")
    x = ndi.gaussian_filter(image.astype(np.float64) / 255.0, 1.0, mode="reflect")
    low, high = np.percentile(x, (2, 98))
    contrast = float(high - low)
    pore = np.zeros_like(x, dtype=bool)
    if contrast > 1e-8:
        candidates = label(x < low - 0.10 * contrast)
        for region in regionprops(candidates):
            if 2 <= region.area <= 128 and region.eccentricity < 0.9 and region.solidity > 0.8:
                box = tuple(slice(a, b) for a, b in zip(region.bbox[:2], region.bbox[2:]))
                pore[box] |= region.image
        pore = ndi.binary_dilation(pore, structure=np.ones((3, 3), dtype=bool))
    ridge = ndi.grey_closing(x, size=(7, 7), mode="reflect") - x
    has_ridges = float(np.ptp(ridge)) > 1e-12
    threshold = float(threshold_otsu(ridge)) if has_ridges else 0.0
    result = []
    for factor in BOUNDARY_MULTIPLIERS:
        boundary = (ridge > factor * threshold) if has_ridges else np.zeros_like(pore)
        boundary = ndi.binary_dilation(boundary, structure=np.ones((3, 3), dtype=bool))
        interior = ~(boundary | pore)
        fitted = _fit_populations(x[interior])
        result.extend((*fitted[:6], float(interior.mean()), float(boundary.mean()),
                       float(pore.mean()), fitted[6]))
    features = np.asarray(result, dtype=np.float64)
    if features.shape != (30,) or not np.isfinite(features).all():
        raise RuntimeError("Invalid interior-phase descriptors")
    return features, list(FEATURE_NAMES)
