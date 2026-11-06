"""Image-local line-intercept morphology proxies (36 deterministic features).

No labels, dataset fitting, file access, or optional dependencies. Distances are
pixels, not calibrated physical grain diameters. ``directional_cv`` measures
anisotropy of four sampled directions, not an absolute orientation angle.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

_SIGMAS = (1.0, 2.0)
_THRESHOLDS = (0.8, 1.0, 1.2)
_METRICS = (
    "gap_mean", "gap_cv", "gap_inv_sqrt_mean", "directional_cv",
    "uncensored_coverage", "crossings_per_pixel",
)

# Configuration-major order: reshape values to (6, 6) for sensitivity summaries.
CONFIGURATIONS = tuple((s, t) for s in _SIGMAS for t in _THRESHOLDS)
DESCRIPTORS = _METRICS
FEATURE_NAMES = tuple(
    f"intercept_s{s:g}_t{t:g}_{metric}"
    for s, t in CONFIGURATIONS for metric in DESCRIPTORS
)


def _line_stats(boundary, valid, step):
    """Measure complete clear runs bounded by boundary bands on both sides."""
    clear = valid & ~boundary
    changes = np.diff(np.pad(clear.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(changes == 1)
    end_rows, ends = np.nonzero(changes == -1)
    # Every clear run has one start/end; padding retains image-edge runs until
    # the explicit censoring check below. End indices are exclusive.
    if not np.array_equal(rows, end_rows):
        raise RuntimeError("Unpaired line runs")
    complete = (starts > 0) & (ends < clear.shape[1])
    ix = np.flatnonzero(complete)
    complete[ix] &= valid[rows[ix], starts[ix] - 1] & valid[rows[ix], ends[ix]]
    gaps = (ends[complete] - starts[complete]).astype(np.float64) * step
    # Count boundary bands, rather than individual boundary pixels. All bands
    # count toward density; coverage only counts uncensored clear intervals.
    bands = valid & boundary
    band_changes = np.diff(np.pad(bands.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    crossings = int(np.count_nonzero(band_changes == 1))
    sampled_length = float(valid.sum()) * step
    return gaps, crossings, sampled_length


def _intercept_stats(edge):
    n = edge.shape[0]
    directions = []
    valid = np.ones_like(edge, dtype=bool)
    for boundary in (edge, edge.T):
        directions.append(_line_stats(boundary, valid, 1.0))
    # Each diagonal is sampled once. Ignore short corner lines (<16 samples),
    # whose predominantly censored intervals would distort support statistics.
    xx = np.broadcast_to(np.arange(n)[None, :], (2 * n - 1, n))
    offsets = np.arange(-(n - 1), n)[:, None]
    for yy in (xx + offsets, (n - 1 - xx) + offsets):
        valid = (yy >= 0) & (yy < n)
        valid &= valid.sum(axis=1, keepdims=True) >= 16
        boundary = edge[np.clip(yy, 0, n - 1), xx] & valid
        directions.append(_line_stats(boundary, valid, np.sqrt(2.0)))
    gaps = np.concatenate([d[0] for d in directions])
    total_length = sum(d[2] for d in directions)
    if gaps.size:
        mean = float(gaps.mean())
        cv = float(gaps.std() / mean)
        inverse_sqrt = float(np.mean(1.0 / np.sqrt(gaps)))
    else:
        mean = cv = inverse_sqrt = 0.0
    # Include zero-crossing directions: parallel scans through aligned bands
    # carry useful anisotropy information even when they yield no complete gap.
    direction_density = np.array([d[1] / d[2] for d in directions])
    directional_cv = (float(direction_density.std() / direction_density.mean())
                      if direction_density.mean() > 0.0 else 0.0)
    return (
        mean, cv, inverse_sqrt, directional_cv,
        float(gaps.sum() / total_length),
        float(sum(d[1] for d in directions) / total_length),
    )


def extract_latent(image):
    """Return (36 float64 values, unique names) for one 256x256 uint8 image.

    Gaussian smoothing and a 7x7 grayscale closing identify dark local ridges.
    A whole-image Otsu threshold, multiplied by .8/1/1.2, tests segmentation
    sensitivity. At each of two smoothing scales, horizontal, vertical and both
    diagonal scans yield complete boundary-to-boundary clear intervals.

    Rotation/reflection invariance is with respect to the square's D4 group;
    arbitrary-angle invariance is approximate. ``directional_cv`` is the CV of
    boundary-band crossing density across four directions, including zero
    densities. ``gap_inv_sqrt_mean`` is only a
    Hall-Petch-shaped proxy, without imposed coefficient/sign. Gap statistics
    are intercept-weighted, not number- or area-weighted true grain sizes.
    Constant images produce zeros; no complete interval means zero gap stats.
    """
    image = np.asarray(image)
    if image.shape != (256, 256) or image.dtype != np.uint8:
        raise ValueError("Expected a 256x256 uint8 grayscale image")
    x = image.astype(np.float64) / 255.0
    values, names = [], []
    for sigma in _SIGMAS:
        smooth = ndi.gaussian_filter(x, sigma=sigma, mode="reflect")
        ridge = ndi.grey_closing(smooth, size=(7, 7), mode="reflect") - smooth
        has_contrast = float(np.ptp(ridge)) > 1e-12
        threshold = float(threshold_otsu(ridge)) if has_contrast else 0.0
        for factor in _THRESHOLDS:
            stats = (_intercept_stats(ridge > factor * threshold)
                     if has_contrast else (0.0,) * len(_METRICS))
            prefix = f"intercept_s{sigma:g}_t{factor:g}"
            names.extend(f"{prefix}_{metric}" for metric in _METRICS)
            values.extend(stats)
    features = np.asarray(values, dtype=np.float64)
    if features.shape != (36,) or len(set(names)) != 36 or not np.isfinite(features).all():
        raise RuntimeError("Invalid line-intercept feature schema")
    return features, names
