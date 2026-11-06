import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import h_minima
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import watershed

TV_MULTIPLIERS = (0.5, 1.0)
PROMINENCE_MULTIPLIERS = (0.1, 0.2)
CONFIGURATIONS = tuple((w, h) for w in TV_MULTIPLIERS for h in PROMINENCE_MULTIPLIERS)
DESCRIPTORS = (
    "grain_count", "interior_count", "interior_diam_mean",
    "interior_diam_cv", "interior_inv_sqrt_diam",
    "interior_area_weighted_diam", "aspect_area_weighted_mean",
    "alignment", "boundary_fraction", "within_region_std",
    "region_mean_std", "region_mean_iqr", "small_region_area_fraction",
    "interior_area_fraction", "interior_diam_q90_q10_ratio",
)
_DESCRIPTOR_BLOCKS = (
    "size", "support", "size", "size", "size", "size", "shape",
    "shape", "support", "texture", "texture", "texture", "support",
    "support", "size",
)
FEATURE_NAMES = tuple(
    f"denoised_grain_{summary}_{name}"
    for name in DESCRIPTORS for summary in ("median", "range")
) + ("denoised_noise_sigma",)
BLOCKS = tuple(
    f"denoised_grain_{block}" for block in _DESCRIPTOR_BLOCKS for _ in range(2)
) + ("denoised_noise",)


def _canonical_d4(image):
    candidates = [np.ascontiguousarray(a) for k in range(4)
                  for a in (np.rot90(image, k), np.fliplr(np.rot90(image, k)))]
    return min(candidates, key=lambda a: a.tobytes())


def _noise_sigma(x):
    hh = (x[0::2, 0::2] - x[0::2, 1::2]
          - x[1::2, 0::2] + x[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(hh)) / 0.6744897501960817)


def _summarize(segmentation, x):
    all_regions = regionprops(segmentation, intensity_image=x)
    regions = [r for r in all_regions if r.area >= 16]
    inside = [r for r in regions if r.bbox[0] > 0 and r.bbox[1] > 0
              and r.bbox[2] < 256 and r.bbox[3] < 256]
    diam = np.array([r.equivalent_diameter_area for r in inside], dtype=float)
    interior_areas = np.array([r.area for r in inside], dtype=float)
    if diam.size:
        dmean = float(diam.mean())
        dcv = float(diam.std() / dmean)
        dinv = float(np.mean(1.0 / np.sqrt(diam)))
        dweighted = float(np.average(diam, weights=interior_areas))
        q10, q90 = np.percentile(diam, (10, 90))
        dratio = float(q90 / q10)
    else:
        dmean = dcv = dinv = dweighted = dratio = 0.0
    if regions:
        areas = np.array([r.area for r in regions], dtype=float)
        major = np.array([r.axis_major_length for r in regions])
        minor = np.array([r.axis_minor_length for r in regions])
        aspect = float(np.average(major / np.maximum(minor, 1.0), weights=areas))
        angles = np.array([r.orientation for r in regions])

        elongation = (major - minor) / np.maximum(major + minor, 1e-12)
        alignment = float(abs(np.sum(areas * elongation * np.exp(2j * angles))) / areas.sum())
        means = np.array([r.intensity_mean for r in regions])
        center = float(np.average(means, weights=areas))
        mean_std = float(np.sqrt(np.average((means - center) ** 2, weights=areas)))
        mean_iqr = float(np.percentile(means, 75) - np.percentile(means, 25))
        within = float(np.average([np.std(r.image_intensity[r.image]) for r in regions], weights=areas))
    else:
        aspect = alignment = mean_std = mean_iqr = within = 0.0
    return (
        float(len(regions)), float(len(inside)), dmean, dcv, dinv, dweighted,
        aspect, alignment, float(np.mean(segmentation == 0)), within,
        mean_std, mean_iqr,
        float(sum(r.area for r in all_regions if r.area < 16) / x.size),
        float(interior_areas.sum() / x.size), dratio,
    )


def extract_denoised(image):
    image = np.asarray(image)
    if image.shape != (256, 256) or image.dtype != np.uint8:
        raise ValueError("Expected a 256x256 uint8 grayscale image")
    x = _canonical_d4(image).astype(np.float64) / 255.0
    noise = _noise_sigma(x)
    summaries = []
    for factor in TV_MULTIPLIERS:
        weight = factor * noise
        denoised = (denoise_tv_chambolle(x, weight=weight, eps=1e-4,
                                       max_num_iter=50, channel_axis=None)
                    if weight > 1e-12 else x.copy())
        ridge = ndi.grey_closing(denoised, size=(7, 7), mode="reflect") - denoised
        ridge_mask = (ridge > threshold_otsu(ridge)
                      if float(np.ptp(ridge)) > 1e-12 else np.zeros_like(x, dtype=bool))
        gradient = ndi.gaussian_gradient_magnitude(denoised, sigma=0.7, mode="reflect")
        spread = float(np.ptp(gradient))
        q90 = float(np.percentile(gradient, 90))
        for prominence in PROMINENCE_MULTIPLIERS:
            if spread <= 1e-12:
                segmentation = np.ones(x.shape, dtype=np.int32)
            else:
                h = max(prominence * q90, 1e-12)
                minima = h_minima(gradient, h).astype(bool) & ~ridge_mask
                markers = label(minima, connectivity=2)
                if not markers.any():
                    # Preserve the seed exclusion in the degenerate fallback.
                    eligible = np.where(~ridge_mask, gradient, np.inf)
                    markers.flat[int(np.argmin(eligible))] = 1
                segmentation = watershed(gradient, markers, watershed_line=True)
            summaries.append(_summarize(segmentation, denoised))
    matrix = np.asarray(summaries, dtype=np.float64)
    result = np.column_stack((np.median(matrix, axis=0), np.ptp(matrix, axis=0))).ravel()
    result = np.append(result, noise)
    if result.shape != (31,) or not np.isfinite(result).all():
        raise RuntimeError("Invalid adaptive watershed descriptors")
    return result, list(FEATURE_NAMES), list(BLOCKS)
