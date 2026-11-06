"""Connected dark-boundary network and enclosed-interior proxies, image-local.

Unlike gradient-minimum watershed, interiors are connected components of the
complement of one linked dark-ridge mask. No markers or basin proliferation.
No labels, file I/O, global fitting, or hardness direction assumptions.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import disk, skeletonize
from skimage.restoration import denoise_tv_chambolle

DESCRIPTORS = (
    'basin_count', 'closed_count', 'closed_area_fraction', 'edge_area_fraction',
    'largest_basin_fraction', 'closed_diam_mean', 'closed_diam_cv',
    'closed_inv_sqrt_diam', 'closed_minor_mean', 'closed_aspect',
    'closed_alignment', 'region_mean_std', 'within_region_std',
    'network_fraction', 'network_largest_share', 'endpoints_per1000',
    'junctions_per1000', 'ridge_q90', 'compact_spot_fraction',
    'noise_sigma', 'small_basin_fraction', 'bridge_fraction',
)
FEATURE_NAMES = tuple('connected_' + n for n in DESCRIPTORS)
BLOCKS = tuple('connected_' + b for b in (
    'size','support','support','support','support','size','size','size','size',
    'shape','shape','texture','texture','network','network','network','network',
    'network','support','noise','support','network',
))
PARAMETERS = dict(tv_noise_multiplier=1.0, tv_iterations=80, blackhat_size=9,
                  hysteresis_low_multiplier=0.5, hysteresis_high_multiplier=1.0,
                  close_disk_radius=1, minimum_network_area=24, minimum_basin_area=16,
                  valley_offset_pixels=3.0, valley_directions=8, valley_smoothing_sigma=0.6)


def _canonical(image):
    options = [np.ascontiguousarray(a) for k in range(4)
               for a in (np.rot90(image, k), np.fliplr(np.rot90(image, k)))]
    return min(options, key=lambda a: a.tobytes())


def segment_connected(image):
    """Return image-local QA arrays; no output is written by this function."""
    image = np.asarray(image)
    if image.shape != (256, 256) or image.dtype != np.uint8:
        raise ValueError('Expected a 256x256 uint8 grayscale image')
    canonical = _canonical(image)
    x = canonical.astype(np.float64) / 255.
    hh = (x[::2,::2] - x[::2,1::2] - x[1::2,::2] + x[1::2,1::2]) / 2.
    noise = float(np.median(np.abs(hh)) / .6744897501960817)
    smooth = denoise_tv_chambolle(x, weight=noise, eps=1e-4, max_num_iter=80,
                                 channel_axis=None) if noise > 1e-12 else x.copy()
    blackhat = ndi.grey_closing(smooth, size=(9,9), mode='reflect') - smooth
    # A boundary is a narrow dark valley: opposite sides must BOTH be brighter.
    # A broad dark phase / one-sided intensity step fails this local condition.
    valley_image = ndi.gaussian_filter(smooth, .6, mode='reflect')
    yy, xx = np.indices(x.shape, dtype=float)
    valley = np.zeros_like(x)
    for angle in np.arange(8) * (np.pi / 8.):
        dy, dx = 3. * np.sin(angle), 3. * np.cos(angle)
        plus = ndi.map_coordinates(valley_image, [yy+dy, xx+dx], order=1, mode='reflect')
        minus = ndi.map_coordinates(valley_image, [yy-dy, xx-dx], order=1, mode='reflect')
        valley = np.maximum(valley, np.minimum(plus, minus) - valley_image)
    ridge = np.minimum(blackhat, valley)
    spots = np.zeros_like(x, dtype=bool)
    if float(np.ptp(ridge)) > 1e-12:
        high = float(threshold_otsu(ridge))
        raw = apply_hysteresis_threshold(ridge, .5 * high, high)
        # Compact isolated dark blobs are not a connected grain-boundary web.
        # Do not delete loops: low-solidity ring components are retained.
        for region in regionprops(label(raw, connectivity=2)):
            if 4 <= region.area <= 128 and region.eccentricity < .85 and region.solidity > .8:
                y0,x0,y1,x1 = region.bbox
                spots[y0:y1,x0:x1] |= region.image
        raw &= ~spots
        linked = ndi.binary_dilation(raw, structure=disk(1), border_value=0)
        linked = ndi.binary_erosion(linked, structure=disk(1), border_value=1)
        component = label(linked, connectivity=2)
        counts = np.bincount(component.ravel())
        keep = counts >= 24
        keep[0] = False
        linked = keep[component]
        bridge_fraction = float(np.mean(linked & ~raw))
    else:
        linked = np.zeros_like(x, dtype=bool)
        bridge_fraction = 0.0
        high = 0.0
    basins = label(~linked, connectivity=1)
    return dict(image=canonical, smooth=smooth, ridge=ridge, network=linked,
                basins=basins, spots=spots, noise=noise, threshold=high,
                bridge_fraction=bridge_fraction)


def extract_connected(image):
    """Return 22 float64 values, fixed unique names and feature-family blocks.

    D4 canonicalization resolves iterative-TV order dependence. Only enclosed
    components >=16px enter size/shape/contrast statistics; frame-touching
    components and unsupported/merged area remain explicit support features.
    Closed interiors are not ground-truth grains; gaps can merge real grains,
    while phase edges or residual noise can create false enclosures.
    """
    s = segment_connected(image)
    regions = regionprops(s['basins'], intensity_image=s['smooth'])
    eligible = [r for r in regions if r.area >= 16]
    closed = [r for r in eligible if r.bbox[0] > 0 and r.bbox[1] > 0
              and r.bbox[2] < 256 and r.bbox[3] < 256]
    edge = [r for r in eligible if r.bbox[0] == 0 or r.bbox[1] == 0
            or r.bbox[2] == 256 or r.bbox[3] == 256]
    size = float(image.size)
    if closed:
        areas = np.array([r.area for r in closed], float)
        diam = np.array([r.equivalent_diameter_area for r in closed])
        minor = np.array([r.axis_minor_length for r in closed])
        major = np.array([r.axis_major_length for r in closed])
        angles = np.array([r.orientation for r in closed])
        means = np.array([r.intensity_mean for r in closed])
        mean = float(np.average(means, weights=areas))
        diam_mean = float(diam.mean())
        diam_cv = float(diam.std() / diam_mean)
        hp = float(np.mean(1. / np.sqrt(diam)))
        minor_mean = float(minor.mean())
        aspect = float(np.average(major / np.maximum(minor, 1.), weights=areas))
        elongation = (major - minor) / np.maximum(major + minor, 1e-12)
        alignment = float(abs(np.sum(areas * elongation * np.exp(2j*angles))) / areas.sum())
        region_std = float(np.sqrt(np.average((means-mean)**2, weights=areas)))
        within = float(np.average([np.std(r.image_intensity[r.image]) for r in closed], weights=areas))
    else:
        diam_mean = diam_cv = hp = minor_mean = aspect = alignment = region_std = within = 0.
    network = s['network']
    parts = np.bincount(label(network, connectivity=2).ravel())[1:]
    largest_share = float(parts.max() / parts.sum()) if parts.size and parts.sum() else 0.
    skeleton = skeletonize(network)
    neighbours = ndi.convolve(skeleton.astype(np.uint8), np.ones((3,3), np.uint8), mode='constant') - skeleton
    length = int(skeleton.sum())
    endpoints = float(1000 * np.count_nonzero(skeleton & (neighbours == 1)) / length) if length else 0.
    junctions = float(1000 * np.count_nonzero(skeleton & (neighbours >= 3)) / length) if length else 0.
    values = [len(eligible), len(closed), sum(r.area for r in closed)/size,
              sum(r.area for r in edge)/size, max([r.area for r in eligible], default=0)/size,
              diam_mean, diam_cv, hp, minor_mean, aspect, alignment, region_std, within,
              float(network.mean()), largest_share, endpoints, junctions,
              float(np.percentile(s['ridge'],90)), float(s['spots'].mean()), s['noise'],
              sum(r.area for r in regions if r.area < 16)/size, s['bridge_fraction']]
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (22,) or not np.isfinite(result).all():
        raise RuntimeError('Invalid connected-grain schema')
    return result, list(FEATURE_NAMES), list(BLOCKS)
