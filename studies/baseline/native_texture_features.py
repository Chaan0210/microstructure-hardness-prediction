"""Native 256px directional-spectrum and local-texture descriptors (28).

Image-local NumPy/SciPy only: no labels, dataset fitting, file access, or target
sign assumptions. D4 invariants preserve alignment strength and relative-scale
alignment without using absolute orientation angles.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import ndimage as ndi

FREQUENCY_BANDS = (
    ("coarse", 1.0 / 64.0, 1.0 / 16.0),
    ("middle", 1.0 / 16.0, 1.0 / 4.0),
    ("fine", 1.0 / 4.0, 1.0 / 2.0),
)
BAND_PAIRS = ((0, 1), (0, 2), (1, 2))
TILE_SIZES = (32, 64)
TILE_DESCRIPTORS = (
    "fine_energy_cv", "coarse_energy_cv", "fine_share_q10",
    "fine_share_q50", "fine_share_q90", "mean_fine_energy_corr",
    "fine_coarse_energy_corr", "fine_top_quartile_energy_share",
)
FEATURE_NAMES = tuple(
    [f"native_fft_{name}_{metric}"
     for name, _, _ in FREQUENCY_BANDS
     for metric in ("h2", "h4", "flatness")]
    + [f"native_fft_{FREQUENCY_BANDS[i][0]}_{FREQUENCY_BANDS[j][0]}_alignment"
       for i, j in BAND_PAIRS]
    + [f"native_tile{size}_{metric}"
       for size in TILE_SIZES for metric in TILE_DESCRIPTORS]
)
BLOCKS = ("native_fourier",) * 12 + ("native_tiles",) * 16
_ENERGY_EPS = 1e-20


def _correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if denom <= _ENERGY_EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _spectrum_features(centered):
    n = centered.shape[0]
    window = np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(n))
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.hypot(fy, fx)
    angle = np.arctan2(fy, fx)
    values, h2_by_band = [], []
    for index, (_, low, high) in enumerate(FREQUENCY_BANDS):
        band = (radius >= low) & ((radius <= high) if index == 2 else (radius < high))
        p = power[band]
        theta = angle[band]
        total = float(p.sum())
        if total <= _ENERGY_EPS:
            h2 = 0.0j
            h4 = 0.0j
            flatness = 0.0
        else:
            h2 = np.sum(p * np.exp(2.0j * theta)) / total
            h4 = np.sum(p * np.exp(4.0j * theta)) / total
            mean = float(p.mean())
            # Relative floor regularizes exact spectral zeros without adding
            # global intensity statistics or claiming a noise calibration.
            floor = max(mean * 1e-12, 1e-30)
            flatness = float(np.clip(np.exp(np.log(np.maximum(p, floor)).mean()) / mean, 0.0, 1.0))
        values.extend((float(abs(h2)), float(abs(h4)), flatness))
        h2_by_band.append(h2)
    for i, j in BAND_PAIRS:
        # Weighted alignment avoids unstable angle estimates in isotropic bands.
        values.append(float(np.real(h2_by_band[i] * np.conj(h2_by_band[j]))))
    return values


def _tile_mean(array, size):
    n = array.shape[0] // size
    return array.reshape(n, size, n, size).mean(axis=(1, 3)).ravel()


def _tile_features(x, centered):
    # Filter the entire native image first; cutting tiles before filtering
    # would introduce artificial local boundary responses.
    fine = (0.7 ** 2 * ndi.gaussian_laplace(centered, 0.7, mode="reflect")) ** 2
    coarse = (2.8 ** 2 * ndi.gaussian_laplace(centered, 2.8, mode="reflect")) ** 2
    values = []
    for size in TILE_SIZES:
        f = _tile_mean(fine, size)
        c = _tile_mean(coarse, size)
        mean_intensity = _tile_mean(x, size)
        f_mean, c_mean = float(f.mean()), float(c.mean())
        f_cv = float(f.std() / f_mean) if f_mean > _ENERGY_EPS else 0.0
        c_cv = float(c.std() / c_mean) if c_mean > _ENERGY_EPS else 0.0
        total = f + c
        share = np.divide(f, total, out=np.zeros_like(f), where=total > _ENERGY_EPS)
        q10, q50, q90 = np.percentile(share, (10, 50, 90))
        concentration = (float(np.sort(f)[-(f.size // 4):].sum() / f.sum())
                         if f_mean > _ENERGY_EPS else 0.0)
        values.extend((f_cv, c_cv, float(q10), float(q50), float(q90),
                       _correlation(mean_intensity, f), _correlation(f, c), concentration))
    return values


def extract_native(image):
    """Return (28 float64 values, names, blocks) for 256x256 uint8 input.

    Fourier features use the mean-centered image with a symmetric 2D Hann
    window. Three radial bands retain angular harmonics |H2| and |H4|,
    spectral flatness, and pairwise Re(H2_i * conjugate(H2_j)). No duplicated
    radial-band total powers or raw axis-autocorrelations are included.

    Tile features use scale-normalized LoG energy at sigma .7 and 2.8 pixels,
    computed before partitioning into 32px/64px native tiles. They measure
    spatial heterogeneity and cross-scale co-location. Correlations without
    variance and all zero-energy quantities are defined as zero. Centering
    before LoG guarantees exact zero descriptors for constant images.

    Tile-grid location is fixed; D4 invariance does not imply translation or
    arbitrary-angle invariance. Flatness reflects spectral dispersion and is
    not a standalone sensor-noise estimate. Texture and blur may both affect
    these descriptors; no hardness-effect direction is assumed.
    """
    image = np.asarray(image)
    if image.shape != (256, 256) or image.dtype != np.uint8:
        raise ValueError("Expected a 256x256 uint8 grayscale image")
    x = image.astype(np.float64) / 255.0
    # Subtract an exactly representable sample first so uniform uint8 inputs
    # remain exactly zero despite floating-point mean accumulation.
    centered = x - x[0, 0]
    centered -= centered.mean()
    result = np.asarray(_spectrum_features(centered) + _tile_features(x, centered), dtype=np.float64)
    if result.shape != (28,) or not np.isfinite(result).all():
        raise RuntimeError("Invalid native texture descriptors")
    return result, list(FEATURE_NAMES), list(BLOCKS)
