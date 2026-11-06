"""Image-only descriptors: no dataset-level fitting or target access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

FEATURE_VERSION = "morphology-v2"

def read_image(path):
    with Image.open(_repo_io(path)) as im:
        if im.size != (256, 256) or im.mode != "L":
            raise ValueError(f"Expected 256x256 8-bit grayscale PNG: {path}")
        return np.asarray(im, dtype=np.uint8).copy()

def canonical_hash(image):
    """Group exact pixel duplicates, including 90-degree rotations/reflections."""
    return min(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
               for k in range(4) for a in (np.rot90(image, k), np.fliplr(np.rot90(image, k))))

def extract_base(image):
    x = image.astype(np.float64) / 255.
    values, names = [], []
    def add(name, value):
        values.append(float(value)); names.append(name)
    def stats(prefix, a):
        a = np.asarray(a, dtype=float).ravel()
        if not a.size:
            a = np.zeros(1)
        add(prefix+"_mean", a.mean()); add(prefix+"_std", a.std())
        for q, v in zip((0, 10, 25, 50, 75, 90, 100), np.percentile(a, (0,10,25,50,75,90,100))):
            add(prefix+f"_q{q}", v)
    stats("intensity", x)
    hist = np.histogram(x, bins=32, range=(0,1))[0] / x.size
    for i, v in enumerate(hist):
        add(f"hist_{i}", v)
    add("entropy", -(hist*np.log(hist+1e-12)).sum())
    for sigma in (0, 1, 2, 4):
        sm = ndi.gaussian_filter(x, sigma) if sigma else x
        gy, gx = np.gradient(sm)
        mag = np.hypot(gx, gy)
        stats(f"gradient_s{sigma}", mag)
        stats(f"laplace_s{sigma}", ndi.laplace(sm))
        a, b, c = np.mean(gx*gx), np.mean(gy*gy), np.mean(gx*gy)
        add(f"anisotropy_s{sigma}", np.sqrt((a-b)**2+4*c*c)/(a+b+1e-12))
        add(f"residual_s{sigma}", np.std(x-sm))
    # Spectrum in radial bands; preserve isotropic spatial scale information.
    centered = x-x.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered)))**2
    yy, xx = np.indices(x.shape)
    rr = np.hypot(yy-128, xx-128)
    total = power.sum()+1e-12
    edges = (0,2,4,8,12,16,24,32,48,64,96,128,182)
    for lo, hi in zip(edges[:-1], edges[1:]):
        add(f"fft_{lo}_{hi}", power[(rr>=lo)&(rr<hi)].sum()/total)
    # Autocorrelation features use only this image.
    for distance in (1,2,4,8,16,32,64):
        denom = np.var(x)+1e-12
        correlations = [np.mean(centered[:-distance,:]*centered[distance:,:])/denom,
                        np.mean(centered[:,:-distance]*centered[:,distance:])/denom]
        add(f"autocorr_{distance}", np.mean(correlations))
        add(f"autocorr_anisotropy_{distance}", abs(correlations[0]-correlations[1]))
    quant = (image//16).astype(np.uint8)
    glcm = graycomatrix(quant, [1,2,4,8], [0,np.pi/4,np.pi/2,3*np.pi/4],
                       levels=16, symmetric=True, normed=True)
    for prop in ("contrast","dissimilarity","homogeneity","energy","correlation","ASM"):
        p = graycoprops(glcm, prop)
        for i,d in enumerate((1,2,4,8)):
            add(f"glcm_{prop}_{d}", p[i].mean())
            add(f"glcm_{prop}_{d}_angle_std", p[i].std())
    for radius in (1,3,5):
        points = 8*radius
        lbp = local_binary_pattern(image, points, radius, method="uniform")
        h = np.bincount(lbp.astype(int).ravel(), minlength=points+2)/x.size
        for i,v in enumerate(h):
            add(f"lbp_r{radius}_{i}", v)
    smooth = ndi.gaussian_filter(x, 1)
    low, high = np.percentile(smooth, (2,98))
    normalized = np.clip((smooth-low)/max(high-low,1e-6), 0,1)
    thresholds = (("otsu", threshold_otsu(smooth)), ("dark10", .10),
                  ("dark25", .25), ("dark40", .40), ("dark60", .60))
    for name, threshold in thresholds:
        source = smooth if name == "otsu" else normalized
        mask = source < threshold
        add(name+"_fraction", mask.mean())
        add(name+"_boundary", np.mean(np.abs(np.diff(mask.astype(float),axis=0)))+
            np.mean(np.abs(np.diff(mask.astype(float),axis=1))))
        regions = [r for r in regionprops(label(mask)) if r.area>=4]
        add(name+"_count", len(regions))
        for attr in ("area", "eccentricity", "solidity", "axis_major_length"):
            stats(name+"_"+attr, [getattr(r,attr) for r in regions])
        if regions:
            weights = np.array([r.area for r in regions])
            orientations = np.array([r.orientation for r in regions])
            alignment = abs(np.sum(weights*np.exp(2j*orientations)))/weights.sum()
        else:
            alignment = 0.
        add(name+"_alignment", alignment)
        stats(name+"_distance", ndi.distance_transform_edt(mask)[mask])
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite image descriptors")
    return arr, names

from .extra import extract_extra
def extract(image):
    a,n=extract_base(image)
    b,m=extract_extra(image)
    return np.concatenate([a,b]),n+m
