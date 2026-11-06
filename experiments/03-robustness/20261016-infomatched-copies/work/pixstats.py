"""Label-free per-image pixel statistics used to calibrate the information-matched renderings (real noisy TRAIN images only; no labels, no test).
Groups: noise (Haar HH), Hann-windowed radial power spectrum in octave bands (raw, comparable because copy noise is drawn from the real pool),
noise-floor-relative band power, shading-relative darkness quantiles after smoothing (phase contrast / dark-mode separation / pore tail),
fine-scale ridge (dark boundary line) energy, gradient quantiles at two scales (edge sharpness), bright-side spread (matrix mottling),
spectral anisotropy.  stats(img) -> dict; usage: pixstats.py SET_DIR_OR_'real'|'clean' OUT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from PIL import Image
E = _RepoPath(__file__).resolve().parents[3]; P = _RepoPath(__file__).resolve().parents[4]
N = 256; fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing='ij'); FR = np.hypot(fy, fx); ANG = np.arctan2(fy, fx)
HANN = np.outer(np.hanning(N), np.hanning(N)); WE = (HANN ** 2).sum()
BANDS = [(0.004, 0.008), (0.008, 0.016), (0.016, 0.031), (0.031, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.5)]
BM = [(FR >= lo) & (FR < hi) for lo, hi in BANDS]
YY, XX = np.mgrid[0:N, 0:N] / (N - 1.0) - 0.5; DQ = np.c_[np.ones(N * N), XX.ravel(), YY.ravel(), XX.ravel() ** 2, YY.ravel() ** 2, (XX * YY).ravel()]

def noise_hh(a):
    hh = (a[0::2, 0::2] - a[0::2, 1::2] - a[1::2, 0::2] + a[1::2, 1::2]) / 2; return float(np.median(np.abs(hh)) / 0.6745)

def background(g):
    """robust quadratic surface over the bright (matrix) pixels, iterated like realdeg2's grain-level fit"""
    v = g.ravel(); keep = v > np.quantile(v, 0.4)
    for _ in range(4):
        b = np.linalg.lstsq(DQ[keep], v[keep], rcond=None)[0]; S = DQ @ b; keep = v / S > 0.97
    return S.reshape(N, N)

KEYS = None
def stats(a):
    a = a.astype(np.float64); o = {}; sd = noise_hh(a); o['noise'] = sd; o['mean'] = a.mean()
    F = np.abs(np.fft.fft2(HANN * (a - a.mean()))) ** 2 / WE
    for k, m in enumerate(BM):
        pb = F[m].mean(); o[f'lp{k}'] = np.log(pb); o[f'rp{k}'] = np.log(max(pb - sd * sd, 1e-2 * sd * sd + 1e-6))
    m = BM[3] | BM[4]; w = F[m]; c2 = (w * np.cos(2 * ANG[m])).sum() / w.sum(); s2 = (w * np.sin(2 * ANG[m])).sum() / w.sum(); o['aniso_mid'] = np.hypot(c2, s2)
    m = BM[1] | BM[2]; w = F[m]; c2 = (w * np.cos(2 * ANG[m])).sum() / w.sum(); s2 = (w * np.sin(2 * ANG[m])).sum() / w.sum(); o['aniso_low'] = np.hypot(c2, s2)
    for s in (1.5, 3.0):
        g = ndi.gaussian_filter(a, s); S = background(g); z = 1 - g / S
        q = np.quantile(z, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.998])
        for p, v in zip((5, 25, 50, 75, 90, 95, 99, 998), q): o[f'z{s:g}_q{p}'] = v
        o[f'z{s:g}_brightsd'] = z[z < q[2]].std(); o[f'z{s:g}_f06'] = (z > 0.06).mean(); o[f'z{s:g}_f12'] = (z > 0.12).mean(); o[f'z{s:g}_f30'] = (z > 0.30).mean()
        o[f'shade{s:g}'] = S.std() / S.mean()
    for s in (1.0, 2.0):
        Hyy = ndi.gaussian_filter(a, s, order=(2, 0)); Hxx = ndi.gaussian_filter(a, s, order=(0, 2)); Hxy = ndi.gaussian_filter(a, s, order=(1, 1))
        l1 = 0.5 * (Hxx + Hyy) + np.sqrt(0.25 * (Hxx - Hyy) ** 2 + Hxy ** 2); l2 = 0.5 * (Hxx + Hyy) - np.sqrt(0.25 * (Hxx - Hyy) ** 2 + Hxy ** 2)
        ridge = np.maximum(l1, 0) * (np.abs(l2) < 0.5 * np.abs(l1)) * s * s   # dark line: one strong positive curvature, weak other
        o[f'ridge{s:g}_mean'] = ridge.mean(); o[f'ridge{s:g}_q99'] = np.quantile(ridge, 0.99)
    for s in (1.0, 2.5):
        gm = np.hypot(ndi.gaussian_filter(a, s, order=(1, 0)), ndi.gaussian_filter(a, s, order=(0, 1))) * s
        q = np.quantile(gm, [0.5, 0.9, 0.99]); o[f'grad{s:g}_q50'], o[f'grad{s:g}_q90'], o[f'grad{s:g}_q99'] = q
    o['sharp'] = o['grad1_q99'] / max(o['grad2.5_q99'], 1e-6)
    bp = ndi.gaussian_filter(a, 4) - ndi.gaussian_filter(a, 16); o['bp4_16_sd'] = bp.std(); o['bp4_16_kurt'] = ((bp - bp.mean()) ** 4).mean() / bp.var() ** 2
    o['bp4_16_skew'] = ((bp - bp.mean()) ** 3).mean() / bp.std() ** 3
    return o

def stats_dir(files):
    rows = [stats(np.asarray(Image.open(_repo_io(f)))) for f in files]; keys = list(rows[0]); return keys, np.array([[r[k] for k in keys] for r in rows])

if __name__ == '__main__':
    what, out = sys.argv[1], sys.argv[2]
    ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]   # ID column only
    nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255
    if what in ('real', 'clean', 'all'):
        sel = {'real': np.where(nz >= 12)[0], 'clean': np.where(nz < np.median(nz))[0], 'all': np.arange(len(ids))}[what]
        files = [_repo_path(P, f'data/train/{ids[k]}.png') for k in sel]
    else:
        files = sorted(Path(what).glob('*.png'))
    keys, X = stats_dir(files)
    np.savez(_repo_io(out), ids=np.array([Path(f).stem for f in files]), X=X, keys=np.array(keys)); print(what, X.shape)
