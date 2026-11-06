"""Per-grain darkness distribution from zero-shot masks (clean images): each grain's interior mean intensity I_i, matrix reference = area-weighted
75th percentile of grain intensities (features.py rule), z_i = (ref - I_i)/ref. Area-weighted darkness moments and histogram, darkness x size terms.
usage: grain_dark.py SPLIT -> gdark-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
E = Path(str(_repo_path(_REPO_ROOT, 'experiments'))); split = sys.argv[1]
M = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]
KEYS = ['gd_wmean', 'gd_wsd', 'gd_wskew', 'gd_f10', 'gd_f20', 'gd_f30', 'gd_f45', 'gd_f60', 'gd_area_x_z', 'gd_dark_size_ratio', 'gd_ref_rel', 'gd_cont']
def feats(img, m):
    a = ndi.gaussian_filter(img.astype(float), 0.7); m = m.astype(np.int64); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    n = m.max() + 1; A = np.bincount(m.ravel(), minlength=n)[1:].astype(float); cnt = np.bincount(inner.ravel(), minlength=n)[1:].astype(float)
    I = np.bincount(inner.ravel(), weights=a.ravel(), minlength=n)[1:] / np.maximum(cnt, 1); ok = (cnt >= 3) & (A >= 10)
    A, I = A[ok], I[ok]
    if len(A) < 10: return [np.nan] * len(KEYS)
    o = np.argsort(I); cw = np.cumsum(A[o]) / A.sum(); ref = I[o][np.searchsorted(cw, 0.75)]; z = np.clip((ref - I) / ref, -0.2, 1.0); w = A / A.sum()
    mu = (w * z).sum(); sd = np.sqrt((w * (z - mu) ** 2).sum()); sk = (w * (z - mu) ** 3).sum() / max(sd, 1e-6) ** 3
    fr = [(w * (z > t)).sum() for t in (0.10, 0.20, 0.30, 0.45, 0.60)]
    dk = z > 0.2; dsr = (A[dk].mean() / A[~dk].mean()) if dk.any() and (~dk).any() else np.nan
    return [mu, sd, sk] + fr + [(w * z * np.log(A)).sum(), dsr, ref / a.mean(), (w * np.clip(z, 0, None) ** 2).sum()]
X = np.array([feats(np.array(Image.open(_repo_path(E.parent, f'data/{split}/{i}.png'))), M['masks'][k]) for k, i in enumerate(ids)])
np.savez(_repo_path(E, f'push-1400/research/gdark-{split}.npz'), ids=np.array(ids), X=X, keys=np.array(KEYS)); print(split, X.shape, np.isnan(X).sum())
