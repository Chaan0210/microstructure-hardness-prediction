"""Precise mask-based 'generator-level' quantities on clean images (zero-shot masks-v3): phase area fraction at several darkness thresholds,
pore area fraction, grain size moments (log area), aspect ratio, orientation order parameter, size-field heterogeneity, phase clustering."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, csv
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
E = Path(str(_repo_path(_REPO_ROOT, 'experiments'))); split = sys.argv[1]
M = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]
KEYS = ['q_ph10', 'q_ph20', 'q_pore', 'q_lA_mean', 'q_lA_sd', 'q_lA_skew', 'q_ar', 'q_lar', 'q_order', 'q_het32', 'q_het64', 'q_n', 'q_ph_cnt10', 'q_ph_size', 'q_lA_q90', 'q_lA_q10']
def f(img, m):
    a = ndi.gaussian_filter(img.astype(float), 0.7); m = m.astype(np.int64); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    P = measure.regionprops(m); n = m.max() + 1
    if len(P) < 12: return [np.nan] * len(KEYS)
    cnt = np.bincount(inner.ravel(), minlength=n).astype(float); I = np.bincount(inner.ravel(), weights=a.ravel(), minlength=n) / np.maximum(cnt, 1)
    lab = np.array([p.label for p in P], dtype=np.int64); A = np.array([p.area for p in P], float); ok = (cnt[lab] >= 3) & (A >= 10); lab, A = lab[ok], A[ok]; P = [p for p, o in zip(P, ok) if o]
    I = I[lab]; o = np.argsort(I); cw = np.cumsum(A[o]) / A.sum(); ref = I[o][np.searchsorted(cw, 0.75)]; z = (ref - I) / ref
    sol = np.array([p.solidity for p in P]); ar = np.array([p.axis_major_length / max(p.axis_minor_length, 1) for p in P]); th = np.array([p.orientation for p in P])
    pore = (z > 0.45) & (sol > 0.85) & (ar < 2.5) & (A < 3 * np.median(A)); g = ~pore; w = A[g] / A[g].sum(); la = np.log(A[g])
    ph10 = (z > 0.10) & g; ph20 = (z > 0.20) & g; tot = A.sum()
    c2 = (w * np.cos(2 * th[g])).sum(); s2 = (w * np.sin(2 * th[g])).sum(); order = np.hypot(c2, s2)
    cy = np.array([p.centroid[0] for p in P])[g]; cx = np.array([p.centroid[1] for p in P])[g]
    def het(bs):
        nb = 256 // bs; vals = []
        for i in range(nb):
            for j in range(nb):
                s = (cy // bs == i) & (cx // bs == j)
                if s.sum() >= 3: vals.append(np.log(A[g][s].mean()))
        return np.std(vals) if len(vals) > 2 else np.nan
    mu = (w * la).sum(); sd = np.sqrt((w * (la - mu) ** 2).sum())
    return [A[ph10].sum() / tot, A[ph20].sum() / tot, A[pore].sum() / tot, la.mean(), la.std(), ((la - la.mean()) ** 3).mean() / max(la.std(), 1e-6) ** 3,
            (w * ar[g]).sum(), (w * np.log(ar[g])).sum(), order, het(32), het(64), float(g.sum()), ph10.sum() / g.sum(), (A[ph10].mean() / A[g & ~ph10].mean()) if ph10.any() else 1.0,
            np.quantile(la, 0.9), np.quantile(la, 0.1)]
X = np.array([f(np.array(Image.open(_repo_path(E.parent, f'data/{split}/{i}.png'))), M['masks'][k]) for k, i in enumerate(ids)])
np.savez(_repo_path(E, f'push-1400/research/exactq-{split}.npz'), ids=np.array(ids), X=X, keys=np.array(KEYS)); print(split, X.shape)
