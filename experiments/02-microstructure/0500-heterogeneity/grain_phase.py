"""Grain-level phase measure: watershed instances from U-Net boundary/grain maps, per-instance mean of U-Net phase share and of
raw intensity; per-image 2-component area-weighted intensity mixture initialised by the U-Net assignment. Label-free.
usage: grain_phase.py NET SPLIT  -> gph-NET-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
H = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]; net, split = sys.argv[1], sys.argv[2]
M = dict(np.load(_repo_path(H.parent, f'push-400/unet2maps-{net}-{split}.npz'))); ids = [str(i) for i in M['ids']]
KEYS = ['g_ph_inst', 'g_ph_gmm', 'g_ph_inst_cnt', 'g_n_inst', 'g_sep', 'g_ph_inst_big']
X = []
for n, i in enumerate(ids):
    m = M['maps'][n].astype(np.float32); up = lambda a: ndi.zoom(a, 2, order=1)
    p0, p1, p2, p3 = up(m[0]), up(m[1]), up(m[2]), up(m[3])
    a = np.array(Image.open(_repo_path(P, f'data/{split}/{i}.png'))).astype(np.float32); sm = ndi.gaussian_filter(a, 1.0)
    grain = (p0 + p1) > 0.35
    dist = ndi.gaussian_filter(1 - p3, 1.5)
    pk = peak_local_max(dist, min_distance=4, threshold_abs=0.5, labels=grain.astype(int))
    mk = np.zeros(a.shape, int); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(p3, mk, mask=(p2 < 0.5))
    idx = np.arange(1, lab.max() + 1)
    A = ndi.sum(np.ones_like(sm), lab, idx); ok = A >= 12
    I = ndi.mean(sm, lab, idx)[ok]; S = ndi.sum(p1, lab, idx)[ok] / np.maximum(ndi.sum(p0 + p1, lab, idx)[ok], 1e-3); A = A[ok]
    o = {}
    ph = S > 0.5
    o['g_ph_inst'] = A[ph].sum() / A.sum(); o['g_ph_inst_cnt'] = ph.mean(); o['g_n_inst'] = float(len(A))
    big = A >= np.median(A); o['g_ph_inst_big'] = A[ph & big].sum() / max(A[big].sum(), 1)
    # area-weighted 2-component 1D mixture on instance intensities, init from U-Net split
    if ph.sum() >= 2 and (~ph).sum() >= 2:
        mu = np.array([np.average(I[ph], weights=A[ph]), np.average(I[~ph], weights=A[~ph])]); sd = np.array([np.sqrt(np.average((I[ph] - mu[0]) ** 2, weights=A[ph])), np.sqrt(np.average((I[~ph] - mu[1]) ** 2, weights=A[~ph]))]) + 1.0
        pi = np.array([A[ph].sum(), A[~ph].sum()]) / A.sum()
        for _ in range(30):
            L = np.stack([pi[c] * np.exp(-0.5 * ((I - mu[c]) / sd[c]) ** 2) / sd[c] for c in range(2)]) + 1e-12; R_ = L / L.sum(0)
            w = R_ * A; pi = w.sum(1) / w.sum(); mu = (w * I).sum(1) / w.sum(1); sd = np.sqrt((w * (I - mu[:, None]) ** 2).sum(1) / w.sum(1)) + 0.5
        dark = int(np.argmin(mu)); o['g_ph_gmm'] = (R_[dark] * A).sum() / A.sum(); o['g_sep'] = abs(mu[1] - mu[0]) / np.sqrt(sd[0] * sd[1])
    else:
        o['g_ph_gmm'] = o['g_ph_inst']; o['g_sep'] = 0.0
    X.append([float(o[k]) for k in KEYS])
    if (n + 1) % 100 == 0: print(n + 1, flush=True)
np.savez(_repo_path(H, f'gph-{net}-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(KEYS)); print('done')
