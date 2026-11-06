"""Is there hardness information in the SHAPE of the per-grain darkness distribution beyond the 044 features?
Clean dev images: per-grain shading-corrected darkness d (gray), area-weighted histogram (14 bins); fold-honest RidgeCV on the 044 residual
(5-fold inside the clean set, repeated), compared with shuffled-row controls. Also the cumulative form (area fraction with d > t)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'; P = E + '../'
R = np.load(_repo_io('eda/resid044.npz'), allow_pickle=True); ids = [str(i) for i in R['ids']]; r = R['r']; X0 = R['X']; k0 = list(R['keys'])
nz = X0[:, k0.index('noise')] * 255; clean = np.where(nz < 9.3)[0]
U = np.load(_repo_io(E + 'push-400/unet4maps-z1-train.npz')); upos = {str(i): k for k, i in enumerate(U['ids'])}
Mc = np.load(_repo_io(E + 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
edges = np.array([-1e9, -8, -4, 0, 4, 8, 12, 16, 20, 25, 30, 35, 40, 50, 1e9]); H = []; keep = []
for k in clean:
    i = ids[k]; im = np.asarray(Image.open(_repo_io(P + f'data/train/{i}.png')), np.float32); mask = Mc['masks'][cpos[i]].astype(int)
    mp = U['maps'][upos[i]].astype(np.float32); ph = np.kron(mp[1], np.ones((2, 2))); po = np.kron(mp[2], np.ones((2, 2)))
    inner = np.where(ndi.minimum_filter(mask, 5) == ndi.maximum_filter(mask, 5), mask, 0); labs = np.unique(inner); labs = labs[labs > 0]
    if len(labs) < 25: continue
    I = np.asarray(ndi.mean(im, inner, labs)); pph = np.asarray(ndi.mean(ph, inner, labs)); ppo = np.asarray(ndi.mean(po, inner, labs))
    A = np.asarray(ndi.sum(np.ones_like(im), mask, labs)); cen = np.array(ndi.center_of_mass(np.ones_like(im), inner, labs))
    kp = ppo < 0.3; I, pph, A, cen = I[kp], pph[kp], A[kp], cen[kp]; mat = pph < 0.15
    if mat.sum() < 15: continue
    d2 = ((cen[:, None, :] - cen[mat][None, :, :]) ** 2).sum(-1); W = np.exp(-d2 / (2 * 48 ** 2)); W[np.where(mat)[0], np.arange(mat.sum())] = 0
    d = (W @ I[mat]) / W.sum(1) - I
    H.append(np.histogram(d, edges, weights=A)[0] / A.sum()); keep.append(k)
H = np.array(H); keep = np.array(keep); rr = r[keep]
C = np.cumsum(H[:, ::-1], 1)[:, ::-1][:, 1:]   # area fraction with d > edge
def cv_r2(Z, y, reps=20, shuffle=False):
    out = []
    for s in range(reps):
        yy = np.random.RandomState(s).permutation(y) if shuffle else y; pred = np.zeros_like(y)
        for tr, va in KFold(5, shuffle=True, random_state=s).split(Z):
            m = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Z[tr], yy[tr]); pred[va] = m.predict(Z[va])
        out.append(1 - np.mean((yy - pred) ** 2) / np.var(yy))
    return np.mean(out), np.std(out)
print('clean images used', len(keep), 'resid sd', rr.std().round(2))
for nm, Z in (('hist14', H), ('cum13', C)):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    print(f'{nm}: CV R2 real {cv_r2(Zs, rr)[0]:+.4f}  shuffled-label {cv_r2(Zs, rr, 20, True)[0]:+.4f}')
for j in range(C.shape[1]):
    print(f'  frac(d>{edges[j + 1]:+.0f}) corr with resid {np.corrcoef(C[:, j], rr)[0, 1]:+.3f}')
