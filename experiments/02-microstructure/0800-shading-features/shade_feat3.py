"""Shading-corrected grain-threshold second-phase area fraction (definition fixed from p4_shade.py / NOTES §5; label-free).
Per image: grains from cellpose masks (noise <= train-median: zero-shot masks-v3; noisier: fine-tuned ft-H300 masks), per-grain mean intensity
over the eroded interior; drop pore-like grains (U-Net z1 pore prob >= 0.3); matrix grains = U-Net z1 phase prob < 0.15; smooth matrix-intensity
field = Gaussian kernel regression (bw 48 px) over matrix-grain centroids (leave-one-out for matrix grains); darkness d = field - I;
sh_cor15 = area fraction with d > 15 gray; sh_raw15 = same with the image's median matrix intensity instead of the field.
usage: shade_feat.py SPLIT [H300_MASKS_NPZ]  -> shade-<split>.npz (ids, X, keys)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from PIL import Image
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
split = sys.argv[1]; hpath = sys.argv[2] if len(sys.argv) > 2 else str(_repo_path(E, f'push-300/ftseg-ft-H300-{split}-noisy-n1-masks.npz'))
U = np.load(_repo_path(E, f'push-400/unet4maps-z1-{split}.npz')); upos = {str(i): k for k, i in enumerate(U['ids'])}
Mc = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
Mn = np.load(_repo_io(hpath), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
F = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
Ftr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
nz = F['X'][:, fk.index('noise')]
KEYS = ['sh_cor12', 'sh_cor15', 'sh_cor18', 'sh_raw15']; X = np.full((len(ids), 4), np.nan); src = []
for k, i in enumerate(ids):
    noisy = nz[k] > thr
    if noisy and i not in npos: raise SystemExit(f'missing ft-H300 mask for noisy {i}')
    mask = (Mn['masks'][npos[i]] if noisy else Mc['masks'][cpos[i]]).astype(int); src.append('H300' if noisy else 'v3')
    im = np.asarray(Image.open(_repo_path(P, f'data/{split}/{i}.png')), np.float32)
    mp = U['maps'][upos[i]].astype(np.float32); ph = np.kron(mp[1], np.ones((2, 2))); po = np.kron(mp[2], np.ones((2, 2)))
    inner = np.where(ndi.minimum_filter(mask, 5) == ndi.maximum_filter(mask, 5), mask, 0)
    labs = np.unique(inner); labs = labs[labs > 0]
    if len(labs) < 25: continue
    I = np.asarray(ndi.mean(im, inner, labs)); pph = np.asarray(ndi.mean(ph, inner, labs)); ppo = np.asarray(ndi.mean(po, inner, labs))
    A = np.asarray(ndi.sum(np.ones_like(im), mask, labs)); com = np.array(ndi.center_of_mass(np.ones_like(im), inner, labs)); cy, cx = com[:, 0], com[:, 1]
    keep = ppo < 0.3; I, pph, A, cx, cy = I[keep], pph[keep], A[keep], cx[keep], cy[keep]
    mat = pph < 0.15
    if mat.sum() < 15: continue
    d2 = (cx[:, None] - cx[mat][None]) ** 2 + (cy[:, None] - cy[mat][None]) ** 2; W = np.exp(-d2 / (2 * 48 ** 2))
    W[np.where(mat)[0], np.arange(mat.sum())] = 0
    field = (W @ I[mat]) / W.sum(1)
    X[k] = [A[field - I > 12].sum() / A.sum(), A[field - I > 15].sum() / A.sum(), A[field - I > 18].sum() / A.sum(), A[np.median(I[mat]) - I > 15].sum() / A.sum()]
np.savez(_repo_path(HERE, f'shade3-{split}.npz'), ids=np.array(ids), X=X, keys=np.array(KEYS), mask_source=np.array(src), noise_thr=thr)
print(split, 'rows', len(ids), 'nan rows', int(np.isnan(X[:, 0]).sum()), 'noisy(H300) rows', src.count('H300'), 'mean', np.nanmean(X, 0).round(4))
