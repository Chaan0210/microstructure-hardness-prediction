"""Absolute-scale size dispersion from U-Net log-area maps (label-free): grain diameter d = sqrt(exp(la)) per pixel (pixel-weighted by g).
d_sd: pixel-weighted SD of d; d_cv; blk4_dsd: SD over 4x4 blocks of mean d; blk4_asd: SD over blocks of mean area; a_sd: pixel-weighted SD of area.
usage: abs_het.py NET[,NET..]  -> ahet-<NETS>-{train,test}.npz (averaged over nets)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; U = _repo_path(H.parent, 'push-400'); nets = sys.argv[1].split(',')
KEYS = ['d_sd', 'd_cv', 'blk4_dsd', 'blk4_asd', 'a_sd', 'd_mean', 'blk4_dmax']
def feats(m):
    g = m[0] + m[1]; la = m[4]; d = np.exp(la / 2); a = np.exp(la); w = g / g.sum(); o = {}
    dm = (w * d).sum(); o['d_mean'] = dm; o['d_sd'] = np.sqrt((w * (d - dm) ** 2).sum()); o['d_cv'] = o['d_sd'] / dm
    am = (w * a).sum(); o['a_sd'] = np.sqrt((w * (a - am) ** 2).sum())
    den = g.reshape(4, 32, 4, 32).sum((1, 3)); ok = den > 0.2 * 32 * 32
    bd = (g * d).reshape(4, 32, 4, 32).sum((1, 3))[ok] / den[ok]; ba = (g * a).reshape(4, 32, 4, 32).sum((1, 3))[ok] / den[ok]
    o['blk4_dsd'] = bd.std(); o['blk4_asd'] = ba.std(); o['blk4_dmax'] = bd.max() - bd.mean()
    return [o[k] for k in KEYS]
for split in ('train', 'test'):
    D = [dict(np.load(_repo_path(U, f'unet4maps-{n[2:]}-{split}.npz' if n.startswith('u4') else f'unet2maps-{n}-{split}.npz'))) for n in nets]
    ids = D[0]['ids']
    X = np.mean([[feats(d['maps'][i].astype(np.float32)) for i in range(len(ids))] for d in D], 0)
    np.savez(_repo_path(H, f'ahet-{"+".join(nets)}-{split}.npz'), ids=ids, X=X, keys=np.array(KEYS)); print(split, X.shape, flush=True)
