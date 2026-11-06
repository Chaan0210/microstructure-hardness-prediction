"""Absolute elongation-direction features from multi-task U-Net shape-tensor maps (doubled-angle channels c2=ch5, s2=ch6),
averaged over nets m1, m2, h1 (unet2maps) and z1 (unet4maps). Label-free; D4-TTA maps already inverse-transformed.
Global: (C,S) = grain-prob-weighted mean of (c2,s2); ang_c4 = |v| cos 4theta = (C^2-S^2)/|v|, ang_s4 = |v| sin 4theta = 2CS/|v|.
Local: grain-prob-weighted mean over pixels of |v_p| cos 4theta_p and |v_p| sin 4theta_p.
usage: angle_feats.py SPLIT -> angle-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
split = sys.argv[1]
src = [f'../push-400/unet2maps-{n}-{split}.npz' for n in ('m1', 'm2', 'h1')] + [f'../push-400/unet4maps-z1-{split}.npz']
acc = None; ids = None
for f in src:
    M = np.load(_repo_io(f), allow_pickle=True); ii = [str(i) for i in M['ids']]
    if ids is None: ids = ii
    assert ii == ids
    maps = M['maps']; out = np.zeros((len(ids), 4))
    for n in range(len(ids)):
        m = maps[n].astype(np.float32); g = m[0] + m[1]; c2, s2 = m[5], m[6]
        C = (g * c2).sum() / g.sum(); S = (g * s2).sum() / g.sum(); vv = np.hypot(C, S) + 1e-9
        vp = np.hypot(c2, s2) + 1e-9
        out[n] = [(C * C - S * S) / vv, 2 * C * S / vv, (g * (c2 * c2 - s2 * s2) / vp).sum() / g.sum(), (g * 2 * c2 * s2 / vp).sum() / g.sum()]
    acc = out if acc is None else acc + out
    print(f, out[:3].round(3), flush=True)
acc /= len(src)
np.savez(_repo_io(f'angle-{split}.npz'), ids=np.array(ids), X=acc, keys=np.array(['ang_c4', 'ang_s4', 'ang_c4_loc', 'ang_s4_loc']))
print('done', split, acc.mean(0).round(4), acc.std(0).round(4))
