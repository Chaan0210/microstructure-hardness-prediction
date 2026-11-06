"""Angle columns weighted by (AR - 1) with the CORRECT inverse of the U-Net shape-tensor target: |(c2,s2)| = log(AR), so AR - 1 = exp(L) - 1,
L = grain-prob-weighted mean of |(c2,s2)| (mean log AR). Direction from the mean doubled-angle vector (C,S). Same nets as angle_feats.py (m1,m2,h1,z1 averaged).
usage: extra_feats2.py SPLIT -> extra2-SPLIT.npz (keys angL_c4, angL_s4, L_logar)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
split = sys.argv[1]
src = [f'../push-400/unet2maps-{n}-{split}.npz' for n in ('m1', 'm2', 'h1')] + [f'../push-400/unet4maps-z1-{split}.npz']
acc = None; ids = None
for f in src:
    M = np.load(_repo_io(f), allow_pickle=True); ii = [str(i) for i in M['ids']]
    if ids is None: ids = ii
    assert ii == ids
    maps = M['maps']; out = np.zeros((len(ids), 3))
    for n in range(len(ids)):
        m = maps[n].astype(np.float64); g = m[0] + m[1]; c2, s2 = m[5], m[6]
        C = (g * c2).sum() / g.sum(); S = (g * s2).sum() / g.sum(); vv2 = C * C + S * S + 1e-12
        L = (g * np.hypot(c2, s2)).sum() / g.sum(); w = np.expm1(L)
        out[n] = [w * (C * C - S * S) / vv2, w * 2 * C * S / vv2, L]
    acc = out if acc is None else acc + out
acc /= len(src)
np.savez(_repo_io(f'extra2-{split}.npz'), ids=np.array(ids), X=acc, keys=np.array(['angL_c4', 'angL_s4', 'L_logar']))
print('done', split, acc.mean(0).round(4), acc.std(0).round(4), 'L range', acc[:, 2].min().round(3), acc[:, 2].max().round(3))
