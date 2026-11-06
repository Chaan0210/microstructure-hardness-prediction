"""Extra label-free columns from U-Net maps for train/test: het at K=8,16 (Hall-Petch block std, z1 + m1/m2/h1 maps averaged), angle weighted by (AR-1).
usage: extra_feats.py SPLIT -> extra-SPLIT.npz"""
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
    maps = M['maps']; out = np.zeros((len(ids), 4))
    for n in range(len(ids)):
        m = maps[n].astype(np.float64); g = m[0] + m[1]; e = np.exp(-0.5 * m[4]); row = []
        for K in (8, 16):
            s = 128 // K; W = g.reshape(K, s, K, s).sum((1, 3)); num = (g * e).reshape(K, s, K, s).sum((1, 3)); ok = W > 0.2 * s * s
            row.append(np.std(-2 * np.log(num[ok] / W[ok])))
        # (AR-1)-weighted angle: local eccentricity -> AR, global direction from mean doubled-angle vector
        c2, s2 = m[5], m[6]; C = (g * c2).sum() / g.sum(); S = (g * s2).sum() / g.sum(); vv = np.hypot(C, S) + 1e-9
        ecc = np.clip((g * np.hypot(c2, s2)).sum() / g.sum(), 0, 0.97); ar1 = np.sqrt((1 + ecc) / (1 - ecc)) - 1
        row += [ar1 * (C * C - S * S) / vv ** 2, ar1 * 2 * C * S / vv ** 2]
        out[n] = row
    acc = out if acc is None else acc + out
acc /= len(src)
np.savez(_repo_io(f'extra-{split}.npz'), ids=np.array(ids), X=acc, keys=np.array(['hetu_k8', 'hetu_k16', 'angar_c4', 'angar_s4']))
print('done', split, acc.mean(0).round(4), acc.std(0).round(4))
