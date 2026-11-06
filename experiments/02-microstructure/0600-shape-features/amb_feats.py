"""Per-image phase-measurement reliability from existing U-Net maps (label-free): ambiguous-pixel fractions.
usage: amb_feats.py SPLIT -> amb-SPLIT.npz (ids, X, keys).  unet2 maps m1,m2,h1,m3 averaged; unet4 z1 map for z."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
split = sys.argv[1]; U = '../push-400'
M0 = [np.load(_repo_io(f'{U}/unet2maps-{n}-{split}.npz')) for n in ['m1', 'm2', 'h1', 'm3']]; ids = [str(i) for i in M0[0]['ids']]
for m in M0: assert [str(i) for i in m['ids']] == ids
M = [{'maps': m['maps'][:, :2]} for m in M0]
Z0 = np.load(_repo_io(f'{U}/unet4maps-z1-{split}.npz')); assert [str(i) for i in Z0['ids']] == ids; Z = {'maps': Z0['maps']}
keys = ['ph_amb28', 'ph_amb37', 'ph_ent', 'z_amb', 'ph_var_nets']; X = []
for k in range(len(ids)):
    ps = np.stack([m['maps'][k, 1].astype(np.float32) for m in M]); g = np.mean([m['maps'][k, 0].astype(np.float32) + m['maps'][k, 1].astype(np.float32) for m in M], 0)
    p = ps.mean(0); q = p / np.maximum(g, 1e-3); w = g / g.sum()
    ent = -(q * np.log(q + 1e-6) + (1 - q) * np.log(1 - q + 1e-6))
    z = Z['maps'][k, 7].astype(np.float32); gz = Z['maps'][k, 0].astype(np.float32) + Z['maps'][k, 1].astype(np.float32); wz = gz / gz.sum()
    X.append([(w * ((q > 0.2) & (q < 0.8))).sum(), (w * ((q > 0.3) & (q < 0.7))).sum(), (w * ent).sum(), (wz * ((z > 0.15) & (z < 0.35))).sum(), (w * ps.var(0)).sum()])
np.savez(_repo_io(f'amb-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(keys)); print('done', split, np.array(X).mean(0))
