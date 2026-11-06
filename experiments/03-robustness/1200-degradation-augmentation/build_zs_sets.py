"""Append ZS-net columns (pre-registered: zs_het_b, zs_het_lam, zs_ph_std, zs_ph_mean) or random N(0,1) null columns to the 044-slot feature
sets (push-800 feat-<v>-anghcor-<split>.npz, 98 cols) -> push-1200/feat-<v>-anghcor<TAG>-<split>.npz.
ZS values: clean train images used to train fold models get the OTHER fold's prediction (cross-fit); all other images the mean of folds A,B.
usage: build_zs_sets.py zs [COLS]   |   build_zs_sets.py rnd SEED"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from zs_feats import image_feats, KF
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
mode = sys.argv[1]
COLS = (sys.argv[2].split(',') if len(sys.argv) > 2 else ['zs_het_b', 'zs_het_lam', 'zs_ph_std', 'zs_ph_mean']) if mode == 'zs' else None
D = np.load(_repo_path(H, 'zs-prep.npz')); cid = [str(i) for i in D['ids']]; n = len(cid); p = np.random.RandomState(0).permutation(n); A = set(p[:n // 2])
pick = {cid[k]: (1 if k in A else 0) for k in range(n)}
for split in ('train', 'test'):
    if mode == 'zs':
        R = np.load(_repo_path(H, f'zs-{split}.npz')); rid = [str(i) for i in R['ids']]; assert list(R['folds']) == ['A', 'B']
        Z = image_feats(R['R'], rid, ['A', 'B'], pick if split == 'train' else 'mean'); zpos = {i: k for k, i in enumerate(rid)}
        tag = 'ZS' if len(sys.argv) <= 2 else 'ZS' + ''.join(c.split('_')[-1][0] for c in COLS)
    else:
        seed = int(sys.argv[2]); tag = f'R{seed}'
    for v in V:
        F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-{split}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
        if mode == 'zs':
            cols = Z[[zpos[i] for i in ids]][:, [KF.index(c) for c in COLS]]; names = COLS
        else:
            rng = np.random.RandomState(seed * 10 + (0 if split == 'train' else 1)); cols = rng.randn(len(ids), 4); names = [f'rnd{seed}_{j}' for j in range(4)]
        np.savez(_repo_path(H, f'feat-{v}-anghcor{tag}-{split}.npz'), ids=np.array(ids), X=np.c_[F['X'].astype(float), cols], keys=np.array(keys + list(names)))
    print(split, tag, 'done')
