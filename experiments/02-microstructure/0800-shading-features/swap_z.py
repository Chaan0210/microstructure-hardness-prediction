"""Build 044-based feature sets with the z1 U-Net columns swapped for a retrained net (z1s: shading-corrected targets, z1c: control, original targets).
usage: swap_z.py NET TAG MODE SPLIT   MODE: z (replace z_f25,z_f50) | zp (replace z_f25,z_f50 and append m_ph, m_ph_hard of NET)
-> feat-<v>-angh<TAG>-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
net, tag, mode, split = sys.argv[1:5]
Z = np.load(_repo_path(E, f'push-400/unet4-{net}-{split}.npz'), allow_pickle=True); zk = [str(k) for k in Z['keys']]; zpos = {str(i): k for k, i in enumerate(Z['ids'])}
for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']:
    F = np.load(_repo_path(HERE, f'feat-{v}-anghcor-{split}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]; X = F['X'].astype(float).copy()
    j = [zpos[i] for i in ids]
    for c in ('z_f25', 'z_f50'): X[:, keys.index(c)] = Z['X'][j, zk.index(c)]
    if mode == 'zp':
        X = np.c_[X, Z['X'][j][:, [zk.index('m_ph'), zk.index('m_ph_hard')]]]; keys = keys + [f'{net}_m_ph', f'{net}_m_ph_hard']
    np.savez(_repo_path(HERE, f'feat-{v}-angh{tag}-{split}.npz'), ids=np.array(ids), X=X, keys=np.array(keys))
    print(v, tag, split, X.shape)
