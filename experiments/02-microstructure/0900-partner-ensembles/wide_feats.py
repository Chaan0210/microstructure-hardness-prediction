"""Wide feature matrix for Ridge partners (Ridge copes with many correlated columns; TabPFN gets the 98-column sets): the 98-column average-variant
set (push-800 feat-u1dmavgH2Z-anghcor) + u1 phase U-Net (10) + multi-task U-Nets m1, m2, h1, m3 (19 each) + z1 (26) + block het columns of m1, m2,
h1, m3, z1 (9 each). All label-free, computed per image. usage: wide_feats.py SPLIT -> wide-{split}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
E = _RepoPath(__file__).resolve().parents[2]; s = sys.argv[1]
B = np.load(_repo_path(E, f'push-800/feat-u1dmavgH2Z-anghcor-{s}.npz'), allow_pickle=True); ids = [str(i) for i in B['ids']]; cols = [np.asarray(B['X'], float)]; keys = [str(k) for k in B['keys']]
def add(path, tag):
    D = np.load(_repo_io(path), allow_pickle=True); pos = {str(i): k for k, i in enumerate(D['ids'])}; cols.append(np.asarray(D['X'], float)[[pos[i] for i in ids]]); keys.extend(f'{tag}:{k}' for k in D['keys'])
add(_repo_path(E, f'push-400/unet-u1-{s}.npz'), 'u1')
for n in ('m1', 'm2', 'h1', 'm3'): add(_repo_path(E, f'push-400/unet2-{n}-{s}.npz'), n); add(_repo_path(E, f'push-500/uhet-{n}-{s}.npz'), 'het_' + n)
add(_repo_path(E, f'push-400/unet4-z1-{s}.npz'), 'z1'); add(_repo_path(E, f'push-500/uhet-u4z1-{s}.npz'), 'het_z1')
X = np.hstack(cols); np.savez(_repo_io(Path(__file__).with_name(f'wide-{s}.npz')), ids=np.array(ids), X=X, keys=np.array(keys)); print(s, X.shape)
