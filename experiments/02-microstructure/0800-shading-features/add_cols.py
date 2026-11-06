"""Append columns to the four 039 feature variants (push-700 feat-<v>-angh-<split>.npz).
usage: add_cols.py TAG SPLIT SOURCE  SOURCE: cor | raw | rnd (N(0,1), seed 800, label-free control)
-> push-800/feat-<v>-angh<TAG>-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
tag, split, src = sys.argv[1:4]
S = np.load(_repo_path(HERE, f'shade-{split}.npz'), allow_pickle=True); spos = {str(i): k for k, i in enumerate(S['ids'])}; sk = [str(k) for k in S['keys']]
for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']:
    F = np.load(_repo_path(E, f'push-700/feat-{v}-angh-{split}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
    if src == 'rnd':
        rs = int(__import__('os').environ.get('RSEED', '800')); rng = np.random.RandomState(rs + (0 if split == 'train' else 1)); col = rng.randn(len(ids)); name = f'rnd{rs}'
    else:
        name = 'sh_cor15' if src == 'cor' else 'sh_raw15'; col = S['X'][[spos[i] for i in ids], sk.index(name)]
    np.savez(_repo_path(HERE, f'feat-{v}-angh{tag}-{split}.npz'), ids=np.array(ids), X=np.c_[F['X'].astype(float), col], keys=np.array(keys + [name]))
    print(v, split, name, 'nan', int(np.isnan(col).sum()))
