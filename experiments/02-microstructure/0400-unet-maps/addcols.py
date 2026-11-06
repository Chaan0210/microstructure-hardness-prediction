"""Append columns from extra npz files (same ids) to the 030 feature file. usage: addcols.py SPLIT OUT EXTRA.npz[:col1,col2] ..."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
split, out = sys.argv[1], sys.argv[2]
F = np.load(_repo_io(f'../push-300/feat-a1s250n4vstd-stcoh-{split}.npz'), allow_pickle=True); X = [F['X'].astype(float)]; keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
for spec in sys.argv[3:]:
    fn, _, cols = spec.partition(':'); E = np.load(_repo_io(fn.replace('{split}', split)), allow_pickle=True); ek = [str(k) for k in E['keys']]
    assert [str(i) for i in E['ids']] == ids
    sel = cols.split(',') if cols else ek
    X.append(E['X'][:, [ek.index(c) for c in sel]].astype(float)); keys += sel
np.savez(_repo_io(out), X=np.hstack(X), keys=np.array(keys), ids=np.array(ids)); print(out, np.hstack(X).shape)
