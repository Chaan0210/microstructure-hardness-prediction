"""Append columns to an arbitrary base feature npz. usage: addcols2.py BASE.npz OUT.npz EXTRA.npz[:cols] ..."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
F = np.load(_repo_io(sys.argv[1]), allow_pickle=True); X = [F['X'].astype(float)]; keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
for spec in sys.argv[3:]:
    fn, _, cols = spec.partition(':'); E = np.load(_repo_io(fn), allow_pickle=True); ek = [str(k) for k in E['keys']]
    assert [str(i) for i in E['ids']] == ids
    sel = cols.split(',') if cols else ek
    X.append(E['X'][:, [ek.index(c) for c in sel]].astype(float)); keys += sel
np.savez(_repo_io(sys.argv[2]), X=np.hstack(X), keys=np.array(keys), ids=np.array(ids)); print(sys.argv[2], np.hstack(X).shape)
