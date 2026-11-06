"""Probe: directional mean intercept lengths from zero-shot Cellpose masks (train). Boundary crossings along rows/cols/diagonals."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mids = [str(i) for i in M['ids']]; masks = M['masks']
out = np.zeros((len(mids), 4))
for n in range(len(mids)):
    m = masks[n].astype(np.int64)
    h = (m[:, 1:] != m[:, :-1]).sum() / m[:, 1:].size            # crossings per px along x (horizontal lines)
    v = (m[1:, :] != m[:-1, :]).sum() / m[1:, :].size            # along y
    d1 = (m[1:, 1:] != m[:-1, :-1]).sum() / m[1:, 1:].size / np.sqrt(2)   # per unit length along diagonal
    d2 = (m[1:, :-1] != m[:-1, 1:]).sum() / m[1:, 1:].size / np.sqrt(2)
    out[n] = [h, v, d1, d2]
np.savez(_repo_io('intercept-probe-train.npz'), ids=np.array(mids), X=out, keys=np.array(['PLx', 'PLy', 'PLd1', 'PLd2']))
print(out.mean(0))
