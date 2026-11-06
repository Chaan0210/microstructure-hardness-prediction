"""A-priori topology of the U-Net (u1) second-phase probability map q = p_phase / (p_matrix + p_phase) at 128 px: for t in (0.35, 0.5, 0.65):
b0 = number of phase components (>= 2 px) per 1000 px, b1 = number of holes (bounded non-phase components) per 1000 px, lcf = largest component
area / total phase area. Label-free, per image. usage: topo_feats.py train|test"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy import ndimage as ndi
split = sys.argv[1]; M = np.load(_repo_io(f'../push-400/unetmaps-u1-{split}.npz')); maps = M['maps'].astype(np.float32); out = []
st = np.ones((3, 3), int)
for mp in maps:
    q = mp[1] / np.maximum(mp[0] + mp[1], 1e-3); row = []
    for t in (0.35, 0.5, 0.65):
        ph = q > t; lab, n = ndi.label(ph, st); sz = np.bincount(lab.ravel())[1:]; keep = sz >= 2
        bg, nb = ndi.label(~ph); bsz = np.bincount(bg.ravel())[1:]; border = set(np.unique(np.r_[bg[0], bg[-1], bg[:, 0], bg[:, -1]])) - {0}
        holes = sum(1 for k in range(1, nb + 1) if k not in border and bsz[k - 1] >= 2)
        row += [keep.sum() / 16.384, holes / 16.384, (sz.max() / sz.sum()) if sz.size else 0.0]
    out.append(row)
keys = [f'topo_{n}_t{int(t * 100)}' for t in (0.35, 0.5, 0.65) for n in ('b0', 'b1', 'lcf')]
np.savez(_repo_io(f'topo-{split}.npz'), ids=M['ids'], X=np.array(out), keys=np.array(keys)); print(split, np.array(out).shape)
