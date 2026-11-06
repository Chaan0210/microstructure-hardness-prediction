"""Rule-of-mixtures x Hall-Petch block (a priori, label-free) from each variant's U-Net maps:
f_ph (soft phase area fraction), hp_ph / hp_mx = phase- / matrix-pixel-weighted mean exp(-logA/4) (= d^-1/2 up to a constant),
rom_ph = f_ph*hp_ph, rom_mx = (1-f_ph)*hp_mx.   usage: rom_feats.py SPLIT NET[,NET..] OUTTAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
split, nets, tag = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
M0 = [np.load(_repo_io(f'../push-400/unet2maps-{n}-{split}.npz')) for n in nets]; ids = [str(i) for i in M0[0]['ids']]
P = np.mean([m['maps'][:, [0, 1, 4]].astype(np.float32) for m in M0], 0)
p0, p1, la = P[:, 0], P[:, 1], P[:, 2]; h = np.exp(-la / 4)
f = p1.sum((1, 2)) / (p0 + p1).sum((1, 2)); hp_ph = (p1 * h).sum((1, 2)) / np.maximum(p1.sum((1, 2)), 1e-3); hp_mx = (p0 * h).sum((1, 2)) / (p0.sum((1, 2)))
X = np.stack([hp_ph, hp_mx, f * hp_ph, (1 - f) * hp_mx], 1)
np.savez(_repo_io(f'rom-{tag}-{split}.npz'), ids=np.array(ids), X=X, keys=np.array(['hp_ph', 'hp_mx', 'rom_ph', 'rom_mx'])); print(tag, split, X.mean(0))
