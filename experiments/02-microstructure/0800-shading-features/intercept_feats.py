"""Linear-intercept (Heyn / ASTM E112) grain size along image rows, columns and both diagonals, from cellpose masks
(zero-shot masks-v3 for noise <= train median, ft-H300 1-view for noisier). For each line, zeros (boundary gaps) are dropped and label changes counted.
N_L = crossings per pixel; features: NL_x, NL_y, NL_d1, NL_d2, hp_xy = mean(sqrt(NL_x), sqrt(NL_y)), hp_d = mean(sqrt(NL_d1), sqrt(NL_d2)), hp_4 = mean over 4.
usage: intercept_feats.py SPLIT -> eda/intercept-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
split = sys.argv[1]
Mc = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
Mn = np.load(_repo_path(E, f'push-300/ftseg-ft-H300-{split}-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
F = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
Ftr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
nz = F['X'][:, fk.index('noise')]
def crossings(lines):
    c = 0; L = 0
    for v in lines:
        v = v[v > 0]
        if len(v) < 2: continue
        c += int((v[1:] != v[:-1]).sum()); L += len(v)
    return c / max(L, 1)
def diag_lines(m):
    return [np.diagonal(m, k) for k in range(-200, 201, 4)]
KEYS = ['NL_x', 'NL_y', 'NL_d1', 'NL_d2', 'hp_xy', 'hp_d', 'hp_4', 'NL_xy_ratio']; X = []
for n, i in enumerate(ids):
    m = (Mn['masks'][npos[i]] if nz[n] > thr else Mc['masks'][cpos[i]]).astype(int)
    nx = crossings(m[::2]); ny = crossings(m.T[::2]); d1 = crossings(diag_lines(m)) / np.sqrt(2) * 2 / 2; d2 = crossings(diag_lines(m[:, ::-1]))
    # diagonal steps are sqrt(2) px long: crossings per diagonal-step -> per pixel length
    d1 = crossings(diag_lines(m)) / np.sqrt(2); d2 = d2 / np.sqrt(2)
    X.append([nx, ny, d1, d2, 0.5 * (np.sqrt(nx) + np.sqrt(ny)), 0.5 * (np.sqrt(d1) + np.sqrt(d2)), 0.25 * (np.sqrt(nx) + np.sqrt(ny) + np.sqrt(d1) + np.sqrt(d2)), np.log(max(nx, 1e-6) / max(ny, 1e-6))])
(_repo_path(HERE, 'eda')).mkdir(exist_ok=True)
np.savez(_repo_path(HERE, f'eda/intercept-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(KEYS))
print('done', np.array(X).shape, np.nanmean(X, 0).round(4))
