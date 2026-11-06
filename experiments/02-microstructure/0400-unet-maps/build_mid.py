"""030 features with additional ft 4-view rows (mean + vstd) for images in an extra noise range. usage: build_mid.py SPLIT OUT FTR_NPZ [FTR_NPZ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
split, out = sys.argv[1], sys.argv[2]
F = np.load(_repo_io(f'../push-300/feat-a1s250n4vstd-stcoh-{split}.npz'), allow_pickle=True); X = F['X'].astype(float).copy(); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
pos = {i: k for k, i in enumerate(ids)}
VSTD = [k[5:] for k in keys if k.startswith('vstd_')]
for src in sys.argv[3:]:
    V = np.load(_repo_io(src), allow_pickle=True); vk = [str(k) for k in V['keys']]; assert vk == keys[:len(vk)]
    for j, i in enumerate(V['ids']):
        k = pos[str(i)]; X[k, :len(vk)] = np.nanmean(V['X'][j], 0)
        for c in VSTD: X[k, keys.index('vstd_' + c)] = np.nanstd(V['X'][j][:, vk.index(c)])
np.savez(_repo_io(out), X=X, keys=np.array(keys), ids=np.array(ids)); print(out, X.shape)
