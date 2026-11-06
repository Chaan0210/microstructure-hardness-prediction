"""Per-feature shift under noise: real (noisy>=12 mean - clean<9.5 mean, unpaired train rows) vs sims (copy - its clean source, paired, copies with
measured noise >= 12), in units of the clean-row SD. A copy set is realistic in a feature when the two shifts agree. usage: realism_shift.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; XA = F['X'].astype(float); ids = [str(i) for i in F['ids']]
pos = {i: j for j, i in enumerate(ids)}; ni = keys.index('noise'); nz = np.nan_to_num(XA[:, ni] * 255, nan=0); C, N = XA[nz < 9.5], XA[nz >= 12]
sd = np.nanstd(C, 0); sd[sd == 0] = 1
real = (np.nanmean(N, 0) - np.nanmean(C, 0)) / sd
sims = {}
for s in 'deg,deg2,deg3,deg4,deg5,deg6,deg7'.split(','):
    S = np.load(_repo_io(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); X = S['X'].astype(float); m = X[:, ni] * 255 >= 12
    src = XA[[pos[str(i)] for i in S['ids']]]; sims[s] = (np.nanmean(X[m] - src[m], 0)) / sd
allsim = np.mean(list(sims.values()), 0)
order = np.argsort(-np.abs(real - allsim))
print(f'{"feature":30s} {"real":>7s} {"sims":>7s} ' + ' '.join(f'{s:>6s}' for s in sims))
for j in order[:25]: print(f'{keys[j]:30s} {real[j]:+7.2f} {allsim[j]:+7.2f} ' + ' '.join(f'{sims[s][j]:+6.2f}' for s in sims))
print('corr(real shift, sim shift) over features:', np.corrcoef(real[np.isfinite(allsim)], allsim[np.isfinite(allsim)])[0, 1].round(3))
