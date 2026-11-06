"""Noisy-row view augmentation test (harness folds; TabPFN v2 n8). Variants:
 base : 030 features (mean of views)            [cached seeds 0,1,2]
 V2   : train on 4 view rows per noisy image (+mean-row cols vstd/stcoh shared), predict each view and average
 V3   : train on mean row + 4 view rows; predict mean row
 V4   : train on mean rows (as base) but predict = mean over the 4 view rows + mean row (prediction-level TTA only)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-300'))
import harness
F = np.load(_repo_io('../push-300/feat-a1s250n4vstd-stcoh-train.npz'), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
V = np.load(_repo_io('../push-300/ftseg-ft-a1s250-train-noisy-n4.npz'), allow_pickle=True); vid = [str(i) for i in V['ids']]; VX = V['X']
y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy')); pos = {i: k for k, i in enumerate(ids)}
rows = np.array([pos[i] for i in vid]); isn = np.zeros(len(ids), bool); isn[rows] = True
vrow = {pos[i]: j for j, i in enumerate(vid)}
def view_mat(k):   # 4 x 88 for noisy image k
    base = np.repeat(X[k][None], 4, 0); base[:, :64] = VX[vrow[k]]; return base
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
variants = sys.argv[2:] if len(sys.argv) > 2 else ['V2', 'V3', 'V4']
for var in variants:
    O = np.zeros((3, len(y)))
    for r in range(3):
        f = harness.folds(len(y), r)
        for j in range(5):
            tr, va = np.where(f != j)[0], np.where(f == j)[0]
            Xt, yt = [], []
            for k in tr:
                if isn[k] and var in ('V2', 'V3'):
                    vm = view_mat(k); Xt.append(vm); yt += [y[k]] * 4
                    if var == 'V3': Xt.append(X[k][None]); yt.append(y[k])
                else:
                    Xt.append(X[k][None]); yt.append(y[k])
            Xt = np.vstack(Xt); yt = np.array(yt)
            m = harness.tabpfn(8, seed, 'mps'); m.fit(Xt, yt)
            pm = m.predict(X[va])
            if var in ('V2', 'V4'):
                nv = [k for k in va if isn[k]]
                if nv:
                    P = m.predict(np.vstack([view_mat(k) for k in nv])).reshape(-1, 4)
                    for q, k in enumerate(nv):
                        pi = np.where(va == k)[0][0]
                        pm[pi] = P[q].mean() if var == 'V2' else (P[q].sum() + pm[pi]) / 5
            O[r, va] = pm
    np.save(_repo_io(f'oof_viewrows_{var}_s{seed}.npy'), O)
    harness.report(f'{var} seed {seed}', O, y, nz)
b = np.load(_repo_io(f'../push-300/oof_none_v2_feat-a1s250n4vstd-stcoh-train_s{seed}.npy')); harness.report(f'base seed {seed}', b, y, nz)
