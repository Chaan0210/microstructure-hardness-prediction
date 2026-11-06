"""Variant: all real rows + synthetic rows of train-fold clean images (label duplication), optional is_synth flag column."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-300'))
import harness
flag = len(sys.argv) > 1 and sys.argv[1] == 'flag'
R = np.load(_repo_io('feat-u1phdrop-train.npz'), allow_pickle=True); keys = [str(k) for k in R['keys']]; ids = [str(i) for i in R['ids']]; X = R['X'].astype(float)
y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy')); noisy = nz >= np.median(nz); pos = {i: k for k, i in enumerate(ids)}
A = np.load(_repo_io('synth/feat-ftA.npz'), allow_pickle=True); ak = [str(k) for k in A['keys']]; U = np.load(_repo_io('unet-u1-synth.npz'), allow_pickle=True); uk = [str(k) for k in U['keys']]
sid = [str(i) for i in A['ids']]; XS = np.stack([A['X'][:, ak.index(k)] if k in ak else U['X'][:, uk.index(k)] for k in keys], 1)
src = np.array([pos[i] for i in sid]); ys = y[src]
if flag: X = np.c_[X, np.zeros(len(X))]; XS = np.c_[XS, np.ones(len(XS))]
base = np.stack([np.load(_repo_io(f'oof_v2_feat-u1phdrop-train_s{s}.npy')) for s in (0, 1, 2)]).mean(0); O = np.zeros_like(base)
for r in range(3):
    f = harness.folds(len(y), r)
    for j in range(5):
        tr = f != j; va = f == j; keep = tr[src]
        Xt = np.vstack([X[tr], XS[keep]]); yt = np.r_[y[tr], ys[keep]]; ps = []
        for s in (0, 1, 2):
            m = harness.tabpfn(8, s, 'mps'); m.fit(Xt, yt); ps.append(m.predict(X[va]))
        O[r, va] = np.mean(ps, 0)
harness.report('base', base, y, nz); harness.report('all-real + synth' + (' +flag' if flag else ''), O, y, nz); harness.report('avg', (O + base) / 2, y, nz)
