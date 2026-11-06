"""Harness: does a noisy-row model trained on [real noisy rows + synthetic degraded copies of clean rows] beat the all-real model on noisy rows?
No label duplication: clean originals are NOT in the noisy model's training set. Synthetic rows of validation-fold images are excluded.
usage: synth_harness.py SYNTH_FEAT_NPZ SYNTH_UNET_NPZ [seeds=0,1,2]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-300'))
import harness
sf, su = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '0,1,2').split(',')]
R = np.load(_repo_io('feat-u1phdrop-train.npz'), allow_pickle=True); keys = [str(k) for k in R['keys']]; ids = [str(i) for i in R['ids']]; X = R['X'].astype(float)
y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy')); noisy = nz >= np.median(nz); pos = {i: k for k, i in enumerate(ids)}
A = np.load(_repo_io(sf), allow_pickle=True); ak = [str(k) for k in A['keys']]; U = np.load(_repo_io(su), allow_pickle=True); uk = [str(k) for k in U['keys']]
sid = [str(i) for i in A['ids']]; assert sid == [str(i) for i in U['ids']]
XS = np.full((len(sid), len(keys)), np.nan)
for j, k in enumerate(keys):
    if k in ak: XS[:, j] = A['X'][:, ak.index(k)]
    elif k in uk: XS[:, j] = U['X'][:, uk.index(k)]
    else: raise SystemExit(f'missing {k}')
src = np.array([pos[i] for i in sid]); ys = y[src]
base = np.stack([np.load(_repo_io(f'oof_v2_feat-u1phdrop-train_s{s}.npy')) for s in seeds]).mean(0)
O = base.copy()
for r in range(3):
    f = harness.folds(len(y), r)
    for j in range(5):
        tr = f != j; va = (f == j) & noisy
        keep = tr[src]
        Xt = np.vstack([X[tr & noisy], XS[keep]]); yt = np.r_[y[tr & noisy], ys[keep]]
        ps = []
        for s in seeds:
            m = harness.tabpfn(8, s, 'mps'); m.fit(Xt, yt); ps.append(m.predict(X[va]))
        O[r, va] = np.mean(ps, 0)
rm = lambda P, m: np.mean([np.sqrt(np.mean((P[r][m] - y[m]) ** 2)) for r in range(3)])
print(f'noisy rows: base(all-real) {rm(base, noisy):.4f} | noisy-model(real noisy + synth) {rm(O, noisy):.4f} | avg {rm((O + base) / 2, noisy):.4f}')
print(f'overall   : base {rm(base, np.ones(len(y), bool)):.4f} | with noisy-model {rm(O, np.ones(len(y), bool)):.4f} | avg {rm((O + base) / 2, np.ones(len(y), bool)):.4f}')
# control: noisy-only real model without synthetic rows
C = base.copy()
for r in range(3):
    f = harness.folds(len(y), r)
    for j in range(5):
        tr = f != j; va = (f == j) & noisy; ps = []
        for s in seeds:
            m = harness.tabpfn(8, s, 'mps'); m.fit(X[tr & noisy], y[tr & noisy]); ps.append(m.predict(X[va]))
        C[r, va] = np.mean(ps, 0)
print(f'control noisy-only real model: noisy rows {rm(C, noisy):.4f}')
np.save(_repo_io('oof_synth_noisymodel.npy'), O)
