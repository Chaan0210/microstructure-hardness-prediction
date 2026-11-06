"""Self-distillation / label smoothing inside outer folds: y' = a*y + (1-a)*inner_OOF_pred (TabPFN, 1 seed), refit TabPFN on (X, y'), predict val.
Arms a = 1 (plain, reference), 0.7, 0.5. usage: selfdistill.py PARTITION"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
part = sys.argv[1]
D = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
ids = [str(i) for i in D[part + '_ids']]; y = D[part + '_y']; fold = D[part + '_fold']; p021 = D[part + '_p021']; failed = D[part + '_failed']
F = np.load(_repo_io('feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; X = F['X'][[fp[i] for i in ids]].astype(float)
A = (1.0, 0.7, 0.5); out = {a: np.zeros(len(y)) for a in A}
for k in range(5):
    tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
    fi = np.arange(len(tr)) % 5; np.random.RandomState(1000 + k).shuffle(fi); oof = np.zeros(len(tr))
    for j in range(5):
        m = harness.tabpfn_new('v3.5', k, 'mps'); m.fit(X[tr][fi != j], y[tr][fi != j]); oof[fi == j] = m.predict(X[tr][fi == j])
    for a in A:
        m = harness.tabpfn_new('v3.5', k, 'mps'); m.fit(X[tr], a * y[tr] + (1 - a) * oof); out[a][va] = m.predict(X[va])
    print('fold', k, ' '.join('a%.1f %.3f' % (a, np.sqrt(np.mean((out[a][va] - y[va]) ** 2))) for a in A), flush=True)
bl = lambda g: np.sqrt(np.mean((0.25 * p021 + 0.75 * np.where(failed, p021, g) - y) ** 2))
for a in A:
    d = (0.25 * p021 + 0.75 * np.where(failed, p021, out[a]) - y) ** 2 - (0.25 * p021 + 0.75 * np.where(failed, p021, out[1.0]) - y) ** 2
    print('a=%.1f grain %.4f W.75 %.4f | dMSE vs a=1: %+.2f (se %.2f), folds better %d/5' % (a, np.sqrt(np.mean((out[a] - y) ** 2)), bl(out[a]), d.mean(), d.std() / np.sqrt(len(d)), sum(np.mean(d[fold == k]) < 0 for k in range(5))))
