"""E2 (frozen 2026-10-16 21:35, before any result): pairwise-difference regression (PADRE) as a new learning problem.
Same features as the linear partners' base (push-800 feat-u1dmavgH2Z-anghcor, 98 columns), Study900 dev 400 x official 3 partitions x 5 folds.
Learners (hyperparameters fixed a priori, no tuning): HistGradientBoostingRegressor(max_iter=400, lr=0.04, max_leaf_nodes=15, min_samples_leaf=15,
l2=1.0). DIRECT: fit (x_i -> y_i). PADRE: fit ([x_i, x_i - x_j] -> y_i - y_j) on 40,000 random ordered pairs (i != j) of the training fold;
predict v as mean over all training anchors j of (y_j + f([x_v, x_v - x_j])). Seeds 0 and 1 averaged.
Decision (see e2_eval.py): PADRE must beat DIRECT clearly and add >= 0.03 fold-honest blend gain over 076 (P >= 0.9); otherwise rejected.
usage: padre.py -> work/e2/oof-padre.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os; os.environ.setdefault('OMP_NUM_THREADS', '2')
import csv, time, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[4]
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); XA = F['X'].astype(float); pos = {str(i): k for k, i in enumerate(F['ids'])}
mk = lambda s: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=1.0, random_state=s)
splits = {'original': _repo_path(E, 'resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(E, 'resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(E, 'resolution-repartition-217/20261101/ensemble-OOF.npz')}
save = {}; t0 = time.time()
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]
    gd, gp = np.zeros(len(y)), np.zeros(len(y))
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]; pd_, pp_ = [], []
        for s in (0, 1):
            pd_.append(mk(s).fit(X[tr], y[tr]).predict(X[va]))
            rng = np.random.RandomState(1000 * s + k); i = rng.randint(0, len(tr), 40000); j = rng.randint(0, len(tr), 40000); ok = i != j; i, j = tr[i[ok]], tr[j[ok]]
            m = mk(s).fit(np.c_[X[i], X[i] - X[j]], y[i] - y[j])
            Xv = np.repeat(X[va], len(tr), 0); Xa = np.tile(X[tr], (len(va), 1)); ya = np.tile(y[tr], len(va))
            pp_.append((ya + m.predict(np.c_[Xv, Xv - Xa])).reshape(len(va), len(tr)).mean(1))
        gd[va], gp[va] = np.mean(pd_, 0), np.mean(pp_, 0)
        print(name, k, f'{time.time() - t0:.0f}s', flush=True)
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_direct': gd, f'{name}_padre': gp})
np.savez(_repo_path(HERE, 'oof-padre.npz'), **save); print('saved')
