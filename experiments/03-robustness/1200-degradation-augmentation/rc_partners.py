"""Ridge / log-Ridge partners on the regression-calibrated (RC) feature sets (real rows only, production settings), dev OOF in Study900 order."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
P = Path(_repo_io('../..')); B = load(3); sid = B['ids']
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; tag = sys.argv[1]
def oof(v, model):
    F = np.load(_repo_io(f'feat-{v}-anghcor{tag}-train.npz'), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); O = np.zeros((3, 400))
    for p_, path in enumerate(splits):
        d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; g = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k
            if model == 'ridge': g[va] = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])).fit(X[tr], y[tr]).predict(X[va])
            else: g[va] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(X[tr], np.log(y[tr])).predict(X[va])
        pp = {i: j for j, i in enumerate(ids)}; O[p_] = g[[pp[i] for i in sid]]
    return O
np.savez(_repo_io(f'sr-A-{tag}.npz'), ridge=oof('u1dmavgH2Z', 'ridge'), logridge=np.exp(np.mean([oof(v, 'log') for v in V], 0)), ids=np.array(sid)); print('saved', tag)
