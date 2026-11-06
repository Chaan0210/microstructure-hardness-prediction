"""Diagnosis of the reproduced 076 (component rebuild c14b-orig): band errors and SSE shares, calibration slopes, residual shape, cross-model residual
covariance (TabPFN / CNN021 / set NN; bootstrap SE), residual correlation matrix, dev-vs-test prediction spread. Reads only; writes diag.json."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, numpy as np
from pathlib import Path
from scipy import stats
HERE = Path(__file__).resolve().parent; B = np.load(_repo_path(HERE.parents[1], 'repro/components/c14b-orig.npz'))
y, nz, f = B['y'], B['nz_dev'], B['fail_dev'][0]; r = B['dev'] - y; out = {}; rng = np.random.RandomState(0)
bands = {'clean': (nz < 9.5) & ~f, 'mid': (nz >= 9.5) & (nz < 12) & ~f, 'noisy': (nz >= 12) & ~f}
for k, m in bands.items():
    R = {c: (B['dev_' + c] - y)[:, m].mean(0) for c in ['cnn', 'set', 'tab']}; mr = r[:, m].mean(0)
    cv = lambda a, b, ix: float(np.cov(R[a][ix], R[b][ix])[0, 1]); n = m.sum(); boot = [rng.randint(0, n, n) for _ in range(1000)]
    out[k] = dict(n=int(n), rmse=float(np.sqrt((r[:, m] ** 2).mean())), sse_share=float((r[:, m] ** 2).sum() / (r ** 2).sum()),
                  slope_y_on_pred=float(np.polyfit(B['dev'][:, m].ravel(), np.tile(y[m], 3), 1)[0]), resid_skew=float(stats.skew(mr)), resid_exkurt=float(stats.kurtosis(mr)),
                  tab_mse=float(np.mean(R['tab'] ** 2)), cov={f'{a}-{b}': [cv(a, b, np.arange(n)), float(np.std([cv(a, b, ix) for ix in boot]))] for a, b in [('cnn', 'tab'), ('set', 'tab'), ('cnn', 'set')]})
comps = ['cnn', 'tab', 'ridge', 'logr', 'set', 'gmm']; RR = {c: (B['dev_' + c] - y)[:, ~f].ravel() for c in comps}; RR['blend'] = r[:, ~f].ravel()
out['resid_corr'] = {a: {b: float(np.corrcoef(RR[a], RR[b])[0, 1]) for b in RR} for a in RR}
nzt, ft = B['nz_test'], B['failed_test']
out['pred_sd_dev_vs_test'] = {band: [float(B['dev'][:, md & ~f].std()), float(B['test'][mt & ~ft].std())] for band, md, mt in [('clean+mid', nz < 12, nzt < 12), ('noisy', nz >= 12, nzt >= 12)]}
print(json.dumps(out, indent=1)); json.dump(out, open(_repo_path(HERE, 'diag.json'), 'w'), indent=1)
