from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-300')); import harness
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); X = F['X'].astype(float); y = np.random.RandomState(0).randn(len(X)) * 20 + 200
for n, nq in [(320, 80), (1500, 124)]:
    Xc = np.vstack([X] * 4)[:n]; yc = np.r_[y, y, y, y][:n]
    t = time.time(); m = harness.tabpfn_new('v3.5', 0, 'mps'); m.fit(Xc, yc); p = m.predict(X[:nq]); print(n, 'fit+pred', round(time.time() - t, 1), 's', 'n_est', m.n_estimators, flush=True)
