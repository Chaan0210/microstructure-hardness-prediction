"""Train on the 400 dev ids, predict the 100 non-dev train ids (never used for dev OOF); compare feature sets (grain-only RMSE), 3 seeds x 4 variants."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-300'))
import harness
d = np.load(_repo_io('../push-300/dev-u1dmavgH2Z-v35.npz'), allow_pickle=True); dev = set(map(str, d['original_ids']))
out = {}
for tag, pat in [('038', '../push-500/feat-{v}-train.npz'), ('039', 'feat-{v}-angh-train.npz')]:
    P = []
    for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']:
        F = np.load(_repo_io(pat.format(v=v)), allow_pickle=True); ids = [str(i) for i in F['ids']]; X = F['X'].astype(float); y = harness.labels(ids)
        tr = np.array([i in dev for i in ids]); ho = ~tr
        ps = []
        for sd in (0, 1, 2):
            m = harness.tabpfn_new('v3.5', 100 * sd, 'mps'); m.fit(X[tr], y[tr]); ps.append(m.predict(X[ho]))
        P.append(np.mean(ps, 0))
    g = np.mean(P, 0); out[tag] = g - y[ho]
    nz = X[ho, [str(k) for k in F['keys']].index('noise')] * 255
    print(tag, 'holdout n=%d rmse %.3f  HQ %.3f LQ %.3f' % (ho.sum(), np.sqrt(np.mean(out[tag] ** 2)), np.sqrt(np.mean(out[tag][nz < 11.9] ** 2)), np.sqrt(np.mean(out[tag][nz >= 11.9] ** 2))), flush=True)
dd = out['039'] ** 2 - out['038'] ** 2
print('paired MSE diff %.2f (SE %.2f) -> RMSE diff ~%.3f' % (dd.mean(), dd.std() / np.sqrt(len(dd)), np.sqrt(np.mean(out['039'] ** 2)) - np.sqrt(np.mean(out['038'] ** 2))))
