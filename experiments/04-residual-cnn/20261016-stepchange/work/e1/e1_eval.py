"""E1 evaluation exactly as frozen in PLAN.md: gated-row (noise >= 12 & ~failed) RMSE pooled over the 3 partitions, paired image bootstrap vs F0,
band RMSEs for context. usage: e1_eval.py -> prints and writes work/e1/e1-results.json"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ME = HERE.parents[1]
B = np.load(_repo_path(ME, 'repro/components/c14b-orig.npz')); y = B['y']; nz = B['nz_dev']; fail = B['fail_dev'][0]
sid = [str(i) for i in np.load(_repo_path(HERE.parents[2], 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
gate = (nz >= 12) & ~fail; clean = (nz < 9.5) & ~fail; mid = (nz >= 9.5) & (nz < 12) & ~fail
def load(arm):
    d = np.load(_repo_path(HERE, f'oof-{arm}.npz'), allow_pickle=True); out = np.zeros((3, 400))
    for k, n in enumerate(['original', '20261031', '20261101']):
        p = {i: j for j, i in enumerate(str(x) for x in d[f'{n}_ids'])}; out[k] = d[f'{n}_g'][[p[i] for i in sid]]
        assert np.allclose(d[f'{n}_y'][[p[i] for i in sid]], y)
    return out
rng = np.random.RandomState(0); BI = [rng.randint(0, gate.sum(), gate.sum()) for _ in range(4000)]
rm = lambda a, m: float(np.sqrt(((a - y)[:, m] ** 2).mean()))
arms = [a for a in ['F0', 'F1', 'A0', 'A1', 'A2', 'C1'] if (_repo_path(HERE, f'oof-{a}.npz')).exists()]; P = {a: load(a) for a in arms}
res = {}
for a in arms:
    se = ((P[a] - y) ** 2).mean(0)[gate]; se0 = ((P['F0'] - y) ** 2).mean(0)[gate]; d = se - se0
    res[a] = dict(gated=rm(P[a], gate), clean=rm(P[a], clean), mid=rm(P[a], mid), parts_gated=[float(np.sqrt(((P[a][k] - y)[gate] ** 2).mean())) for k in range(3)],
                  dmse_vs_F0=float(d.mean()), se_dmse=float(d.std(ddof=1) / np.sqrt(len(d))), P_better_than_F0=float(np.mean([d[b].mean() < 0 for b in BI])))
    print(f"{a}: gated {res[a]['gated']:.3f} (parts {np.round(res[a]['parts_gated'], 3)}) clean {res[a]['clean']:.3f} mid {res[a]['mid']:.3f} | dMSE vs F0 {res[a]['dmse_vs_F0']:+.2f} +- {res[a]['se_dmse']:.2f} P {res[a]['P_better_than_F0']:.3f}")
json.dump(res, open(_repo_path(HERE, 'e1-results.json'), 'w'), indent=1)
