"""E3 evaluation as frozen in transfer_ridge.py: standalone gated RMSE vs production DTA RidgeCV and plain Ridge, and the 076 dev blend with the
DTA RidgeCV partner (weight 0.128) replaced on gated rows by TR / TRD (same weight). Paired image bootstrap (4000)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ME = HERE.parents[1]; E = _RepoPath(__file__).resolve().parents[4]
B = np.load(_repo_path(ME, 'repro/components/c14b-orig.npz')); y = B['y']; nz = B['nz_dev']; fail = B['fail_dev'][0]; base = B['dev']
PT = np.load(_repo_path(ME, 'repro/components/c14p-orig.npz'))
sid = [str(i) for i in np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
d = np.load(_repo_path(HERE, 'oof-tr.npz'), allow_pickle=True)
def al(key):
    out = np.zeros((3, 400))
    for k, n in enumerate(['original', '20261031', '20261101']):
        p = {str(i): j for j, i in enumerate(d[f'{n}_ids'])}; out[k] = d[f'{n}_{key}'][[p[i] for i in sid]]
    return out
g = (nz >= 12) & ~fail; rm = lambda a, m: float(np.sqrt(((a - y)[:, m] ** 2).mean())); rng = np.random.RandomState(0); BI = [rng.randint(0, 400, 400) for _ in range(4000)]
res = {'gated_alone': {'dta_ridge(prod)': rm(PT['dta_ridge_dev'], g), 'ridge_full(plain)': rm(PT['ridge_full_dev'], g)}}
for key in ('tr', 'trd'):
    P = al(key); res['gated_alone'][key] = rm(P, g)
    new = base + np.where(g, 0.128 * (P - PT['dta_ridge_dev']), 0)
    dd = ((new - y) ** 2).mean(0) - ((base - y) ** 2).mean(0)
    res[key] = dict(blend=rm(new, slice(None)), gain=rm(base, slice(None)) - rm(new, slice(None)),
                    parts=[float(np.sqrt(((base[k] - y) ** 2).mean()) - np.sqrt(((new[k] - y) ** 2).mean())) for k in range(3)],
                    P=float(np.mean([dd[b].mean() < 0 for b in BI])), gated_dmse=float(dd[g].mean()))
print(json.dumps(res, indent=1)); json.dump(res, open(_repo_path(HERE, 'e3-results.json'), 'w'), indent=1)
