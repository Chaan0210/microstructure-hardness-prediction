"""E2 evaluation (frozen with padre.py): standalone RMSE by band for DIRECT and PADRE, and the fold-honest blend gain over the 076 dev blend:
for each partition and outer fold k, w in {0, .025, ..., .5} minimizes MSE of (1-w)*076 + w*partner on the other four folds and is applied to
fold k (failed rows keep 076). Paired image bootstrap (4000) of the pooled MSE change."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ME = HERE.parents[1]; E = _RepoPath(__file__).resolve().parents[4]
B = np.load(_repo_path(ME, 'repro/components/c14b-orig.npz')); y = B['y']; nz = B['nz_dev']; fail = B['fail_dev'][0]; base = B['dev']
sid = [str(i) for i in np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
d = np.load(_repo_path(HERE, 'oof-padre.npz'), allow_pickle=True); names = ['original', '20261031', '20261101']
def al(key):
    out = np.zeros((3, 400)); F = np.zeros((3, 400), int)
    for k, n in enumerate(names):
        p = {str(i): j for j, i in enumerate(d[f'{n}_ids'])}; ix = [p[i] for i in sid]; out[k] = d[f'{n}_{key}'][ix]
        F[k] = np.load(_repo_path(E, 'resolution-216/ensemble-OOF.npz' if n == 'original' else f'resolution-repartition-217/{n}/ensemble-OOF.npz'))['fold'][ix]
    return out, F
bands = {'all': ~fail, 'clean': (nz < 9.5) & ~fail, 'mid': (nz >= 9.5) & (nz < 12) & ~fail, 'noisy': (nz >= 12) & ~fail}
rm = lambda a, m: float(np.sqrt(((a - y)[:, m] ** 2).mean()))
W = np.arange(0, 0.501, 0.025); res = {}
for key in ('direct', 'padre'):
    P, FO = al(key); out = base.copy(); ws = []
    for k in range(3):
        for f in range(5):
            tr = (FO[k] != f) & ~fail; va = (FO[k] == f) & ~fail
            w = W[np.argmin([(((1 - w) * base[k] + w * P[k] - y)[tr] ** 2).mean() for w in W])]; ws.append(float(w))
            out[k, va] = (1 - w) * base[k, va] + w * P[k, va]
    dd = ((out - y) ** 2).mean(0) - ((base - y) ** 2).mean(0); rng = np.random.RandomState(0)
    pb = np.mean([dd[rng.randint(0, 400, 400)].mean() < 0 for _ in range(4000)])
    res[key] = dict(alone={b: rm(P, m) for b, m in bands.items()}, blend_076=rm(base, ~fail | fail), blend_new=rm(out, ~fail | fail),
                    gain=rm(base, ~fail | fail) - rm(out, ~fail | fail), P_better=float(pb), weights=ws,
                    noisy_gain=rm(base, bands['noisy']) - rm(out, bands['noisy']), clean_gain=rm(base, bands['clean']) - rm(out, bands['clean']))
    print(key, json.dumps({a: (np.round(b, 4) if not isinstance(b, (list, dict)) else b) for a, b in res[key].items() if a != 'weights'}), 'median w', np.median(ws))
P1, _ = al('direct'); P2, _ = al('padre'); print('corr of residuals direct vs padre', np.corrcoef((P1 - y)[:, ~fail].ravel(), (P2 - y)[:, ~fail].ravel())[0, 1],
                                                  '| padre vs 076', np.corrcoef((P2 - y)[:, ~fail].ravel(), (base - y)[:, ~fail].ravel())[0, 1])
json.dump(res, open(_repo_path(HERE, 'e2-results.json'), 'w'), indent=1)
