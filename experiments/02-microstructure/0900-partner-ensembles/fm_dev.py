"""Dev OOF for non-TabPFN tabular foundation models (run with experiments/tabular-fm/.venv): TabICL regressor v2 (jingang/TabICL,
tabicl-regressor-v2-20260212.ckpt) and TabDPT 1.3 (Layer6/TabDPT tabdpt1_3.safetensors), both public before 2026-09-29, local weights only.
Same partitions / folds / failed rule as dev900.py (push-300 format output dev-<TAG>.npz).
usage: fm_dev.py tabicl|tabdpt FEATURE_TRAIN_NPZ TAG [seeds=0,1,2]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, json, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]; W = _repo_path(E, 'tabular-fm/weights')
model, fpath, tag = sys.argv[1], sys.argv[2], sys.argv[3]; seeds = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else '0,1,2').split(',')]
F = np.load(_repo_io(fpath), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
o3 = np.load(_repo_path(E, 'grain-cellpose-220/oof-v3.npz'))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
def make(seed):
    if model == 'tabicl':
        from tabicl import TabICLRegressor
        return TabICLRegressor(model_path=str(_repo_path(W, 'tabicl-regressor-v2-20260212.ckpt')), allow_auto_download=False, device='cpu', random_state=seed)
    from tabdpt import TabDPTRegressor
    return TabDPTRegressor(model_weight_path=str(_repo_path(W, 'tabdpt1_3.safetensors')), device='cpu', use_flash=False, compile=False, verbose=False)
def fit_predict(m, Xtr, ytr, Xva, seed):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med); Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    m.fit(Xtr.astype(np.float32), ytr.astype(np.float32))
    if model == 'tabdpt': return m.predict(Xva.astype(np.float32), seed=seed)
    return m.predict(Xva.astype(np.float32))
rm = lambda v, y: float(np.sqrt(np.mean((v - y) ** 2)))
save = {}; res = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'
    p = 2 * o3[a + 'b'] - o3[a + 'g']; X = XA[[pos[i] for i in ids]]; failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
    g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        g[va] = np.mean([fit_predict(make(100 * s + k), X[tr], y[tr], X[va], 100 * s + k) for s in seeds], 0)
    gf = np.where(failed, p, g); res[name] = dict(grain=rm(g, y), b021=rm(p, y), w075=rm(0.25 * p + 0.75 * gf, y))
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_fold': fold, f'{name}_p021': p, f'{name}_g': g, f'{name}_failed': failed})
    print(name, json.dumps(res[name]), flush=True)
np.savez(_repo_path(H, f'dev-{tag}.npz'), **save, feature_file=str(fpath), columns=np.array(keys)); json.dump(res, open(_repo_path(H, f'dev-{tag}.json'), 'w'), indent=1)
