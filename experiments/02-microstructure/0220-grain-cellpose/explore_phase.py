from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, socket, warnings
import numpy as np, torch
from pathlib import Path
warnings.filterwarnings('ignore')
def blocked(*a, **k): raise RuntimeError('offline')
socket.create_connection = blocked; socket.socket.connect = blocked; socket.getaddrinfo = blocked
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]; T = _repo_path(P, 'experiments/tabpfn-v2')
from tabpfn import TabPFNRegressor, ModelSpecs
from tabpfn import model_loading
models, criterion, configs, ic = model_loading.load_model_criterion_config(model_path=_repo_path(T, 'weights/tabpfn-v2-regressor.ckpt'), version='v2', estimator_type='regressor', check_bar_distribution_criterion=True, download_if_not_exists=False)
spec = ModelSpecs(model=models[0], architecture_config=configs[0], inference_config=ic, norm_criterion=criterion); torch.set_num_threads(4)
F = np.load(_repo_path(R, 'features-v3-train.npz')); keys = list(F['keys']); pos = {i: k for k, i in enumerate(F['ids'])}
Ph = np.load(_repo_path(R, 'phase-train.npz')); assert (Ph['ids'] == F['ids']).all()
XA = np.c_[F['X'], Ph['X']]
c = np.load(_repo_path(R, 'oof-v4-tabpfn.npz')); o3 = np.load(_repo_path(R, 'oof-v3.npz')); out = {}; per = {}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = d['ids'], d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'
    p = 2 * o3[a + 'b'] - o3[a + 'g']; bC = c[f'{name}_bC']
    X = XA[[pos[i] for i in ids]]; failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
    g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        est = TabPFNRegressor(model_path=spec, n_estimators=8, device='cpu', inference_precision=torch.float32, random_state=k, show_progress_bar=False)
        est.fit(X[tr], y[tr]); g[va] = est.predict(X[va])
    graw = g.copy(); g = np.where(failed, p, g); bP = 0.5 * p + 0.5 * g
    rm = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
    out[name] = dict(P_standalone=rm(graw), P_blend=rm(bP), C_blend=rm(bC)); per[name] = dict(ids=ids, y=y, bC=bC, bP=bP)
    print(name, json.dumps(out[name]), flush=True)
ref = per['original']['ids']
def se(n, key):
    ix = {i: k for k, i in enumerate(per[n]['ids'])}; j = [ix[i] for i in ref]; return (per[n][key][j] - per[n]['y'][j]) ** 2
s0 = np.mean([se(n, 'bC') for n in per], 0); s1 = np.mean([se(n, 'bP') for n in per], 0)
out['pooled'] = dict(C=float(np.sqrt(s0.mean())), P=float(np.sqrt(s1.mean())), gain=float(np.sqrt(s0.mean()) - np.sqrt(s1.mean())), all3_better=bool(all(out[n]['P_blend'] < out[n]['C_blend'] for n in per)))
out['adopt'] = bool(out['pooled']['all3_better'] and out['pooled']['gain'] >= 0.05)
print(json.dumps(out['pooled']), 'adopt', out['adopt']); json.dump(out, open(_repo_path(R, 'explore-phase.json'), 'x'), indent=1)
