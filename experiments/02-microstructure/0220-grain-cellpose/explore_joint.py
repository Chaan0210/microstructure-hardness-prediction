from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, time, socket, warnings
import numpy as np, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
def blocked(*a, **k): raise RuntimeError('offline')
socket.create_connection = blocked; socket.socket.connect = blocked; socket.getaddrinfo = blocked
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]; T = _repo_path(P, 'experiments/tabpfn-v2')
from tabpfn import TabPFNRegressor, ModelSpecs
from tabpfn import model_loading
models, criterion, configs, ic = model_loading.load_model_criterion_config(model_path=_repo_path(T, 'weights/tabpfn-v2-regressor.ckpt'), version='v2', estimator_type='regressor', check_bar_distribution_criterion=True, download_if_not_exists=False)
spec = ModelSpecs(model=models[0], architecture_config=configs[0], inference_config=ic, norm_criterion=criterion)
torch.set_num_threads(4)
F = np.load(_repo_path(R, 'features-v3-train.npz')); keys = list(F['keys']); pos = {i: k for k, i in enumerate(F['ids'])}
D = np.load(_repo_path(P, 'experiments/candidate021-178/train-features.npz')); dpos = {i: k for k, i in enumerate(D['ids'])}
c = np.load(_repo_path(R, 'oof-v4-tabpfn.npz')); out = {}; per = {}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
o3 = np.load(_repo_path(R, 'oof-v3.npz')); t0 = time.time()
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = d['ids'], d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'
    p = 2 * o3[a + 'b'] - o3[a + 'g']; bC = c[f'{name}_bC']; gC = c[f'{name}_gt']
    X = F['X'][[pos[i] for i in ids]]; Z = D['X'][[dpos[i] for i in ids]]
    failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
    g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        sc = StandardScaler().fit(Z[tr]); pca = PCA(32, svd_solver='full').fit(sc.transform(Z[tr]))
        Xtr = np.c_[X[tr], pca.transform(sc.transform(Z[tr]))]; Xva = np.c_[X[va], pca.transform(sc.transform(Z[va]))]
        est = TabPFNRegressor(model_path=spec, n_estimators=8, device='cpu', inference_precision=torch.float32, random_state=k, show_progress_bar=False)
        est.fit(Xtr, y[tr]); g[va] = est.predict(Xva)
    gE_raw = g.copy()
    g = np.where(failed, p, g); bE = 0.5 * p + 0.5 * g
    rm = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
    out[name] = dict(E_standalone=rm(gE_raw), C_standalone=rm(gC), E_blend=rm(bE), C_blend=rm(bC), sec=round(time.time() - t0))
    per[name] = dict(ids=ids, y=y, bC=bC, bE=bE, gE=gE_raw); print(name, json.dumps(out[name]), flush=True)
ref = per['original']['ids']
def se(n, key):
    ix = {i: k for k, i in enumerate(per[n]['ids'])}; j = [ix[i] for i in ref]; return (per[n][key][j] - per[n]['y'][j]) ** 2
s0 = np.mean([se(n, 'bC') for n in per], 0); s1 = np.mean([se(n, 'bE') for n in per], 0); rng = np.random.default_rng(2); gains = []
for _ in range(5000):
    i = rng.integers(0, len(ref), len(ref)); gains.append(np.sqrt(s0[i].mean()) - np.sqrt(s1[i].mean()))
out['pooled'] = dict(C=float(np.sqrt(s0.mean())), E=float(np.sqrt(s1.mean())), gain=float(np.sqrt(s0.mean()) - np.sqrt(s1.mean())), ci95=[float(np.quantile(gains, .025)), float(np.quantile(gains, .975))],
                     all3_better=bool(all(out[n]['E_blend'] < out[n]['C_blend'] for n in per)))
out['adopt_E'] = bool(out['pooled']['all3_better'] and out['pooled']['gain'] >= 0.10)
print(json.dumps(out['pooled']), 'adopt_E', out['adopt_E'])
json.dump(out, open(_repo_path(R, 'explore-joint.json'), 'x'), indent=1)
np.savez(_repo_path(R, 'oof-v5-joint.npz'), **{f'{n}_{k}': v for n, dct in per.items() for k, v in dct.items()})
