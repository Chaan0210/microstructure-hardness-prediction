from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, time, socket, warnings
import numpy as np, torch
from pathlib import Path
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
o = np.load(_repo_path(R, 'oof-v3.npz')); out = {}; per = {}
splits = {'original': (_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz')), '20261031': (_repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz')), '20261101': (_repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz'))}
t0 = time.time()
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = d['ids'], d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'
    assert (o[a + 'ids'] == ids).all(); gk = o[a + 'g']; b3 = o[a + 'b']; p = 2 * b3 - gk
    X = F['X'][[pos[i] for i in ids]]; failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
    gt = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        est = TabPFNRegressor(model_path=spec, n_estimators=8, device='cpu', inference_precision=torch.float32, random_state=k, show_progress_bar=False)
        est.fit(X[tr], y[tr]); gt[va] = est.predict(X[va])
    gt = np.where(failed, p, gt); gk2 = np.where(failed, p, gk)
    rm = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
    bC = 0.5 * p + 0.5 * gt; bD = 0.5 * p + 0.5 * (0.5 * gk2 + 0.5 * gt)
    out[name] = dict(tabpfn_standalone=rm(gt), krr_standalone=rm(gk), v3_blend=rm(b3), C_blend=rm(bC), D_blend=rm(bD), sec=round(time.time() - t0))
    per[name] = dict(ids=ids, y=y, b3=b3, bC=bC, bD=bD, gt=gt)
    print(name, json.dumps(out[name]), flush=True)
ref = per['original']['ids']
def se(n, key):
    ix = {i: k for k, i in enumerate(per[n]['ids'])}; j = [ix[i] for i in ref]; return (per[n][key][j] - per[n]['y'][j]) ** 2
s0 = np.mean([se(n, 'b3') for n in per], 0)
for arm in ['bC', 'bD']:
    s1 = np.mean([se(n, arm) for n in per], 0); rng = np.random.default_rng(1); gains = []
    for _ in range(5000):
        i = rng.integers(0, len(ref), len(ref)); gains.append(np.sqrt(s0[i].mean()) - np.sqrt(s1[i].mean()))
    out[arm + '_pooled'] = dict(v3=float(np.sqrt(s0.mean())), arm=float(np.sqrt(s1.mean())), gain=float(np.sqrt(s0.mean()) - np.sqrt(s1.mean())), ci95=[float(np.quantile(gains, .025)), float(np.quantile(gains, .975))],
                                all3_better=bool(all(out[n][('C' if arm == 'bC' else 'D') + '_blend'] < out[n]['v3_blend'] for n in per)))
    print(arm, json.dumps(out[arm + '_pooled']))
json.dump(out, open(_repo_path(R, 'explore-tabpfn.json'), 'x'), indent=1)
np.savez(_repo_path(R, 'oof-v4-tabpfn.npz'), **{f'{n}_{k}': v for n, dct in per.items() for k, v in dct.items()})
