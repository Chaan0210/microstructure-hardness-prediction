"""Fast evaluation harness on all 500 train images: repeated 5-fold CV, per-noise-tercile RMSE."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, socket, warnings, time
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
P = _RepoPath(__file__).resolve().parents[3]
G = _repo_path(P, 'experiments/grain-cellpose-220')
T = _repo_path(P, 'experiments/tabpfn-v2')

def labels(ids):
    lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
    return np.array([lab[i] for i in ids])

def load_feats(prefix='features-v3tta'):
    F = np.load(_repo_path(G, f'{prefix}-train.npz'), allow_pickle=True)
    return F['X'].astype(float), [str(k) for k in F['keys']], [str(i) for i in F['ids']]

def folds(n, rep, k=5):
    rng = np.random.RandomState(1000 + rep); f = np.arange(n) % k; rng.shuffle(f); return f

def noise_of(ids, split='train'):
    from PIL import Image
    out = []
    for i in ids:
        f = np.asarray(Image.open(_repo_path(P, f'data/{split}/{i}.png')), np.float32) / 255
        hh = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2
        out.append(np.median(np.abs(hh)) / 0.6745)
    return np.array(out)

_spec = None
def tabpfn(n_est=8, seed=0, device='cpu'):
    global _spec
    def blocked(*a, **k): raise RuntimeError('offline')
    socket.create_connection = blocked; socket.getaddrinfo = blocked
    import torch
    from tabpfn import TabPFNRegressor, ModelSpecs, model_loading
    if _spec is None:
        models, criterion, configs, ic = model_loading.load_model_criterion_config(model_path=_repo_path(T, 'weights/tabpfn-v2-regressor.ckpt'), version='v2', estimator_type='regressor', check_bar_distribution_criterion=True, download_if_not_exists=False)
        _spec = ModelSpecs(model=models[0], architecture_config=configs[0], inference_config=ic, norm_criterion=criterion)
    return TabPFNRegressor(model_path=_spec, n_estimators=n_est, device=device, inference_precision=torch.float32, random_state=seed, show_progress_bar=False)

def cv_predict(make, X, y, reps=(0,), k=5):
    """returns oof [len(reps), n]"""
    O = np.zeros((len(reps), len(y)))
    for r_i, r in enumerate(reps):
        f = folds(len(y), r, k)
        for j in range(k):
            tr, va = f != j, f == j
            m = make(); m.fit(X[tr], y[tr]); O[r_i, va] = m.predict(X[va])
    return O

def report(name, O, y, nz=None):
    rm = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    per = [rm(o) for o in O]
    s = f'{name:40s} RMSE mean {np.mean(per):.4f} ({" ".join(f"{v:.3f}" for v in per)})'
    if nz is not None:
        q = np.quantile(nz, [1/3, 2/3]); t = np.digitize(nz, q)
        s += ' | terciles ' + ' '.join(f'{np.mean([np.sqrt(np.mean((o[t==j]-y[t==j])**2)) for o in O]):.3f}' for j in range(3))
    print(s, flush=True); return float(np.mean(per))

W3 = _repo_path(P, 'experiments/tabpfn-v3/weights')
def tabpfn_new(version='v3.5', seed=0, device='mps', n_est=None):
    """TabPFN >= 2.5 from a locally downloaded checkpoint (offline)."""
    def blocked(*a, **k): raise RuntimeError('offline')
    socket.create_connection = blocked; socket.getaddrinfo = blocked
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    files = {'v3.5': 'tabpfn-v3.5-20260909.safetensors', 'v3.5-fast': 'tabpfn-v3.5-fast-20260909.safetensors', 'v3': 'tabpfn-v3-regressor-v3_default.ckpt',
             'v2.6': 'tabpfn-v2.6-regressor-v2.6_default.ckpt', 'v2.5': 'tabpfn-v2.5-regressor-v2.5_default.ckpt', 'v2.5-real': 'tabpfn-v2.5-regressor-v2.5_real.ckpt',
             'v2.5-small': 'tabpfn-v2.5-regressor-v2.5_small-samples.ckpt'}
    mv = {'v2.5-real': 'v2.5', 'v2.5-small': 'v2.5'}.get(version, version)
    kw = dict(model_path=str(_repo_path(W3, files[version])), device=device, random_state=seed, show_progress_bar=False)
    if n_est is not None: kw['n_estimators'] = n_est
    return TabPFNRegressor.create_default_for_version(ModelVersion(mv), **kw)
