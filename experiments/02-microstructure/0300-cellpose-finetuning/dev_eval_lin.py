"""Official dev procedure (3 partitions x 5 folds on the 400 dev ids, best021 OOF from resolution-216/217) for a grain-feature TabPFN model.
Saves the raw grain OOF per partition so blends / weights can be compared later without refitting.
usage: dev_eval_lin.py FEATURE_TRAIN_NPZ MODEL TAG [device=cpu] [n_est for v3.5]
[lin variant] per fold: ridge (standardized, alpha LIN_ALPHA) on LIN_COLS fitted on outer-train rows; TabPFN fitted on the ridge residual; pred = ridge + TabPFN.
  MODEL: v2 (n_est 32 x seeds 0,1,2 as finalize_tabpfn/explore_tta) | v3.5 (default n_est x seeds 0,1,2)
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, hashlib, warnings, numpy as np, torch
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import harness
P = Path(_repo_io('../..')); R = Path(_repo_io('../grain-cellpose-220'))
fpath, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]; device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'; n_est = int(sys.argv[5]) if len(sys.argv) > 5 else None
F = np.load(_repo_io(fpath), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X']
o3 = np.load(_repo_path(R, 'oof-v3.npz'))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
def make(sd, k):
    if model == 'v2':
        m = harness.tabpfn(32, 100 * sd + k, device)
    else:
        m = harness.tabpfn_new(model, 100 * sd + k, device, n_est)
    return m
rm = lambda v, y: float(np.sqrt(np.mean((v - y) ** 2)))
save = {}; res = {}
import os
LIN_COLS = os.environ.get('LIN_COLS', 'u_ph_hard,m_ph_hard,m_la_mean,het4_u,m_el_glob,stcoh_s3_w8_locmed,pore_frac,noise').split(','); LIN_ALPHA = float(os.environ.get('LIN_ALPHA', '1.0'))
li = [keys.index(c) for c in LIN_COLS]
def ridge_fit(Xl, yl):
    mu = np.nanmean(Xl, 0); sd = np.nanstd(Xl, 0) + 1e-9; Z = np.nan_to_num((Xl - mu) / sd); A = np.c_[np.ones(len(Z)), Z]; R_ = LIN_ALPHA * np.eye(A.shape[1]); R_[0, 0] = 0
    b = np.linalg.solve(A.T @ A + R_, A.T @ yl); return lambda Xq: np.c_[np.ones(len(Xq)), np.nan_to_num((Xq - mu) / sd)] @ b
prov = json.load(open(_repo_io(Path(fpath).with_suffix('.provenance.json')))) if Path(fpath).with_suffix('.provenance.json').exists() else None
res['inputs'] = dict(feature_file=str(fpath), feature_sha256=hashlib.sha256(open(_repo_io(fpath), 'rb').read()).hexdigest(), n_columns=len(keys), columns=keys,
                     feature_provenance=prov, model=model, device=device, seeds='100*sd+fold, sd in 0,1,2', n_est=('32' if model == 'v2' else (str(n_est) if n_est else 'default')))
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'
    p = 2 * o3[a + 'b'] - o3[a + 'g']
    X = XA[[pos[i] for i in ids]]; failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
    g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k; ps = []
        for sd in (0, 1, 2):
            lf = ridge_fit(X[tr][:, li], y[tr]); m = make(sd, k); m.fit(X[tr], y[tr] - lf(X[tr][:, li])); ps.append(m.predict(X[va]) + lf(X[va][:, li]))
        g[va] = np.mean(ps, 0)
    gf = np.where(failed, p, g)
    res[name] = dict(grain=rm(g, y), b021=rm(p, y), blend05=rm(0.5 * p + 0.5 * gf, y))
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_fold': fold, f'{name}_p021': p, f'{name}_g': g, f'{name}_failed': failed})
    print(name, json.dumps(res[name]), flush=True)
ref = save['original_ids']
def se(n, w):
    ix = {i: k for k, i in enumerate(save[f'{n}_ids'])}; j = [ix[i] for i in ref]
    p, g, f, y = save[f'{n}_p021'][j], save[f'{n}_g'][j], save[f'{n}_failed'][j], save[f'{n}_y'][j]
    return ((1 - w) * p + w * np.where(f, p, g) - y) ** 2
res['pooled05'] = float(np.sqrt(np.mean([se(n, 0.5) for n in splits], 0).mean()))
print('pooled 0.5 blend', res['pooled05'], flush=True)
np.savez(_repo_io(f'dev-{tag}.npz'), **save, feature_file=str(fpath), feature_sha256=res['inputs']['feature_sha256'], columns=np.array(keys)); json.dump(res, open(_repo_io(f'dev-{tag}.json'), 'w'), indent=1)
