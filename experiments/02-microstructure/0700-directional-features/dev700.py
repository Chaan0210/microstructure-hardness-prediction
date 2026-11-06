"""Official dev procedure (same partitions/folds/seeds as push-300/dev_eval.py) with optional target/feature transforms.
usage: dev600.py FEATURE_TRAIN_NPZ TAG [device=mps]
env: RESID=col1,col2,...  -> in-fold RidgeCV on these columns (median-imputed, standardised), TabPFN fits y - ridge, prediction = ridge + TabPFN
     ADD=name             -> extra columns from extra_cols.py (computed from the feature matrix only)
     SEEDS=0,1,2
Saves dev-<TAG>.npz in push-300 format (usable with push-300/dev_combine.py when copied/symlinked)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, json, hashlib, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-300')); sys.path.insert(0, _repo_io('../push-600'))
import harness
from sklearn.linear_model import RidgeCV
P = Path(_repo_io('../..')); R = Path(_repo_io('../grain-cellpose-220'))
fpath, tag = sys.argv[1], sys.argv[2]; device = sys.argv[3] if len(sys.argv) > 3 else 'mps'
F = np.load(_repo_io(fpath), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
if os.environ.get('ADD'):
    import extra_cols
    XA, keys = extra_cols.add(os.environ['ADD'], XA, keys, [str(i) for i in F['ids']])
DROP = [c for c in os.environ.get('DROP', '').split(',') if c]
if DROP:
    keep = [j for j, k in enumerate(keys) if k not in DROP]; XA = XA[:, keep]; keys = [keys[j] for j in keep]
REFL = os.environ.get('REFLECT')   # column(s) whose sign flips under mirror reflection: average predictions over both signs
refl_idx = [keys.index(c) for c in REFL.split(',')] if REFL else []
o3 = np.load(_repo_path(R, 'oof-v3.npz'))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
rcols = [keys.index(c) for c in os.environ['RESID'].split(',')] if os.environ.get('RESID') else None
seeds = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
KW = json.loads(os.environ.get('KW', '{}')); IC = json.loads(os.environ.get('IC', '{}'))
def make(seed):
    if os.environ.get('MODEL') == 'v2': return harness.tabpfn(32, seed, device)
    m = harness.tabpfn_new('v3.5', seed, device)
    for k_, v_ in KW.items(): setattr(m, k_, v_)
    if IC:
        from tabpfn.inference_config import InferenceConfig
        ic = dict(IC)
        if 'PREPROCESS_TRANSFORMS' in ic:
            from tabpfn.preprocessing import PreprocessorConfig
            ic['PREPROCESS_TRANSFORMS'] = [PreprocessorConfig(**p) for p in ic['PREPROCESS_TRANSFORMS']]
        m.inference_config = ic
    return m
rm = lambda v, y: float(np.sqrt(np.mean((v - y) ** 2)))
save = {}; res = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'
    p = 2 * o3[a + 'b'] - o3[a + 'g']
    assert len(set(ids)) == len(ids) and all(i in pos for i in ids)
    X = XA[[pos[i] for i in ids]]; failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
    g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        if os.environ.get('TRAIN_NOISE'):   # train only on rows whose noise is in [lo, hi) (NaN noise -> excluded)
            lo_, hi_ = [float(v) for v in os.environ['TRAIN_NOISE'].split(',')]; nz_ = X[:, keys.index('noise')]
            tr = tr & (nz_ >= lo_) & (nz_ < hi_)
        base_tr = np.zeros(tr.sum()); base_va = np.zeros(va.sum())
        if rcols is not None:
            Z = X[:, rcols]; med = np.nanmedian(Z[tr], 0); Z = np.where(np.isnan(Z), med, Z); mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9; Z = (Z - mu) / sd
            lr = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[tr], y[tr]); base_tr = lr.predict(Z[tr]); base_va = lr.predict(Z[va])
        ps = []
        for sd_ in seeds:
            m = make(100 * sd_ + k); m.fit(X[tr], y[tr] - base_tr); pv = m.predict(X[va])
            if refl_idx:
                Xm = X[va].copy(); Xm[:, refl_idx] = -Xm[:, refl_idx]; pv = 0.5 * (pv + m.predict(Xm))
            ps.append(pv + base_va)
        g[va] = np.mean(ps, 0)
    gf = np.where(failed, p, g)
    res[name] = dict(grain=rm(g, y), b021=rm(p, y), w075=rm(0.25 * p + 0.75 * gf, y))
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_fold': fold, f'{name}_p021': p, f'{name}_g': g, f'{name}_failed': failed})
    print(name, json.dumps(res[name]), flush=True)
np.savez(_repo_io(f'dev-{tag}.npz'), **save, feature_file=str(fpath), columns=np.array(keys)); json.dump(res, open(_repo_io(f'dev-{tag}.json'), 'w'), indent=1)
