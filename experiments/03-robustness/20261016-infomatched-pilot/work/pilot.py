"""Pilot evaluation exactly as frozen in PLAN.md (2026-10-16 21:50).
usage (from the project root):
  .venv/bin/python   experiments/infomatched-pilot-20261016/work/pilot.py ridge          -> work/oof-ridge.npz   (arms R0 R1 R2 R3 RD RS, Ridge + logRidge)
  tabpfn-v2 python   experiments/infomatched-pilot-20261016/work/pilot.py tab T1|T2|TD   -> work/oof-tab-<arm>.npz
  .venv/bin/python   experiments/infomatched-pilot-20261016/work/pilot.py diag           -> work/diag/d1d2.json
  .venv/bin/python   experiments/infomatched-pilot-20261016/work/pilot.py eval           -> work/pilot-results.json
Copies enter a training fold only if their source is in that outer training fold (dev ids). Holdout-100 sources are never used."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, csv, json, time, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
W = Path(__file__).resolve().parents[1]; E = _RepoPath(__file__).resolve().parents[3]; ROOT = _RepoPath(__file__).resolve().parents[4]; OUT = _repo_path(W, 'work')
V = 'u1dmavgH2Z'; NEW = ['v1a', 'v2a']; PROD = ['deg', 'deg2', 'deg3', 'deg4']   # NEW per amendment A1
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(ROOT, 'data/train.csv')))}
F = np.load(_repo_path(E, f'push-800/feat-{V}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; XA = F['X'].astype(float); pos = {str(i): k for k, i in enumerate(F['ids'])}
ni = keys.index('noise')
SPLITS = {'original': _repo_path(E, 'resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(E, 'resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(E, 'resolution-repartition-217/20261101/ensemble-OOF.npz')}
def copies(sets, src_dir):
    X, S = [], []
    for s in sets:
        f = (_repo_path(W, f'feat/synthrows-{s}-{V}-anghcor.npz')) if src_dir == 'new' else (_repo_path(E, f'push-1200/synthrows-{s}-{V}-anghcor.npz'))
        D = np.load(_repo_io(f), allow_pickle=True); assert [str(k) for k in D['keys']] == keys, f
        m = D['X'][:, ni].astype(float) * 255 >= 12; X.append(D['X'].astype(float)[m]); S += [str(i) for i in np.array(D['ids'])[m]]
    return np.vstack(X), np.array(S)

# ------------------------------------------------------------------ DTA Ridge / logRidge (production c14_partners.fitB)
GR = {'ridge': [0.1, 0.3, 1, 3, 10, 30, 100, 1000], 'log': [0.3, 1, 2, 3, 10, 30, 100]}
def pipe(kind, a):
    from sklearn.pipeline import make_pipeline; from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import Ridge
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True) if kind == 'ridge' else SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=a))
tf = lambda kind, y: np.log(y) if kind == 'log' else y; itf = lambda kind, p: np.exp(p) if kind == 'log' else p
def fitB(kind, Xr, yr, sr, nzr, Xc, yc, sc):
    from sklearn.model_selection import GroupKFold
    X = np.vstack([Xr, Xc]) if len(Xc) else Xr; y = np.r_[yr, yc]; srcs = np.r_[sr, sc]; isreal = np.r_[np.ones(len(yr), bool), np.zeros(len(yc), bool)]
    nzall = np.r_[nzr, np.full(len(yc), 99.0)]; best = None
    for a in GR[kind]:
        err = []
        for tr, va in GroupKFold(5).split(X, y, srcs):
            m = pipe(kind, a).fit(X[tr], tf(kind, y[tr])); sel = va[isreal[va] & (nzall[va] >= 12)]
            if sel.size: err += list((itf(kind, m.predict(X[sel])) - y[sel]) ** 2)
        e = np.mean(err)
        if best is None or e < best[0]: best = (e, a)
    return pipe(kind, best[1]).fit(X, tf(kind, y)), best[1]
def train_extra(arm, trs, PX, PS, NX, NS, perm):
    """extra (copy / duplicate) rows for an arm given the training-fold source set trs."""
    kp = np.array([s in trs for s in PS]); kn = np.array([s in trs for s in NS]); Z = np.zeros((0, len(keys)))
    if arm in ('R0', 'T0'): return Z, np.zeros(0), np.array([], str)
    if arm == 'R1' or arm == 'T1': return PX[kp], np.array([lab[s] for s in PS[kp]]), PS[kp]
    if arm == 'R2' or arm == 'T2': return NX[kn], np.array([lab[s] for s in NS[kn]]), NS[kn]
    if arm == 'R3': return np.vstack([PX[kp], NX[kn]]), np.array([lab[s] for s in PS[kp]] + [lab[s] for s in NS[kn]]), np.r_[PS[kp], NS[kn]]
    if arm == 'RS': return np.vstack([PX[kp], NX[kn]]), np.array([lab[s] for s in PS[kp]] + [lab[perm[s]] for s in NS[kn]]), np.r_[PS[kp], NS[kn]]
    if arm in ('RD', 'TD'):   # duplicates of each new-copy source's own clean row, same multiplicity as its new copies (+ production copies for RD)
        D = XA[[pos[s] for s in NS[kn]]]; yd = np.array([lab[s] for s in NS[kn]])
        if arm == 'RD': return np.vstack([PX[kp], D]), np.r_[np.array([lab[s] for s in PS[kp]]), yd], np.r_[PS[kp], NS[kn]]
        return D, yd, NS[kn]
    raise ValueError(arm)

def run_ridge():
    PX, PS = copies(PROD, 'prod'); NX, NS = copies(NEW, 'new'); srcs = sorted(set(NS)); perm = dict(zip(srcs, np.random.RandomState(20261016).permutation(srcs)))
    out = {}; t0 = time.time()
    for kind in ('ridge', 'log'):
        for arm in ('R0', 'R1', 'R2', 'R3', 'RD', 'RS'):
            for name, path in SPLITS.items():
                d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; nz = np.nan_to_num(X[:, ni] * 255, nan=0)
                g = np.zeros(len(y)); al = []
                for k in range(5):
                    tr, va = fold != k, fold == k; trs = set(np.array(ids)[tr])
                    Xc, yc, sc = train_extra(arm, trs, PX, PS, NX, NS, perm)
                    m, a = fitB(kind, X[tr], y[tr], np.array(ids)[tr], nz[tr], Xc, yc, sc); g[va] = itf(kind, m.predict(X[va])); al.append(a)
                out[f'{kind}_{arm}_{name}'] = g; out[f'{kind}_{arm}_{name}_alphas'] = np.array(al); out[f'ids_{name}'] = np.array(ids); out[f'y_{name}'] = y; out[f'fold_{name}'] = fold
            print(kind, arm, f'{time.time() - t0:.0f}s', flush=True)
    out['n_new_copies'] = len(NS); out['n_prod_copies'] = len(PS)
    np.savez(_repo_path(OUT, 'oof-ridge.npz'), **out)

def run_tab(arm):
    import torch; sys.path.insert(0, str(_repo_path(E, 'push-300'))); cwd = os.getcwd(); os.chdir(_repo_path(E, 'push-300')); import harness; os.chdir(_repo_io(cwd)); torch.set_num_threads(2)
    PX, PS = copies(PROD, 'prod'); out = {}; t0 = time.time(); o = _repo_path(OUT, f'oof-tab-{arm}.npz'); assert not o.exists()
    NX, NS = copies(NEW, 'new') if arm in ('T2', 'TD') else (np.zeros((0, len(keys))), np.array([], str))   # T1 needs only production copies
    for name, path in SPLITS.items():
        d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; g = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k; trs = set(np.array(ids)[tr]); Xc, yc, _ = train_extra(arm, trs, PX, PS, NX, NS, None)
            Xt, yt = np.vstack([X[tr], Xc]), np.r_[y[tr], yc]; ps = []
            for sd in (0, 1, 2):
                m = harness.tabpfn_new('v3.5', 100 * sd + k, 'mps'); m.fit(Xt, yt); ps.append(m.predict(X[va]))
            g[va] = np.mean(ps, 0); print(arm, name, k, len(yt), f'{time.time() - t0:.0f}s', flush=True)
        out[f'{name}_ids'] = np.array(ids); out[f'{name}_y'] = y; out[f'{name}_g'] = g
    np.savez(_repo_io(o), **out)

# ------------------------------------------------------------------ diagnostics D1/D2 (structure preservation, label information)
def run_diag():
    from sklearn.linear_model import RidgeCV; from sklearn.pipeline import make_pipeline; from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_predict
    key = ['u_ph', 'het4_u', 'grain_eqd_mean', 'm_el_pix', 'px_pore_frac', 'ang_c4', 'sizefield_std_s32']; res = {'D1_copy_source_corr': {}, 'D1_median_shift_vs_real_noisy': {}}
    nzall = np.nan_to_num(XA[:, ni] * 255, nan=0); real_noisy = XA[nzall >= 12]
    for tag, sets, sd in [('V1', ['v1a'], 'new'), ('V2', ['v2a'], 'new'), ('prod_deg4', ['deg4'], 'prod'), ('prod_all', PROD, 'prod')]:
        CX, CS = copies(sets, sd); SX = XA[[pos[s] for s in CS]]
        res['D1_copy_source_corr'][tag] = {k: float(np.corrcoef(np.nan_to_num(CX[:, keys.index(k)]), np.nan_to_num(SX[:, keys.index(k)]))[0, 1]) for k in key}
        res['D1_median_shift_vs_real_noisy'][tag] = {k: float((np.nanmedian(CX[:, keys.index(k)]) - np.nanmedian(real_noisy[:, keys.index(k)])) / np.nanstd(real_noisy[:, keys.index(k)])) for k in key}
    mk = lambda: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
    yr = np.array([lab[str(i)] for i in np.array(F['ids'])[nzall >= 12]])
    rr = [float(np.sqrt(((cross_val_predict(mk(), real_noisy, yr, cv=KFold(5, shuffle=True, random_state=s)) - yr) ** 2).mean())) for s in range(20)]
    res['D2_cv_rmse'] = {'real_noisy_n161': [float(np.mean(rr)), float(np.std(rr))]}
    for tag, sets, sd in [('V1', ['v1a'], 'new'), ('V2', ['v2a'], 'new'), ('prod_all', PROD, 'prod')]:
        CX, CS = copies(sets, sd); r = []
        for s in range(20):
            rg = np.random.RandomState(s); src = rg.choice(sorted(set(CS)), min(161, len(set(CS))), replace=False)
            j = np.array([rg.choice(np.where(CS == q)[0]) for q in src]); yc = np.array([lab[q] for q in src])
            r.append(float(np.sqrt(((cross_val_predict(mk(), CX[j], yc, cv=KFold(5, shuffle=True, random_state=s)) - yc) ** 2).mean())))
        # the same sources' CLEAN rows (structure ceiling of these sources)
        cl = []
        for s in range(20):
            rg = np.random.RandomState(s); src = rg.choice(sorted(set(CS)), min(161, len(set(CS))), replace=False); yc = np.array([lab[q] for q in src])
            cl.append(float(np.sqrt(((cross_val_predict(mk(), XA[[pos[q] for q in src]], yc, cv=KFold(5, shuffle=True, random_state=s)) - yc) ** 2).mean())))
        res['D2_cv_rmse'][f'{tag}_copies_n161'] = [float(np.mean(r)), float(np.std(r))]; res['D2_cv_rmse'][f'{tag}_sources_clean_n161'] = [float(np.mean(cl)), float(np.std(cl))]
    (_repo_path(OUT, 'diag')).mkdir(exist_ok=True); json.dump(res, open(_repo_path(OUT, 'diag/d1d2.json'), 'w'), indent=1); print(json.dumps(res, indent=1))

# ------------------------------------------------------------------ evaluation (frozen statistics)
def run_eval():
    sid = [str(i) for i in np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
    B = np.load(_repo_path(E, 'stepchange-20261016/repro/components/c14b-orig.npz')); y = B['y']; nz = B['nz_dev']; fail = B['fail_dev'][0]; g = (nz >= 12) & ~fail
    names = list(SPLITS); FO = np.zeros((3, 400), int)
    def al(d, key, kp):   # align a (partition) array to sid order
        out = np.zeros((3, 400))
        for k, n in enumerate(names):
            p = {str(i): j for j, i in enumerate(d[kp.format(n=n)])}; ix = [p[i] for i in sid]; out[k] = d[key.format(n=n)][ix]
        return out
    R = np.load(_repo_path(OUT, 'oof-ridge.npz'), allow_pickle=True)
    for k, n in enumerate(names):
        p = {str(i): j for j, i in enumerate(R[f'ids_{n}'])}; FO[k] = R[f'fold_{n}'][[p[i] for i in sid]]
    P = {f'{kind}_{a}': al(R, f'{kind}_{a}_{{n}}', 'ids_{n}') for kind in ('ridge', 'log') for a in ('R0', 'R1', 'R2', 'R3', 'RD', 'RS')}
    P['tab_T0'] = al(np.load(_repo_path(E, 'stepchange-20261016/work/e1/oof-F0.npz'), allow_pickle=True), '{n}_g', '{n}_ids')
    for a in ('T1', 'T2', 'TD'):
        f = _repo_path(OUT, f'oof-tab-{a}.npz')
        if f.exists(): P[f'tab_{a}'] = al(np.load(_repo_io(f), allow_pickle=True), '{n}_g', '{n}_ids')
    rng = np.random.RandomState(0); BI = [rng.randint(0, g.sum(), g.sum()) for _ in range(4000)]
    def compare(new, ref):
        se_n = ((P[new] - y) ** 2)[:, g]; se_r = ((P[ref] - y) ** 2)[:, g]; d = (se_n - se_r).mean(0)   # partition-averaged per image
        o = np.argsort(d); fo = FO[0][g]
        return dict(dMSE=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), P_better=float(np.mean([d[b].mean() < 0 for b in BI])),
                    per_partition=[float((se_n[k] - se_r[k]).mean()) for k in range(3)], per_fold_original=[float((se_n[0] - se_r[0])[fo == f].mean()) for f in range(5)],
                    drop_top5=float(np.delete(d, o[:5]).mean()), drop_top10=float(np.delete(d, o[:10]).mean()),
                    rmse=[float(np.sqrt(se_n.mean())), float(np.sqrt(se_r.mean()))])
    pairs = [('ridge_R3', 'ridge_R1'), ('log_R3', 'log_R1'), ('tab_T2', 'tab_T0'),                              # primary
             ('ridge_R2', 'ridge_R1'), ('log_R2', 'log_R1'), ('ridge_R1', 'ridge_R0'), ('log_R1', 'log_R0'),   # secondary / context
             ('ridge_R3', 'ridge_RD'), ('log_R3', 'log_RD'), ('ridge_R3', 'ridge_RS'), ('log_R3', 'log_RS'),   # controls
             ('tab_T1', 'tab_T0'), ('tab_T2', 'tab_TD'), ('tab_TD', 'tab_T0')]
    res = {'gated_rmse': {k: float(np.sqrt(((v - y) ** 2)[:, g].mean())) for k, v in P.items()}}
    for a, b in pairs:
        if a in P and b in P: res[f'{a} vs {b}'] = compare(a, b)
    res['alphas_median'] = {k: float(np.median(R[k])) for k in R.files if k.endswith('_alphas')}
    json.dump(res, open(_repo_path(OUT, 'pilot-results.json'), 'w'), indent=1)
    print(json.dumps(res['gated_rmse'], indent=1))
    for k, v in res.items():
        if ' vs ' in k: print(f"{k:24s} dMSE {v['dMSE']:+7.2f} +- {v['se']:.2f}  P {v['P_better']:.3f}  parts {np.round(v['per_partition'], 2)}  folds {np.round(v['per_fold_original'], 1)}  drop5 {v['drop_top5']:+.2f}")

if __name__ == '__main__':
    {'ridge': run_ridge, 'diag': run_diag, 'eval': run_eval}.get(sys.argv[1], lambda: run_tab(sys.argv[2]))()
