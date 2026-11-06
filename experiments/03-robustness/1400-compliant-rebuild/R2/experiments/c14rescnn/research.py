"""Isolated, train-only RMSE audit and nested linear-domain experiments.

Run from repository root with .venv/bin/python. No shared output is modified.
All new estimators fit preprocessing within the relevant training fold.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import numpy as np
from scipy.linalg import solve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = _repo_path(ROOT, 'experiments')
H13 = _repo_path(EXP, 'push-1300')
SEED = 20261015
ALPHAS = (0.1, 1.0, 10.0, 100.0)
SCALES = (0.0, 0.1, 0.3, 1.0)
PENALTIES = (0.0, 0.1, 1.0, 10.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def metrics(pred, y, fold):
    return dict(pooled=rmse(pred, y), partitions=[rmse(p, y) for p in pred],
                folds=[[rmse(pred[s, fold[s] == k], y[fold[s] == k])
                        for k in range(5)] for s in range(len(fold))])


def compare(base, new, y, fold, draws=10000):
    olderr = ((base - y) ** 2).mean(0)
    newerr = ((new - y) ** 2).mean(0)
    gains = olderr - newerr
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), (draws, len(y)))
    bg = np.sqrt(olderr[idx].mean(1)) - np.sqrt(newerr[idx].mean(1))
    b, n = metrics(base, y, fold), metrics(new, y, fold)
    order = np.argsort(gains)[::-1]
    out = dict(baseline=b, candidate=n, gain=b['pooled'] - n['pooled'],
               partition_gains=(np.array(b['partitions']) - n['partitions']).tolist(),
               fold_gains=(np.array(b['folds']) - n['folds']).tolist(),
               improved_folds=int((np.array(b['folds']) > n['folds']).sum()),
               conditional_id_bootstrap_gain_ci95=np.quantile(bg, [.025, .975]).tolist(),
               bootstrap_fraction_positive=float(np.mean(bg > 0)),
               prediction_delta_rms=float(np.sqrt(np.mean((new-base)**2))),
               error_correlation=float(np.corrcoef((base-y).ravel(), (new-y).ravel())[0, 1]),
               improved_ids=int((gains > 0).sum()),
               top5_gain_fraction=float(gains[order[:5]].sum()/gains.sum()) if gains.sum() else None)
    for k in (5, 10):
        keep = order[k:]
        out[f'gain_without_top{k}_improvements'] = float(np.sqrt(olderr[keep].mean()) - np.sqrt(newerr[keep].mean()))
    return out


def snapshot():
    if (_repo_path(HERE, 'inputs.npz')).exists():
        raise FileExistsError('The immutable input snapshot already exists.')
    inputs = []

    def read(path):
        inputs.append(path)
        return np.load(_repo_io(path), allow_pickle=True)

    base = read(_repo_path(EXP, 'probabilistic-grains-900/base.npz'))
    a = read(_repo_path(H13, 'audit061.npz'))
    b = read(_repo_path(EXP, 'probabilistic-grains-900/ridge/predictions.npz'))
    assert list(b['ids']) == list(base['ids'])
    assert np.array_equal(a['y'], base['y'])
    assert np.array_equal(a['fold'], base['fold'])
    sid = base['ids'].astype(str)
    labels_path = _repo_path(ROOT, 'data/train.csv')
    inputs.append(labels_path)
    rows = list(csv.DictReader(labels_path.open()))
    labels = {r['ID']: float(r['hardness']) for r in rows}
    f = read(_repo_path(EXP, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'))
    ids, x, keys = f['ids'].astype(str), f['X'].astype(float), f['keys'].astype(str)
    pos = {v: i for i, v in enumerate(ids)}
    dev = np.array([pos[i] for i in sid])
    y = np.array([labels[i] for i in ids])
    assert len(set(ids)) == 500 and len(set(sid)) == 400
    assert np.array_equal(y[dev], a['y'])
    assert np.allclose(x[dev], base['X98'], equal_nan=True)
    noise = np.nan_to_num(x[:, list(keys).index('noise')] * 255, nan=0)
    assert np.allclose(noise[dev], a['nz'])
    nb = read(_repo_path(H13, 'nb-partners.npz'))
    groups_path = _repo_path(H13, 'abl/groups.json')
    inputs.append(groups_path)
    groups = json.loads(groups_path.read_text())
    drop = set((groups['het'] + ' ' + groups['intensity']).split())
    keep = np.array([k not in drop for k in keys])
    gate = noise[dev] >= 12
    fail = a['fail'].astype(bool)
    assert np.array_equal(fail, np.broadcast_to(fail[0], fail.shape))
    ng = ~gate & ~fail[0]
    tabs = []
    for slot in ('038', '039', '044'):
        for var in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4'):
            d = read(_repo_path(H13, f'abl/dev-nb-{slot}-{var}.npz'))
            pred = []
            for split in ('original', '20261031', '20261101'):
                ii = list(d[split + '_ids'].astype(str))
                pred.append(d[split + '_g'][[ii.index(i) for i in sid]])
            tabs.append(pred)
    c = {name: a['C_' + name].copy() for name in ('cnn', 'tab', 'ridge', 'logr', 'set', 'gmm')}
    w = {'cnn': .16, 'tab': .384, 'ridge': .128, 'logr': .128, 'set': .05, 'gmm': .15}
    c61 = np.where(fail, c['cnn'], sum(w[k]*c[k] for k in w))
    assert np.max(np.abs(c61-a['c61'])) < 1e-10
    c['tab'][:, ng] = np.mean(tabs, axis=0)[:, ng]
    for name, key in (('ridge', 'ridge'), ('logr', 'logridge'), ('gmm', 'gmm')):
        # The actual component, not a delta transported onto a different CV seed.
        c[name][:, ng] = nb[key + '_nb_dev'][:, ng]
    c73 = np.where(fail, c['cnn'], sum(w[k]*c[k] for k in w))
    legacy = read(_repo_path(H13, 'dev-nb-candidate.npy'))
    sx, src, sets = [], [], []
    for tag in ('deg', 'deg2', 'deg3', 'deg4'):
        s = read(_repo_path(EXP, f'push-1200/synthrows-{tag}-u1dmavgH2Z-anghcor.npz'))
        assert list(s['keys'].astype(str)) == list(keys)
        sel = s['X'][:, list(keys).index('noise')].astype(float) * 255 >= 12
        sx.append(s['X'].astype(float)[sel])
        src.extend(pos[str(i)] for i in s['ids'][sel])
        sets.extend([tag] * int(sel.sum()))
    sx, src = np.vstack(sx), np.array(src)
    np.savez_compressed(_repo_path(HERE, 'inputs.npz'), ids=ids, X=x, y=y, noise=noise,
                        keys=keys, keep=keep, dev=dev, fold=a['fold'], fail=fail,
                        c61=c61, c73=c73, legacy73=legacy,
                        ridge73=c['ridge'], sx=sx, src=src, degradation=np.array(sets),
                        **{f'C73_{k}': v for k, v in c.items()})
    protected = [_repo_path(ROOT, f'artifacts/submission-candidate-{k}.{suffix}')
                 for k in ('021', '061', '073') for suffix in ('csv', 'manifest.json')]
    inputs += protected + [_repo_path(HERE, 'PLAN.md'), Path(__file__)]
    save_json(_repo_path(HERE, 'provenance.json'), {str(p.relative_to(ROOT)): digest(p) for p in inputs})
    audit = dict(n_train=len(ids), n_dev=len(dev), feature_count=len(keys),
                 non_noisy_feature_count=int(keep.sum()), target_range=[float(y.min()), float(y.max())],
                 target_mean=float(y.mean()), target_sd=float(y.std()),
                 fold_sizes=[[int((a['fold'][p]==k).sum()) for k in range(5)] for p in range(3)],
                 baseline061=metrics(c61, y[dev], a['fold']),
                 baseline073_actual_components=metrics(c73, y[dev], a['fold']),
                 baseline073_legacy_delta=metrics(legacy, y[dev], a['fold']),
                 legacy_vs_direct_max_difference=float(np.max(np.abs(legacy-c73))),
                 baseline061_vs073=compare(c61, c73, y[dev], a['fold']),
                 missing_feature_cells=int(np.isnan(x).sum()), failed_dev_ids=sid[fail[0]].tolist(),
                 degradation_rows=len(src), degradation_unique_sources=int(len(set(src))),
                 degradation_sources_outside_dev=int(len(set(src)-set(dev))),
                 degradation_rows_outside_dev=int((~np.isin(src, dev)).sum()),
                 augmentation_validation_source_overlap=0)
    audit['per_fold_augmentation_counts'] = []
    for p in range(3):
        for k in range(5):
            tr, va = dev[a['fold'][p] != k], dev[a['fold'][p] == k]
            historical = ~np.isin(src, va)
            strict = np.isin(src, tr)
            assert not np.any(np.isin(src[historical], va))
            audit['per_fold_augmentation_counts'].append(dict(partition=p, fold=k,
                historical_rows=int(historical.sum()), strict_rows=int(strict.sum()),
                historical_unique_training_source_count=len(set(tr)|set(src[historical]))))
    audit['noise_bands'] = {}
    e = (c73-y[dev])**2
    for name, m in [('clean_lt9.5', noise[dev]<9.5), ('mid9.5_12', (noise[dev]>=9.5)&~gate), ('noisy_ge12', gate)]:
        audit['noise_bands'][name] = dict(n=int(m.sum()), rmse=rmse(c73[:,m], y[dev][m]),
            sse_share=float(e[:,m].sum()/e.sum()), bias=float((c73-y[dev])[:,m].mean()),
            components={k: rmse(c[k][:,m], y[dev][m]) for k in c},
            error_correlation=np.corrcoef([(c[k]-y[dev])[:,m].ravel() for k in c]).tolist())
    audit['target_quintiles'] = []
    edges = np.quantile(y[dev], np.linspace(0,1,6))
    for i in range(5):
        m=(y[dev]>=edges[i]) & (y[dev]<=edges[i+1] if i==4 else y[dev]<edges[i+1])
        audit['target_quintiles'].append(dict(bounds=edges[i:i+2].tolist(), n=int(m.sum()),
            rmse=rmse(c73[:,m], y[dev][m]), bias=float((c73-y[dev])[:,m].mean())))
    order = np.argsort(e.mean(0))[::-1]
    audit['top10_sse_share']=float(e[:,order[:10]].sum()/e.sum())
    with (_repo_path(HERE, 'residuals.csv')).open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['ID','hardness','noise','prediction_mean','residual_mean','mean_squared_error'])
        for j in order: writer.writerow([sid[j], y[dev][j], noise[dev][j], c73[:,j].mean(), (c73[:,j]-y[dev][j]).mean(), e[:,j].mean()])
    save_json(_repo_path(HERE, 'audit.json'), audit)
    print(json.dumps({k:audit[k] for k in ('baseline061','baseline073_actual_components','baseline073_legacy_delta','degradation_sources_outside_dev','degradation_rows_outside_dev','noise_bands')},indent=2))


def training(data, train, valid, noisy, strict=False):
    """Return real rows first, then copies; all copies of held-out sources excluded."""
    x, y, noise, sx, src = [data[k] for k in ('X','y','noise','sx','src')]
    cols = np.ones(x.shape[1],bool) if noisy else data['keep']
    cp = (np.isin(src, train) if strict else ~np.isin(src, valid)) if noisy else np.zeros(len(src),bool)
    source = np.r_[train, src[cp]]
    assert not np.any(np.isin(source, valid))
    z = np.vstack([x[train][:,cols],sx[cp][:,cols]])
    target = y[source]
    dom = np.r_[(noise[train]>=12).astype(int),np.full(cp.sum(),2)]
    isreal = np.arange(len(source))<len(train)
    # A pair uses its real source only when that source is in the training set.
    pairs=[]; real_pos={s:i for i,s in enumerate(train)}
    for j,s in enumerate(src[cp], start=len(train)):
        if s in real_pos: pairs.append([real_pos[s],j])
    return z,target,dom,source,isreal,np.array(pairs,dtype=int).reshape(-1,2),cols


class DomainRidge:
    """Shared ridge coefficients, optional shrunk domain deviations / pair penalty."""
    def __init__(self, alpha, scale=0., penalty=0.):
        self.alpha, self.scale, self.penalty = alpha, scale, penalty

    def design(self, z, domain):
        if not self.scale: return z
        q=np.c_[np.ones(len(z)),z]
        return np.column_stack([z]+[self.scale*q*(domain==d)[:,None] for d in (0,1,2)])

    def fit(self, x, y, domain, pairs):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        self.scaler=StandardScaler().fit(self.imputer.transform(x))
        z=self.scaler.transform(self.imputer.transform(x))
        q=self.design(z,domain)
        self.xmean=q.mean(0); self.ymean=float(np.mean(y))
        qc=q-self.xmean
        lhs=qc.T@qc
        lhs.flat[::lhs.shape[0]+1]+=self.alpha
        if self.penalty and len(pairs):
            diff=q[pairs[:,0]]-q[pairs[:,1]]
            lhs+=self.penalty*len(x)/len(pairs)*(diff.T@diff)
        self.coef=solve(lhs,qc.T@(y-self.ymean),assume_a='pos',check_finite=False)
        return self

    def predict(self,x,domain):
        z=self.scaler.transform(self.imputer.transform(x))
        return (self.design(z,domain)-self.xmean)@self.coef+self.ymean


def fit_nested(x,y,dom,source,isreal,pairs,noisy,family):
    """Select on real target-domain validation rows, with source-grouped folds."""
    values=SCALES if family=='domain' else PENALTIES
    grid=[(a,v) for v in values for a in ALPHAS]
    loss=np.zeros(len(grid)); count=0
    cv=GroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    for tr,va in cv.split(x,y,source):
        assert not set(source[tr])&set(source[va])
        va=va[isreal[va] & (dom[va]==int(noisy))]
        assert len(va)
        old_to_new={v:i for i,v in enumerate(tr)}
        ip=np.array([[old_to_new[a],old_to_new[b]] for a,b in pairs
                     if a in old_to_new and b in old_to_new],dtype=int).reshape(-1,2)
        for j,(alpha,value) in enumerate(grid):
            kw={'scale':value} if family=='domain' else {'penalty':value}
            m=DomainRidge(alpha,**kw).fit(x[tr],y[tr],dom[tr],ip)
            loss[j]+=np.sum((m.predict(x[va],dom[va])-y[va])**2)
        count+=len(va)
    best=int(np.argmin(loss)); null=int(np.argmin(loss[:len(ALPHAS)]))
    result=[]
    for j in (null,best):
        alpha,value=grid[j]; kw={'scale':value} if family=='domain' else {'penalty':value}
        result.append(DomainRidge(alpha,**kw).fit(x,y,dom,pairs))
    return result,dict(alpha=grid[best][0],value=grid[best][1],control_alpha=grid[null][0],
                       inner_mse=(loss/count).tolist(),grid=grid)


def production(x,y,dom,pairs):
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
    ss=StandardScaler().fit(imp.transform(x)); z=ss.transform(imp.transform(x))
    m=RidgeCV(alphas=[.1,1,10,100,1000]).fit(z,y)
    return DomainRidge(float(m.alpha_)).fit(x,y,dom,pairs)


def run(family,strict=False,partitions=(0,1,2)):
    start=time.monotonic(); d=np.load(_repo_path(HERE, 'inputs.npz'))
    name=family+('-strict' if strict else '')
    outpath=_repo_path(HERE, name+'.npz')
    if outpath.exists(): raise FileExistsError(outpath)
    pred={k:d['ridge73'].copy() for k in ('production','nested_control','candidate')}
    selections=[]; dev=d['dev']; fold=d['fold']; noise=d['noise']; xs=d['X']
    for p in partitions:
        for k in range(5):
            train,valid=dev[fold[p]!=k],dev[fold[p]==k]
            for noisy in ((True,) if family=='consistency' else (False,True)):
                xv,yt,dom,source,isreal,pairs,cols=training(d,train,valid,noisy,strict)
                (ctrl,model),sel=fit_nested(xv,yt,dom,source,isreal,pairs,noisy,family)
                prod=production(xv,yt,dom,pairs)
                indices=np.where((fold[p]==k)&((noise[dev]>=12)==noisy))[0]
                query=xs[dev[indices]][:,cols]; qdom=(noise[dev[indices]]>=12).astype(int)
                for key,m in [('production',prod),('nested_control',ctrl),('candidate',model)]:
                    pred[key][p,indices]=m.predict(query,qdom)
                sel.update(partition=p,fold=k,noisy=noisy,n_real=len(train),n_copy=int((~isreal).sum()),n_sources=len(set(source)))
                selections.append(sel)
            print(name,'partition',p,'fold',k,'seconds',round(time.monotonic()-start,1),flush=True)
    # Preserve the same segmentation-failure fallback and fixed Ridge weight.
    blends={k:np.where(d['fail'],d['c73'],d['c73']+.128*(v-d['ridge73'])) for k,v in pred.items()}
    y=d['y'][dev]
    result={k:compare(d['c73'],v,y,fold) for k,v in blends.items()}
    result['candidate_vs_nested_control']=compare(blends['nested_control'],blends['candidate'],y,fold)
    result['candidate_vs_matched_production']=compare(blends['production'],blends['candidate'],y,fold)
    result['selections']=selections; result['seed']=SEED; result['seconds']=time.monotonic()-start
    result['strict_sources']=strict; result['partitions_executed']=list(partitions)
    result['standalone_rmse']={k:rmse(v,y) for k,v in pred.items()}
    result['production_max_diff_vs_stored_usable_rows']=float(np.max(np.abs(pred['production']-d['ridge73'])[~d['fail']]))
    result['production_max_diff_at_failed_rows_before_fallback']=float(np.max(np.abs(pred['production']-d['ridge73'])[d['fail']]))
    result['code_sha256']=digest(__file__)
    np.savez_compressed(_repo_io(outpath),**pred,**{k+'_blend':v for k,v in blends.items()})
    save_json(_repo_path(HERE, name+'.json'),result)
    print(json.dumps({k:result[k] for k in ('standalone_rmse','production_max_diff_vs_stored_usable_rows','candidate_vs_matched_production','candidate_vs_nested_control','seconds')},indent=2),flush=True)


def self_check():
    from sklearn.linear_model import Ridge
    rng=np.random.default_rng(SEED); x=rng.normal(size=(60,7)); y=rng.normal(size=60)
    dom=np.arange(60)%3; pairs=np.array([[0,1],[3,4],[6,7]])
    m=DomainRidge(2.).fit(x,y,dom,pairs)
    z=m.scaler.transform(m.imputer.transform(x)); ref=Ridge(alpha=2).fit(z,y).predict(z)
    assert np.allclose(ref,m.predict(x,dom),atol=1e-12)
    # Consistency implementation equals direct pseudo-observation least squares.
    m=DomainRidge(2.,penalty=.3).fit(x,y,dom,pairs)
    diff=z[pairs[:,0]]-z[pairs[:,1]]
    A=np.vstack([z-z.mean(0),np.sqrt(.3*len(x)/len(pairs))*diff,np.sqrt(2)*np.eye(7)])
    b=np.r_[y-y.mean(),np.zeros(len(pairs)+7)]
    assert np.allclose(m.coef,np.linalg.lstsq(A,b,rcond=None)[0],atol=1e-12)
    # Snapshot design: a source's copies never enter an outer validation fold.
    if (_repo_path(HERE, 'inputs.npz')).exists():
        d=np.load(_repo_path(HERE, 'inputs.npz')); dev=d['dev']; f=d['fold'][0]
        tr,va=dev[f!=0],dev[f==0]
        args=training(d,tr,va,True,True)
        assert set(args[3])<=set(tr)
    print('Self-check passed: ridge equivalence, paired penalty algebra, source exclusion.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['snapshot','audit','domain','consistency','check'])
    ap.add_argument('--strict',action='store_true'); ap.add_argument('--partitions',default='0,1,2'); args=ap.parse_args()
    with threadpool_limits(limits=2):
        if args.action=='snapshot': snapshot()
        elif args.action=='audit': print((_repo_path(HERE, 'audit.json')).read_text())
        elif args.action=='check': self_check()
        else: run(args.action,args.strict,tuple(map(int,args.partitions.split(','))))
