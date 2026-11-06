"""One bounded hypothesis round, dev-only selection and two train-only CV repeats."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[n]='2'
import sys,time,json,csv,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(_repo_path(ROOT, 'src')))
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from microstructure.pipeline import read_image,rmse,write_json,digest
from grain_features import extract_grains

def one(ident):return extract_grains(read_image(_repo_path(ROOT, 'data/train', f'{ident}.png')))[0]
def ridge(alpha):return make_pipeline(MinMaxScaler(clip=True),StandardScaler(),Ridge(alpha=alpha))
def score(y,p,folds):return dict(oof_rmse=rmse(y,p),fold_rmse=[rmse(y[v],p[v]) for _,v in folds])
def cv(model,X,y,folds):
    p=np.zeros(len(y))
    for tr,va in folds:p[va]=clone(model).fit(X[tr],y[tr]).predict(X[va])
    return p

def main():
    out=_repo_path(ROOT, 'artifacts/run-006');out.mkdir(exist_ok=False)
    sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())
    cached=np.load(_repo_path(ROOT, 'artifacts/run-003/train_features.npz'))
    dev=np.array([i for i,s in enumerate(cached['ids']) if s in sp['development_folds']])
    ids=cached['ids'][dev];X=cached['X'][dev];y=cached['y'][dev]
    groups=np.array([sp['group_by_id'][s] for s in ids]);fid=np.array([sp['development_folds'][s] for s in ids])
    folds=[(np.flatnonzero(fid!=k),np.flatnonzero(fid==k)) for k in range(5)]
    plan=dict(seed=20260930,repeated_seeds=[20261001,20261002],development_rows=len(dev),holdout_used=False,test_used=False,
        hypothesis='Dark polygonal ridges approximate grain boundaries; watershed size/shape, compact dark spots, phase and tensor proxies may explain residual hardness.',
        official_generator_details='Unavailable through public web tool; no exact formula assumed.',
        candidates=['grain_ridge100','augmented_ridge1000','baseline75_grain25','baseline75_augmented25'],
        advancement_rule='Fixed-fold RMSE improves >=0.15 and both predeclared repeated 5-fold CVs improve >=0.10; otherwise retain run-005.',
        existing_model_sha256=digest(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib')),existing_csv_sha256=digest(_repo_path(ROOT, 'artifacts/submission-candidate-002.csv')))
    write_json(out/'plan.json',plan);start=time.perf_counter()
    chunks=[]
    for first in range(0,len(ids),40):
        chunks.extend(joblib.Parallel(n_jobs=2)(joblib.delayed(one)(s) for s in ids[first:first+40]));print(f'Grain features {min(first+40,len(ids))}/{len(ids)}',flush=True)
    G=np.stack(chunks);names=extract_grains(read_image(_repo_path(ROOT, 'data/train', f'{ids[0]}.png')))[1]
    np.savez_compressed(_repo_io(out/'development_features.npz'),X=G,ids=ids)
    feature_seconds=time.perf_counter()-start
    with (_repo_path(ROOT, 'artifacts/run-005/development_oof.csv')).open() as f:prior={r['ID']:float(r['stable_hybrid']) for r in csv.DictReader(f)}
    base=np.array([prior[s] for s in ids]);Z=np.hstack([X,G])
    pgrain=cv(ridge(100),G,y,folds);paug=cv(ridge(1000),Z,y,folds)
    preds={'baseline':base,'grain_ridge100':pgrain,'augmented_ridge1000':paug,'baseline75_grain25':.75*base+.25*pgrain,'baseline75_augmented25':.75*base+.25*paug}
    scores={k:score(y,p,folds) for k,p in preds.items()};winner=min(plan['candidates'],key=lambda k:scores[k]['oof_rmse'])
    print(json.dumps({'fixed_cv':scores,'best_new':winner}),flush=True)
    # Same fixed best candidate only; no selecting among repeats or further hyperparameters.
    bundle=joblib.load(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib'));repeats=[]
    for seed in plan['repeated_seeds']:
        ff=list(GroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,groups))
        pb=sum(w*cv(m,X,y,ff) for n,m in bundle['models'].items() for w in [bundle['weights'][n]])
        pg=cv(ridge(100) if 'grain' in winner else ridge(1000),G if 'grain' in winner else Z,y,ff)
        pn=.75*pb+.25*pg if winner.startswith('baseline75') else pg
        rec=dict(seed=seed,baseline=score(y,pb,ff),candidate=score(y,pn,ff),improvement=rmse(y,pb)-rmse(y,pn))
        repeats.append(rec);np.savez_compressed(_repo_io(out/f'repeat-{seed}.npz'),ids=ids,baseline=pb,candidate=pn)
        print(json.dumps(rec),flush=True)
    gain=scores['baseline']['oof_rmse']-scores[winner]['oof_rmse'];advance=gain>=.15 and all(r['improvement']>=.1 for r in repeats)
    rng=np.random.default_rng(20261001);delta=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y));delta.append(rmse(y[ix],base[ix])-rmse(y[ix],preds[winner][ix]))
    corr=[]
    for j,n in enumerate(names):
        if np.std(G[:,j])>1e-10:corr.append(dict(feature=n,residual_spearman=float(spearmanr(G[:,j],y-base).statistic)))
    corr.sort(key=lambda d:abs(d['residual_spearman']),reverse=True)
    q=np.quantile(y,[0,.25,.5,.75,1]);strata=[]
    for k in range(4):
        mask=(y>=q[k])&(y<=q[k+1] if k==3 else y<q[k+1]);strata.append(dict(target_range=q[k:k+2].tolist(),n=int(mask.sum()),baseline_rmse=rmse(y[mask],base[mask]),candidate_rmse=rmse(y[mask],preds[winner][mask])))
    report=dict(**plan,feature_names=names,n_new_features=G.shape[1],feature_seconds=feature_seconds,fixed_cv=scores,best_new=winner,improvement=gain,repeats=repeats,advance=advance,
        selected_run='run-006' if advance else 'run-005',paired_bootstrap_improvement_95=np.percentile(delta,[2.5,97.5]).tolist(),
        uncertainty_note='Conditional on reused folds and chosen candidate; not an unbiased post-selection confidence interval. Repeats share the same 400 rows.',
        residual_correlations=corr[:15],target_strata=strata,elapsed_seconds=time.perf_counter()-start,
        source_sha256={p:digest(_repo_path(ROOT, p)) for p in ['grain_features.py','research_grains.py']})
    write_json(out/'report.json',report)
    with (out/'development_oof.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*preds]);w.writerows([s,y[i],*[p[i] for p in preds.values()]] for i,s in enumerate(ids))
    assert digest(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib'))==plan['existing_model_sha256']
    assert digest(_repo_path(ROOT, 'artifacts/submission-candidate-002.csv'))==plan['existing_csv_sha256']
    print(json.dumps({'advance':advance,'improvement':gain,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
