"""Test whether explicit phase moments are useful to linear/log-linear predictors.

Fixed regularization and fixed incumbent block masses; no fitting of correction
weights or in-sample residuals. The combined predictor is an explicit addition,
not a claim that a one-bank control equals the incumbent's entire linear block.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import json,hashlib,time
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from phase_readout import augment

def main():
    d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];fold=d['fold'];y=d['y'][dev];out=_repo_path(HERE, 'phase-linear-readout');out.mkdir(exist_ok=True);assert (y>0).all()
    file=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(file));pos={str(v):i for i,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in ids]].astype(float)
    spec=dict(hypothesis='Directphase/geometrymomentsmaybenefitregularizedlinearandloglinearpredictorsdifferentlyfromTabPFN; testbothblockschangingtogether.',
        predictor='EqualmeanofRidge(alpha100)ony andexp(Ridge(alpha100)onlogy);source-onlymedian/StandardScaler. Alpha100fixed,noinnerselection.',
        formula='083 +.256*(augmentedlinear-controlLinear) +.384*(augmentedTab-controlTab),existingfailedmaskfallback. Coefficientsfixedbeforethisreadout.',
        interpretation='Definedcompletecombinedsystem;one-bankcontrolsarenotrepresentedasthewholeincumbent. ConditionallegacyP104features;newphasefitstrictlysource-local.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));pred={k:np.zeros(400) for k in ['control','phase','tab_system']}
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);tag=f'phase-boundary-nll-f{k}-s89';feature=_repo_path(HERE, f'runs/{tag}/frozen-features.npz');tab=_repo_path(HERE, f'phase-production-{tag}/predictions.npz');start=time.monotonic()
        while not (_repo_path(HERE, f'phase-production-{tag}/results.json')).exists():
            if time.monotonic()-start>7200:raise RuntimeError('Phasepredictionstage didnotcomplete.')
            time.sleep(5)
        p=np.load(_repo_io(feature));assert np.array_equal(p['train'],dev[tr]) and np.array_equal(p['query'],dev[q]);t=np.load(_repo_io(tab));assert np.array_equal(t['query'],dev[q]);pred['tab_system'][q]=t['system']
        for name,A,Q in [('control',X[tr],X[q]),('phase',np.c_[X[tr],augment(p['train_features'])],np.c_[X[q],augment(p['query_features'])])]:
            ps=[]
            for kind in ['linear','log']:
                model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.));model.fit(A,y[tr] if kind=='linear' else np.log(y[tr]));v=model.predict(Q);ps.append(v if kind=='linear' else np.exp(v));joblib.dump(model,_repo_io(out/f'{name}-{kind}-f{k}.joblib'))
            pred[name][q]=np.mean(ps,0)
        elig=~d['failed'][q];old=d['base083'][q];new=pred['tab_system'][q]+.256*elig*(pred['phase'][q]-pred['control'][q]);print('FOLD',k,json.dumps(dict(base083=float(np.mean((old-y[q])**2)**.5),tab_only=float(np.mean((pred['tab_system'][q]-y[q])**2)**.5),tab_and_linear=float(np.mean((new-y[q])**2)**.5))),flush=True)
    delta=.256*(~d['failed'])*(pred['phase']-pred['control']);pred['linear_system']=d['base083']+delta;pred['combined_system']=pred['tab_system']+delta;pred['base083']=d['base083'];r={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
