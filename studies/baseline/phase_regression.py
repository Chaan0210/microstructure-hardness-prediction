"""Compact image physics proxies and nested smooth regression; development-only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,time,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.linalg import solve
from sklearn.base import BaseEstimator,RegressorMixin,clone
from sklearn.preprocessing import StandardScaler,MinMaxScaler,SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import joblib

def write(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
class BalancedKernel(RegressorMixin,BaseEstimator):
    def __init__(self,blocks=None,bandwidth=1.,alpha=.1):self.blocks=blocks;self.bandwidth=bandwidth;self.alpha=alpha
    def fit(self,X,y):
        self.bound_=MinMaxScaler(clip=True).fit(X);self.scale_=StandardScaler().fit(self.bound_.transform(X))
        blocks=np.array(self.blocks if self.blocks is not None else [0]*X.shape[1]);self.weights_=np.array([1/np.sqrt(np.sum(blocks==b)) for b in blocks])
        self.X_=self.scale_.transform(self.bound_.transform(X))*self.weights_
        dist=pdist(self.X_,'sqeuclidean');self.gamma_=1/max(float(np.median(dist[dist>1e-12]))*self.bandwidth**2,1e-12)
        self.ym_=float(np.mean(y));self.ys_=max(float(np.std(y)),1e-8);K=np.exp(-self.gamma_*squareform(dist));K.flat[::len(y)+1]+=self.alpha
        self.coef_=solve(K,(y-self.ym_)/self.ys_,assume_a='pos');self.n_features_in_=X.shape[1];return self
    def predict(self,X):
        z=self.scale_.transform(self.bound_.transform(X))*self.weights_;return np.exp(-self.gamma_*cdist(z,self.X_,'sqeuclidean'))@self.coef_*self.ys_+self.ym_
def compact_existing(G,names):
    def g(n):return G[:,names.index(n)]
    cols=[.5*(g('grain_s1_interior_hall_petch')+g('grain_s2_interior_hall_petch')),g('grain_s1_interior_diam_std')/np.maximum(g('grain_s1_interior_diam_mean'),1.),g('phase_s1_0_fraction'),g('phase_s1_2_fraction'),g('pore_s1_t0.15_fraction'),g('pore_s1_t0.3_fraction'),g('pore_s1_t0.15_diam_mean'),g('grain_s1_all_aspect_mean'),g('grain_s1_alignment'),g('grain_s1_within_std_mean'),g('scale_robustness_diam'),g('grain_s1_boundary_fraction')]
    names=['inverse_sqrt_grain_diameter','grain_size_cv','phase_dark','phase_light','pore_fraction15','pore_fraction30','pore_diameter','grain_aspect','grain_alignment','within_grain_contrast','grain_scale_sensitivity','boundary_fraction']
    blocks=['size','size','phase','phase','phase','phase','phase','shape','shape','intensity','size','shape']
    return np.column_stack(cols),names,blocks

def extract_one(ident):
    from latent_features import extract_latent
    path=_repo_path(ROOT, 'data/train', f'{ident}.png')
    with Image.open(_repo_io(path)) as im:a=np.asarray(im,dtype=np.uint8).copy()
    v,n=extract_latent(a)
    return v,n,[float(a.mean()/255),float(a.std()/255)],sha(path)
def get_base():
    with (_repo_path(ROOT, 'artifacts/run-005/development_oof.csv')).open() as f:return {r['ID']:r for r in csv.DictReader(f)}
def main():
    out=_repo_path(ROOT, 'artifacts/run-014');out.mkdir(exist_ok=False);start=time.perf_counter()
    base=get_base();ids=sorted(base);y=np.array([float(base[s]['hardness']) for s in ids]);baseline=np.array([float(base[s]['stable_hybrid']) for s in ids]);sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());assert set(ids)==set(sp['development_folds'])
    fid=np.array([sp['development_folds'][s] for s in ids]);groups=np.array([sp['group_by_id'][s] for s in ids])
    plan={'hypothesis':'Add boundary/pore-masked intensity mixture to existing compact panel, comparing identical nested regressors' ,'outer_folds':'same development 5 folds','inner_folds':3,'inner_seed':20261005,'kernel_bandwidth_multipliers':[.25,1.,4.],'kernel_alpha':[.01,.1,1.],'kernel_formula':'gamma=1/(median train squared distance * bandwidth^2); train-standardized block weights1/sqrt(block columns), standardized target','other_models':['fixed additive splines knots4 degree3 Ridgealpha10','fixed shallow HGB leaves7 minleaf20 l2=10 iter200 lr.05'],'recipes':['alone','baseline75_candidate25','baseline50_grain25_candidate25','baseline50_cnn80_25_candidate25'],'no_test_holdout':True,'provenance':'Parent-transcribed official PDF: Hall-Petch, phase mixture, porosity; direct PDF retrieval403. No effect sign constrained.','prior_choice_bias':'Existing baseline/CNN/grain and feature hypotheses developed on same400 rows; inner CV controls current kernel hyperparameters, not all historical choices.'}
    write(out/'plan.json',plan)
    from phase_features import extract_phase,DESCRIPTORS
    def phase_one(ident):
        path=_repo_path(ROOT, 'data/train', f'{ident}.png')
        with Image.open(_repo_io(path)) as im:img=np.asarray(im,dtype=np.uint8).copy()
        v,n=extract_phase(img)
        return v,n,sha(path)
    rows=joblib.Parallel(n_jobs=2)(joblib.delayed(phase_one)(s) for s in ids)
    raw=np.stack([z[0] for z in rows]);rawnames=rows[0][1];assert all(z[1]==rawnames for z in rows)
    summaries=[];newnames=[]
    for name in DESCRIPTORS:
        ix=[j for j,n in enumerate(rawnames) if n.endswith('_'+name)];assert len(ix)==3,(name,ix)
        summaries.extend([np.median(raw[:,ix],axis=1),np.ptp(raw[:,ix],axis=1)]);newnames.extend(['interior_median_'+name,'interior_sensitivity_'+name])
    cache=np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'));ci={s:i for i,s in enumerate(cache['ids'])}
    prior=json.loads((_repo_path(ROOT, 'artifacts/run-012/report.json')).read_text());X=np.column_stack([cache['X'][[ci[s] for s in ids]],*summaries]);names=prior['features']+newnames;blocks=prior['blocks']+['separated_phase']*len(newnames)
    assert np.isfinite(X).all() and len(names)==X.shape[1]
    np.savez_compressed(_repo_io(out/'development_features.npz'),X=X,y=y,ids=np.array(ids),groups=groups)
    write(out/'input-provenance.json',{'dev_image_sha256':dict(zip(ids,[z[2] for z in rows])),'split_manifest_sha256':sha(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')),'base_feature_cache_sha256':sha(_repo_path(ROOT, 'artifacts/run-012/development_features.npz')),'baseline_oof_sha256':sha(_repo_path(ROOT, 'artifacts/run-005/development_oof.csv')),'source_sha256':{n:sha(_repo_path(ROOT, n)) for n in ['phase_features.py','phase_regression.py']}})
    feature_seconds=time.perf_counter()-start
    fixed={'spline':TransformedTargetRegressor(regressor=make_pipeline(MinMaxScaler(clip=True),StandardScaler(),SplineTransformer(n_knots=4,degree=3,include_bias=False),StandardScaler(),Ridge(alpha=10)),transformer=StandardScaler()),'shallow_hgb':TransformedTargetRegressor(regressor=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261005),transformer=StandardScaler())}
    preds={n:np.zeros(len(y)) for n in [*fixed,'nested_kernel']};choices=[]
    for k in range(5):
        tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k)
        for name,model in fixed.items():preds[name][va]=clone(model).fit(X[tr],y[tr]).predict(X[va])
        inner=list(GroupKFold(3,shuffle=True,random_state=20261005+k).split(X[tr],y[tr],groups[tr]));scores=[]
        for width in plan['kernel_bandwidth_multipliers']:
            for alpha in plan['kernel_alpha']:
                pv=np.zeros(len(tr))
                for ti,vi in inner:
                    m=BalancedKernel(blocks,width,alpha).fit(X[tr[ti]],y[tr[ti]]);pv[vi]=m.predict(X[tr[vi]])
                scores.append({'bandwidth':width,'alpha':alpha,'inner_rmse':rmse(y[tr],pv)})
        best=min(scores,key=lambda x:x['inner_rmse']);model=BalancedKernel(blocks,best['bandwidth'],best['alpha']).fit(X[tr],y[tr]);preds['nested_kernel'][va]=model.predict(X[va])
        choices.append({'fold':k,'selected':best,'inner_scores':scores,'outer_rmse':rmse(y[va],preds['nested_kernel'][va])});print(json.dumps(choices[-1]),flush=True)
    with (_repo_path(ROOT, 'artifacts/run-006/development_oof.csv')).open() as f:grainrows={r['ID']:r for r in csv.DictReader(f)}
    with (_repo_path(ROOT, 'artifacts/run-009/development_oof.csv')).open() as f:cnnrows={r['ID']:r for r in csv.DictReader(f)}
    grain=np.array([float(grainrows[s]['grain_ridge100']) for s in ids]);cnn=np.array([float(cnnrows[s]['cnn']) for s in ids]);allpred={'baseline':baseline}
    for name,p in preds.items():
        allpred[name]=p;allpred[name+'_blend25']=.75*baseline+.25*p;allpred[name+'_grainblend']=.5*baseline+.25*grain+.25*p;allpred[name+'_cnnblend']=.5*baseline+.25*cnn+.25*p
    scores={n:{'oof_rmse':rmse(y,p),'fold_rmse':[rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in allpred.items()}
    with (out/'development_oof.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*allpred]);w.writerows([s,y[i],*[p[i] for p in allpred.values()]] for i,s in enumerate(ids))
    result={**plan,'scores':scores,'inner_choices':choices,'features':names,'blocks':blocks,'feature_seconds':feature_seconds,'elapsed_seconds':time.perf_counter()-start};write(out/'report.json',result);print(json.dumps({'scores':scores,'seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
