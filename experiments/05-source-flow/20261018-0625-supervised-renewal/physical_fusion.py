"""Direct joint prediction using the incumbent's actual physical feature bank."""
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
import argparse,hashlib,json
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
import dense_fusion_v3 as image_model
import grain_attention_v2 as ga

def context(bags,weights,G,y,tr):
    m=GridSearchCV(make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge()),
        {'ridge__alpha':[10.,100.,1000.]},cv=KFold(3,shuffle=True,random_state=2026101853),scoring='neg_mean_squared_error',n_jobs=1)
    m.fit(G[tr],y[tr]);im,sc,rr=[x[1] for x in m.best_estimator_.steps];ym=float(y[tr].mean());ys=float(y[tr].std())
    return dict(global_median=im.statistics_,global_mean=sc.mean_,global_std=sc.scale_,ym=ym,ys=ys,base_w=rr.coef_/ys,base_b=(rr.intercept_-ym)/ys,alpha=m.best_params_['ridge__alpha'])

class Data(image_model.ImageData):
    def __init__(self,bags,weights,G,ctx):
        super().__init__(bags,weights,np.where(np.isfinite(G),G,ctx['global_median']),ctx)

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    bags,weights,_,d,_=ga.load();path=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(path))
    lookup={str(v):i for i,v in enumerate(z['ids'])};G=z['X'][[lookup[str(v)] for v in d['ids']]].astype(float);G=np.where(np.isfinite(G),G,np.nan)
    image_model.ImageData.context_only=a.kind=='context'
    if a.kind!='context':
        image_model.RAW=image_model.prepare_raw();image_model.NECK=np.load(_repo_path(HERE, 'cpsam-neck-c4/maps.npy'),mmap_mode='r')
    ga.context=context;ga.Data=Data;ga.Model=image_model.Fusion;ga.predict=image_model.predict
    for f in a.folds:
        tag=f'physical-{a.kind}-f{f}-s{a.seed}';out=_repo_path(HERE, 'runs', tag);out.mkdir(exist_ok=True)
        if (out/'done.json').exists():continue
        tr=d['dev'][d['fold']!=f];q=d['dev'][d['fold']==f];A,B=train_test_split(tr,test_size=.2,random_state=2026101847+f)
        ga.save(_repo_io(out/'recipe.json'),vars(a)|dict(fold=f,A=A.tolist(),B=B.tolist(),train=tr.tolist(),query=q.tolist(),feature_keys=z['keys'].tolist(),
            feature_file=str(path.relative_to(ROOT)),feature_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope='Conditional on exactly the historical physical bank used by one incumbent member. New preprocessing/alpha/epoch selection and predictor fit are train-only. Not an end-to-end auxiliary refit.',
            objective='Jointly optimized original-hardness prediction; Ridge initialization is trainable, no fixed teacher/residual labels.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        r=ga.fit(bags,weights,G,d['y'],A,B,a.kind,a.seed,80,out/'selector',select=True)
        s=ga.fit(bags,weights,G,d['y'],tr,q,a.kind,a.seed,r['selected_epoch'],out/'refit')
        pred=np.load(_repo_io(out/'refit/predictions.npz'))['prediction'];base=d['base083'][d['fold']==f];eligible=~d['failed'][d['fold']==f]
        result=dict(fold=f,kind=a.kind,selector=r,refit=s,base083_rmse=ga.rmse(base,d['y'][q]),quarter_blend_rmse=ga.rmse(base+.25*eligible*(pred-base),d['y'][q]))
        ga.save(_repo_io(out/'done.json'),result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['context','fusion','raw'],default='context');p.add_argument('--folds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--seed',type=int,default=2026101891)
    with threadpool_limits(limits=2):main(p.parse_args())
