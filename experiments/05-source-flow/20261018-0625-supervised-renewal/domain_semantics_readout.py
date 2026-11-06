"""Compare original versus SEM-adapted continuous representations on hardness."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME']=str(_repo_path(HERE, 'tmp/huggingface'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import json,hashlib,time,gc
import numpy as np
import torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold
from threadpoolctl import threadpool_limits

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];fold=d['fold'];ids=d['ids'][dev];y=d['y'][dev];out=_repo_path(HERE, 'domain-semantics/readout');out.mkdir(exist_ok=True);F={};provenance={}
    for arm in ['original','H300']:
        path=_repo_path(HERE, f'domain-semantics/{arm}-features.npz');z=np.load(_repo_io(path));assert np.array_equal(z['ids'],d['ids']);F[arm]=dict(semantic=z['semantic'][dev],flow=z['flow'][dev]);provenance[arm]=dict(feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),recipe=json.loads((_repo_path(HERE, f'domain-semantics/{arm}-recipe.json')).read_text()))
    fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(fp));pos={str(v):i for i,v in enumerate(z['ids'])};P=z['X'][[pos[str(v)] for v in ids]].astype(float)
    spec=dict(primary='H300continuousfeaturesplusproductionphysical104; complete083+.384*(newP104head -matchedP104control),existingfailedmaskfallback.',
        controls='IdenticaloriginalCPSAMdescriptor; image-onlyTabPFN anduncompressedRidgecontrols;P104+flow-onlyablation. PCA32fitinsideeachtrainingfold,46continuousflow/photometryfeaturesuncompressed. Ridgealphas10/100/1000/10000 chosenbyinner3foldwithrefittedscaler.',
        model='localTabPFN3.5n8,seeds0/100/200+fold,CPU',scope='Originalimage-onlybranchusesfrozenexternalweights andtraining-onlyhead/PCA. H300andphysical104carryhistoricalglobalrepresentationfitting:conditionalcomparisons,notend-to-endvalidation.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provenance=provenance)
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    pred={k:np.zeros(400) for k in ['physical']+[arm+'_'+kind for arm in ['original','H300'] for kind in ['image','image_ridge','augmented','flow_only']]}
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);ps=[]
        for seed in [0,100,200]:
            c=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{k}-s{seed}.npz'));assert np.array_equal(c['train'],tr) and np.array_equal(c['query'],q);ps.append(c['prediction'])
        pred['physical'][q]=np.mean(ps,0)
        for arm in ['original','H300']:
            normalizer=make_pipeline(StandardScaler(),PCA(32,svd_solver='full'));normalizer.fit(F[arm]['semantic'][tr]);E=np.c_[normalizer.transform(F[arm]['semantic']),F[arm]['flow']];joblib.dump(normalizer,_repo_io(out/f'{arm}-pca-f{k}.joblib'))
            raw=np.c_[F[arm]['semantic'],F[arm]['flow']];ridge=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[10.,100.,1000.,10000.]},cv=KFold(3,shuffle=True,random_state=2026101981+k),scoring='neg_mean_squared_error',n_jobs=1)
            ridge.fit(raw[tr],y[tr]);pred[arm+'_image_ridge'][q]=ridge.predict(raw[q]);joblib.dump(ridge,_repo_io(out/f'{arm}-ridge-f{k}.joblib'))
            for kind,A in [('image',E),('augmented',np.c_[P,E]),('flow_only',np.c_[P,F[arm]['flow']])]:
                name=arm+'_'+kind;ps=[]
                for seed in [0,100,200]:
                    path=out/f'{name}-f{k}-s{seed}.npz'
                    if path.exists():v=np.load(_repo_io(path))['prediction']
                    else:
                        start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=8);m.fit(A[tr],y[tr]);v=m.predict(A[q]);np.savez(_repo_io(path),train=tr,query=q,prediction=v);del m;gc.collect();print(name,k,seed,time.monotonic()-start,flush=True)
                    ps.append(v)
                pred[name][q]=np.mean(ps,0)
    p21=np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz'))['C73_cnn'][0];eligible=~d['failed']
    for arm in ['original','H300']:
        pred[arm+'_system']=d['base083']+.384*eligible*(pred[arm+'_augmented']-pred['physical'])
        pred[arm+'_flow_system']=d['base083']+.384*eligible*(pred[arm+'_flow_only']-pred['physical'])
        pred[arm+'_image_blend']=.75*d['base083']+.25*pred[arm+'_image']
        pred[arm+'_image_replacement']=d['base083']+np.where(d['failed'],1.,.16)*(pred[arm+'_image']-p21)
        pred[arm+'_ridge_replacement']=d['base083']+np.where(d['failed'],1.,.16)*(pred[arm+'_image_ridge']-p21)
    pred['base083']=d['base083'];r={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
