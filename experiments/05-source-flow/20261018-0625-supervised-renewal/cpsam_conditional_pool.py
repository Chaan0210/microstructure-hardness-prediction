"""Frozen semantic geometry conditioned on co-located original brightness.

The previous frozen descriptor lost associations between grayscale phase and
semantic geometry. Soft per-image brightness bins expose those associations;
no new neural representation is fit. PCA is fit within each training fold.
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
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME']=str(_repo_path(HERE, 'tmp/huggingface'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import argparse,json,hashlib,time,gc
import numpy as np
import torch,joblib
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

def features():
    out=_repo_path(HERE, 'cpsam-conditional-pool');out.mkdir(exist_ok=True);dest=out/'features.npz'
    if dest.exists():return
    d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');maps=np.load(_repo_path(HERE, 'cpsam-neck-c4/maps.npy'),mmap_mode='r');assert np.load(_repo_path(HERE, 'cpsam-neck-c4/done.npy')).all();assert maps.shape==(500,2,256,32,32)
    glob=[];cond=[];fractions=[];start=time.monotonic()
    for j in range(500):
        a=px[j,0].astype(np.float32)/255;lo,hi=np.percentile(a,[10,90]);scale=max(float((hi-lo)/2.563),.01);mean=float(a.mean());rows=[];cr=[];fs=[]
        for view in [0,1]:
            im=np.rot90(a,view);local=im.reshape(32,8,32,8).mean((1,3));z=np.clip((local-mean)/scale,-8,8).reshape(-1)
            w=softmax(-.5*((z[None]-np.array([-1.5,-.5,.5,1.5])[:,None])/.75)**2,axis=0).astype(np.float32);mass=w.mean(1);w=w/w.sum(1,keepdims=True)
            f=np.array(maps[j,view],dtype=np.float32).reshape(256,1024);mu=f.mean(1);sd=np.sqrt(np.maximum(f.var(1),1e-6));cm=w@f.T;cs=np.sqrt(np.maximum(w@(f*f).T-cm*cm,1e-6))
            delta=np.c_[np.clip((cm-mu)/np.maximum(sd,.01),-5,5),np.clip(cs/np.maximum(sd,.01)-1,-5,5)]
            g=np.r_[mu,sd];rows.append(g);cr.append(np.r_[g,delta.reshape(-1)]);fs.append(mass)
        glob.append(np.mean(rows,0));cond.append(np.mean(cr,0));fractions.append(np.mean(fs,0))
    np.savez(_repo_io(dest),ids=d['ids'],global_features=np.array(glob),conditional_features=np.array(cond),fractions=np.array(fractions));r=dict(seconds=time.monotonic()-start,global_shape=list(np.shape(glob)),conditional_shape=list(np.shape(cond)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_provenance=json.loads((_repo_path(HERE, 'cpsam-neck-c4/complete.json')).read_text()),scope='Frozen eligibleCPSAM original/rot90maps; no label/representationfit;softbinsfitonlywithineachoriginalimage.');(out/'extraction.json').write_text(json.dumps(r,indent=2));print('EXTRACT',json.dumps(r),flush=True)

def readout():
    out=_repo_path(HERE, 'cpsam-conditional-pool');z=np.load(_repo_io(out/'features.npz'));d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];y=d['y'][dev];fold=d['fold'];assert np.array_equal(z['ids'],d['ids'])
    fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');p=np.load(_repo_io(fp));pos={str(v):i for i,v in enumerate(p['ids'])};X=p['X'][[pos[str(v)] for v in ids]].astype(float)
    recipe=dict(primary='Physical104+32train-fittedPCAconditionalcomponents+4softbinfractions',control='Sameviewsphysical104+32train-fittedPCAglobalcomponents+same4fractions;physical104alonecachedmatchedn8x3head.',
        readout='localTabPFN3.5n8, seeds0/100/200+fold,CPU;PCAfullSVDwithtraining-onlyStandardScaler',formula='Complete083+.384*(augmentedphysical104prediction -matchedphysical104prediction),existingfailedmaskfallback.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features_sha256=hashlib.sha256((out/'features.npz').read_bytes()).hexdigest(),physical_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),scope='Newpreprocessingtrain-only,fixedexternalimagefeatures; inheritedphysical104 remainsconditional.')
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2));sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness;pred={k:np.zeros(400) for k in ['physical','global','conditional']}
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);pp=[]
        for seed in [0,100,200]:
            c=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{k}-s{seed}.npz'));assert np.array_equal(c['train'],tr) and np.array_equal(c['query'],q);pp.append(c['prediction'])
        pred['physical'][q]=np.mean(pp,0)
        for name in ['global','conditional']:
            F=z[name+'_features'][dev];sc=make_pipeline(StandardScaler(),PCA(32,svd_solver='full'));sc.fit(F[tr]);V=sc.transform(F);joblib.dump(sc,_repo_io(out/f'{name}-pca-f{k}.joblib'));A=np.c_[X,V,z['fractions'][dev]];ps=[]
            for seed in [0,100,200]:
                path=out/f'{name}-f{k}-s{seed}.npz'
                if path.exists():v=np.load(_repo_io(path))['prediction']
                else:
                    start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=8);m.fit(A[tr],y[tr]);v=m.predict(A[q]);np.savez(_repo_io(path),train=tr,query=q,prediction=v);del m;gc.collect();print(name,k,seed,time.monotonic()-start,flush=True)
                ps.append(v)
            pred[name][q]=np.mean(ps,0)
    for name in ['global','conditional']:pred[name+'_system']=d['base083']+.384*(~d['failed'])*(pred[name]-pred['physical'])
    pred['base083']=d['base083'];scores={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(scores,indent=2));print('DONE',json.dumps(scores),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['features','readout']);a=ap.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(2):features() if a.action=='features' else readout()
