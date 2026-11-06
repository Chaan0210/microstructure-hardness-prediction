"""Paired physical-only/structure-augmented readouts with train-local PCA."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import json,sys,time,gc,socket,importlib.metadata
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def blocked(*a,**kw):raise RuntimeError('Network disabled in local experiment')
socket.create_connection=blocked;socket.getaddrinfo=blocked
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
WEIGHT=_repo_path(e.ROOT, 'experiments/tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors')
ARMS=['spatial','permuted','random']
def physical():
    z=np.load(_repo_io(e.OLD/'inputs.npz'));d={k:z[k] for k in z.files};base=_repo_path(e.ROOT, 'experiments/push-1300');record=np.load(_repo_io(base/'abl/dev-h9phys-full-044-u1dmavg.npz'));path=(base/str(record['feature_file'])).resolve();assert path.is_relative_to(base);assert e.sha(path)==str(record['feature_sha256'])
    f=np.load(_repo_io(path),allow_pickle=True);pos={str(v):i for i,v in enumerate(f['ids'])};d['X']=f['X'][[pos[str(i)] for i in d['ids']]].astype(float);assert d['X'].shape==(500,100);d['feature_source']=str(path.relative_to(e.ROOT));d['feature_sha256']=e.sha(path);return d
def predict_tab(X,y,Q,seed,path,train,query):
    start=time.monotonic();m=TabPFNRegressor.create_default_for_version(ModelVersion('v3.5'),model_path=str(WEIGHT),device='mps',inference_precision=torch.float32,memory_saving_mode=True,random_state=seed,n_preprocessing_jobs=1,show_progress_bar=False)
    m.fit(X,y.copy());p=np.asarray(m.predict(Q),float);assert np.isfinite(p).all();torch.mps.synchronize();mem=e.memory();record=dict(seed=seed,seconds=time.monotonic()-start,train=train.tolist(),query=query.tolist(),train_X_sha256=e.ah(X),train_y_sha256=e.ah(y),query_X_sha256=e.ah(Q),inference_precision='float32',memory_saving_mode=True,n_estimators=str(m.n_estimators),peak_memory=mem)
    e.arrays(str((path/'prediction.npz').relative_to(e.HERE)),prediction=p,train=train,query=query);e.save(_repo_io(str((path/'contract.json').relative_to(e.HERE))),record);del m;gc.collect();torch.mps.empty_cache();return p
def main():
    completed=json.loads((e.HERE/'encoders-complete.json').read_text());assert completed['complete'];assert e.sha(e.HERE/'encoder.py')==completed['source_sha256'];d=physical();graphs=json.loads((e.OLD/'graphs.json').read_text());start=time.monotonic();deadline=start+7200-completed['wall_seconds'];assert deadline>start
    e.save(_repo_io('readout-code-lock.json'),dict(source_sha256=e.sha(__file__),encoder_source_sha256=e.sha(e.HERE/'encoder.py'),plan_sha256=e.sha(e.HERE/'PLAN.md'),foundation_weight_sha256=e.sha(WEIGHT),feature_source=d['feature_source'],feature_sha256=d['feature_sha256'],environment=dict(python=sys.version,packages={x.metadata['Name']:x.version for x in importlib.metadata.distributions()}),config=dict(PCA=16,tab_seeds='100*j+fold, j=0,1,2',dtype='float32',memory_saving_mode=True)))
    allrecords=[]
    for k,g in enumerate(graphs):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);assert set(tr).isdisjoint(q);X=d['X'][tr];Q=d['X'][q];y=d['y'][tr].copy();folder=e.HERE/f'readouts/fold{k}-{stage}';folder.mkdir(parents=True,exist_ok=False);base=[];out={a:[] for a in ARMS}
            e.arrays(str((folder/'physical-context.npz').relative_to(e.HERE)),X=X,y=y,Q=Q,train=tr,query=q)
            for j in range(3):
                if time.monotonic()>deadline:raise RuntimeError('Frozen120-minute compute budget reached')
                path=folder/f'base-tab{j}';path.mkdir();base.append(predict_tab(X,y,Q,100*j+k,path,tr,q))
            for arm in ARMS:
                for repeat in range(2):
                    src=e.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('spatial' if arm=='random' else arm))
                    # Paths are resolved explicitly; no historical output destination is imported.
                    f=np.load(_repo_io(src/('random-features.npz' if arm=='random' else 'features.npz')));assert np.array_equal(f['train'],tr) and np.array_equal(f['query'],q)
                    sc=StandardScaler().fit(f['train_features']);pca=PCA(n_components=16,svd_solver='full').fit(sc.transform(f['train_features']));zt=pca.transform(sc.transform(f['train_features']));zq=pca.transform(sc.transform(f['query_features']));xt=np.c_[X,zt];xq=np.c_[Q,zq];path=folder/f'{arm}-seed{repeat}';path.mkdir()
                    e.arrays(str((path/'context.npz').relative_to(e.HERE)),X=xt,y=y,Q=xq,train=tr,query=q,scaler_mean=sc.mean_,scaler_scale=sc.scale_,pca_mean=pca.mean_,pca_components=pca.components_,pca_explained_variance=pca.explained_variance_,pca_explained_variance_ratio=pca.explained_variance_ratio_)
                    pp=[]
                    for j in range(3):
                        if time.monotonic()>deadline:raise RuntimeError('Frozen120-minute compute budget reached')
                        sub=path/f'tab{j}';sub.mkdir();pp.append(predict_tab(xt,y,xq,100*j+k,sub,tr,q))
                    out[arm].append(np.mean(pp,0));print('READOUT',k,stage,arm,repeat,'seconds',round(time.monotonic()-start,1),flush=True)
            e.arrays(str((folder/'predictions.npz').relative_to(e.HERE)),train=tr,query=q,base=np.mean(base,0),**{a:np.array(v) for a,v in out.items()});allrecords.append(str(folder.relative_to(e.HERE)))
    e.save(_repo_io('readouts-complete.json'),dict(complete=True,stages=allrecords,seconds=time.monotonic()-start,encoder_wall_seconds=completed['wall_seconds'],total_compute_seconds=time.monotonic()-start+completed['wall_seconds'],source_sha256=e.sha(__file__)))
if __name__=='__main__':
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
