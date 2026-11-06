"""Paired physical+learned connectivity readouts, all transformations train-local."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import graph
import numpy as np,json,sys,time,gc,importlib.metadata
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
WEIGHT=_repo_path(c.ROOT, 'experiments/tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors');ARMS=['topology','local','hardness','random','raw']
def physical():
    d=c.inputs(labels=True);base=_repo_path(c.ROOT, 'experiments/push-1300');record=np.load(_repo_io(base/'abl/dev-h9phys-full-044-u1dmavg.npz'));path=(base/str(record['feature_file'])).resolve();assert path.is_relative_to(base) and c.sha(path)==str(record['feature_sha256']);f=np.load(_repo_io(path),allow_pickle=True);pos={str(v):i for i,v in enumerate(f['ids'])};d['X']=f['X'][[pos[str(i)] for i in d['ids']]].astype(float);assert d['X'].shape==(500,100);d['feature_source']=str(path.relative_to(c.ROOT));d['feature_sha256']=c.sha(path);return d
def predict_tab(X,y,Q,seed,path,train,query):
    start=time.monotonic();m=TabPFNRegressor.create_default_for_version(ModelVersion('v3.5'),model_path=str(WEIGHT),device='mps',inference_precision=torch.float32,memory_saving_mode=True,random_state=seed,n_preprocessing_jobs=1,show_progress_bar=False);m.fit(X,y.copy());p=np.asarray(m.predict(Q),float);assert np.isfinite(p).all();torch.mps.synchronize();record=dict(seed=seed,seconds=time.monotonic()-start,train=train.tolist(),query=query.tolist(),train_X_sha256=c.ah(X),query_X_sha256=c.ah(Q),train_y_sha256=c.ah(y),peak_memory=c.memory(),inference_precision='float32',memory_saving_mode=True,n_estimators=str(m.n_estimators));c.arrays(str((path/'prediction.npz').relative_to(c.HERE)),prediction=p,train=train,query=query);c.save(_repo_io(str((path/'contract.json').relative_to(c.HERE))),record);del m;gc.collect();torch.mps.empty_cache();return p
def raw_signatures():
    z=np.load(_repo_io(c.HERE/'targets.npz'));pairs,radii=graph.fixed_pairs();out=[]
    for b,bw,q,qw in zip(z['boundary'],z['boundary_weight'],z['edge_target'],z['edge_weight']):out.append(graph.signature(.5+(b-.5)*bw,.5+(q-.5)*qw,pairs,radii))
    return dict(zip(map(int,z['dev']),out))
def main():
    complete=json.loads((c.HERE/'encoders-complete.json').read_text());assert complete['complete'];assert complete['source_sha256']==c.sha(c.HERE/'encoder.py');d=physical();start=time.monotonic();deadline=start+7200-complete['seconds']-json.loads((c.HERE/'resource-probe-b32.json').read_text())['seconds']-600;assert deadline>start
    c.save(_repo_io('readout-code-lock.json'),dict(source_sha256=c.sha(__file__),encoder_source_sha256=c.sha(c.HERE/'encoder.py'),foundation_weight_sha256=c.sha(WEIGHT),feature_source=d['feature_source'],feature_sha256=d['feature_sha256'],environment=dict(python=sys.version,packages={x.metadata['Name']:x.version for x in importlib.metadata.distributions()}),plan_sha256=c.sha(c.HERE/'PLAN.md'),base_reuse='Exact prior physical-only contexts, identical source counts/features/targets/seeds.'))
    raw=raw_signatures();c.arrays('raw-signatures.npz',indices=np.array(list(raw)),signature=np.array(list(raw.values())));records=[]
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);folder=c.HERE/f'readouts/fold{k}-{stage}';folder.mkdir(parents=True,exist_ok=False);old=c.PREV/f'readouts/fold{k}-{stage}';base=np.load(_repo_io(old/'predictions.npz'));ctx=np.load(_repo_io(old/'physical-context.npz'));X=d['X'][tr];Q=d['X'][q];y=d['y'][tr];assert np.array_equal(ctx['X'],X,equal_nan=True) and np.array_equal(ctx['Q'],Q,equal_nan=True) and np.array_equal(ctx['y'],y);assert np.array_equal(base['train'],tr) and np.array_equal(base['query'],q);c.save(_repo_io(str((folder/'baseline-reference.json').relative_to(c.HERE))),dict(path=str(old.relative_to(c.ROOT)),context_sha256=c.sha(old/'physical-context.npz'),predictions_sha256=c.sha(old/'predictions.npz'),train=tr.tolist(),query=q.tolist()));out={}
            for arm in ARMS:
                vv=[]
                for repeat in ([0] if arm=='raw' else [0,1]):
                    path=folder/f'{arm}-seed{repeat}';path.mkdir();saved={}
                    if arm=='raw':
                        st=np.array([raw[int(i)] for i in tr]);sq=np.array([raw[int(i)] for i in q]);pt=np.zeros((len(tr),16));pq=np.zeros((len(q),16))
                    else:
                        src=c.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('topology' if arm=='random' else arm));f=np.load(_repo_io(src/('random-features.npz' if arm=='random' else 'features.npz')));assert np.array_equal(f['train'],tr) and np.array_equal(f['query'],q);sc=StandardScaler().fit(f['train_latent']);pca=PCA(n_components=16,svd_solver='full').fit(sc.transform(f['train_latent']));pt=pca.transform(sc.transform(f['train_latent']));pq=pca.transform(sc.transform(f['query_latent']));st=f['train_signature'];sq=f['query_signature'];saved=dict(scaler_mean=sc.mean_,scaler_scale=sc.scale_,pca_mean=pca.mean_,pca_components=pca.components_,pca_explained_variance_ratio=pca.explained_variance_ratio_)
                    xt=np.c_[X,st,pt];xq=np.c_[Q,sq,pq];assert xt.shape==(len(tr),144);c.arrays(str((path/'context.npz').relative_to(c.HERE)),X=xt,y=y,Q=xq,train=tr,query=q,**saved);pp=[]
                    for j in range(3):
                        if time.monotonic()>deadline:raise RuntimeError('Fixed readout budget exhausted')
                        sub=path/f'tab{j}';sub.mkdir();pp.append(predict_tab(xt,y,xq,100*j+k,sub,tr,q))
                    vv.append(np.mean(pp,0));print('READOUT',k,stage,arm,repeat,round(time.monotonic()-start,1),flush=True)
                out[arm]=np.repeat(vv,2,axis=0) if arm=='raw' else np.array(vv)
            c.arrays(str((folder/'predictions.npz').relative_to(c.HERE)),train=tr,query=q,base=base['base'],**out);records.append(str(folder.relative_to(c.HERE)))
    c.save(_repo_io('readouts-complete.json'),dict(complete=True,contexts=270,reused_base_contexts=30,records=records,seconds=time.monotonic()-start,encoder_seconds=complete['seconds'],total_compute_seconds=complete['seconds']+json.loads((c.HERE/'resource-probe-b32.json').read_text())['seconds']+time.monotonic()-start,source_sha256=c.sha(__file__)))
if __name__=='__main__':
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
