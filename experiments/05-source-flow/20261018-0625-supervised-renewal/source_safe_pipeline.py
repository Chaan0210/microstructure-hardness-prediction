"""A complete replacement from external zero-shot and fold-local image features.

No historical dataset-trained H300/U-Net features or incumbent predictions enter
training. The complete083 is a conditional reference, clearly distinguished
from this pipeline's source-local fits. All images remain in primary scoring.
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
import json,time,gc,hashlib
import numpy as np
import torch,joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from phase_readout import augment

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];y=d['y'][dev];fold=d['fold'];out=_repo_path(HERE, 'source-safe-pipeline');out.mkdir(exist_ok=True)
    file=_repo_path(ROOT, 'experiments/grain-cellpose-220/features-v3-train.npz');z=np.load(_repo_io(file));pos={str(v):i for i,v in enumerate(z['ids'])};G=z['X'][[pos[str(v)] for v in ids]].astype(float)
    cp=np.load(_repo_path(HERE, 'domain-semantics/original-features.npz'));assert np.array_equal(cp['ids'],d['ids']);X=np.c_[G,cp['flow'][dev]]
    spec=dict(primary='Completepredictor=.75TabPFNmean+.25Ridge100,trainedonexternalzero-shotgrainmeasurements+46originalCPSAMflowdescriptors+source-localphasegeometry/uncertaintyfeatures.',
        control='Samecompletepredictorwithoutnewphasefeatures.',tabpfn='Existinglocal3.5n8,seeds0/100/200+fold;nativepreprocessing.',ridge='MedianimputerandStandardScalerfitonlyT,Ridgealpha100fixed;noinnerhyperparametersearch.',
        source_images='Phasebase4000+boundaryNLL2000trainedonlyT320;originalCPweightsareeligibleexternal,frozen. OldH300andglobaldataset-trainedU-Netfeaturesarenotused.',
        evaluation='All400originalheldoutsincludingfailedmaskimages. Newpipelinefitisouter-source-local; reference083hasconditionalhistoricalrepresentations. These400labelshavebeenreusedhistorically,notuntouched.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),grain_feature_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),flow_feature_sha256=hashlib.sha256((_repo_path(HERE, 'domain-semantics/original-features.npz')).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness;pred={k:np.zeros(400) for k in ['control_tab','control_ridge','control','phase_tab','phase_ridge','phase']}
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);path=_repo_path(HERE, f'runs/phase-boundary-nll-f{k}-s89/frozen-features.npz');start=time.monotonic()
        while not path.exists():
            if time.monotonic()-start>7200:raise RuntimeError('Source-localphasefeaturestage didnotcomplete.')
            time.sleep(5)
        p=np.load(_repo_io(path));assert np.array_equal(p['train'],dev[tr]) and np.array_equal(p['query'],dev[q]);rec=json.loads((path.parent/'recipe.json').read_text());assert rec['train']==dev[tr].tolist() and rec['validation']==dev[q].tolist()
        for name,T,Q in [('control',X[tr],X[q]),('phase',np.c_[X[tr],augment(p['train_features'])],np.c_[X[q],augment(p['query_features'])])]:
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.));model.fit(T,y[tr]);pred[name+'_ridge'][q]=model.predict(Q);joblib.dump(model,_repo_io(out/f'{name}-ridge-f{k}.joblib'));ps=[]
            for seed in [0,100,200]:
                cache=out/f'{name}-f{k}-s{seed}.npz'
                if cache.exists():v=np.load(_repo_io(cache))['prediction']
                else:
                    start=time.monotonic();model=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=8);model.fit(T,y[tr]);v=model.predict(Q);np.savez(_repo_io(cache),train=dev[tr],query=dev[q],prediction=v);del model;gc.collect();print(name,k,seed,time.monotonic()-start,flush=True)
                ps.append(v)
            pred[name+'_tab'][q]=np.mean(ps,0);pred[name][q]=.75*pred[name+'_tab'][q]+.25*pred[name+'_ridge'][q]
            print('FOLD',k,name,'rawRMSE',float(np.mean((pred[name][q]-y[q])**2)**.5),'083conditional',float(np.mean((d['base083'][q]-y[q])**2)**.5),flush=True)
    pred['base083_conditional']=d['base083'];pred['phase_quarter_blend_conditional']=.75*d['base083']+.25*pred['phase'];r={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
