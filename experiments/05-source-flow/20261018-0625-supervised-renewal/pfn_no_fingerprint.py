"""Isolate a data-independent hashed nuisance while keeping native preprocessing.

The differentiable-input investigation exposed discontinuous fingerprint
changes. This is a prediction check, not an assumption that gradients prove
an RMSE benefit. Native feature/target transforms, seeds and training rows are
unchanged from the cached physical104 control.
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
import json,hashlib,time,gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];y=d['y'][dev];fold=d['fold'];out=_repo_path(HERE, 'pfn-no-fingerprint');out.mkdir(exist_ok=True)
    file=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(file));pos={str(v):i for i,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in ids]].astype(float)
    spec=dict(hypothesis='Disableunnecessaryhashedrowfingerprintforcontinuousuniquephysicalmeasurements;gradientchecksrevealedinputdiscontinuities butdo notestablishpredictiveharm.',
        unchanged='Actualphysical104inputs,nativefeature/targetpreprocessing,n8,seeds0/100/200+fold,T320/V80. OnlyFINGERPRINT_FEATURE=False.',
        primary='Complete083+.384*(noFingerprintP104 -matchedstandardP104),existingnonfailedfallback.',scope='Conditionalhistoricalrepresentations;noPublicfeedbackorcoefficientfit.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness;new=np.zeros(400);control=np.zeros(400)
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);ps=[];cs=[]
        for seed in [0,100,200]:
            c=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{k}-s{seed}.npz'));assert np.array_equal(c['train'],tr) and np.array_equal(c['query'],q);cs.append(c['prediction']);cache=out/f'f{k}-s{seed}.npz'
            if cache.exists():pr=np.load(_repo_io(cache))['prediction']
            else:
                start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=8);m.inference_config={'FINGERPRINT_FEATURE':False};m.fit(X[tr],y[tr]);assert m.inference_config_.FINGERPRINT_FEATURE is False;pr=m.predict(X[q]);np.savez(_repo_io(cache),train=tr,query=q,prediction=pr);del m;gc.collect();print(k,seed,time.monotonic()-start,flush=True)
            ps.append(pr)
        new[q]=np.mean(ps,0);control[q]=np.mean(cs,0)
    system=d['base083']+.384*(~d['failed'])*(new-control);pred=dict(control=control,new=new,system=system,base083=d['base083']);r={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
