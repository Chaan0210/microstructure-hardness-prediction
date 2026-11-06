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
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];fold=d['fold'];y=d['y'][dev]
    fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(fp));pos={str(v):i for i,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in ids]].astype(float)
    f=np.load(_repo_path(HERE, 'conductivity/features.npz'));pos={str(v):i for i,v in enumerate(f['ids'])};ix=[pos[str(v)] for v in ids];S=f['marginal'][ix];P=f['pde'][ix]
    sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    out=_repo_path(HERE, 'conductivity/readout');out.mkdir(exist_ok=True)
    spec=dict(seeds=[0,100,200],seed_fold_offset=True,n_estimators=8,device='CPU',feature_counts=[104,113,149],
        delta_weight=.384,formula='Complete083 +.384*(new single-bank prediction -matched single-bank physical104 control),onlyonexistingnonfailedrows.',
        interpretation='This is a defined full combined predictor; it does not pretend that one bank is the whole incumbent or that all incumbent TabPFN members were replaced.',
        primary='physical+marginals+PDE',control='physical+marginals',scope='Conditional frozen-representation screen onreused400originals.',
        physics_file_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),pde_file_sha256=hashlib.sha256((_repo_path(HERE, 'conductivity/features.npz')).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));ps={k:np.zeros(400) for k in ['physical','marginal','pde']}
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k)
        for kind,A in [('physical',X),('marginal',np.c_[X,S]),('pde',np.c_[X,S,P])]:
            pp=[]
            for seed in [0,100,200]:
                path=out/f'{kind}-f{k}-s{seed}.npz'
                if path.exists():pred=np.load(_repo_io(path))['prediction']
                else:
                    start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=8);m.fit(A[tr],y[tr]);pred=m.predict(A[q]);np.savez(_repo_io(path),train=tr,query=q,prediction=pred)
                    del m;gc.collect();print(kind,k,seed,'seconds',round(time.monotonic()-start,1),flush=True)
                pp.append(pred)
            ps[kind][q]=np.mean(pp,0)
    ps['base083']=d['base083'];eligible=~d['failed']
    for kind in ['marginal','pde']:ps[kind+'_system']=d['base083']+.384*eligible*(ps[kind]-ps['physical'])
    results={k:dict(rmse=float(np.sqrt(np.mean((v-y)**2))),fold_rmse=[float(np.sqrt(np.mean((v[fold==f]-y[fold==f])**2))) for f in range(5)]) for k,v in ps.items()}
    np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**ps);(out/'results.json').write_text(json.dumps(results,indent=2));print('DONE',json.dumps(results),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
