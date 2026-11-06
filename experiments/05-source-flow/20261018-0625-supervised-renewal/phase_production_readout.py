"""Complementarity of source-local phase uncertainty and production physical inputs."""
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
import torch
from threadpoolctl import threadpool_limits
from phase_readout import augment

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));folder=_repo_path(HERE, 'runs', a.run);z=np.load(_repo_io(folder/'frozen-features.npz'));tr=z['train'];q=z['query'];f=a.fold
    assert np.array_equal(tr,d['dev'][d['fold']!=f]) and np.array_equal(q,d['dev'][d['fold']==f]);ix=np.flatnonzero(d['fold']==f)
    fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');ph=np.load(_repo_io(fp));pos={str(v):i for i,v in enumerate(ph['ids'])};X=ph['X'][[pos[str(v)] for v in d['ids']]].astype(float)
    sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    out=_repo_path(HERE, f'phase-production-{a.run}');out.mkdir(exist_ok=True)
    F=augment(z['train_features']);Q=augment(z['query_features'])
    recipe=dict(run=a.run,fold=f,train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),seed=[0,100,200],seed_fold_offset=True,n_estimators=8,feature_dimension=104+F.shape[1],
        phase_feature_sha256=hashlib.sha256((folder/'frozen-features.npz').read_bytes()).hexdigest(),production_feature_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),
        primary='Complete083 +.384*(physical104+newphase prediction -physical104control), preserve existingfailed-maskfallback.',
        scope='Conditional screening; newphase trained onlyouterT, inheritedphysical104 hassharedhistoricalrepresentations. No claimend-to-endvalidation.',
        reason='Earliernewphaseheadsusedweakerzero-shotglobalfeatures. This assessesadditionalinformationwiththeactualproductionbank and matchedcontrols.')
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2));ps=[];controls=[]
    for seed in [0,100,200]:
        c=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{f}-s{seed}.npz'));assert np.array_equal(d['dev'][c['train']],tr) and np.array_equal(d['dev'][c['query']],q);controls.append(c['prediction'])
        path=out/f'seed{seed}.npz'
        if path.exists():p=np.load(_repo_io(path))['prediction']
        else:
            start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+f,'cpu',n_est=8);m.fit(np.c_[X[tr],F],d['y'][tr]);p=m.predict(np.c_[X[q],Q]);np.savez(_repo_io(path),train=tr,query=q,prediction=p);del m;gc.collect();print(seed,'seconds',time.monotonic()-start,flush=True)
        ps.append(p)
    p=np.mean(ps,0);control=np.mean(controls,0);base=d['base083'][ix];eligible=~d['failed'][ix];system=base+.384*eligible*(p-control);y=d['y'][q];rm=lambda v:float(np.sqrt(np.mean((v-y)**2)))
    r=dict(raw_rmse=rm(p),matched_control_rmse=rm(control),base083_rmse=rm(base),system_rmse=rm(system),perseed=[rm(v) for v in ps]);np.savez(_repo_io(out/'predictions.npz'),train=tr,query=q,prediction=p,control=control,base083=base,system=system);(out/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--fold',type=int,default=0)
    with threadpool_limits(limits=2):main(ap.parse_args())
