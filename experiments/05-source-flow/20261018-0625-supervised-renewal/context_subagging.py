"""Balanced source-row subagging versus equal-member full-context seed control."""
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
import numpy as np,torch
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];fold=d['fold'];y=d['y'][dev];ids=d['ids'][dev]
    source=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(source));index={str(v):j for j,v in enumerate(z['ids'])};X=z['X'][[index[str(v)] for v in ids]].astype(float)
    sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    out=_repo_path(HERE, 'context-subagging');out.mkdir(exist_ok=True);pred={k:np.empty(400) for k in ['full','subbag']};start=time.monotonic()
    recipe=dict(members=5,n_estimators=8,seeds=[0,100,200,300,400],partition_seed=2026101931,
        training='SameouterT320pool;5balanceddelete-one-blockcontexts,n256each,eachoriginalappears4times. Control5fullT320contexts. No sourcegroupsorvalidationrowscrossboundaries.',
        formula='Complete083+.384*(subbagmean-fullseedmean)onexistingnonfailedrows',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),comparison_scope='Conditionalonhistoricalphysicalfeatures;availableoriginaltrainingpoolandqueryimagesmatch.')
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2))
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);bags=list(KFold(5,shuffle=True,random_state=2026101931+k).split(tr));coverage=np.zeros(len(tr),int);ps={a:[] for a in pred}
        for member,(bt,_) in enumerate(bags):
            coverage[bt]+=1;seed=100*member+k
            for arm,tt in [('full',tr),('subbag',tr[bt])]:
                target=out/f'{arm}-f{k}-m{member}.npz';old=_repo_path(HERE, f'conductivity/readout/physical-f{k}-s{100*member}.npz')
                if target.exists():r=np.load(_repo_io(target));assert np.array_equal(r['train'],tt) and np.array_equal(r['query'],q);p=r['prediction']
                elif arm=='full' and old.exists():
                    r=np.load(_repo_io(old));assert np.array_equal(r['train'],tt) and np.array_equal(r['query'],q);p=r['prediction'];np.savez(_repo_io(target),train=tt,query=q,prediction=p,reused_from=np.array(str(old.relative_to(ROOT))))
                else:
                    t=time.monotonic();m=harness.tabpfn_new('v3.5',seed,'cpu',n_est=8);m.fit(X[tt],y[tt]);p=m.predict(X[q]);del m;gc.collect();np.savez(_repo_io(target),train=tt,query=q,prediction=p,seed=np.array(seed));print(arm,k,member,'seconds',round(time.monotonic()-t,1),flush=True)
                ps[arm].append(p)
        assert np.all(coverage==4)
        for arm in pred:pred[arm][q]=np.mean(ps[arm],0)
    base=d['base083'];eligible=~d['failed'];system=base+.384*eligible*(pred['subbag']-pred['full']);rm=lambda a:float(np.sqrt(np.mean((a-y)**2)))
    result=dict(full_single_bank_rmse=rm(pred['full']),subbag_single_bank_rmse=rm(pred['subbag']),base083_rmse=rm(base),system_rmse=rm(system),system_fold_rmse=[float(np.sqrt(np.mean((system[fold==k]-y[fold==k])**2))) for k in range(5)],seconds=time.monotonic()-start)
    np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,base083=base,system=system,**pred);(out/'results.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
