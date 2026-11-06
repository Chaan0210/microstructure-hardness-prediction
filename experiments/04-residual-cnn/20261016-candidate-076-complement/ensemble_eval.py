"""Fixed50:50 family averaging; no new fitted coefficient."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
from confidence import compare
H=Path(__file__).resolve().parent


def main():
    z=np.load(_repo_path(H, 'control-repeat-oof.npz'));y=z['y'];fold=z['fold'];a=z['current'];b=z['duplicate_mean2']
    mean=(a+b)/2;frozen=(a+z['duplicate_mean2_frozen'])/2
    expected=np.mean((a-y)**2)/2+np.mean((b-y)**2)/2-np.mean((a-b)**2)/4
    assert abs(np.mean((mean-y)**2)-expected)<1e-10
    effects={'direct_duplicate':compare(a,b,y,fold),'fixed_family_mean':compare(a,mean,y,fold)}
    rng=np.random.default_rng(2026101633);samples=[]
    for _ in range(100):
        draw=rng.integers(0,len(y),(100,len(y)));base=np.sqrt(((a-y)**2)[draw].mean(1))
        samples.append(np.c_[base-np.sqrt(((b-y)**2)[draw].mean(1)),base-np.sqrt(((mean-y)**2)[draw].mean(1))])
    samples=np.concatenate(samples);gains=np.array([effects[n]['gain'] for n in effects]);sd=np.maximum(samples.std(0,ddof=1),1e-12)
    critical=float(np.quantile(np.max(np.abs((samples-gains)/sd),axis=1),.95))
    bands={n:[float(g-critical*t),float(g+critical*t)] for n,g,t in zip(effects,gains,sd)}
    q=effects['fixed_family_mean'];f=compare(a,frozen,y,fold)
    adopt=q['gain']>=.01 and sum(v>1e-10 for v in q['fold_gains'])>=3 and q['ci95'][0]>0 and f['gain']>0 and bands['fixed_family_mean'][0]>0
    result=dict(scores={'current':float(np.sqrt(np.mean((a-y)**2))),'duplicate':float(np.sqrt(np.mean((b-y)**2))),
        'fixed_family_mean':float(np.sqrt(np.mean((mean-y)**2))),'fixed_family_mean_frozen':float(np.sqrt(np.mean((frozen-y)**2)))},
        effects=effects,frozen_effect=f,simultaneous_bands=bands,adoption_gate_passed=bool(adopt),
        residual_correlation=float(np.corrcoef(a-y,b-y)[0,1]),mean_squared_prediction_disagreement=float(np.mean((a-b)**2)),
        MSE_averaging_identity_verified=True,coefficient_search=False,
        limitation='Same400 explored labels; no new-data or Public performance claim.')
    (_repo_path(H, 'ensemble-results.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(_repo_path(H, 'ensemble-oof.npz'),ids=z['ids'],fold=fold,y=y,current=a,mean=mean,frozen_mean=frozen)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
