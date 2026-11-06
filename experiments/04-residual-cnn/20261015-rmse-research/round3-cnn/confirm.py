"""Repeat only screen-qualified CNN objectives; do not modify the training recipe."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import gc
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import complement_cnn as c

ARMS=('residual','aux','context_only')


def summarize():
    d=dict(np.load(_repo_io(c.HERE.parent/'inputs.npz')));out={a:[]for a in ('base',*ARMS)};folds=[];parts=[]
    for part in range(3):
        root=c.HERE/f'partition-{part}'
        if not all((root/f'fold{k}-{a}/training.json').exists()for k in range(5)for a in ARMS):continue
        pred={a:np.zeros(500)for a in out};fold=np.zeros(500,int)
        fail=np.nan_to_num(d['X'][:,list(d['keys']).index('n_obj')],nan=0)<20
        for k in range(5):
            anchor=np.load(_repo_io(c.ROUND2/f'full500-{c.SEEDS[part]}/fold{k}.npz'));va=anchor['validation_index'];pred['base'][va]=anchor['base'];fold[va]=k
            for arm in ARMS:
                z=np.load(_repo_io(root/f'fold{k}-{arm}/predictions.npz'));assert np.array_equal(z['validation_index'],va)
                p=anchor['base']+c.SHRINK*z['validation_prediction'];p[fail[va]]=anchor['base'][fail[va]];pred[arm][va]=p
        for arm in out:out[arm].append(pred[arm])
        folds.append(fold);parts.append(part)
    arrays={a:np.array(v)for a,v in out.items()};folds=np.array(folds);y=d['y']
    result={a:c.r.compare(arrays['base'],arrays[a],y,folds)for a in ARMS}
    for a in ('residual','aux'):result[a+'_vs_context_only']=c.r.compare(arrays['context_only'],arrays[a],y,folds)
    result['executed_partitions']=parts;result['training_source_sha256']=c.r.digest(c.HERE/'complement_cnn.py')
    result['scope']='Honest cached tabular anchor, 500 repeated images; no full021 or full073 revalidation claim.'
    np.savez_compressed(_repo_io(c.HERE/'confirmation-summary.npz'),ids=d['ids'],fold=folds,**arrays)
    c.r.save_json(c.HERE/'confirmation-summary.json',result)
    for a in ARMS:
        q=result[a];print(a,'rmse',q['candidate']['pooled'],'gain',q['gain'],'parts',q['partition_gains'],'folds',q['improved_folds'],'CI',q['conditional_id_bootstrap_gain_ci95'],flush=True)


def main():
    d=c.load_data();started=time.monotonic()
    for part in (1,2):
        root=c.HERE/f'partition-{part}';root.mkdir(exist_ok=True)
        for k in range(5):
            data=c.fold_data(d,part,k)
            for arm in ARMS:
                output=root/f'fold{k}-{arm}'
                if(output/'training.json').exists():continue
                if output.exists():raise RuntimeError(f'Incomplete earlier fit: {output}')
                pred=c.fit_one(data,arm,c.SEEDS[part]+k,output)
                value=data['base'][data['valid']]+c.SHRINK*pred
                print('confirmation completed',part,k,arm,'RMSE',round(c.r.rmse(value,d['y'][data['valid']]),5),flush=True)
                if time.monotonic()-started>2700:raise RuntimeError('45 minute confirmation guard; completed arms retained.')
            del data;gc.collect()
        summarize()


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
