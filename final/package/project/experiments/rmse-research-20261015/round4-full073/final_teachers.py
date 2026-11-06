"""Train-only cross-fitted full073 teachers for the final500 CNN refit."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import full_branch as b
HERE=Path(__file__).resolve().parent
FINAL=HERE/'final500'


def forbid_test_inputs(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).absolute()
        if path.is_relative_to(b.ROOT/'data/test') or (path.is_relative_to(b.EXP) and path.name.endswith('-test.npz')):
            raise RuntimeError('Test inputs are forbidden while fitting final500 models: '+str(path))


def main():
    sys.addaudithook(forbid_test_inputs)
    d,y=b.load();FINAL.mkdir(exist_ok=True)
    assert len(d['ids'])==500 and all(str(i).startswith('TRAIN_') for i in d['ids'])
    records=[]
    for k,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=2026101607).split(y)):
        records.append(dict(fold=k,train=tr.tolist(),query=va.tolist(),
                            train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][va].tolist()))
    manifest=dict(split_seed=2026101607,training_ids=d['ids'].tolist(),folds=records,
                  source_hashes=d['source_hashes'],script_sha256=b.r.digest(__file__),
                  branch_script_sha256=b.r.digest(HERE/'full_branch.py'),plan_sha256=b.r.digest(HERE/'FINAL500.md'),
                  test_inputs_blocked=True,device='cpu',threads=2)
    path=FINAL/'teacher-plan.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:b.r.save_json(path,manifest)
    # Only this module's output root changes; the original branch source and
    # dataset/configuration remain untouched. Existing CV caches stay immutable.
    b.HERE=FINAL
    for row in records:
        tr=np.array(row['train']);va=np.array(row['query'])
        b.fit(d,tr,va,y[tr].copy(),row['fold'],'full500')
        print('FINAL500 TEACHER COMPLETE',row['fold'],flush=True)
    b.r.save_json(FINAL/'teachers-complete.json',dict(complete=True,teacher_count=3,test_inputs_read=False))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
