"""Repeat only the unexpected duplicate-input control; no new hyperparameters."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import stream_cnn as s
from confidence import compare
p=s.p


def evaluate():
    d=dict(np.load(_repo_io(p.R4.parent/'inputs.npz')));z=dict(np.load(_repo_io(p.HERE/'baseline076.npz')))
    y=d['y'][d['dev']];fold=z['fold'];fixed=json.loads((p.R4/'noise-calibration/summary.json').read_text())['aux']['selection']
    current=np.load(_repo_io(p.HERE/'probe-oof.npz'))['old4'];out={'current':current};records={}
    for tag,repeats in [('seed0',[0]),('seed1',[1]),('mean2',[0,1])]:
        for family in ('legacy','duplicate'):
            matched=z['p76'].copy();frozen=z['p76'].copy();selections=[]
            for k in range(5):
                inner=np.load(_repo_io(p.R4/f'aux-mean2/fold{k}-selection.npz'));B=inner['B'];good=inner['eligible']
                means=[]
                for stage in ('selector','final'):
                    ps=[]
                    for j in repeats:
                        path=p.R4/f'aux-seed{j}/fold{k}-{stage}/predictions.npz' if family=='legacy' else p.HERE/f'stream/duplicate-seed{j}/fold{k}-{stage}/predictions.npz'
                        q=np.load(_repo_io(path));assert np.array_equal(q['validation_index'],B if stage=='selector' else z['dev'][fold==k])
                        ps.append(q['validation_prediction'])
                    means.append(np.mean(ps,axis=0))
                rb,rv=means;ix=np.flatnonzero(fold==k)
                w,meta=s.nc.select(rb[good],(d['y'][B]-inner['base073'])[good],d['noise'][B][good]>=12)
                poison=d['y'].copy();poison[z['dev'][ix]]+=10000
                ww,mm=s.nc.select(rb[good],(poison[B]-inner['base073'])[good],d['noise'][B][good]>=12)
                assert np.array_equal(w,ww) and meta==mm
                active=(z['noise'][ix]>=12)&~z['failed'][ix]
                matched[ix[active]]+=w[1]*rv[active];frozen[ix[active]]+=fixed[k]['addition']['weights'][1]*rv[active]
                selections.append(dict(fold=k,weights=w.tolist(),**meta,outer_poison_exact=True))
            out[family+'_'+tag]=matched;out[family+'_'+tag+'_frozen']=frozen;records[family+'_'+tag]=selections
    assert np.max(np.abs(out['legacy_mean2']-current))<1e-10
    effect=compare(current,out['duplicate_mean2'],y,fold)
    frozen_effect=compare(current,out['duplicate_mean2_frozen'],y,fold)
    seed_effects={tag:compare(out['legacy_'+tag],out['duplicate_'+tag],y,fold) for tag in ('seed0','seed1')}
    adopt=effect['gain']>=.01 and sum(g>1e-10 for g in effect['fold_gains'])>=3 and effect['ci95'][0]>0 and frozen_effect['gain']>0
    scores={name:dict(rmse=s.r.rmse(pr,y),folds=[s.r.rmse(pr[fold==k],y[fold==k]) for k in range(5)]) for name,pr in out.items()}
    result=dict(scores=scores,effect=effect,frozen_weight_effect=frozen_effect,seed_effects=seed_effects,
        adoption_gate_passed=bool(adopt),selections=records,limitation='Post-analysis seed confirmation; same400 explored labels.')
    p.save(_repo_io('control-repeat-results.json'),result)
    np.savez_compressed(_repo_io(p.HERE/'control-repeat-oof.npz'),ids=z['ids'],fold=fold,y=y,**out)
    print(json.dumps(dict(scores=scores,effect=effect,frozen=frozen_effect,seed_effects=seed_effects,adopt=bool(adopt)),indent=2),flush=True)
    p.protect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate']);args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.mode=='train':s.ARMS=('duplicate',);s.train(1)
        else:evaluate()
