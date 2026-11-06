"""Numerical replay, matched-old control, and held-out-label poison refit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import study as s


def main(repeat_arms):
    s.protect(); fit=s.fitter(); s.sys.addaudithook(s.no_test)
    d=s.e.c.load_data(); labels=d.pop('y')
    graphs=json.loads((s.R4/'label-graph.json').read_text())
    records=[]
    for arm,repeat in [(a,0) for a in s.ARMS]+[(a,1) for a in repeat_arms]:
        for graph in graphs:
            for stage in ('selector','final'):
                k=graph['fold'];path=s.HERE/f'models/{arm}-seed{repeat}/fold{k}-{stage}'
                data,contract=s.prepare(d,labels,graph,stage);data=s.configure(data,d,arm)
                z=np.load(_repo_io(path/'predictions.npz'))
                ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)
                assert ck['normalization']==data['meta']
                assert np.array_equal(z['train_index'],contract['train'])
                assert np.array_equal(z['validation_index'],contract['query'])
                assert set(contract['train']).isdisjoint(contract['query'])
                model=fit.ComplementCNN();model.load_state_dict(ck['state_dict'])
                p=fit.predict(model,data['images'],data['context'],data['valid'],'aux')*data['meta']['target_std']
                assert np.array_equal(p,z['validation_prediction'])
                # Query label mutation cannot change a single training tensor.
                altered=labels.copy();altered[contract['query']]+=10000
                poisoned,_=s.prepare(d,altered,graph,stage);poisoned=s.configure(poisoned,d,arm)
                for key in ('images','context','main_weights','train','valid'):
                    assert np.array_equal(data[key],poisoned[key]),key
                for key in data['targets']:
                    assert np.array_equal(data['targets'][key],poisoned['targets'][key]),key
                assert data['meta']==poisoned['meta']
                records.append(dict(arm=arm,repeat=repeat,fold=k,stage=stage,reload_max_error=0.,
                                    query_label_poison_all_fit_inputs_exact=True))
    # Validate that an all-ones loss weight preserves the previous algorithm.
    data,contract=s.prepare(d,labels,graphs[0],'selector')
    data['main_weights']=np.ones(len(data['train']),np.float32)
    control=s.HERE/'verification/control-old-seed0-selector'
    control.parent.mkdir(exist_ok=True)
    if not (control/'training.json').exists():fit.fit_one(data,'aux',2026101501,control)
    old=np.load(_repo_io(s.R4/'aux-seed0/fold0-selector/predictions.npz'))
    new=np.load(_repo_io(control/'predictions.npz'))
    old_error=float(np.max(np.abs(old['validation_prediction']-new['validation_prediction'])))
    assert old_error<1e-4,old_error
    # Actually repeat a full40-epoch fit with poisoned held-out labels.
    altered=labels.copy();altered[np.r_[graphs[0]['B'],graphs[0]['V']]]+=10000
    data,_=s.prepare(d,altered,graphs[0],'selector');data=s.configure(data,d,'high_loss')
    poison=s.HERE/'verification/poison-high-loss-selector'
    if not (poison/'training.json').exists():fit.fit_one(data,'aux',2026101501,poison)
    clean=np.load(_repo_io(s.HERE/'models/high_loss-seed0/fold0-selector/predictions.npz'))
    dirty=np.load(_repo_io(poison/'predictions.npz'))
    assert np.array_equal(clean['validation_prediction'],dirty['validation_prediction'])
    clean_state=torch.load(_repo_io(s.HERE/'models/high_loss-seed0/fold0-selector/model.pt'),map_location='cpu',weights_only=True)['state_dict']
    dirty_state=torch.load(_repo_io(poison/'model.pt'),map_location='cpu',weights_only=True)['state_dict']
    assert all(torch.equal(clean_state[k],dirty_state[k]) for k in clean_state)
    s.protect()
    s.save(_repo_io('verification.json'),dict(all_passed=True,replays=records,old_algorithm_control_max_error=old_error,
                                    actual40epoch_query_poison_state_identical=True,protected_sources_unchanged=True))
    print('VERIFIED',len(records),'models; old-control max error',old_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeat-arms',nargs='*',choices=s.ARMS,default=[])
    a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):main(a.repeat_arms)
