"""Reproduce the adopted, unchanged two-seed full500 CNNs in this folder."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import study as s


def main():
    s.protect();fit=s.fitter();s.sys.addaudithook(s.no_test)
    d=s.e.c.load_data();labels=d.pop('y');all_train=np.arange(500);empty=np.array([],int)
    plan=json.loads((s.R4/'final500/teacher-plan.json').read_text())
    base=np.full(500,np.nan);gap=np.full(500,np.nan);count=np.zeros(500,int);sources=[]
    for row in plan['folds']:
        path=s.R4/f"final500/teachers/fold{row['fold']}-full500/predictions.npz"
        z=np.load(_repo_io(path));tr=z['train'];va=z['query']
        assert np.array_equal(tr,row['train']) and np.array_equal(va,row['query'])
        assert set(tr).isdisjoint(va) and set(tr)|set(va)==set(all_train)
        base[va]=z['base'];gap[va]=z['tab']-z['ridge'];count[va]+=1
        sources.append(dict(train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][va].tolist(),
                            file=str(path.relative_to(s.R4)),sha256=s.r.digest(path)))
    assert np.all(count==1) and np.isfinite(base).all() and np.isfinite(gap).all()
    keys=list(d['keys']);X=d['X']
    raw=np.c_[base,gap,d['noise'],X[:,keys.index('u_ph')],
              np.log1p(np.maximum(np.nan_to_num(X[:,keys.index('n_obj')],nan=0),0)),X[:,keys.index('coverage')]]
    data=s.e.cnn_data(d,all_train,empty,labels.copy(),base,sources,-1,'full500_refit',raw)
    data['meta'].update(test_inputs_read=False,teacher_split_seed=2026101607,
                        validation_role='none; full training fit')
    data['main_weights']=np.ones(500,np.float32)
    recipe=json.loads((s.R4/'final500/recipe.json').read_text())
    assert data['meta']==recipe['normalization']
    folder=s.HERE/'retrained-selected';folder.mkdir(exist_ok=True)
    records=[]
    for j,seed in enumerate((2026101501,2026201501)):
        path=folder/f'aux-seed{j}'
        if not (path/'training.json').exists():fit.fit_one(data,'aux',seed,path)
        a=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)
        b=torch.load(_repo_io(s.R4/f'final500/models/aux-seed{j}/model.pt'),map_location='cpu',weights_only=True)
        differences={k:float((a['state_dict'][k]-b['state_dict'][k]).abs().max()) for k in a['state_dict']}
        assert max(differences.values())==0.,max(differences.values())
        assert a['normalization']==b['normalization'] and a['seed']==b['seed']
        records.append(dict(seed=seed,all_parameter_tensors_bitwise_identical=True,max_parameter_error=0.,
                            model_file=str((path/'model.pt').relative_to(s.ROOT)),sha256=s.r.digest(path/'model.pt')))
        print('SELECTED FULL500 REPRODUCED',seed,'parameter error0',flush=True)
    s.save(_repo_io('selected-training-reproduction.json'),dict(complete=True,models=records,
        teachers_reused_from_verified_train_only_caches=True,no_test_inputs_read=True,
        final_coefficient_unchanged=recipe['production_weights']['addition'],
        no_validation_score_from_full500_refit=True,source_recipe_sha256=s.r.digest(s.R4/'final500/recipe.json')))
    s.protect()


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
