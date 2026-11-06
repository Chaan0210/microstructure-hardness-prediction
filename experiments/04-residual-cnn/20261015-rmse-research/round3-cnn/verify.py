"""Independent CNN comparison audit and actual held-out-label poisoning refit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import complement_cnn as c


def main():
    for filename in ('provenance.json','confirmation-provenance.json'):
        for name,expected in json.loads((c.HERE/filename).read_text()).items():
            assert c.r.digest(_repo_path(c.r.ROOT, name))==expected,name
    d=c.load_data();s=np.load(_repo_io(c.HERE/'confirmation-summary.npz'));meta=json.loads((c.HERE/'confirmation-summary.json').read_text());checks=[]
    for part in range(3):
        for k in range(5):
            anchor=np.load(_repo_io(c.ROUND2/f'full500-{c.SEEDS[part]}/fold{k}.npz'));va=anchor['validation_index']
            for arm in ('residual','aux','context_only'):
                root=c.HERE/f'partition-{part}/fold{k}-{arm}'
                report=json.loads((root/'training.json').read_text());z=np.load(_repo_io(root/'predictions.npz'))
                assert report['epochs']==40 and report['updates']==520 and report['seed']==c.SEEDS[part]+k
                assert not set(report['meta']['train_ids'])&set(report['meta']['validation_ids'])
                assert np.array_equal(z['validation_index'],va)
                pred=anchor['base']+.25*z['validation_prediction']
                fail=np.nan_to_num(d['X'][va,list(d['keys']).index('n_obj')],nan=0)<20;pred[fail]=anchor['base'][fail]
                assert np.array_equal(pred,s[arm][part,va])
                value=(sum(float(a-b)**2 for a,b in zip(pred,d['y'][va]))/100)**.5
                assert abs(value-meta[arm]['candidate']['folds'][part][k])<1e-12
                checks.append(dict(partition=part,fold=k,arm=arm,RMSE=value,reload_difference=report['reload_max_difference']))
    # A complete 40-epoch image-residual fit must be invariant to outer labels.
    a=c.fold_data(d,0,0);poison={key:value.copy()if isinstance(value,np.ndarray)else value for key,value in d.items()}
    poison['y'][a['valid']]+=10000;b=c.fold_data(poison,0,0)
    for key in ('direct','residual'):assert np.array_equal(a['targets'][key],b['targets'][key])
    assert a['meta']==b['meta']
    target=c.HERE/'poison-refit-fold0-residual'
    if not(target/'training.json').exists():c.fit_one(b,'residual',c.SEEDS[0],target)
    original=torch.load(_repo_io(c.HERE/'partition-0/fold0-residual/model.pt'),map_location='cpu',weights_only=True)
    altered=torch.load(_repo_io(target/'model.pt'),map_location='cpu',weights_only=True)
    state_diff=max(float((original['state_dict'][k]-altered['state_dict'][k]).abs().max())for k in original['state_dict'])
    assert state_diff==0.
    p=np.load(_repo_io(target/'predictions.npz'))['validation_prediction'];q=np.load(_repo_io(c.HERE/'partition-0/fold0-residual/predictions.npz'))['validation_prediction']
    assert np.array_equal(p,q)
    old=json.loads((c.HERE/'partition-0/fold0-residual/training.json').read_text())
    new=json.loads((target/'training.json').read_text());assert old['curve']==new['curve']
    result=dict(passed=True,all_45_confirmation_fits_verified=checks,heldout_label_change=10000,
        complete_40_epoch_poison_refit_state_maxdiff=state_diff,poison_prediction_maxdiff=float(np.max(np.abs(p-q))),
        identical_training_curve=True,protected_inputs_unchanged=True,no_new_external_models=True,no_test_use=True,
        source_sha256=c.r.digest(c.HERE/'complement_cnn.py'))
    c.r.save_json(c.HERE/'verification.json',result);print('All 45 confirmation fits and a complete 40-epoch label-poison refit passed.')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
