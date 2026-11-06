"""Reconstruct H9 seed means and test a real TabPFN fit with held-out labels poisoned."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from full500 import HERE,SEEDS,W,harness
from models import inputs
import research as r


def main():
    d=inputs();saved=np.load(_repo_io(HERE/'h9-three-seeds.npz'));report=json.loads((HERE/'h9-three-seeds.json').read_text())
    failed=np.nan_to_num(d['X'][:,list(d['keys']).index('n_obj')],nan=0)<20
    reconstructed={key:np.zeros((3,500))for key in ('base','physical','random')}
    checked=[]
    for rep,seed in enumerate(SEEDS):
        for k in range(5):
            old=np.load(_repo_io(HERE/f'full500-{seed}/fold{k}.npz'));one=np.load(_repo_io(HERE/f'h9/part{rep}-fold{k}.npz'));vi=old['validation_index']
            vals={'base':[old['tab']],'physical':[one['physical']],'random':[one['random']]}
            for add in (100,200):
                z=np.load(_repo_io(HERE/f'h9-replicates/part{rep}-fold{k}-seed{add}.npz'))
                assert not set(z['train_ids'])&set(z['validation_ids'])
                assert np.array_equal(z['train_ids'],old['train_ids'])and np.array_equal(z['validation_ids'],old['validation_ids'])
                assert int(z['seed'])==seed+k+add
                for key in vals:vals[key].append(z[key+'_tab'])
            for key,v in vals.items():
                # Explicit addition in place of the training script's np.mean.
                average=(v[0]+v[1]+v[2])/3
                p=old['base']+W[0]*(average-old['tab']);p[failed[vi]]=old['base'][failed[vi]]
                reconstructed[key][rep,vi]=p
            checked.append([rep,k])
    differences={key:float(np.max(np.abs(v-saved[key])))for key,v in reconstructed.items()}
    assert max(differences.values())<1e-12
    for key in ('physical','random'):
        direct=float(np.linalg.norm(saved[key]-d['y'])/np.sqrt(1500))
        assert abs(direct-report[key]['candidate']['pooled'])<1e-12
    # A saved full-seed physical head is rerun with outer held-out targets changed.
    old=np.load(_repo_io(HERE/f'full500-{SEEDS[0]}/fold0.npz'));stored=np.load(_repo_io(HERE/'h9-replicates/part0-fold0-seed100.npz'))
    pos={i:j for j,i in enumerate(d['ids'])};tr=np.array([pos[i]for i in old['train_ids']]);va=old['validation_index']
    y=d['y'].copy();y[va]+=10000
    extra=np.load(_repo_io(HERE/'cumulants-train.npz'))['X'][:,[1,4]];pred=np.zeros(len(va))
    for noisy in (False,True):
        cols=np.ones(len(d['keys']),bool)if noisy else d['keep'];mask=(d['noise'][va]>=12)==noisy;vi=va[mask]
        model=harness.tabpfn_new('v3.5',SEEDS[0]+100,'cpu',n_est=4)
        model.fit(np.c_[d['X'][tr][:,cols],extra[tr]],y[tr]);pred[mask]=model.predict(np.c_[d['X'][vi][:,cols],extra[vi]])
        del model;gc.collect()
    poison_diff=float(np.max(np.abs(pred-stored['physical_tab'])))
    assert poison_diff<1e-8,poison_diff
    r.save_json(HERE/'verification-h9.json',dict(fold_pairs_verified=checked,all_3_seeds_used=True,
        reconstruction_max_differences=differences,physical_TabPFN_heldout_label_poison_maxdiff=poison_diff,
        independent_metric_recalculation=True,public_upload=False,code_sha256=r.digest(__file__)))
    print('H9 verified:',differences,'poison maxdiff',poison_diff)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
