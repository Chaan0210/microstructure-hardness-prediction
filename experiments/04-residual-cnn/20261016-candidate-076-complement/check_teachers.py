"""Check every completed teacher without reading any test inputs."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib
import json
import numpy as np
import probe as p
import full_branch as b


def main():
    d=dict(np.load(_repo_io(p.R4.parent/'inputs.npz')));records=[]
    for path in sorted((p.HERE/'teachers076').glob('*/predictions.npz')):
        z=np.load(_repo_io(path));tr=z['train'];query=z['query'];contract=json.loads(path.with_name('contract.json').read_text())
        assert set(tr).isdisjoint(query)
        assert contract['train_ids']==d['ids'][tr].tolist() and contract['query_ids']==d['ids'][query].tolist()
        assert contract['target_sha256']==hashlib.sha256(d['y'][tr].tobytes()).hexdigest()
        assert contract['code_sha256']==p.r.digest(p.HERE/'retarget_teachers.py')
        old=np.load(_repo_io(p.R4/'teachers'/path.parent.name/'predictions.npz'))
        assert contract['old_prediction_sha256']==p.r.digest(p.R4/'teachers'/path.parent.name/'predictions.npz')
        for key in ('gmm','set'):assert np.array_equal(z[key],old[key])
        low=d['noise'][query]<12
        for key in ('ridge','logridge'):assert np.array_equal(z[key][low],old[key][low])
        for noisy in (False,True):
            mask=(d['noise'][query]>=12)==noisy;members=[]
            for slot in b.SLOTS:
                for variant in b.VARIANTS:
                    for seed in range(3):
                        q=np.load(_repo_io(path.parent/f'tab-{int(noisy)}-{slot}-{variant}-seed{seed}.npz'))
                        assert np.array_equal(q['train'],tr) and np.array_equal(q['query'],query[mask])
                        assert int(q['seed'])==100*seed+contract['fold'];members.append(q['prediction'])
            assert np.array_equal(np.mean(members,axis=0),z['tab'][mask])
        details=json.loads(path.with_name('details.json').read_text())
        for row in details['linear']:
            for linear in [row['ridge'],*row['logridge']]:assert set(linear['source_ids'])<=set(d['ids'][tr])
        rebuilt=sum(b.WEIGHTS[k]*z[k] for k in b.WEIGHTS)/.84
        failed=np.nan_to_num(d['X'][query,list(d['keys']).index('n_obj')],nan=0)<20
        rebuilt[failed]=d['y'][tr].mean();assert np.array_equal(rebuilt,z['base'])
        records.append(dict(teacher=path.parent.name,all72_Tab_members_present=True,unchanged_components_exact=True,
            strict_source_scope=True,complete076_sum_exact=True,rms_shift=details['rms_shift_vs073']))
    p.save(_repo_io('teacher076-verification.json'),dict(n_complete=len(records),all_checked_passed=True,teachers=records))
    print('Verified complete076 teachers:',len(records),flush=True)


if __name__=='__main__':main()
