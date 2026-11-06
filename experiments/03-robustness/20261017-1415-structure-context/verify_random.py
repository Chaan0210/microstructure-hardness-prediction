"""Verify the saved no-pretraining control descriptors."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import json,numpy as np,torch,sys,gc
def main():
    d=e.data();rows=[]
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V'])
            for repeat in range(2):
                path=e.HERE/f'encoders/fold{k}-{stage}-seed{repeat}-spatial';ck=torch.load(_repo_io(path/'initial.pt'),map_location='cpu',weights_only=True);r=json.loads((path/'training.json').read_text());m=e.ContextNet().to('mps');m.load_state_dict(ck['state_dict']);assert e.sh(m.state_dict())==r['initial_sha256'];f=np.load(_repo_io(path/'random-features.npz'));actual=e.features(m,d['pixels'][q],q,ck['normalization'],'random');error=float(np.max(abs(actual-f['query_features'])));assert error<2e-5
                spot=np.r_[np.arange(8),np.arange(len(tr)-8,len(tr))];actual=e.features(m,d['pixels'][tr[spot]],tr[spot],ck['normalization'],'random');terr=float(np.max(abs(actual-f['train_features'][spot])));assert terr<2e-5;rows.append(dict(fold=k,stage=stage,repeat=repeat,initial_state_exact=True,query_feature_max_error=error,train_spot_max_error=terr));del m;gc.collect();torch.mps.empty_cache()
    e.save(_repo_io('random-control-verification.json'),dict(all_passed=True,models=rows));print('RANDOM CONTROLS VERIFIED',len(rows))
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
