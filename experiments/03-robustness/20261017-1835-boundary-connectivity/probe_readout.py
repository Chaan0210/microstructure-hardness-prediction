"""Check the fixed144-column reader's memory before the full unscored matrix."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import readout as r
import numpy as np,json,sys,time,gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
def main():
    assert json.loads((c.HERE/'encoder-verification.json').read_text())['all_passed'];d=r.physical();start=time.monotonic();rows=[]
    for stage in ['selector','refit']:
        g=d['graphs'][0];tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);f=np.load(_repo_io(c.HERE/f'encoders/fold0-{stage}-seed0-topology/features.npz'));sc=StandardScaler().fit(f['train_latent']);pc=PCA(n_components=16,svd_solver='full').fit(sc.transform(f['train_latent']));X=np.c_[d['X'][tr],f['train_signature'],pc.transform(sc.transform(f['train_latent']))];Q=np.c_[d['X'][q],f['query_signature'],pc.transform(sc.transform(f['query_latent']))];path=c.HERE/f'readout-resource-probe/{stage}';path.mkdir(parents=True,exist_ok=False);c.arrays(str((path/'context.npz').relative_to(c.HERE)),X=X,y=d['y'][tr],Q=Q,train=tr,query=q)
        try:p=r.predict_tab(X,d['y'][tr],Q,0,path,tr,q)
        except (AssertionError,RuntimeError) as error:
            c.save(_repo_io('readout-resource-probe.json'),dict(passed=False,stage=stage,error=repr(error),seconds=time.monotonic()-start,hardness_accuracy_not_computed=True,readout_source_sha256=c.sha(r.__file__)));raise
        rows.append(dict(stage=stage,train=len(tr),query=len(q),columns=X.shape[1],finite=bool(np.isfinite(p).all()),contract=json.loads((path/'contract.json').read_text())));print('READOUT RESOURCE',stage,'PASS',flush=True)
    c.save(_repo_io('readout-resource-probe.json'),dict(passed=True,contexts=rows,seconds=time.monotonic()-start,hardness_accuracy_not_computed=True,readout_source_sha256=c.sha(r.__file__),scope='Two identical planned seed0 topology contexts; resource check only. Main matrix repeats them without using accuracy for any choice.'))
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
