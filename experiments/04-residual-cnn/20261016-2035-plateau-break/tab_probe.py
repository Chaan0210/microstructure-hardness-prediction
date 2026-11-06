"""Isolate hardware vs query grouping in frozen-model reproduction only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import reproduce as r
import numpy as np,torch,sys,time,json,gc

def main():
    torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.3)
    sys.path.insert(0,str(r.EXP/'push-300'));import harness
    root=r.EXP/'push-1300';t=r.nz(root/'abl/feat-h9-nb-038-u1dm1sz-train.npz');q=r.nz(root/'abl/feat-h9-nb-038-u1dm1sz-test.npz')
    labels={v['ID']:float(v['hardness']) for v in r.csv.DictReader(r.track(_repo_path(r.ROOT, 'data/train.csv')).open())};y=np.array([labels[str(i)] for i in t['ids']]);active=r.nz(r.HERE/'incumbent-components.npz')['low']
    original=r.nz(r.HERE/'tab-replay/038-nb-u1dm1sz-0.npz')['pred'];begin=time.monotonic()
    force=r.os.environ.get('TAB_PROBE_FLOAT32')=='1';suffix='-float32' if force else ''
    model=harness.tabpfn_new('v3.5',0,'mps');model.set_params(memory_saving_mode=True)
    if force:model.set_params(inference_precision=torch.float32)
    model.fit(t['X'].astype(float),y)
    full=model.predict(q['X'].astype(float));sub=model.predict(q['X'].astype(float)[active])
    r.pack('tab-backend-query-probe'+suffix+'.npz',ids=q['ids'],active=active,cpu_full=original,mps_full=full,mps_active=sub)
    errors=dict(mps_vs_cpu_full_max=float(np.max(abs(full-original))),mps_subset_vs_mps_full_max=float(np.max(abs(full[active]-sub))),mps_subset_vs_cpu_full_max=float(np.max(abs(original[active]-sub))))
    r.save(_repo_io('tab-backend-query-probe'+suffix+'.json'),dict(errors=errors,seconds=time.monotonic()-begin,source_sha256=r.USED,model_random_state=0,parameters_changed=False,force_float32=force,autocast=getattr(model,'use_autocast_',None),forced_dtype=str(getattr(model,'forced_inference_dtype_',None)),n_estimators=int(model.n_estimators_)))
    print(errors,flush=True)

if __name__=='__main__':
    sys.addaudithook(r.guard)
    with r.threadpool_limits(limits=1):main()
