"""Matched320-source validation of the inference precision mismatch."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import reproduce as r
import numpy as np,torch,json,sys,time,gc

def main():
    torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.3)
    sys.path.insert(0,str(r.EXP/'push-300'));import harness
    d=r.nz(r.EXP/'rmse-research-20261015/inputs.npz');ids=d['ids'];dev=d['dev'];fold=d['fold'][0];y=d['y'];high=d['noise'][dev]>=12
    pred={arm:np.zeros(400) for arm in ['cpu','mps_auto','mps_float32']};historical=np.zeros(400);records=[];start=time.monotonic();folder=r.HERE/'precision-train';folder.mkdir(exist_ok=True)
    banks={}
    for gate in ['nb','full']:
        bank=r.nz(r.EXP/f'push-1300/abl/feat-h9-{gate}-038-u1dm1sz-train.npz');assert np.array_equal(bank['ids'],ids);banks[gate]=bank['X'].astype(float)
        old=r.nz(r.EXP/f'push-1300/abl/dev-h9-{gate}-038-u1dm1sz.npz');pos={str(i):j for j,i in enumerate(old['original_ids'])};order=[pos[str(i)] for i in ids[dev]]
        assert np.array_equal(old['original_fold'][order],fold)
        use=~high if gate=='nb' else high;historical[use]=old['original_g'][order][use]
    for k in range(5):
        tr=dev[fold!=k];ix=np.flatnonzero(fold==k);q=dev[ix]
        for gate in ['nb','full']:
            take=(~high[ix]) if gate=='nb' else high[ix]
            for arm in pred:
                vals=[]
                for s in range(3):
                    seed=100*s+k;path=folder/f'fold{k}-{gate}-{arm}-seed{s}.npz'
                    if path.exists():z=np.load(_repo_io(path));assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],q);pp=z['pred'];meta=json.loads(str(z['metadata']))
                    else:
                        device='cpu' if arm=='cpu' else 'mps';model=harness.tabpfn_new('v3.5',seed,device);model.set_params(memory_saving_mode=True)
                        if arm=='mps_float32':model.set_params(inference_precision=torch.float32)
                        tick=time.monotonic();model.fit(banks[gate][tr],y[tr].copy());pp=model.predict(banks[gate][q])
                        meta=dict(fold=k,gate=gate,arm=arm,seed=seed,seconds=time.monotonic()-tick,autocast=bool(model.use_autocast_),n_estimators=int(model.n_estimators_),forced_dtype=str(model.forced_inference_dtype_))
                        np.savez_compressed(_repo_io(path),train=tr,query=q,pred=pp,metadata=json.dumps(meta));del model;gc.collect();torch.mps.empty_cache()
                    vals.append(pp);records.append(meta)
                    if time.monotonic()-start>900:raise RuntimeError('Registered15min diagnostic limit; completed contexts retained')
                pred[arm][ix[take]]=np.mean(vals,0)[take]
            print('PRECISION TRAIN fold',k,'gate',gate,'calls',len(records),'/90','seconds',round(time.monotonic()-start,1),flush=True)
    target=y[dev];rng=np.random.default_rng(2026101638);draw=rng.integers(0,400,(10000,400));metrics={};base=pred['cpu']
    for arm,pp in pred.items():
        diff=pp-base;gain=np.sqrt(((base-target)**2)[draw].mean(1))-np.sqrt(((pp-target)**2)[draw].mean(1))
        metrics[arm]=dict(rmse=float(np.sqrt(np.mean((pp-target)**2))),difference_from_cpu_rms=float(np.sqrt(np.mean(diff**2))),difference_from_cpu_max=float(np.max(abs(diff))),
            cpu_minus_arm_rmse=float(np.sqrt(np.mean((base-target)**2))-np.sqrt(np.mean((pp-target)**2))),conditional_gain_ci95=np.percentile(gain,[2.5,97.5]).tolist(),
            difference_from_saved_legacy_max=float(np.max(abs(pp-historical))))
    r.pack('precision-train-oof.npz',ids=ids[dev],y=target,fold=fold,historical=historical,**pred)
    r.save(_repo_io('precision-train-results.json'),dict(all_complete=True,n_contexts=len(records),metrics=metrics,records=records,source_sha256=r.USED,seconds=time.monotonic()-start,
        registered_plan_sha256=r.sha(r.HERE/'PRECISION-DIAGNOSTIC.md'),code_sha256=r.sha(__file__),
        limits='One of12 Tab feature/variant banks, three seeds, same320-source fits. Diagnostic only, no candidate selection or full-ensemble RMSE claim. No independent holdout.'))
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':
    sys.addaudithook(r.guard)
    with r.threadpool_limits(limits=1):main()
