"""Finish the source-local phase screen after the finite CPSAM long run."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,time,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];gpu_python=_repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python');cpu_python=_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python');start=time.monotonic();long=_repo_path(HERE, 'runs/cpsam-global-c4-f0-s11-e120')
while not ((_repo_path(long, 'done.json')).exists() or (_repo_path(long, 'stopped.json')).exists()):
    if time.monotonic()-start>14400:raise RuntimeError('Long GPU screen did not finish; no additional phase training started.')
    time.sleep(5)
for fold in [2,3,4]:
    base=f'phase-base-f{fold}-s81';new=f'phase-boundary-nll-f{fold}-s89'
    for python,log,args in [(gpu_python,f'phase-base-f{fold}.log',['phase_base_fold.py','--fold',str(fold),'--tag',base]),(gpu_python,f'phase-boundary-nll-f{fold}.log',['phase_boundary_general.py','--parent',base,'--tag',new]),(cpu_python,f'phase-production-{base}.log',['phase_production_readout.py','--run',base,'--fold',str(fold)]),(cpu_python,f'phase-production-{new}.log',['phase_production_readout.py','--run',new,'--fold',str(fold)])]:
        print('START',args,flush=True)
        with (_repo_path(HERE, log)).open('w') as stream:subprocess.run([str(python),'-u',str(_repo_path(HERE, args[0])),*args[1:]],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(_repo_path(HERE, 'tmp'))),stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('DONE',args,flush=True)
d=np.load(_repo_path(HERE, 'data.npz'));y=d['y'][d['dev']];fold=d['fold'];result={};pred={};out=_repo_path(HERE, 'phase-full');out.mkdir(exist_ok=True)
for kind in ['point','boundary']:
    v=np.zeros(400)
    for f in range(5):
        tag=('phase-joint-b4-f0-s81' if f==0 else f'phase-base-f{f}-s81') if kind=='point' else f'phase-boundary-nll-f{f}-s89'
        z=np.load(_repo_path(HERE, f'phase-production-{tag}/predictions.npz'));ii=np.flatnonzero(fold==f);assert np.array_equal(z['query'],d['dev'][ii]);assert np.array_equal(z['base083'],d['base083'][ii]);v[ii]=z['system']
    pred[kind]=v;loss=(v-y)**2;old=(d['base083']-y)**2;gain=old-loss;keep=np.ones(400,bool);keep[np.argsort(gain)[-10:]]=False;indices=np.random.default_rng(2026101971).integers(400,size=(5000,400));bs=np.sqrt(old[indices].mean(1))-np.sqrt(loss[indices].mean(1));result[kind]=dict(rmse=float(loss.mean()**.5),baseline_rmse=float(old.mean()**.5),gain=float(old.mean()**.5-loss.mean()**.5),fold_rmse=[float(loss[fold==f].mean()**.5) for f in range(5)],paired_id_bootstrap_ci95=np.percentile(bs,[2.5,97.5]).tolist(),top10_positive_contributors_removed_gain=float(old[keep].mean()**.5-loss[keep].mean()**.5),caveat='Reused400conditionalOOF,notfinalconfirmation;pairedIDbootstrapdoesnotremoveadaptiveoverfitting.')
np.savez(_repo_path(out, 'oof.npz'),ids=d['ids'][d['dev']],y=y,fold=fold,base083=d['base083'],**pred);(_repo_path(out, 'results.json')).write_text(json.dumps(result,indent=2));print('FULL',json.dumps(result),flush=True)
