"""Serial queue for this study's local GPU jobs only; finite wait and exit on error."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python')
start=time.monotonic();done=_repo_path(HERE, 'runs/phase-boundary-nll-f0-s89/done.json')
while not done.exists():
    if time.monotonic()-start>1800:raise RuntimeError('Prior study job did not complete; no new GPU job started.')
    time.sleep(5)
commands=[('deep-kernel-micronet-f1.log',['deep_kernel.py','train','--arm','micronet','--fold','1','--tag','deep-kernel-micronet-f1']),('dino-kernel-check-mps.log',['dino_kernel.py','check']),('dino-kernel-f0.log',['dino_kernel.py','train'])]
for log,args in commands:
    print('START',args,flush=True)
    with (_repo_path(HERE, log)).open('w') as stream:subprocess.run([str(python),'-u',str(_repo_path(HERE, args[0])),*args[1:]],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',args,flush=True)
