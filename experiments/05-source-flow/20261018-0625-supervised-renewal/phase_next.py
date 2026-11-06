"""One source-local confirmation screen after this study's DINO GPU job."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python')
start=time.monotonic()
while not (_repo_path(HERE, 'runs/dino-kernel-f0/done.json')).exists():
    if time.monotonic()-start>5400:raise RuntimeError('Study queue did not complete; no new GPU work started.')
    time.sleep(5)
for log,args in [('phase-base-f1.log',['phase_base_fold.py','--fold','1','--tag','phase-base-f1-s81']),('phase-boundary-nll-f1.log',['phase_boundary_general.py','--parent','phase-base-f1-s81','--tag','phase-boundary-nll-f1-s89'])]:
    print('START',args,flush=True)
    with (_repo_path(HERE, log)).open('w') as stream:subprocess.run([str(python),'-u',str(_repo_path(HERE, args[0])),*args[1:]],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',args,flush=True)
