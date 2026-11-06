from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python');start=time.monotonic()
while not (_repo_path(HERE, 'runs/phase-boundary-nll-f1-s89/done.json')).exists():
    if time.monotonic()-start>5400:raise RuntimeError('Earlier study GPU jobs did not finish; no moment training started.')
    time.sleep(5)
with (_repo_path(HERE, 'phase-moments-f0.log')).open('w') as log:subprocess.run([str(python),'-u',str(_repo_path(HERE, 'phase_moments.py'))],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
print('DONE',flush=True)
