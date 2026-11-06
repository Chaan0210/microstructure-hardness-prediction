from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,time,json,math
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python');start=time.monotonic()
while not all(p.exists() for p in [_repo_path(HERE, 'runs/pfn-head-f0/done.json'),_repo_path(HERE, 'scattering-c4/results.json')]):
    if time.monotonic()-start>3600:raise RuntimeError('Earlier study CPU jobs did not finish; no new training started.')
    time.sleep(5)
for action in ['probe','train']:
    if action=='train':
        p=json.loads((_repo_path(HERE, 'runs/soft-pfn-f0/probe/probe.json')).read_text());assert p['memory']<6*2**30 and p['seconds']<30 and p['gradient_norm']>0
    print('START',action,flush=True)
    with (_repo_path(HERE, f'soft-pfn-{action}-f0.log')).open('w') as log:subprocess.run([str(python),'-u',str(_repo_path(HERE, 'soft_pfn_learning.py')),action],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',action,flush=True)
