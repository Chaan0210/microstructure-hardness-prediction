from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python')
def wait_for(paths,timeout=3600):
    start=time.monotonic()
    while not all(p.exists() for p in paths):
        if time.monotonic()-start>timeout:raise RuntimeError('Earlier study jobs did not complete; no dependent readout started.')
        time.sleep(5)
wait_for([_repo_path(HERE, 'soft-measurement-readout/results.json'),_repo_path(HERE, 'cpsam-conditional-pool/results.json')])
for tag,fold in [('phase-base-f1-s81',1),('phase-boundary-nll-f1-s89',1),('phase-moments-f0-s89',0)]:
    wait_for([_repo_path(HERE, 'runs', tag, 'done.json')]);print('START',tag,flush=True)
    with (_repo_path(HERE, f'phase-production-{tag}.log')).open('w') as log:subprocess.run([str(python),'-u',str(_repo_path(HERE, 'phase_production_readout.py')),'--run',tag,'--fold',str(fold)],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',tag,flush=True)
