from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, subprocess, time
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
start = time.monotonic()
required = [_repo_path(HERE, 'flow-full-block/results-f1.json'), _repo_path(HERE, 'source-h300/proper-original400-f0-s1811/system38-results.json')]
while not all(p.exists() for p in required):
    if time.monotonic()-start > 18000:
        raise RuntimeError('Earlier readouts did not release CPU resources; frozen38 fitting not started.')
    time.sleep(5)
for action in ['fit', 'predict', 'replay']:
    print('START', action, flush=True)
    with (_repo_path(HERE, f'flow38-{action}.log')).open('w') as log:
        subprocess.run([str(_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python')), '-u', str(_repo_path(HERE, 'flow38_finalize.py')), action],
                       cwd=_repo_io(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp'))),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print('DONE', action, flush=True)
