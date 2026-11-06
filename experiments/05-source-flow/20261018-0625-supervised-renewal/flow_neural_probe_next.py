from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, subprocess, time
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
start = time.monotonic()
while not (_repo_path(HERE, 'flow-full-block/results-f1.json')).exists():
    if time.monotonic()-start > 14400:
        raise RuntimeError('Earlier readouts did not finish; neural probe not started.')
    time.sleep(5)
with (_repo_path(HERE, 'flow-neural-probe.log')).open('w') as log:
    subprocess.run([str(_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python')), '-u', str(_repo_path(HERE, 'flow_neural_learning.py')), 'probe'],
                   cwd=_repo_io(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp'))),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
print('PROBE_DONE', flush=True)
