"""Resource-ordered local continuation, bounded by the predeclared window."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, subprocess, time
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
start = time.monotonic()
while not (_repo_path(HERE, 'flow-geometry/full-done.json')).exists():
    if time.monotonic()-start > 3600:
        raise RuntimeError('Geometry extraction did not finish; no dependent work started.')
    time.sleep(5)
jobs = [
    ('experiments/tabpfn-v2/.venv/bin/python', 'flow_geometry_readout.py', ['--folds', '0', '1'], 'flow-geometry-readout.log'),
    ('experiments/grain-cellpose-220/.venv/bin/python', 'cpsam_long_rotation.py', ['--tag', 'cpsam-global-c4-f0-s11-e120', '--epochs', '120'], 'cpsam-final-resume.log'),
    ('experiments/tabpfn-v2/.venv/bin/python', 'semantic_pfn_learning.py', ['train', '--tag', 'semantic-pfn-f0-c160', '--context', '160'], 'semantic-pfn-f0-c160.log'),
]
for python, script, args, logfile in jobs:
    print('START', script, args, flush=True)
    with (_repo_path(HERE, logfile)).open('w') as log:
        subprocess.run([str(_repo_path(ROOT, python)), '-u', str(_repo_path(HERE, script)), *args], cwd=_repo_io(ROOT),
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp'))),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print('DONE', script, flush=True)
