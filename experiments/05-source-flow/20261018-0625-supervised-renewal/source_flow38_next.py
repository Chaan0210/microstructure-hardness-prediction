from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, subprocess, time
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
start = time.monotonic()
while not (_repo_path(HERE, 'flow-block-stress/results.json')).exists():
    if time.monotonic()-start > 18000:
        raise RuntimeError('Earlier CPU readouts did not finish; secondary source heads not started.')
    time.sleep(5)
for fold in range(5):
    tag = f'proper-original400-f{fold}-s1811'
    while not (_repo_path(HERE, 'source-h300', tag, 'system-results.json')).exists():
        if time.monotonic()-start > 32400:
            raise RuntimeError('Primary source-local fold did not finish; secondary head not started.')
        time.sleep(5)
    print('START', tag, flush=True)
    with (_repo_path(HERE, f'{tag}-source_h300_evaluate38.log')).open('w') as log:
        subprocess.run([str(_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python')), '-u', str(_repo_path(HERE, 'source_h300_evaluate38.py')), '--tag', tag],
                       cwd=_repo_io(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp'))),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print('DONE', tag, flush=True)
