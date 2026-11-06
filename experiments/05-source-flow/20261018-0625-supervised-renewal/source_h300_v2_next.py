"""Five fixed original400 folds; one encoder seed, paired native heads."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, subprocess, time, json
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
start = time.monotonic()
while not (_repo_path(HERE, 'runs/semantic-pfn-f0-c160/done.json')).exists():
    if time.monotonic()-start > 10800:
        raise RuntimeError('Earlier memory-heavy job did not complete; auxiliary fitting not started.')
    time.sleep(5)
jobs = [
    ('experiments/grain-cellpose-220/.venv/bin/python', 'source_h300_v2.py', ['train']),
    ('experiments/grain-cellpose-220/.venv/bin/python', 'source_h300_extract_v2.py', []),
    ('experiments/tabpfn-v2/.venv/bin/python', 'source_h300_evaluate_v2.py', []),
]
for fold in range(5):
    tag = f'proper-original400-f{fold}-s1811'
    for python, script, extra in jobs:
        args = [*extra, '--tag', tag]
        if script == 'source_h300_v2.py':
            args += ['--protocol', 'original400', '--fold', str(fold), '--seed', '2026101811']
        print('START', tag, script, flush=True)
        with (_repo_path(HERE, f'{tag}-{script[:-3]}.log')).open('w') as log:
            subprocess.run([str(_repo_path(ROOT, python)), '-u', str(_repo_path(HERE, script)), *args], cwd=_repo_io(ROOT),
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp'))),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('DONE', tag, script, flush=True)
    # The final model and exact training recipe remain. The completed optimizer
    # checkpoint is redundant for inference/reproduction and is the only file
    # removed, to keep the next fold's atomic checkpoint above the disk reserve.
    out = _repo_path(HERE, 'source-h300', tag)
    assert (out/'trained.json').exists() and (out/'flow-complete.json').exists() and (out/'system-results.json').exists()
    resume = out/'resume.pt'
    if resume.exists():
        size = resume.stat().st_size
        resume.unlink()
        (out/'completed-optimizer-cleanup.json').write_text(json.dumps(dict(file='resume.pt', bytes=size, reason='Completed model.pt, recipe, curves and outputs retained; no further optimizer continuation required. Reserve disk for next source-local fold.'), indent=2))
print('SOURCE_FOLDS_DONE', flush=True)
