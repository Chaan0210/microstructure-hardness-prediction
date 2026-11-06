"""One predefined second encoder seed, only if evidence/time justify it."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, subprocess, time, json, datetime, fcntl
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
CUTOFF = datetime.datetime(2026, 10, 18, 15, 15, tzinfo=datetime.timezone.utc)  # Oct19 00:15KST
required = [_repo_path(HERE, 'source-h300/proper-original400-f4-s1811', name)
            for name in ['system-results.json', 'system38-results.json']]
while not all(p.exists() for p in required):
    if datetime.datetime.now(datetime.timezone.utc) >= CUTOFF:
        (_repo_path(HERE, 'source-second-seed-decision.json')).write_text(json.dumps(dict(started=False, reason='First source-local repetition did not finish before00:15KST; preserve final-fit/inference/recovery reserve.'), indent=2))
        raise SystemExit(0)
    time.sleep(5)
summary = _repo_path(HERE, 'source-h300/proper-summary-s1811.json')
if not summary.exists():
    with (_repo_path(HERE, 'source-summary-1811.log')).open('w') as log:
        subprocess.run([str(_repo_path(ROOT, '.venv/bin/python')), str(_repo_path(HERE, 'summarize_source_flow.py')), '--seeds', '1811'],
                       cwd=_repo_io(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=subprocess.STDOUT, check=True)
r = json.loads(summary.read_text())
gain = max(r['flow46']['gain'], r['flow38']['gain'])
decision = dict(started=gain >= .10, encoder_seed=2026101823, seed_suffix=1823, first_seed_gains={k:r[k]['gain'] for k in ['flow46','flow38']},
                allocation_rule='Continue one predefined second encoder seed if the completed source-local system gain is at least0.10 for either already-declared field variant and first repetition is ready before00:15KST. This is a resource-allocation rule, not a confirmation pass.',
                full_system_confirmation_passed=False, labels_reused=True)
(_repo_path(HERE, 'source-second-seed-decision.json')).write_text(json.dumps(decision, indent=2))
print('DECISION', json.dumps(decision), flush=True)
if not decision['started']:
    raise SystemExit(0)
jobs = [
    ('experiments/grain-cellpose-220/.venv/bin/python', 'source_h300_v2.py', ['train'], 'trained.json'),
    ('experiments/grain-cellpose-220/.venv/bin/python', 'source_h300_extract_v2.py', [], 'flow-complete.json'),
    ('experiments/tabpfn-v2/.venv/bin/python', 'source_h300_evaluate_v2.py', [], 'system-results.json'),
    ('experiments/tabpfn-v2/.venv/bin/python', 'source_h300_evaluate38.py', [], 'system38-results.json'),
]
for fold in range(5):
    tag = f'proper-original400-f{fold}-s1823'; out = _repo_path(HERE, 'source-h300', tag)
    for python, script, extra, marker in jobs:
        if (out/marker).exists():
            continue
        args = [*extra, '--tag', tag]
        if script == 'source_h300_v2.py':
            args += ['--protocol', 'original400', '--fold', str(fold), '--seed', '2026101823']
        print('START', tag, script, flush=True)
        with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            with (_repo_path(HERE, f'{tag}-{script[:-3]}.log')).open('w') as log:
                subprocess.run([str(_repo_path(ROOT, python)), '-u', str(_repo_path(HERE, script)), *args], cwd=_repo_io(ROOT),
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp'))),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        print('DONE', tag, script, flush=True)
    resume = out/'resume.pt'
    if resume.exists():
        size = resume.stat().st_size; resume.unlink()
        (out/'completed-optimizer-cleanup.json').write_text(json.dumps(dict(file='resume.pt', bytes=size,
            reason='Completed model, exact recipe, curves, feature caches and both head evaluations retained. Only redundant optimizer checkpoint removed for disk reserve.'), indent=2))
with (_repo_path(HERE, 'source-summary-mean2.log')).open('w') as log:
    subprocess.run([str(_repo_path(ROOT, '.venv/bin/python')), str(_repo_path(HERE, 'summarize_source_flow.py')), '--seeds', '1811', '1823'],
                   cwd=_repo_io(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=subprocess.STDOUT, check=True)
print('SECOND_SEED_COMPLETE', flush=True)
