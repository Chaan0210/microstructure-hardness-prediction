"""Bounded dispatch of already specified checks, freeze, and local final fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
CUTOFF = datetime.datetime.fromisoformat('2026-10-18T19:30:00+00:00')
SUMMARY = _repo_path(HERE, 'source-h300/proper-summary-s1811-1823.json')
OUT = _repo_path(HERE, 'candidate-source-flow38-v1')
DECISION = _repo_path(HERE, 'source-final-dispatch-decision.json')
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp')))


def save(**value):
    DECISION.write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        full_system_protocol_passed=False, no_upload_or_final_selection_change=True, **value), indent=2)+'\n')


def run(python, script, args, name):
    print('START', name, flush=True)
    with (_repo_path(HERE, f'{name}.log')).open('w') as log:
        result = subprocess.run([str(_repo_path(ROOT, python)), '-u', str(_repo_path(HERE, script)), *args], cwd=_repo_io(ROOT),
                                env=ENV, stdout=log, stderr=subprocess.STDOUT)
    print('END', name, 'exit', result.returncode, flush=True)
    return result.returncode


def main():
    save(status='waiting_for_fixed_second_seed')
    while not SUMMARY.exists():
        if datetime.datetime.now(datetime.timezone.utc) >= CUTOFF:
            save(status='not_started', reason='04:30KST final-fit start cutoff reached; retain the verified globalF38 deliverable.')
            return
        time.sleep(10)
    code = run('.venv/bin/python','compare_current_best.py',['--seeds','1811','1823'],'source-mean2-current-best-comparison')
    if code:
        save(status='not_started', reason='Matched comparison did not complete; inspect its log, keep globalF38.')
        return
    code = run('.venv/bin/python','freeze_source_flow38_v2.py',['--candidate',str(OUT)],'source-candidate-freeze')
    if code:
        save(status='not_started', reason='Frozen final-fit resource checks or time limit did not pass; exact assertion preserved in source-candidate-freeze.log. No thresholds changed.')
        return
    save(status='frozen_final_fit_running', candidate=str(OUT.relative_to(HERE)))
    code = run('experiments/grain-cellpose-220/.venv/bin/python','fit_source_flow38.py',['--candidate',str(OUT)],'source-candidate-full-fit')
    save(status='complete' if code == 0 else 'fit_needs_recovery', candidate=str(OUT.relative_to(HERE)), fit_exit_code=code)


if __name__ == '__main__':
    main()
