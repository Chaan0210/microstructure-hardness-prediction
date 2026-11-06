"""Run only the locked replay/resplit checks after the two teacher extensions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
sys.dont_write_bytecode=True
import json
import subprocess
import time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]


if __name__=='__main__':
    start=time.monotonic()
    while True:
        path=_repo_path(H, 'extension-verification.json')
        if path.exists() and len(json.loads(path.read_text()))==2:break
        if 'Traceback (most recent call last)' in (_repo_path(H, 'extensions.log')).read_text():raise RuntimeError('Teacher extension failed; inspect it before proceeding')
        if time.monotonic()-start>3600:raise TimeoutError('Pre-registered extension budget exhausted')
        time.sleep(5)
    for mode in ('replay','resplit'):
        print('START',mode,flush=True)
        with (_repo_path(H, f'{mode}.log')).open('w') as log:
            subprocess.run([str(_repo_path(ROOT, '.venv/bin/python')),'-u',str(_repo_path(H, 'replay.py')),mode],cwd=_repo_io(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',mode,flush=True)
    (_repo_path(H, 'checks-complete.json')).write_text(json.dumps(dict(completed=True,seconds=time.monotonic()-start))+'\n')
