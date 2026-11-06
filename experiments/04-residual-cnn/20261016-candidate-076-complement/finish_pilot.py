"""Continue the authorized pilot sequentially once its teacher stage is complete."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]


def main():
    started=time.monotonic();marker=_repo_path(HERE, 'teachers076-complete-01.json')
    while not marker.exists():
        log=(_repo_path(HERE, 'retarget-teachers01-resumed.log')).read_text()
        if 'Traceback (most recent call last)' in log or 'KeyboardInterrupt' in log:
            raise RuntimeError('Teacher stage interrupted; retain caches and inspect the resumed log')
        if time.monotonic()-started>4500:raise TimeoutError('Teacher wait exceeded75min')
        time.sleep(5)
    assert json.loads(marker.read_text())['complete']
    commands=[('check-teachers-pilot.log','check_teachers.py',[]),
              ('retarget-cnn-pilot.log','retarget_cnn.py',['train','--folds','0','1']),
              ('retarget-evaluation-pilot.log','retarget_cnn.py',['evaluate','--folds','0','1'])]
    for log,script,args in commands:
        print('START',script,args,flush=True)
        with (_repo_path(HERE, log)).open('w') as output:
            subprocess.run([sys.executable,str(_repo_path(HERE, script)),*args],cwd=_repo_io(ROOT),stdout=output,stderr=subprocess.STDOUT,check=True)
        print('DONE',script,args,flush=True)
    (_repo_path(HERE, 'pilot-complete.json')).write_text(json.dumps(dict(complete=True,seconds=time.monotonic()-started))+'\n')


if __name__=='__main__':main()
