"""Train/infer only after the final cross-fitted teacher stage has closed."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import subprocess
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[4]


def run(script,args,log):
    print('START',script,args,flush=True)
    with (_repo_path(HERE, log)).open('a') as out:
        subprocess.run([str(_repo_path(ROOT, '.venv/bin/python')),'-u',str(_repo_path(HERE, script)),*args],cwd=_repo_io(ROOT),
                       stdout=out,stderr=subprocess.STDOUT,check=True)
    print('DONE',script,args,flush=True)


if __name__=='__main__':
    start=time.monotonic()
    while not (_repo_path(HERE, 'final500/teachers-complete.json')).exists():
        if time.monotonic()-start>7200:raise TimeoutError('Final teacher stage not complete')
        time.sleep(10)
    run('final_model.py',['fit'],'final-model-fit.log')
    run('final_model.py',['infer'],'final-model-infer.log')
    run('verify_final.py',[],'final-independent-verification.log')
    (_repo_path(HERE, 'final500/pipeline-complete.json')).write_text(json.dumps(dict(complete=True,seconds=time.monotonic()-start,uploaded=False),indent=2)+'\n')
    print('FINAL500 CANDIDATES COMPLETE AND VERIFIED',flush=True)
