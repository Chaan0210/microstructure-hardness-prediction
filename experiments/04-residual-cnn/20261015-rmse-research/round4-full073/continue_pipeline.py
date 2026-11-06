"""Continue the authorized experiment serially after all teacher caches close."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import subprocess
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=_RepoPath(__file__).resolve().parents[4]
PY=_repo_path(ROOT, '.venv/bin/python')


def run(script,args,log):
    with (_repo_path(HERE, log)).open('a') as out:
        print('START',script,args,flush=True)
        subprocess.run([str(PY),'-u',str(_repo_path(HERE, script)),*args],cwd=_repo_io(ROOT),stdout=out,stderr=subprocess.STDOUT,check=True)
    print('DONE',script,args,flush=True)


def main():
    start=time.monotonic();last=-1
    while True:
        complete=list((_repo_path(HERE, 'teachers')).glob('*/details.json'))
        if len(complete)!=last:
            last=len(complete);print('WAITING FOR TEACHERS',last,'/30',flush=True)
        if len(complete)==30:break
        log=(_repo_path(HERE, 'teachers.log')).read_text()
        if 'Traceback (most recent call last)' in log:raise RuntimeError('Teacher process failed; inspect teachers.log')
        if time.monotonic()-start>6*3600:raise TimeoutError('Teacher stage exceeded six hours')
        time.sleep(10)
    run('verify.py',[],'teacher-verification.log')
    for name,args in [('cnn',[]),('stats',['--stats']),('aux',['--aux'])]:
        run('cnn_experiment.py',args,f'{name}-seed0.log')
        run('verify.py',['--folder',str(_repo_path(HERE, f'{name}-seed0')),'--poison'],f'{name}-seed0-verification.log')
    repeated=[]
    for name,args in [('cnn',[]),('aux',['--aux'])]:
        result=json.loads((_repo_path(HERE, f'{name}-seed0/summary.json')).read_text())
        if result['addition']['gain']>0 or result['replacement_vs_deletion']['gain']>0:
            run('cnn_experiment.py',[*args,'--repeat','1'],f'{name}-seed1.log')
            run('verify.py',['--folder',str(_repo_path(HERE, f'{name}-seed1'))],f'{name}-seed1-verification.log')
            repeated.append(name)
    record=dict(completed=True,repeated=repeated,seconds=time.monotonic()-start,
                teacher_count=30,test_read=False,submission_modified=False)
    (_repo_path(HERE, 'pipeline-completion.json')).write_text(json.dumps(record,indent=2)+'\n')
    print('PIPELINE COMPLETE',json.dumps(record),flush=True)


if __name__=='__main__':main()
