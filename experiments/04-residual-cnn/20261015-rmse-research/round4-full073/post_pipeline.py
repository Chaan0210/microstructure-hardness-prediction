"""Serial pre-registered controls and reporting after the neural seed pipeline."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import subprocess
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[4]


def run(name,args=(),tabpfn=False):
    py=_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python' if tabpfn else '.venv/bin/python')
    print('START',name,list(args),flush=True)
    with (_repo_path(HERE, f'post-{name}.log')).open('a') as log:
        subprocess.run([str(py),'-u',str(_repo_path(HERE, name)),*map(str,args)],cwd=_repo_io(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,list(args),flush=True)


def main():
    start=time.monotonic()
    while not (_repo_path(HERE, 'pipeline-completion.json')).exists():
        if time.monotonic()-start>7200:raise TimeoutError('Neural confirmation not complete')
        time.sleep(10)
    aux=json.loads((_repo_path(HERE, 'aux-seed0/summary.json')).read_text())
    if aux['addition']['gain']>0 or aux['replacement_vs_deletion']['gain']>0:
        run('cnn_experiment.py',['--context-only'])
        run('verify.py',['--folder',_repo_path(HERE, 'context-seed0'),'--poison'])
    run('spotcheck.py',tabpfn=True)
    run('frozen_heads.py')
    run('seed_mean.py')
    run('adaptive_weights.py')
    run('report.py')
    (_repo_path(HERE, 'post-completion.json')).write_text(json.dumps(dict(completed=True,seconds=time.monotonic()-start),indent=2)+'\n')
    print('ALL PRESPECIFIED EXPERIMENTS AND CHECKS COMPLETE',flush=True)


if __name__=='__main__':main()
