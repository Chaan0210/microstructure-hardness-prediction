"""Fixed two-by-two screen, one local MPS child at a time."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent
arms=[('conditional',False),('conditional',True),('global',False),('global',True)]
start=time.monotonic();results=[]
for pooling,frozen in arms:
    name=f'cpsam-{pooling}-'+('frozen' if frozen else 'adapt')+'-f0-s11-e40'
    path=_repo_path(HERE, 'runs', name, 'done.json')
    if not path.exists():
        command=[sys.executable,'-B','-u',str(_repo_path(HERE, 'cpsam_train_v1.py')),'--tag',name,'--pooling',pooling]
        if frozen:command.append('--frozen')
        print('LAUNCH',name,flush=True)
        with (_repo_path(HERE, f'{name}.log')).open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=_repo_io(_RepoPath(__file__).resolve().parents[3]))
    r=json.loads(path.read_text());results.append(dict(name=name,**r));print('COMPLETE',json.dumps(results[-1]),flush=True)
(_repo_path(HERE, 'cpsam-screen-results.json')).write_text(json.dumps(dict(results=results,seconds=time.monotonic()-start),indent=2))
