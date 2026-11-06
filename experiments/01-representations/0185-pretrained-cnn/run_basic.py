from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import subprocess,sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
assert json.loads((_repo_path(R, 'smoke.json')).read_text())['passed']
jobs=[('features.py',[n],n+'-extract') for n in ['resnet50','efficientnet_b0']]+[('train.py',['frozen',n,'original'],n+'-frozen-original') for n in ['resnet50','efficientnet_b0']]+[('train.py',['adapt',n,'original'],n+'-adapt-original') for n in ['resnet50','efficientnet_b0']]
for script,args,name in jobs:
 print(json.dumps({'starting':name}),flush=True)
 with (_repo_path(R, name+'.log')).open('x') as f:result=subprocess.run([str(_repo_path(P, '.venv/bin/python')),'-B','-u',str(_repo_path(R, 'launch.py')),script,*args],stdout=f,stderr=subprocess.STDOUT)
 print(json.dumps({'completed':name,'exit_code':result.returncode}),flush=True)
 if result.returncode:sys.exit(result.returncode)
