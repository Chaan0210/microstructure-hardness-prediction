from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(_repo_path(HERE, 'tmp')))
for python,script,args,log in [(_repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python'),'flow_test_inference.py',[],'flow-test-inference.log'),(_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python'),'flow_candidate_finalize.py',['predict'],'flow-candidate-predict.log'),(_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python'),'flow_candidate_finalize.py',['replay'],'flow-candidate-replay.log')]:
 print('START',script,args,flush=True)
 with (_repo_path(HERE, log)).open('w') as stream:subprocess.run([str(python),'-u',str(_repo_path(HERE, script)),*args],cwd=_repo_io(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,args,flush=True)
