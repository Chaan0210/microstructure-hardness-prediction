from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import subprocess,sys,json
root=_RepoPath(__file__).resolve().parents[1]
jobs=[('cache','candidate010_pipeline.py',['cache']),('fit','candidate010_pipeline.py',['fit']),('predict','candidate010_pipeline.py',['predict']),('verification','verify_candidate010.py',[])]
for name,script,args in jobs:
 with (_repo_path(root, f'reports/candidate010-{name}.log')).open('x') as log:
  result=subprocess.run([sys.executable,'-I','-B',str(_repo_path(_repo_io(root), script)),*args],cwd=_repo_io(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  print(json.dumps({'failed_stage':name,'exit_code':result.returncode}),flush=True)
  raise SystemExit(result.returncode)
 print(json.dumps({'completed_stage':name}),flush=True)
