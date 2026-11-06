from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import subprocess,sys,json
root=_RepoPath(__file__).resolve().parents[1]
for stage,script,args in [('prepare','frozen_complement_085.py',['prepare']),*[(s,'frozen_complement_085.py',[s]) for s in ('original','20261031','20261101')],('validation','validate_frozen_complement_085.py',[])]:
 with (_repo_path(root, f'reports/frozen-complement-085-{stage}.log')).open('x') as log:r=subprocess.run([sys.executable,'-I','-B',str(_repo_path(_repo_io(root), script)),*args],cwd=_repo_io(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps({'stage':stage,'exit_code':r.returncode}),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
