from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import subprocess,sys,json
root=_RepoPath(__file__).resolve().parents[1]
def job(name,script,args):
 with (_repo_path(root, f'reports/dinob-lastblock-083-{name}.log')).open('x') as log:
  p=subprocess.run([sys.executable,'-I','-B',str(_repo_path(_repo_io(root), script)),*args],cwd=_repo_io(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps({'stage':name,'exit_code':p.returncode}),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
job('benchmark','dinob_lastblock_083.py',['benchmark-fold'])
job('original','dinob_lastblock_083.py',['original'])
s=json.loads((_repo_path(root, 'artifacts/dinob-lastblock-083/selection.json')).read_text())
if s['advance']:
 for split in ['20261031','20261101']:job(split,'dinob_lastblock_083.py',[split])
job('validation','validate_dinob_lastblock_083_v2.py',[])
