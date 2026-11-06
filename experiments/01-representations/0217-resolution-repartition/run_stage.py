from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import sys,subprocess,json,datetime
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];stage=sys.argv[1]
assert stage in ['preflight','study','validate','score','finalize']
code="import sys,runpy;sys.path.insert(0,"+repr(str(R))+");runpy.run_path("+repr(str(_repo_path(R, stage+'.py')))+",run_name='__main__')"
with (_repo_path(R, stage+'.log')).open('x') as log:
 p=subprocess.Popen([str(_repo_path(P, '.venv/bin/python')),'-I','-B','-c',code],cwd=_repo_io(P),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
 rc=p.wait()
if rc:
 with (_repo_path(R, stage+'-failure.json')).open('x') as f:json.dump({'stage':stage,'returncode':rc,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
sys.exit(rc)
