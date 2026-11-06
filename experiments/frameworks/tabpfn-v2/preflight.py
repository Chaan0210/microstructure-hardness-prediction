from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,os,socket,json,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;assert Path.cwd().resolve()==R and Path(sys.prefix).resolve()==_repo_path(R, '.venv') and not (_repo_path(R, '.env')).exists()
events=[]
def block(*a,**kw):events.append('network_connection_blocked');raise RuntimeError('Network prohibited during import/inference')
socket.create_connection=block;socket.socket.connect=block;socket.socket.connect_ex=block

def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if (p.name in ['.env','.netrc','.git-credentials','credentials','token','stored_tokens'] or any(v in p.parts for v in ['.aws','.ssh'])) and not p.is_relative_to(R):raise RuntimeError('Outside credential/dotenv access blocked')
sys.addaudithook(guard);start=time.perf_counter();report={'cwd':str(R),'existing_environment_untouched':True,'stage':'lightgbm_import','network_events':events}
try:
 import lightgbm
 report.update(lightgbm_version=lightgbm.__version__,lightgbm_import=True,seconds=time.perf_counter()-start,passed=True)
except Exception as e:
 report.update(passed=False,error_type=type(e).__name__,error=str(e),seconds=time.perf_counter()-start);traceback.print_exc()
with (_repo_path(R, 'preflight.json')).open('x') as fp:json.dump(report,fp,indent=2)
print(json.dumps(report),flush=True)
if not report['passed']:sys.exit(1)
