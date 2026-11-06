"""New outputs only; inherited predictors are immutable read-only inputs."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys,json,hashlib,ast
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];EXP=_RepoPath(__file__).resolve().parents[2]
PREV=_repo_path(EXP, 'model-audit-repair-20261017-2335');R4=_repo_path(EXP, 'rmse-research-20261015/round4-full073');OLD=_repo_path(EXP, '076-complement-20261016');TOPO=_repo_path(EXP, 'boundary-connectivity-20261017-1835')
sys.dont_write_bytecode=True
(_repo_path(HERE, 'tmp')).mkdir(exist_ok=True)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ.update(HF_HUB_OFFLINE='1',TABPFN_DISABLE_TELEMETRY='1')
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(name,d):
 p=_repo_path(HERE, name);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False,ensure_ascii=False)
def arrays(name,**d):
 p=_repo_path(HERE, name);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.savez_compressed(_repo_io(p),**d)
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:];write=(mode is not None and any(k in mode for k in 'wax+')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
  if write and not p.is_relative_to(HERE):raise RuntimeError('Outside experiment write: '+str(p))
  if not write and (p.is_relative_to(_repo_path(ROOT, 'data/test')) or '-test.' in p.name or p.name.startswith('submission-candidate-') or p.name=='sample_submission.csv'):raise RuntimeError('No test input during research: '+str(p))
def definitions(path,names,ns):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
