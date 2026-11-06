from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,subprocess,json,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];name=sys.argv[1];assert name in ['infer','analyze']
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','XDG_CACHE_HOME':str(_repo_path(R, 'cache')),'XDG_CONFIG_HOME':str(_repo_path(R, 'config')),'TMPDIR':str(_repo_path(R, 'tmp')),'TORCH_HOME':str(_repo_path(R, 'cache/torch')),'HF_HUB_OFFLINE':'1','DO_NOT_TRACK':'1','PYTORCH_MPS_HIGH_WATERMARK_RATIO':'0.5','PYTORCH_MPS_LOW_WATERMARK_RATIO':'0.4'}
with (_repo_path(R, name+'.log')).open('x') as f:
 p=subprocess.Popen([str(_repo_path(P, '.venv/bin/python')),'-B','-u',str(_repo_path(R, name+'.py'))],cwd=_repo_io(R),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
 code=p.wait()
with (_repo_path(R, name+'-exit.json')).open('x') as f:json.dump({'exit_code':code,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
sys.exit(code)
