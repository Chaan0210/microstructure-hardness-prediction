"""Run only from this isolated experiment with a credential-free environment."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
assert Path.cwd().resolve()==R and not (_repo_path(R, '.env')).exists()
allowed=['preflight.py','run.py','validate.py']
assert len(sys.argv)>1 and sys.argv[1] in allowed
for name in ['cache','cache/huggingface','cache/tabpfn','cache/matplotlib','config','tmp']: (_repo_path(R, name)).mkdir(parents=True,exist_ok=True)
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_DATASETS_OFFLINE':'1','HF_HOME':str(_repo_path(R, 'cache/huggingface')),'HF_HUB_CACHE':str(_repo_path(R, 'cache/huggingface/hub')),'HF_TOKEN_PATH':str(_repo_path(R, 'cache/huggingface/no-token')),'TABPFN_MODEL_CACHE_DIR':str(_repo_path(R, 'cache/tabpfn')),'TABPFN_MODEL_VERSION':'v2','TABPFN_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','XDG_CACHE_HOME':str(_repo_path(R, 'cache')),'XDG_CONFIG_HOME':str(_repo_path(R, 'config')),'MPLCONFIGDIR':str(_repo_path(R, 'cache/matplotlib')),'TORCH_HOME':str(_repo_path(R, 'cache/torch')),'JOBLIB_TEMP_FOLDER':str(_repo_path(R, 'tmp')),'TMPDIR':str(_repo_path(R, 'tmp')),'PYTHONNOUSERSITE':'1'}
res=subprocess.run([str(_repo_path(R, '.venv/bin/python')),'-B','-u',*sys.argv[1:]],cwd=_repo_io(R),env=env)
sys.exit(res.returncode)
