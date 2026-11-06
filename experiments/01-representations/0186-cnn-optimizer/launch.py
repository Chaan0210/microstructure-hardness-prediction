from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
assert sys.argv[1] in ['run.py','validate.py','summarize.py']
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONNOUSERSITE':'1','XDG_CACHE_HOME':str(_repo_path(R, 'cache')),'XDG_CONFIG_HOME':str(_repo_path(R, 'config')),'TORCH_HOME':str(_repo_path(R, 'cache/torch')),'TMPDIR':str(_repo_path(R, 'tmp')),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','PYTORCH_MPS_HIGH_WATERMARK_RATIO':'0.5','PYTORCH_MPS_LOW_WATERMARK_RATIO':'0.4'}
r=subprocess.run([str(_repo_path(R.parent, 'pretrained-cnn-185/.venv/bin/python')),'-B','-u',*sys.argv[1:]],cwd=_repo_io(R),env=env);sys.exit(r.returncode)
