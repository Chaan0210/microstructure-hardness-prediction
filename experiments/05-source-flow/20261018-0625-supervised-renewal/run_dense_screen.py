from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;start=time.monotonic();results=[]
for kind,fold in [('fusion',1),('context',0),('raw',0),('fusion',2),('fusion',3),('fusion',4)]:
    tag=f'dense-{kind}-f{fold}-s73';done=_repo_path(HERE, 'runs', tag, 'done.json')
    if not done.exists():
        with (_repo_path(HERE, f'{tag}.log')).open('a') as log:subprocess.run([sys.executable,'-B','-u',str(_repo_path(HERE, 'dense_fusion_v2.py')),'--tag',tag,'--kind',kind,'--fold',str(fold)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=_repo_io(_RepoPath(__file__).resolve().parents[3]))
    r=dict(kind=kind,fold=fold,**json.loads(done.read_text()));results.append(r);print(json.dumps(r),flush=True)
(_repo_path(HERE, 'dense-screen-results.json')).write_text(json.dumps(dict(results=results,seconds=time.monotonic()-start),indent=2))
