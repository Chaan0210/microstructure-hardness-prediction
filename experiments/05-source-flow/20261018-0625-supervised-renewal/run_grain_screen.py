from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import subprocess,sys,json,time
HERE=Path(__file__).resolve().parent
start=time.monotonic();result=[]
for kind,fold in [('attention',0),('deepset',0),('attention',1),('deepset',1),('attention',2),('attention',3),('attention',4)]:
    tag=f'grain-{kind}-v2-f{fold}-s41';done=_repo_path(HERE, 'runs', tag, 'done.json')
    if not done.exists():
        with (_repo_path(HERE, f'{tag}.log')).open('a') as log:
            subprocess.run([sys.executable,'-B','-u',str(_repo_path(HERE, 'grain_attention_v2.py')),'--tag',tag,'--kind',kind,'--fold',str(fold)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=_repo_io(_RepoPath(__file__).resolve().parents[3]))
    r=dict(kind=kind,fold=fold,**json.loads(done.read_text()));result.append(r);print(json.dumps(r),flush=True)
(_repo_path(HERE, 'grain-screen-results.json')).write_text(json.dumps(dict(results=result,seconds=time.monotonic()-start),indent=2))
