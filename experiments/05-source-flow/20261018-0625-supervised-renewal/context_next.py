from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,json
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, '.venv/bin/python');p=json.loads((_repo_path(HERE, 'runs/context-supervised-f0-pretrained/probe/probe.json')).read_text());assert p['projected_selector_refit_200_seconds']<1800 and p['peak_memory']<2*2**30 and p['optimizer_updates']==0
for arm in ['pretrained','random']:
    print('START',arm,flush=True)
    with (_repo_path(HERE, f'context-supervised-f0-{arm}.log')).open('w') as log:subprocess.run([str(python),'-u',str(_repo_path(HERE, 'context_supervised.py')),'train','--arm',arm,'--tag',f'context-supervised-f0-{arm}'],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(_repo_path(HERE, 'tmp'))),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',arm,flush=True)
