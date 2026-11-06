from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,json
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python')
for tag,args in [('original-probe',['--arm','original','--probe','2']),('original',['--arm','original']),('H300',['--arm','H300'])]:
    if tag=='original':
        r=json.loads((_repo_path(HERE, 'domain-semantics/original-probe-done.json')).read_text());assert r['peak_memory']<6*2**30 and max(r['original_descriptor_RMS_check'])<.001
    print('START',tag,flush=True)
    with (_repo_path(HERE, f'domain-semantics-{tag}.log')).open('w') as log:subprocess.run([str(python),'-u',str(_repo_path(HERE, 'domain_semantics.py')),*args,'--device','cpu'],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(_repo_path(HERE, 'tmp'))),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',tag,flush=True)
