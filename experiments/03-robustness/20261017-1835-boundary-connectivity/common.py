"""Local-only paths, source graph and output protection for connectivity study."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys,json,hashlib,time,resource,socket
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];OLD=_repo_path(ROOT, 'experiments/highnoise-relational-20261016');PREV=_repo_path(ROOT, 'experiments/structure-context-20261017-1415')
sys.dont_write_bytecode=True
(_repo_path(HERE, 'tmp')).mkdir(exist_ok=True)
for name in ['TMPDIR','MPLCONFIGDIR','TORCHINDUCTOR_CACHE_DIR']:os.environ[name]=str(_repo_path(HERE, 'tmp'))
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TABPFN_DISABLE_TELEMETRY='1',TABPFN_MODEL_CACHE_DIR=str(_repo_path(HERE, 'tmp/tabpfn')))
def blocked(*a,**k):raise RuntimeError('No network in this local experiment')
socket.create_connection=blocked;socket.getaddrinfo=blocked
import numpy as np
from threadpoolctl import threadpool_limits
FIT_ACTIVE=False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(name,d):
    p=_repo_path(HERE, name);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def arrays(name,**d):
    p=_repo_path(HERE, name);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.savez_compressed(_repo_io(p),**d)
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:];writing=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not p.is_relative_to(HERE):raise RuntimeError('Outside new experiment write: '+str(p))
        if p.is_relative_to(_repo_path(ROOT, 'data/test')) or '-test.' in p.name or p.name.startswith('submission-candidate-') or p.name=='sample_submission.csv':raise RuntimeError('No test data in research')
        if FIT_ACTIVE and not writing and p.suffix in ['.npz','.npy','.pt','.csv','.png','.jpg','.joblib']:raise RuntimeError('Dataset access during isolated fitting: '+str(p))
def inputs(labels=False):
    z=np.load(_repo_path(OLD, 'inputs.npz'));keys=['ids','dev','fold','noise','failed']+(['y','base082','base076'] if labels else []);d={k:z[k] for k in keys};d['graphs']=json.loads((_repo_path(OLD, 'graphs.json')).read_text());return d
def memory():
    import torch
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;driver=torch.mps.driver_allocated_memory()
    assert driver<3.5*2**30 and rss+driver<5*2**30,('Own memory cap',rss,driver)
    return dict(RSS_bytes=rss,MPS_driver_bytes=driver)
