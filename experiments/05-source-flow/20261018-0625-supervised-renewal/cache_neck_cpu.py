"""Fixed pretrained spatial maps for a higher-resolution supervised readout."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import json,time,gc
import numpy as np
import torch
from cpsam_train_v1 import Tail,read_prefix
from threadpoolctl import threadpool_limits

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    initial=torch.load(_repo_path(HERE, 'cpsam-tail-initial.pt'),map_location='cpu',weights_only=True);model=Tail(initial).eval();del initial
    out=_repo_path(HERE, 'cpsam-neck-c4');out.mkdir(exist_ok=True);path=out/'maps.npy';shape=(500,2,256,32,32)
    maps=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float16,shape=shape)
    done=np.load(_repo_io(out/'done.npy')) if (out/'done.npy').exists() else np.zeros(500,bool);start=time.monotonic()
    with torch.no_grad():
        for j in range(500):
            if done[j]:continue
            x=torch.tensor(read_prefix(_repo_path(HERE, 'cpsam-prefix/features.npy'),np.array([j,j]),np.array([0,1])))
            maps[j]=model.spatial(x).numpy().astype(np.float16);maps.flush();done[j]=True;np.save(_repo_io(out/'done.npy'),done)
            if (j+1)%25==0:
                del maps;gc.collect();maps=np.load(_repo_io(path),mmap_mode='r+');print('CACHED',j+1,'seconds',round(time.monotonic()-start,1),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(complete=bool(done.all()),shape=shape,views=['original','rot90'],seconds=time.monotonic()-start,scope='Only fixed eligible pretrained inference on train images; no label/representation fit.'),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
