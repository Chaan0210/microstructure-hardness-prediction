"""Frozen eligible encoders, raw versus source-local affine contrast normalization."""
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
import numpy as np,torch,json,time,hashlib,gc
import micronet_direct as direct

def main():
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.35)
    d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');photo=[];contrast=[]
    for im in px[:,0]:
        x=im.astype(np.float32)/255.;lo,hi=np.percentile(x,[1,99]);span=max(float(hi-lo),1/255)
        contrast.append(((x-float(lo))/span).astype(np.float32));photo.append([float(x.mean()),float(x.std()),float((x**3).mean()),float((x**4).mean()),float(lo),float(hi)])
    contrast=np.array(contrast)[:,None];photo=np.array(photo)
    out=_repo_path(HERE, 'microscopy-contrast');out.mkdir(exist_ok=True);np.savez(_repo_io(out/'photometry.npz'),ids=d['ids'],X=photo)
    mean=torch.tensor([.485,.456,.406],device='mps')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='mps')[None,:,None,None]
    for arm in ['micronet','imagenet']:
        if (out/f'{arm}.npz').exists():continue
        model,provenance=direct.backbone(arm);model=model.to('mps').eval();result={};start=time.monotonic()
        with torch.no_grad():
            for kind in ['raw','contrast']:
                rows=[]
                for first in range(0,500,8):
                    a=np.array(px[first:first+8],dtype=np.float32)/255. if kind=='raw' else contrast[first:first+8]
                    x=torch.tensor(a,device='mps');ff=[]
                    for rot in [0,1,2,3]:
                        z=torch.rot90(x,rot,(2,3));ff.append(model((z.expand(-1,3,-1,-1)-mean)/std).cpu().numpy())
                    rows.append(np.mean(ff,0))
                result[kind]=np.concatenate(rows);print(arm,kind,'seconds',round(time.monotonic()-start,1),flush=True)
        np.savez(_repo_io(out/f'{arm}.npz'),ids=d['ids'],photo=photo,**result)
        (out/f'{arm}-provenance.json').write_text(json.dumps(dict(weight=provenance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Unmodified eligible external pretrained weights, eval/no-grad; onlytrainoriginals;nolearnedacross-imagepreprocessing.',input='Native256,C4mean. Contrast=(original-p01)/max(p99-p01,1gray),unclipped;originalmean/std/rawmoments3/4/p01/p99retainedas6scalarphotometryfeatures.',seconds=time.monotonic()-start),indent=2))
        del model;gc.collect();torch.mps.empty_cache()

if __name__=='__main__':main()
