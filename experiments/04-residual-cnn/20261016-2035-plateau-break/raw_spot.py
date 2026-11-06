"""Bounded raw-pixel saved-weight z1 replay; no fitting or hardness labels."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import reproduce as r
import ast,json,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def main():
    source=r.EXP/'push-400/phase_unet4.py';tree=ast.parse(r.track(source).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['blk','UNet','norm']]
    ns=dict(torch=torch,nn=nn,F=F,np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    net=ns['UNet']();net.load_state_dict(torch.load(_repo_io(r.track(r.EXP/'push-400/unet4-z1.pt')),map_location='cpu',weights_only=True));net.eval();net.requires_grad_(False)
    source2=r.EXP/'push-400/unet4_feats.py';tree=ast.parse(r.track(source2).read_text());inv=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inv');exec(compile(ast.Module(body=[inv],type_ignores=[]),str(source2),'exec'),ns)
    selected={};records=[]
    for split in ['train','test']:
        z=r.nz(r.EXP/f'push-400/unet4maps-z1-{split}.npz');ids=z['ids'];maps=z['maps']
        # IDs fixed by index, not by targets/errors or the newly scored results.
        for j in range(3):
            iid=str(ids[j]);a=np.asarray(Image.open(_repo_io(r.track(_repo_path(r.ROOT, f'data/{split}/{iid}.png')))),np.float32);x=ns['norm'](a);acc=np.zeros((8,256,256))
            with torch.inference_mode():
                for k in range(8):
                    v=x
                    if k&1:v=v.T
                    if k&2:v=v[::-1]
                    if k&4:v=v[:,::-1]
                    o=net(torch.from_numpy(np.ascontiguousarray(v))[None,None]);pr=o[:,:4].softmax(1);rg=o[:,4:].clone();rg[:,0]=rg[:,0]*.9691801071166992+5.636784553527832;rg[:,1:3]*=.3;rg[:,3]=rg[:,3]*.15+.15
                    acc+=ns['inv'](torch.cat([pr,rg],1).numpy()[0],k)
            p=acc/8;p=p[:,::2,::2];ref=maps[j].astype(float);err=np.max(abs(p-ref),axis=(1,2))
            # Saved maps are float16, while replay is accumulated float64 over
            # float32 CPU outputs. Allow one half-precision ULP plus 2e-5 CPU drift.
            ulp=np.spacing(np.maximum(abs(maps[j]),np.float16(2**-14))).astype(float)
            bound=ulp+2e-5;assert np.all(abs(p-ref)<=bound),(iid,err.tolist(),float(np.max(abs(p-ref)-bound)))
            selected[f'{split}_{j}']=p;records.append(dict(ID=iid,split=split,max_abs_by_channel=err.tolist(),max_error=float(err.max()),tolerance='One float16 ULP at stored value plus 2e-5 for CPU/MPS float32 drift; no fitted tolerance.'))
        del maps,z
    r.pack('raw-image-spot-replay.npz',**selected);r.save(_repo_io('raw-image-spot-replay.json'),dict(n_images=6,views_per_image=8,all_passed=True,source_sha256=r.USED,records=records,
        scope='Saved z1 weights on six original image files. Not full raw1500 feature extraction or U-Net training reproduction.'))
    print('RAW PIXEL Z1 SPOT REPLAY PASS',flush=True)

if __name__=='__main__':
    sys.addaudithook(r.guard);torch.set_num_threads(1)
    with r.threadpool_limits(limits=1):main()
