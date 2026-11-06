"""Normalize learned image measurements inside each training partition."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import argparse,hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import physical_fusion as physical
import dense_fusion_v3 as image
import grain_attention_v3 as trainer

class Model(image.Fusion):
    def __init__(self,od,gd,ctx,kind):
        super().__init__(od,gd,ctx,kind);self.image_norm=nn.BatchNorm1d(40,eps=1e-5,momentum=.1)
    def image_stats(self,neck,raw,g):
        if self.kind=='context':neck=neck[:1];raw=raw[:1]
        neck=torch.zeros_like(neck) if self.kind in ['raw','context'] else neck
        raw=torch.zeros_like(raw) if self.kind in ['neck','context'] else raw
        z=F.interpolate(self.neck(neck),size=(128,128),mode='bilinear',align_corners=False)
        p=self.local(torch.cat([z,raw],1)).sigmoid();stats=[p.mean((2,3)),(p.var((2,3),correction=0)+1e-6).sqrt()]
        for k in [16,32,64]:stats.append((F.avg_pool2d(p,k).var((2,3),correction=0)+1e-6).sqrt())
        f=torch.cat(stats,1);return f.expand(len(g),-1) if self.kind=='context' else f
    def forward(self,neck,raw,g):
        f=self.image_norm(self.image_stats(neck,raw,g));return self.base(g)[:,0]+self.head(torch.cat([f,g],1))[:,0]
    @torch.no_grad()
    def initialize_stats(self,data,tr):
        self.eval();features=[]
        for first in range(0,len(tr),16):
            for view in [0,1]:
                data.view=view;features.append(self.image_stats(*data.batch(tr[first:first+16])))
        data.view=None;f=torch.cat(features);mean=f.mean(0);var=f.var(0,correction=0)
        self.image_norm.running_mean.copy_(mean);self.image_norm.running_var.copy_(var);self.image_norm.num_batches_tracked.zero_()
        self.feature_scale_record=dict(training_indices=tr.tolist(),original_images=len(tr),views=[0,1],mean=mean.tolist(),std=var.sqrt().tolist(),median_std=float(var.sqrt().median()),epsilon=1e-5,query_images_used=False)

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    bags,weights,_,d,_=trainer.load();path=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(path))
    lookup={str(i):j for j,i in enumerate(z['ids'])};G=z['X'][[lookup[str(i)] for i in d['ids']]].astype(float);G=np.where(np.isfinite(G),G,np.nan)
    image.ImageData.context_only=a.kind=='context'
    if a.kind!='context':image.RAW=image.prepare_raw();image.NECK=np.load(_repo_path(HERE, 'cpsam-neck-c4/maps.npy'),mmap_mode='r')
    trainer.context=physical.context;trainer.Data=physical.Data;trainer.Model=Model;trainer.predict=image.predict
    for f in a.folds:
        out=_repo_path(HERE, 'runs', f'normalized-{a.kind}-f{f}-s{a.seed}');out.mkdir(exist_ok=True)
        if (out/'done.json').exists():continue
        tr=d['dev'][d['fold']!=f];q=d['dev'][d['fold']==f];A,B=train_test_split(tr,test_size=.2,random_state=2026101847+f)
        trainer.save(_repo_io(out/'recipe.json'),vars(a)|dict(fold=f,A=A.tolist(),B=B.tolist(),train=tr.tolist(),query=q.tolist(),
            feature_path=str(path.relative_to(ROOT)),feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            normalization='40 learned image measurements: BatchNorm1d initialized from only the current training originals and their original/rot90 views; updates only during training.',
            scope='Conditional on inherited physical104 bank; new image normalization and learning exclude held-out images.'))
        r=trainer.fit(bags,weights,G,d['y'],A,B,a.kind,a.seed,80,out/'selector',select=True)
        rr=trainer.fit(bags,weights,G,d['y'],tr,q,a.kind,a.seed,r['selected_epoch'],out/'refit')
        p=np.load(_repo_io(out/'refit/predictions.npz'))['prediction'];base=d['base083'][d['fold']==f];eligible=~d['failed'][d['fold']==f]
        result=dict(fold=f,kind=a.kind,selector=r,refit=rr,base083_rmse=trainer.rmse(base,d['y'][q]),quarter_blend_rmse=trainer.rmse(base+.25*eligible*(p-base),d['y'][q]))
        trainer.save(_repo_io(out/'done.json'),result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['fusion','raw','context'],default='fusion');p.add_argument('--folds',type=int,nargs='+',default=[0]);p.add_argument('--seed',type=int,default=2026101891)
    with threadpool_limits(limits=2):main(p.parse_args())
