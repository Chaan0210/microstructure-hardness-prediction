"""Fine-tune an outer-train-only local-boundary encoder on hardness directly."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import train
from boundary_architecture import Net

HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]

class BoundaryHardness(nn.Module):
    def __init__(self,checkpoint,expected_ids,initialize=True):
        super().__init__()
        ck=torch.load(_repo_io(checkpoint),map_location='cpu',weights_only=True)
        assert ck['arm']=='local'
        assert ck['normalization']['train_ids']==expected_ids
        self.body=Net()
        if initialize:self.body.load_state_dict(ck['state_dict'])
        self.mean=float(ck['normalization']['image_mean']);self.std=float(ck['normalization']['image_std'])
        self.head=nn.Sequential(nn.Linear(324,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1))
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        raw=[x.pow(k).mean((2,3)) for k in [1,2,3,4]]
        h=(x*.25+.5-self.mean)/self.std;m=self.body
        a=m.s1(F.avg_pool2d(h,2));b=m.s2(F.avg_pool2d(a,2));d=m.s3(F.avg_pool2d(b,2));h=m.s4(F.avg_pool2d(d,2))
        z=m.norm(m.attention(h.flatten(2).transpose(1,2)+m.position)).transpose(1,2).reshape(-1,96,16,16)
        d=m.d3(torch.cat([F.interpolate(z,scale_factor=2,mode='bilinear',align_corners=False),d],1))
        stats=[z.mean((2,3)),(z.var((2,3),correction=0)+1e-6).sqrt(),d.mean((2,3)),(d.var((2,3),correction=0)+1e-6).sqrt()]
        return self.head(torch.cat(raw+stats,1)).squeeze(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--seed',type=int,default=2026101811);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--scratch',action='store_true');p.add_argument('--lr',type=float,default=.0003)
    args=p.parse_args()
    d=np.load(_repo_path(HERE, 'data.npz'));tr=d['dev'][d['fold']!=args.fold]
    checkpoint=_repo_path(ROOT, f'experiments/boundary-connectivity-20261017-1835/encoders/fold{args.fold}-refit-seed0-local/model.pt')
    recipe=dict(checkpoint=str(checkpoint.relative_to(ROOT)),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                expected_train_ids=d['ids'][tr].tolist(),initialization='random' if args.scratch else 'train-only-local-boundary',
                new_head='raw-four-moments + pooled learned encoder/decoder features -> 128 -> hardness',
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                architecture_sha256=hashlib.sha256((_repo_path(HERE, 'boundary_architecture.py')).read_bytes()).hexdigest())
    folder=_repo_path(HERE, 'runs', args.tag);folder.mkdir(parents=True,exist_ok=True)
    train.save_json(folder/'initialization.json',recipe)
    train.Predictor=lambda kind,width:BoundaryHardness(checkpoint,d['ids'][tr].tolist(),initialize=not args.scratch)
    config=argparse.Namespace(tag=args.tag,kind='boundary-finetune',fold=args.fold,seed=args.seed,epochs=args.epochs,lr=args.lr,wd=.0001,width=16,
                              batch=16,device='mps',eval_every=20,split='historical',split_seed=2026101801)
    train.main(config)
