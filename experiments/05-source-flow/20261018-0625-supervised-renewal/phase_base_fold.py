"""Fold-local original-image phase pretraining for the production-feature screen."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import phase_joint_v3 as p

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.35)
    d=dict(np.load(_repo_path(HERE, 'data.npz')));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');tr=d['dev'][d['fold']!=a.fold];q=d['dev'][d['fold']==a.fold];out=_repo_path(HERE, 'runs', a.tag);out.mkdir(exist_ok=True)
    if (out/'done.json').exists():return
    aux=p.prepare(d,px,tr,out);assert not set(aux['indices'])&set(q)
    spec=vars(a)|dict(train=tr.tolist(),query=q.tolist(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trainer_sha256=hashlib.sha256((_repo_path(HERE, 'phase_joint_v3.py')).read_bytes()).hexdigest(),matches='Exact auxiliary training recipe from fold0 phase-joint-b4-f0-s81; no hardness training or outer-query representation fitting.')
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));start=time.monotonic()
    if (out/'pretrained.pt').exists():
        ck=torch.load(_repo_io(out/'pretrained.pt'),map_location='cpu',weights_only=True);assert ck['train']==tr.tolist() and ck['pretrain_steps']==a.pretrain_steps and ck['seed']==a.seed
        net=p.UNet(a.width).to('mps');net.load_state_dict(ck['state'])
    else:net=p.pretrain(aux,px,a,out)
    f=p.extract(net,px,aux,tr,batch=4);v=p.extract(net,px,aux,q,batch=4)
    np.savez(_repo_io(out/'frozen-features.npz'),train=tr,query=q,train_features=f,query_features=v,train_globals=np.empty((len(tr),0)),query_globals=np.empty((len(q),0)))
    (out/'done.json').write_text(json.dumps(dict(complete=True,seconds=time.monotonic()-start),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--tag',required=True);ap.add_argument('--width',type=int,default=32);ap.add_argument('--seed',type=int,default=2026101881);ap.add_argument('--pretrain-steps',type=int,default=4000);ap.add_argument('--pretrain-batch',type=int,default=4)
    with threadpool_limits(2):main(ap.parse_args())
