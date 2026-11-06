"""Inference-only cache from eligible, unmodified Cellpose-SAM pretrained weights."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'));os.environ['CELLPOSE_LOCAL_MODELS_PATH']=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE']='1'
import argparse,gc,hashlib,json,resource,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cellpose.vit import CPSAM

WEIGHTS=_repo_path(ROOT, 'experiments/grain-cellpose-220/weights/cpsam')
VIEWS=[(0,False),(1,False),(2,False),(3,False)]

def normalize(x):
    a=x.astype(np.float32);lo,hi=np.percentile(a,[1,99]);return ((a-float(lo))/max(float(hi-lo),1.)).astype(np.float32)

def main(args):
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.35)
    d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r')
    net=CPSAM();state=torch.load(_repo_io(WEIGHTS),map_location='cpu',weights_only=True)
    status=net.load_state_dict(state,strict=False)
    assert set(status.missing_keys)=={'diam_labels','diam_mean'} and not status.unexpected_keys
    # These two newer package metadata scalars do not participate in this forward.
    del state;gc.collect()
    split=len(net.encoder.blocks)-2;assert split==22
    tail={f'blocks.{i}.{k}':v.detach().clone() for i,b in enumerate(net.encoder.blocks[split:]) for k,v in b.state_dict().items()}
    tail.update({f'neck.{k}':v.detach().clone() for k,v in net.encoder.neck.state_dict().items()})
    if not (_repo_path(HERE, 'cpsam-tail-initial.pt')).exists():torch.save(tail,_repo_path(HERE, 'cpsam-tail-initial.pt'))
    net=net.to('mps').eval();start=time.monotonic();times=[];errors=[]
    dest=_repo_path(HERE, 'cpsam-prefix-c4');dest.mkdir(exist_ok=True)
    meta_path=dest/'meta.json'
    meta=dict(reused_original90_prefix='cpsam-prefix/features.npy:views0and1; previously verified frozen eligible weights',ids=d['ids'].tolist(),views=VIEWS,shape=[500,4,32,32,1024],dtype='float16',prefix_blocks=22,
              source_weights='experiments/grain-cellpose-220/weights/cpsam',
              original_weight_sha256='e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2',
              weight_bytes=WEIGHTS.stat().st_size,license='BSD-3-Clause',release='April 2025 (official Cellpose model documentation)',
              source_urls=['https://github.com/MouseLand/cellpose/blob/main/docs/models.rst','https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam'],
              preprocessing='original grayscale256; image-local1st/99th percentile scaling; first input channel of pretrained patch projection; C4 rotations0/90/180/270; no reflections',
              scope='Unmodified eligible pretrained weights, eval+no_grad inference on train only. No dataset-specific representation fit or labels read.',
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.probe:
        indices=d['dev'][d['fold']!=0][:args.probe]
    else:
        indices=np.arange(500)
        if meta_path.exists():assert json.loads(meta_path.read_text())==meta
        else:meta_path.write_text(json.dumps(meta,indent=2))
        feature_path=dest/'features.npy'
        features=np.lib.format.open_memmap(feature_path,mode='r+' if feature_path.exists() else 'w+',dtype=np.float16,shape=(500,4,32,32,1024))
        descriptor_path=dest/'frozen-descriptors.npy'
        descriptors=np.lib.format.open_memmap(descriptor_path,mode='r+' if descriptor_path.exists() else 'w+',dtype=np.float32,shape=(500,4,512))
        donepath=dest/'done.npy';done=np.load(_repo_io(donepath)) if donepath.exists() else np.zeros(500,bool)
    old_meta=json.loads((_repo_path(HERE, 'cpsam-prefix/meta.json')).read_text());assert old_meta['ids']==d['ids'].tolist() and old_meta['views'][:2]==[[0,False],[1,False]];assert np.load(_repo_path(HERE, 'cpsam-prefix/done.npy')).all()
    old_prefix=np.load(_repo_path(HERE, 'cpsam-prefix/features.npy'),mmap_mode='r');old_desc=np.load(_repo_path(HERE, 'cpsam-prefix/frozen-descriptors.npy'),mmap_mode='r')
    with torch.no_grad():
        for j in indices:
            if not args.probe and done[j]:continue
            torch.mps.synchronize();t=time.monotonic();a=normalize(px[j,0]);xx=[]
            for rot,flip in VIEWS:
                v=np.rot90(a,rot);xx.append(np.ascontiguousarray(v[:,::-1] if flip else v))
            allprefix=[] if args.probe else [np.array(old_prefix[j,:2])];alldescriptors=[] if args.probe else [np.array(old_desc[j,:2])]
            for first in range(0 if args.probe else 2,4,2):
                x=torch.from_numpy(np.stack(xx[first:first+2])[:,None]).to('mps')
                h=F.conv2d(x,net.encoder.patch_embed.proj.weight[:,:1],bias=net.encoder.patch_embed.proj.bias,stride=8)
                h=h.permute(0,2,3,1)+net.encoder.pos_embed
                for block in net.encoder.blocks[:split]:h=block(h)
                if args.probe:
                    if first==0:
                        previous=np.array(old_prefix[j,:2],dtype=np.float32);fresh=h.cpu().numpy();assert np.max(abs(previous-fresh))<.02
                    hf=h.to(torch.float16).to(torch.float32)
                    for block in net.encoder.blocks[split:]:h=block(h);hf=block(hf)
                    h=net.encoder.neck(h.permute(0,3,1,2));hf=net.encoder.neck(hf.permute(0,3,1,2))
                    errors.append(float(((h-hf).square().mean()/h.square().mean().clamp(min=1e-9)).sqrt().cpu()))
                else:
                    allprefix.append(h.cpu().numpy().astype(np.float16))
                    for block in net.encoder.blocks[split:]:h=block(h)
                    h=net.encoder.neck(h.permute(0,3,1,2))
                    desc=torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()],1)
                    alldescriptors.append(desc.cpu().numpy())
                del x,h
            if not args.probe:
                features[j]=np.concatenate(allprefix);descriptors[j]=np.concatenate(alldescriptors)
                features.flush();descriptors.flush();done[j]=True;np.save(_repo_io(donepath),done)
                if int(done.sum())%10==0:
                    # Bound resident mapped pages while retaining the disk cache.
                    del features,old_prefix,old_desc;gc.collect();features=np.load(_repo_io(feature_path),mmap_mode='r+')
                    old_prefix=np.load(_repo_path(HERE, 'cpsam-prefix/features.npy'),mmap_mode='r');old_desc=np.load(_repo_path(HERE, 'cpsam-prefix/frozen-descriptors.npy'),mmap_mode='r')
            torch.mps.synchronize();times.append(time.monotonic()-t)
            memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+torch.mps.driver_allocated_memory()
            if memory>6*2**30:raise RuntimeError(f'Own combined memory budget: {memory}')
            if args.probe or len(times)%10==0:print(json.dumps(dict(completed=len(times),last_id=str(d['ids'][j]),seconds=time.monotonic()-start,per_image=float(np.median(times)),combined_memory=memory)),flush=True)
    result=dict(probe=args.probe,images=len(indices),seconds=time.monotonic()-start,median_image_seconds=float(np.median(times)) if times else 0,
                projected500_seconds=float(np.median(times)*500) if times else 0,quantization_relative_RMS=errors,
                complete=True if args.probe else bool(done.all()),peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mps_driver=torch.mps.driver_allocated_memory())
    (dest/('probe.json' if args.probe else 'complete.json')).write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=int,default=0);main(p.parse_args())
