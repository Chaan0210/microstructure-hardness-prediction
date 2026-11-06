"""Continuous SEM-adapted encoder/flow information before hard mask extraction.

H300 is an existing competition-trained auxiliary segmenter, not an external
pretrained model. Its all500-source ancestry makes these screens conditional.
Original-CPSAM and H300 use identical descriptors and C4 views. No labels or
test images are used for extraction. All output files remain in this study.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','CELLPOSE_LOCAL_MODELS_PATH']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE']='1'
import argparse,gc,hashlib,json,resource,time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import expit,softmax
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits
from cellpose.vit import CPSAM

PATHS={'original':_repo_path(ROOT, 'experiments/grain-cellpose-220/weights/cpsam'),'H300':_repo_path(ROOT, 'experiments/push-300/ft-H300.pt')}
def normalize(a):
    lo,hi=np.percentile(a,[1,99]);return ((a.astype(np.float32)-float(lo))/max(float(hi-lo),1.)).astype(np.float32)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def descriptors(neck,flow,image):
    im=image.astype(np.float32)/255;lo,hi=np.percentile(im,[10,90]);scale=max(float((hi-lo)/2.563),.01);mean=float(im.mean());local=im.reshape(32,8,32,8).mean((1,3));z=np.clip((local-mean)/scale,-8,8).reshape(-1)
    w=softmax(-.5*((z[None]-np.array([-1.5,-.5,.5,1.5])[:,None])/.75)**2,axis=0).astype(np.float32);mass=w.mean(1);w=w/w.sum(1,keepdims=True);f=neck.astype(np.float32).reshape(256,1024);mu=f.mean(1);sd=np.sqrt(np.maximum(f.var(1),1e-6));cm=w@f.T;cs=np.sqrt(np.maximum(w@(f*f).T-cm*cm,1e-6));conditional=np.c_[np.clip((cm-mu)/np.maximum(sd,.01),-5,5),np.clip(cs/np.maximum(sd,.01)-1,-5,5)]
    semantic=np.r_[mu,sd,conditional.reshape(-1)]
    fy,fx=flow[:2].astype(float)/5;logit=np.clip(flow[2].astype(float),-20,20);prob=expit(logit);dyfy,dxfy=np.gradient(fy);dyfx,dxfx=np.gradient(fx);div=dxfx+dyfy;curl=dyfx-dxfy;a=dxfx-dyfy;b=dyfx+dxfy;strain=np.sqrt(a*a+b*b);speed=np.sqrt(fx*fx+fy*fy);gray=(gaussian_filter(im,1.)-mean)/scale
    values=[];keys=[]
    for name,field in [('cell_logit',logit),('speed',speed),('divergence',div),('curl',curl),('strain',strain),('gray_divergence',gray*div)]:
        values.extend([field.mean(),field.std(),*np.quantile(field,[.1,.5,.9])]);keys.extend([name+'_'+s for s in ['mean','std','q10','q50','q90']])
    values.extend([prob.mean(),prob.std(),np.mean(-prob*np.log(prob+1e-8)-(1-prob)*np.log(1-prob+1e-8))]);keys+=['prob_mean','prob_std','prob_entropy']
    values.extend([np.mean(prob*(a*a-b*b)/(strain+1e-6)),np.mean(prob*(2*a*b)/(strain+1e-6))]);keys+=['strain_c4','strain_s4']
    for name,field in [('prob',prob),('speed',speed),('divergence',div)]:values.append(field.reshape(4,64,4,64).mean((1,3)).std());keys.append(name+'_tile4_std')
    values.extend([np.mean(im**k) for k in [1,2,3,4]]);keys.extend(['gray_moment'+str(k) for k in [1,2,3,4]])
    return semantic.astype(np.float32),np.r_[values,mass].astype(np.float32),keys+['brightness_bin'+str(k) for k in range(4)]

def model(arm,device):
    m=CPSAM();state=torch.load(_repo_io(PATHS[arm]),map_location='cpu',weights_only=True);status=m.load_state_dict(state,strict=False);assert set(status.missing_keys)<={'diam_labels','diam_mean'} and not status.unexpected_keys;del state;gc.collect();m.requires_grad_(False);m.eval()
    if arm=='original':m.encoder.blocks=torch.nn.ModuleList(list(m.encoder.blocks)[-2:])
    return m.to(device)

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.device=='mps':assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.35)
    d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');out=_repo_path(HERE, 'domain-semantics');out.mkdir(exist_ok=True)
    meta=json.loads((_repo_path(HERE, 'cpsam-prefix-c4/meta.json')).read_text());assert meta['ids']==d['ids'].tolist() and np.load(_repo_path(HERE, 'cpsam-prefix-c4/done.npy')).all()
    if a.probe:indices=d['dev'][d['fold']!=0][:a.probe]
    else:indices=np.arange(500)
    spec=dict(arm=a.arm,device=a.device,torch=str(torch.__version__),ids=d['ids'].tolist(),views=[0,90,180,270],source_weights=str(PATHS[a.arm].relative_to(ROOT)),source_weights_sha256=sha(PATHS[a.arm]),
        scope='original is fixedeligibleApril2025CPSAM;H300 iscompetition-trainedon250cleanestprovidedimageswithordinaryoriginalblur/contrast/noise. H300screensareconditional; no newrepresentationtrainingorlabels/testreads.',
        preprocessing='Image-local1/99percentiles,onegraychannel. Original usesverifiedfrozen22blockC4prefixcache plusunchangedlast2blocks/neck/outputhead;H300usesfullnativeforward.',
        descriptors='2560global+brightnessconditionedneuralstatistics;46continuousflow/confidence/spatial/photometrydescriptors. No hard maskthreshold/flowintegration.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    recipe_path=out/f'{a.arm}-recipe.json';partial=out/f'{a.arm}-partial.npz'
    if not a.probe:
        if recipe_path.exists():assert json.loads(recipe_path.read_text())==spec
        else:recipe_path.write_text(json.dumps(spec,indent=2))
        if (out/f'{a.arm}-done.json').exists():return
    recipe_hash=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    net=model(a.arm,a.device);S=[];G=[];start=time.monotonic();check=[];elapsed=0.
    if partial.exists() and not a.probe:
        r=np.load(_repo_io(partial));assert str(r['recipe_hash'])==recipe_hash;S=list(r['semantic']);G=list(r['flow']);keys=r['flow_keys'].tolist();elapsed=float(r['elapsed']);assert np.array_equal(r['ids'],d['ids'][indices[:len(S)]])
    first_row=len(S)
    capture={}
    if a.arm=='H300':handle=net.encoder.neck.register_forward_hook(lambda module,args,value:capture.update(neck=value))
    try:
        with torch.no_grad():
            for j in indices[first_row:]:
                ss=[];gg=[]
                for first in [0,2]:
                    if a.arm=='original':
                        cache=np.load(_repo_path(HERE, 'cpsam-prefix-c4/features.npy'),mmap_mode='r');h=torch.tensor(np.array(cache[j,first:first+2],dtype=np.float32),device=a.device);del cache
                        for block in net.encoder.blocks:h=block(h)
                        neck=net.encoder.neck(h.permute(0,3,1,2));flow=F.conv_transpose2d(net.out(neck),net.W2,stride=net.ps,padding=0)
                    else:
                        image=np.stack([np.ascontiguousarray(np.rot90(normalize(px[j,0]),r)) for r in [first,first+1]])[:,None];flow,_=net(torch.tensor(image,device=a.device));neck=capture.pop('neck')
                    n=neck.cpu().numpy();f=flow.cpu().numpy()
                    if a.arm=='original' and a.probe:
                        expected=np.load(_repo_path(HERE, 'cpsam-prefix-c4/frozen-descriptors.npy'),mmap_mode='r')[j,first:first+2];v=np.c_[n.mean((2,3)),np.sqrt(n.var((2,3))+1e-6)];error=float(np.sqrt(np.mean((v-expected)**2))/max(np.sqrt(np.mean(expected**2)),1e-6));check.append(error);assert error<.001
                    for k,rot in enumerate([first,first+1]):
                        s,g,keys=descriptors(n[k],f[k],np.rot90(px[j,0],rot));ss.append(s);gg.append(g)
                    del neck,flow
                S.append(np.mean(ss,0));G.append(np.mean(gg,0));assert np.isfinite(S[-1]).all() and np.isfinite(G[-1]).all()
                if a.device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
                mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if a.device=='mps' else 0);assert mem<6*2**30
                if len(S)%10==0 or a.probe:
                    print(json.dumps(dict(rows=len(S),seconds=elapsed+time.monotonic()-start,memory=mem)),flush=True)
                    if not a.probe:
                        np.savez(_repo_io(out/f'{a.arm}-partial-new.npz'),ids=d['ids'][indices[:len(S)]],semantic=np.array(S),flow=np.array(G),flow_keys=np.array(keys),recipe_hash=recipe_hash,elapsed=elapsed+time.monotonic()-start)
                        (out/f'{a.arm}-partial-new.npz').replace(partial)
    finally:
        if a.arm=='H300':handle.remove()
    tag=a.arm+('-probe' if a.probe else '');np.savez(_repo_io(out/f'{tag}-features.npz'),ids=d['ids'][indices],semantic=np.array(S),flow=np.array(G),flow_keys=np.array(keys));r=dict(complete=True,images=len(indices),seconds=elapsed+time.monotonic()-start,peak_memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if a.device=='mps' else 0),semantic_shape=list(np.shape(S)),flow_shape=list(np.shape(G)),original_descriptor_RMS_check=check);(out/f'{tag}-done.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['original','H300'],required=True);ap.add_argument('--probe',type=int,default=0);ap.add_argument('--device',choices=['cpu','mps'],default='mps')
    with threadpool_limits(2):main(ap.parse_args())
