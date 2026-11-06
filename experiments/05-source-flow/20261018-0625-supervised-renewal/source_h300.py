"""Source-restricted H300-style auxiliary replication and continuous-flow inference.

Only outer-training originals from the pre-existing clean auxiliary eligibility
pool enter optimization. No hardness labels are read by this trainer. This
checks the newly introduced representation; inherited P104/083 representations
remain conditional and are not silently relabeled end-to-end validation.
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
import argparse,gc,json,hashlib,math,time,subprocess,re,resource
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from cellpose import dynamics,transforms
from cellpose.train import _loss_fn_seg
from cellpose.vit import CPSAM
from scipy import ndimage as ndi
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from domain_semantics import normalize,descriptors
WEIGHTS=_repo_path(ROOT, 'experiments/grain-cellpose-220/weights/cpsam')

def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def pool():
 out=_repo_path(HERE, 'source-h300');out.mkdir(exist_ok=True);path=out/'flow-targets.npz'
 if path.exists():return dict(np.load(_repo_io(path)))
 d=np.load(_repo_path(HERE, 'data.npz'));m=np.load(_repo_path(ROOT, 'experiments/grain-cellpose-220/masks-v3-train.npz'));ids=m['ids'];masks=m['masks'];assert np.array_equal(ids,d['ids']);noise=np.load(_repo_path(ROOT, 'experiments/push-300/noise_train.npy'));assert noise.shape==(500,);indices=np.argsort(noise)[:250]
 start=time.monotonic();flows=dynamics.labels_to_flows([masks[j].astype(np.int32) for j in indices],device=torch.device('cpu'));flows=np.array([f[1:4].astype(np.float32) for f in flows]);assert flows.shape==(250,3,256,256) and np.isfinite(flows).all();np.savez_compressed(_repo_io(path),ids=ids[indices],indices=indices,flows=flows)
 save(out/'flow-targets.json',dict(scope='Pre-existingH300auxiliaryeligibility(250lowesttrainnoise),originalzero-shotmask-derivedflowannotations; nolearnerfit/nohardness. EverytrainingjobintersectsthisfixedpoolwithitsoutertrainingIDs.',seconds=time.monotonic()-start,mask_source_sha256=sha(_repo_path(ROOT, 'experiments/grain-cellpose-220/masks-v3-train.npz')),noise_source_sha256=sha(_repo_path(ROOT, 'experiments/push-300/noise_train.npy'))));return dict(np.load(_repo_io(path)))

def degrade(im,rng):
 f=im.astype(np.float32);sb=rng.uniform(0.,1.6);c=rng.uniform(.55,1.);sn=rng.uniform(0.,24.)
 if rng.rand()<.7:sb,c,sn=rng.uniform(.7,1.5),rng.uniform(.65,.95),rng.uniform(12,22)
 if rng.rand()<.15:sb,c,sn=0.,1.,rng.uniform(0,4)
 g=ndi.gaussian_filter(f,sb) if sb>.05 else f;m=g.mean();g=m+c*(g-m)+rng.uniform(-15,15);g+=rng.normal(0,sn,f.shape);return np.clip(np.round(g),0,255).astype(np.uint8)

def d4(a,fl,t,ud,lr):
 b,fy,fx=fl
 if t:a,b,fy,fx=a.T,b.T,fx.T,fy.T
 if ud:a,b,fy,fx=a[::-1],b[::-1],-fy[::-1],fx[::-1]
 if lr:a,b,fy,fx=a[:,::-1],b[:,::-1],fy[:,::-1],-fx[:,::-1]
 return np.ascontiguousarray(a),np.ascontiguousarray(np.stack([b,fy,fx]))
def norm(a):return transforms.normalize_img(a.astype(np.float32)[...,None],normalize=True,axis=-1)[...,0]

def initial_model():
 # Meta construction avoids simultaneously holding two full CPU copies.
 with torch.device('meta'):net=CPSAM()
 state=torch.load(_repo_io(WEIGHTS),map_location='cpu',weights_only=True,mmap=True);status=net.load_state_dict(state,strict=False,assign=True);assert set(status.missing_keys)<={'diam_labels','diam_mean'} and not status.unexpected_keys
 for key in status.missing_keys:setattr(net,key,nn.Parameter(torch.tensor(30.,dtype=torch.float32),requires_grad=False))
 del state;gc.collect();return net.to('mps')

def forward_train(net,x):
 x=F.conv2d(x,net.encoder.patch_embed.proj.weight.data[:,:x.shape[1]],bias=net.encoder.patch_embed.proj.bias.data,stride=net.ps);x=x.permute(0,2,3,1)+net.encoder.pos_embed;n=len(net.encoder.blocks);rd=(torch.rand((len(x),n),device=x.device)<torch.linspace(0,net.rdrop,n,device=x.device)).to(x.dtype)
 for j,block in enumerate(net.encoder.blocks):
  keep=rd[:,j,None,None,None];h=checkpoint(block,x,use_reentrant=False);x=x*keep+h*(1-keep)
 x=net.encoder.neck(x.permute(0,3,1,2));return F.conv_transpose2d(net.out(x),net.W2,stride=net.ps,padding=0)

def memory():
 rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024
 driver=torch.mps.driver_allocated_memory();return dict(rss=rss,mps_driver=driver,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def train(a):
 d=np.load(_repo_path(HERE, 'data.npz'));p=pool();tr,q=list(KFold(5,shuffle=True,random_state=a.split_seed).split(np.arange(500)))[a.fold];which=np.flatnonzero(np.isin(p['indices'],tr));sources=p['indices'][which];assert not set(sources)&set(q);px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');out=_repo_path(HERE, 'source-h300', a.tag);out.mkdir(exist_ok=True)
 recipe=dict(architecture='OriginalCPSAMfullencoder/neck/output;sameH300-styleheavyaugmentation/segmentationobjective',split_seed=a.split_seed,fold=a.fold,seed=a.seed,steps=300,effective_batch=4,microbatch=1,lr=.00002,weight_decay=.1,optimizer_foreach=False,source_indices=sources.tolist(),source_ids=d['ids'][sources].tolist(),train=tr.tolist(),query=q.tolist(),eligibility='Fixedpre-existing250cleanestauxiliarypoolintersectedwithouterT; neverouterQ. Originalclean-poolselectionishistoricalandnotanuntoucheddataclaim.',objective='Cellpose_loss_fn_seg onoriginalmask-derivedbinary/flowtargets;nohardnesslabels. Ordinaryoriginalblur/contrast/brightness/noise+D4covarianttargets;no renderednewlayout.',initial_weights_sha256='e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2',source_sha256=sha(Path(__file__)),deviations_from_legacy='Explicitseed, source restriction, microbatchgradientaccumulationandforeach=False forlocalmemory;effectivebatch/loss/LRbudgetmaintained. NotbitwiseoldH300trainingreplay.',resource='OneMPSjob;driverfraction.6ofrecommendedworkingset;guarddriver6.5GiB,RSS5GiB. Checkpointedbenchmarkbeforefull300.')
 if (out/'recipe.json').exists():assert json.loads((out/'recipe.json').read_text())==recipe
 else:save(out/'recipe.json',recipe)
 torch.manual_seed(a.seed);torch.mps.manual_seed(a.seed);net=initial_model();torch.manual_seed(a.seed);torch.mps.manual_seed(a.seed);net.train();opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.00002,weight_decay=.1,foreach=False);rng=np.random.RandomState(a.seed);first=0;elapsed=0.;curve=[];resume=out/'resume.pt'
 if resume.exists():
  ck=torch.load(_repo_io(resume),map_location='cpu',weights_only=False);assert ck['recipe']==recipe;net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);first=ck['step'];rng.set_state(ck['rng']);torch.set_rng_state(ck['torch_rng']);torch.mps.set_rng_state(ck['mps_rng']);elapsed=ck['elapsed'];curve=ck['curve']
  for state in opt.state.values():
   for k,v in state.items():
    if torch.is_tensor(v) and k!='step':state[k]=v.to('mps')
  del ck;gc.collect()
 target=min(300,a.max_steps);start=time.monotonic();times=[]
 for step in range(first,target):
  tic=time.monotonic();opt.zero_grad(set_to_none=True);loss_sum=0.
  for j in rng.randint(0,len(which),4):
   row=int(which[j]);im,fl=d4(px[p['indices'][row],0],p['flows'][row],rng.rand()<.5,rng.rand()<.5,rng.rand()<.5);X=torch.tensor(norm(degrade(im,rng))[None,None],device='mps');L=torch.tensor(fl[None],device='mps');prediction=forward_train(net,X);loss=_loss_fn_seg(L,prediction,torch.device('mps'));assert torch.isfinite(loss);(loss/4).backward();loss_sum+=float(loss.detach().cpu())/4;del X,L,prediction,loss
  lr=.00002*min(1.,(step+1)/15)*(.05+.95*.5*(1+math.cos(math.pi*step/300)))
  for group in opt.param_groups:group['lr']=lr
  opt.step();torch.mps.synchronize();torch.mps.empty_cache();times.append(time.monotonic()-tic);mem=memory();assert mem['mps_driver']<6.5*2**30 and mem['rss']<5*2**30
  row=dict(step=step+1,loss=loss_sum,seconds=elapsed+time.monotonic()-start,step_seconds=times[-1],**mem);curve.append(row)
  if step<3 or (step+1)%10==0:print(json.dumps(row),flush=True)
  if (step+1)%25==0 or step+1==target:
   ck=dict(state=net.state_dict(),optimizer=opt.state_dict(),recipe=recipe,step=step+1,rng=rng.get_state(),torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),curve=curve,elapsed=row['seconds']);torch.save(ck,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(resume);save(out/'curve.json',curve)
 if target<300:
  r=dict(completed_steps=target,median_step_seconds=float(np.median(times)),projected300_seconds=float(np.median(times)*300),memory=memory(),actual_source_ids=d['ids'][sources].tolist(),query_ids=d['ids'][q].tolist(),hardness_used=False);save(out/'benchmark.json',r);print('BENCHMARK',json.dumps({k:v for k,v in r.items() if k not in ['actual_source_ids','query_ids']}),flush=True);return
 torch.save(dict(state=net.state_dict(),recipe=recipe),_repo_io(out/'model.pt'));save(out/'trained.json',dict(complete=True,steps=300,seconds=elapsed+time.monotonic()-start,model_sha256=sha(out/'model.pt')));print('TRAINED',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train']);ap.add_argument('--tag',default='repeat0-fold0-seed1811');ap.add_argument('--split-seed',type=int,default=2026101801);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--seed',type=int,default=2026101811);ap.add_argument('--max-steps',type=int,default=300);a=ap.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if a.action=='train':assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.6)
 with threadpool_limits(2):pool() if a.action=='prepare' else train(a)
