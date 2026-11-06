"""Native-pixel masked structural context learning; no hardness in encoder fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];OLD=_repo_path(ROOT, 'experiments/highnoise-relational-20261016')
(_repo_path(HERE, 'tmp')).mkdir(exist_ok=True)
for k in ['TMPDIR','TORCHINDUCTOR_CACHE_DIR','MPLCONFIGDIR']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1';os.environ['TABPFN_MODEL_CACHE_DIR']=str(_repo_path(HERE, 'tmp/tabpfn'))
import json,hashlib,time,resource,gc,argparse,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from threadpoolctl import threadpool_limits
WIDTH=96;EPOCHS=100;SEEDS=[2026101771,2026111771];FIT_ACTIVE=False;DEADLINE=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sh(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def save(name,d):
    p=_repo_path(HERE, name);p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def arrays(name,**d):
    p=_repo_path(HERE, name);p.parent.mkdir(exist_ok=True,parents=True);assert not p.exists();np.savez_compressed(_repo_io(p),**d)
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:];write=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if write and not p.is_relative_to(HERE):raise RuntimeError('Outside experiment write: '+str(p))
        if p.is_relative_to(_repo_path(ROOT, 'data/test')) or '-test.' in p.name or p.name=='sample_submission.csv' or p.name.startswith('submission-candidate-'):raise RuntimeError('No test data in research')
        if FIT_ACTIVE and not write and p.suffix in ['.npz','.npy','.pt','.csv','.png','.jpg','.joblib']:raise RuntimeError('Dataset/model read during isolated encoder fitting: '+str(p))
def memory():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;driver=torch.mps.driver_allocated_memory()
    if driver>3.5*2**30 or rss+driver>5*2**30:raise RuntimeError('Own memory cap exceeded')
    return dict(RSS_bytes=rss,MPS_driver_bytes=driver)
def data():
    z=np.load(_repo_path(OLD, 'inputs.npz'));d={k:z[k] for k in ['ids','dev','fold','noise']};d['graphs']=json.loads((_repo_path(OLD, 'graphs.json')).read_text());d['pixels']=np.zeros((500,1,256,256),np.uint8)
    for j in d['dev']:
        with Image.open(_repo_path(ROOT, f"data/train/{d['ids'][j]}.png")) as im:assert im.mode=='L' and im.size==(256,256);d['pixels'][j,0]=np.asarray(im)
    return d
def positions():
    yy,xx=np.mgrid[:16,:16];freq=1/10000**(np.arange(24)/24)
    pe=np.c_[np.sin(xx.ravel()[:,None]*freq),np.cos(xx.ravel()[:,None]*freq),np.sin(yy.ravel()[:,None]*freq),np.cos(yy.ravel()[:,None]*freq)].astype(np.float32)
    return torch.from_numpy(np.vstack([np.zeros((1,96),np.float32),pe]))[None]
class Block(nn.Module):
    def __init__(self):
        super().__init__();self.n1=nn.LayerNorm(WIDTH);self.qkv=nn.Linear(WIDTH,3*WIDTH);self.out=nn.Linear(WIDTH,WIDTH);self.n2=nn.LayerNorm(WIDTH);self.ff=nn.Sequential(nn.Linear(WIDTH,192),nn.GELU(),nn.Dropout(.1),nn.Linear(192,WIDTH));self.drop=nn.Dropout(.1)
    def forward(self,x):
        b,n,_=x.shape;q,k,v=self.qkv(self.n1(x)).reshape(b,n,3,4,24).permute(2,0,3,1,4).unbind(0)
        z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(b,n,WIDTH);x=x+self.drop(self.out(z));return x+self.drop(self.ff(self.n2(x)))
class ContextNet(nn.Module):
    def __init__(self):
        super().__init__();self.pixel=nn.Sequential(nn.Linear(64,64),nn.GELU());self.merge=nn.Linear(256,WIDTH);self.cls=nn.Parameter(torch.randn(1,1,WIDTH)*.02);self.mask_token=nn.Parameter(torch.randn(1,1,WIDTH)*.02);self.register_buffer('pos',positions());self.blocks=nn.Sequential(*[Block() for _ in range(4)]);self.norm=nn.LayerNorm(WIDTH);self.decoder=nn.Sequential(nn.Linear(WIDTH,64),nn.GELU(),nn.Linear(64,16));nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
    def forward(self,x,perm,mask=None,decode=True):
        b=len(x);p=x.unfold(2,8,8).unfold(3,8,8).squeeze(1).reshape(b,32,32,64);p=self.pixel(p).reshape(b,16,2,16,2,64).permute(0,1,3,2,4,5).reshape(b,256,256);z=self.merge(p)
        z=torch.gather(z,1,perm[:,:,None].expand(-1,-1,WIDTH))
        if mask is not None:z=torch.where(mask[:,:,None],self.mask_token,z)
        z=torch.cat([self.cls.expand(b,-1,-1),z],1)+self.pos;z=self.norm(self.blocks(z));v=z[:,1:];desc=torch.cat([z[:,0],v.mean(1),(v.var(1,correction=0)+1e-6).sqrt()],1)
        return (self.decoder(v) if decode else None),desc
def targets(x,perm):
    b=len(x);t=F.avg_pool2d(x,4).reshape(b,1,16,4,16,4).permute(0,2,4,1,3,5).reshape(b,256,16)
    return torch.gather(t,1,perm[:,:,None].expand(-1,-1,16))
def losses(pred,t,mask,per_image=False):
    pm=pred.mean(-1,keepdim=True);tm=t.mean(-1,keepdim=True);pa=(pred-pm).reshape(-1,256,4,4);ta=(t-tm).reshape(-1,256,4,4)
    dc=(pm-tm).square().squeeze(-1);ac=(pa-ta).square().mean((-2,-1));grad=.5*((torch.diff(pa,dim=-1)-torch.diff(ta,dim=-1)).square().mean((-2,-1))+(torch.diff(pa,dim=-2)-torch.diff(ta,dim=-2)).square().mean((-2,-1)))
    values=torch.stack([(x*mask).sum(1)/mask.sum(1) for x in [dc,ac,grad]],1)
    return values if per_image else (values.mean(0)*torch.tensor([1.,4.,2.],device=pred.device)).sum()
def permutation(indices,arm):
    return np.stack([np.random.RandomState(271828+int(j)).permutation(256) if arm=='permuted' else np.arange(256) for j in indices])
def mask_batch(n,rng):
    m=np.zeros((n,64),bool)
    for row in m:row[rng.choice(64,26,replace=False)]=True
    return m.reshape(n,8,8).repeat(2,1).repeat(2,2).reshape(n,256)
def normalize(p,meta):return (p.astype(np.float32)/255-meta['image_mean'])/meta['image_std']
def seed_all(seed):torch.manual_seed(seed);torch.mps.manual_seed(seed)
def init(seed):seed_all(seed);return ContextNet().to('mps')
@torch.no_grad()
def features(model,px,indices,meta,arm):
    model.eval();out=[]
    for first in range(0,len(px),16):
        raw=normalize(px[first:first+16],meta);perm=torch.from_numpy(permutation(indices[first:first+16],arm)).to('mps');vv=[]
        for rot,flip in [(0,False),(1,False),(0,True),(1,True)]:
            x=np.rot90(raw,rot,axes=(2,3));x=x[:,:,:,::-1] if flip else x
            _,f=model(torch.from_numpy(np.ascontiguousarray(x)).to('mps'),perm,decode=False);vv.append(f.cpu().numpy())
        out.append(np.mean(vv,0))
    return np.concatenate(out)
@torch.no_grad()
def geometry(model,px,indices,meta,arm):
    model.eval();out=[];reference=[]
    for first in range(0,len(px),16):
        ix=indices[first:first+16];x=torch.from_numpy(np.ascontiguousarray(normalize(px[first:first+16],meta))).to('mps');perm=torch.from_numpy(permutation(ix,arm)).to('mps');mask=torch.from_numpy(np.concatenate([mask_batch(1,np.random.default_rng(81173+int(j))) for j in ix])).to('mps');t=targets(x,perm);p,_=model(x,perm,mask)
        out.append(losses(p,t,mask,True).cpu().numpy());tm=t.mean(-1);mean=(tm*(~mask)).sum(1)/(~mask).sum(1);ref=mean[:,None,None].expand_as(t);reference.append(losses(ref,t,mask,True).cpu().numpy())
    return np.concatenate(out),np.concatenate(reference)
def fit_encoder(model,px,indices,meta,arm,seed,epochs=EPOCHS):
    global FIT_ACTIVE
    assert len(px)==len(indices)==len(meta['train_ids']) and len(set(indices))==len(indices)
    FIT_ACTIVE=True;seed_all(seed+41);rng=np.random.default_rng(seed+41);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02);curve=[];average=None;oh=hashlib.sha256();vh=hashlib.sha256();mh=hashlib.sha256();steps=0;torch.mps.synchronize();start=time.monotonic();peak=memory()
    try:
        for epoch in range(epochs):
            if DEADLINE is not None and time.monotonic()>DEADLINE:raise RuntimeError('Frozen encoder-stage runtime cap reached')
            lr=.001*(epoch+1)/5 if epoch<5 else .00005+.5*(.001-.00005)*(1+math.cos(math.pi*(epoch-4)/max(epochs-5,1)))
            for g in opt.param_groups:g['lr']=lr
            model.train();order=rng.permutation(len(px));oh.update(order.tobytes());total=0.
            for ix in np.array_split(order,int(np.ceil(len(order)/16))):
                raw=normalize(px[ix],meta);rot=int(rng.integers(4));flip=bool(rng.random()<.5);raw=np.rot90(raw,rot,axes=(2,3));raw=raw[:,:,:,::-1] if flip else raw;raw=np.ascontiguousarray(raw);mb=mask_batch(len(ix),rng);vh.update(raw.tobytes());mh.update(mb.tobytes())
                x=torch.from_numpy(raw).to('mps');perm=torch.from_numpy(permutation(indices[ix],arm)).to('mps');mask=torch.from_numpy(mb).to('mps');t=targets(x,perm);p,_=model(x,perm,mask);loss=losses(p,t,mask);assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);opt.step();total+=float(loss.detach().cpu())*len(ix);steps+=1
                mem=memory();peak={k:max(peak[k],mem[k]) for k in peak}
            curve.append(dict(epoch=epoch+1,loss=total/len(px),lr=lr))
            if epoch>=epochs-5:
                state={k:v.detach().cpu().clone()/5 for k,v in model.state_dict().items()}
                if average is None:average=state
                else:
                    for k,v in state.items():average[k]+=v
            if (epoch+1)%25==0:print('EPOCH',arm,seed,epoch+1,'seconds',round(time.monotonic()-start,1),'loss',round(total/len(px),6),flush=True)
        model.load_state_dict(average);torch.mps.synchronize()
        return dict(seconds=time.monotonic()-start,epochs=epochs,steps=steps,source_visits=epochs*len(px),curve=curve,order_sha256=oh.hexdigest(),view_sha256=vh.hexdigest(),mask_sha256=mh.hexdigest(),state_sha256=sh(model.state_dict()),peak_memory=peak,labels_used=False,dataset_reads_blocked=True)
    finally:FIT_ACTIVE=False;del opt;gc.collect();torch.mps.empty_cache()
def probe():
    assert (_repo_path(HERE, 'starting-state.json')).exists();d=data();ix=np.array(d['graphs'][0]['A'][:16]);px=d['pixels'][ix];meta=dict(image_mean=float(px.mean()/255),image_std=float(px.std()/255),train_ids=d['ids'][ix].tolist());m=init(SEEDS[0]);x=torch.from_numpy(normalize(px,meta)).to('mps');perm=torch.from_numpy(permutation(ix,'spatial')).to('mps');rng=np.random.default_rng(7);mask=torch.from_numpy(mask_batch(len(ix),rng)).to('mps');t=targets(x,perm)
    raw=x.cpu().numpy();manual=raw.reshape(16,1,16,4,4,16,4,4).mean((4,7)).transpose(0,2,4,1,3,5).reshape(16,256,16)
    assert np.max(abs(manual-t.cpu().numpy()))<1e-6
    exact=losses(t,t,mask);assert float(exact.cpu())==0.;p0=torch.randn_like(t);val=losses(p0,t,mask,True);assert torch.allclose(losses(p0,t,mask),(val.mean(0)*torch.tensor([1.,4.,2.],device='mps')).sum())
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);times=[];ls=[]
    for step in range(40):
        torch.mps.synchronize();start=time.monotonic();p,_=m(x,perm,mask);loss=losses(p,t,mask);opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.linalg.vector_norm(m.pixel[0].weight.grad).cpu());nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();torch.mps.synchronize();times.append(time.monotonic()-start);ls.append(float(loss.detach().cpu()));memory()
    step=float(np.median(times[20:]));project=step*68000*1.25+1200;passed=project<5400 and grad>0 and np.isfinite(ls).all()
    save('probe.json',dict(passed=bool(passed),seconds_per_step=step,projected_main_seconds=project,parameter_count=sum(p.numel() for p in m.parameters()),last_gradient_norm=grad,loss_first=ls[0],loss_last=ls[-1],independent_target_max_error=float(np.max(abs(manual-t.cpu().numpy()))),peak_memory=memory(),source_ids=d['ids'][ix].tolist(),labels_read=False,training_steps=40,source_sha256=sha(__file__)));print('PROBE',passed,step,project,flush=True)
def train():
    global DEADLINE
    probe=json.loads((_repo_path(HERE, 'probe.json')).read_text());assert (_repo_path(HERE, 'AMENDMENT-01.md')).exists() and probe['projected_main_seconds']<7200 and probe['last_gradient_norm']>0 and probe['independent_target_max_error']<1e-6
    d=data();save('encoder-code-lock.json',dict(source_sha256=sha(__file__),plan_sha256=sha(_repo_path(HERE, 'PLAN.md')),amendment_sha256=sha(_repo_path(HERE, 'AMENDMENT-01.md')),seeds=SEEDS,epochs=EPOCHS,torch=str(torch.__version__),device='mps-float32'));elapsed=0.;done=[];wall_start=time.monotonic();DEADLINE=wall_start+6300
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);assert set(tr).isdisjoint(q);px=d['pixels'][tr].copy();qp=d['pixels'][q].copy();meta=dict(image_mean=float(px.mean(dtype=np.float64)/255),image_std=float(px.std(dtype=np.float64)/255),train_ids=d['ids'][tr].tolist())
            for repeat,base in enumerate(SEEDS):
                seed=base+100*k
                for arm in ['spatial','permuted']:
                    if time.monotonic()>DEADLINE:raise RuntimeError('Encoder budget reserve for readouts reached')
                    path=_repo_path(HERE, f'encoders/fold{k}-{stage}-seed{repeat}-{arm}');path.mkdir(parents=True,exist_ok=False);m=init(seed);initial=sh(m.state_dict())
                    if arm=='spatial':
                        torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=meta),_repo_io(path/'initial.pt'));ft=features(m,px,tr,meta,'random');fq=features(m,qp,q,meta,'random');arrays(str((path/'random-features.npz').relative_to(HERE)),train=tr,query=q,train_features=ft,query_features=fq)
                    record=fit_encoder(m,px,tr,meta,arm,seed);elapsed+=record['seconds'];ft=features(m,px,tr,meta,arm);fq=features(m,qp,q,meta,arm);ge,ref=geometry(m,qp,q,meta,arm)
                    torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=meta,arm=arm,seed=seed),_repo_io(path/'model.pt'));arrays(str((path/'features.npz').relative_to(HERE)),train=tr,query=q,train_features=ft,query_features=fq,geometry=ge,geometry_reference=ref)
                    record.update(initial_sha256=initial,normalization=meta,training_pixels_sha256=ah(px),arm=arm,seed=seed,source_sha256=sha(__file__));save(str((path/'training.json').relative_to(HERE)),record);done.append(str(path.relative_to(HERE)));print('ENCODER_COMPLETE',k,stage,repeat,arm,'training_seconds',round(elapsed,1),flush=True);del m;gc.collect();torch.mps.empty_cache()
    save('encoders-complete.json',dict(complete=True,models=done,training_seconds=elapsed,wall_seconds=time.monotonic()-wall_start,source_sha256=sha(__file__)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['probe','train']);args=ap.parse_args();assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(guard)
    with threadpool_limits(limits=2):globals()[args.action]()
