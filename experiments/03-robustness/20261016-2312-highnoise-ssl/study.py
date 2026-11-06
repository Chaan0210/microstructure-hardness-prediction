"""Small isolated SSL study. No external weights, labels, or files enter pretrain()."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
(_repo_path(HERE, 'tmp')).mkdir(exist_ok=True)
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'));os.environ['TORCHINDUCTOR_CACHE_DIR']=str(_repo_path(HERE, 'tmp/torch'));os.environ['MPLCONFIGDIR']=str(_repo_path(HERE, 'tmp/mpl'))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time,resource,gc,math,argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from scipy import ndimage as ndi
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];OLD=_repo_path(ROOT, 'experiments/highnoise-relational-20261016')
SEED=2026101671;EPOCHS=24;ARMS=['none','augment','ssl','long'];CAP=45*60
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(s):
    h=hashlib.sha256()
    for k,v in s.items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
def save(path,obj):
    p=_repo_path(HERE, path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def arrays(path,**kw):
    p=_repo_path(HERE, path);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.savez_compressed(_repo_io(p),**kw)
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:]
        write=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if write and not p.is_relative_to(HERE):raise RuntimeError('Outside experiment write: '+str(p))
        if p.is_relative_to(_repo_path(ROOT, 'data/test')) or (p.is_relative_to(_repo_path(ROOT, 'artifacts')) and p.name.startswith('submission-')):raise RuntimeError('No test data in research')
def limits():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss>2**30:raise RuntimeError('Own RSS exceeded1GiB')
    return rss
def rmse(p,y):return float(np.sqrt(np.mean((p-y)**2)))
def load_pixels():
    d=dict(np.load(_repo_path(OLD, 'inputs.npz')));d['graphs']=json.loads((_repo_path(OLD, 'graphs.json')).read_text())
    d['pixels']=np.zeros((500,1,256,256),np.uint8)
    for j in d['dev']:
        with Image.open(_repo_path(ROOT, f"data/train/{d['ids'][j]}.png")) as im:
            assert im.mode=='L' and im.size==(256,256);d['pixels'][j,0]=np.asarray(im)
    return d
def descriptor(x):
    # Image-local, fixed proxies. They cannot certify the actual unknown physics.
    x=np.asarray(x,np.float32);out=[float(x.mean()),float(x.std()),*np.quantile(x,[.01,.05,.1,.2,.3,.5,.7,.8,.9,.95,.99]).tolist()]
    for s in [1,2,4,8]:
        u=ndi.gaussian_filter(x,s);gx=ndi.sobel(u,0)/8;gy=ndi.sobel(u,1)/8;g=np.hypot(gx,gy)
        out.extend([float(u.std()),float((x-u).std()),float(g.mean()),float(g.std()),float(np.quantile(g,.9))])
    u=ndi.gaussian_filter(x,1);m=float(u.mean());sd=max(float(u.std()),1e-6)
    for t in [-1.5,-1.,-.5,.5,1.,1.5]:
        mask=u<m+t*sd;_,n=ndi.label(mask);out.extend([float(mask.mean()),n/x.size])
    return np.array(out,float)
def regions(pixels):
    n=len(pixels);mask=np.empty_like(pixels,dtype=np.float16);means=np.zeros((n,17),np.float32);sds=np.empty(n,np.float32)
    tile=(np.arange(256)[:,None]//64*4+np.arange(256)[None,:]//64+1).astype(np.int64)
    for j,p in enumerate(pixels):
        x=p[0].astype(np.float32)/255;u=ndi.gaussian_filter(x,1);g=np.hypot(ndi.sobel(u,0),ndi.sobel(u,1));lo,hi=np.quantile(u,[.1,.9]);interior=(g<np.quantile(g,.6))&(u>lo)&(u<hi)
        inside=ndi.binary_erosion(interior);w=ndi.gaussian_filter(inside.astype(np.float32),1)*interior;mask[j,0]=w;sds[j]=x.std()
        for k in range(1,17):
            z=(tile==k)*w;means[j,k]=float((x*z).sum()/max(z.sum(),1e-6))
    return mask,means,sds,tile
def photo(x,mask,means,sds,tile,rng,strength):
    # x is native float[0,1], original-image region analysis, no output clipping.
    n=len(x);choose=rng.random((n,17))<.5;choose[:,0]=False
    a=-strength*rng.uniform(.5,1,(n,17))*choose;b=strength*.25*rng.uniform(-1,1,(n,17))*choose
    aa=a[:,tile][:,None];bb=b[:,tile][:,None];mm=means[:,tile][:,None]
    out=x+mask.astype(np.float32)*(aa*(x-mm)+bb*sds[:,None,None,None])
    assert np.isfinite(out).all();return out.astype(np.float32)
def boundary(x,strength):
    u=ndi.gaussian_filter(x,1);g=np.hypot(ndi.sobel(u,0),ndi.sobel(u,1));m=g>=np.quantile(g,.85)
    # Center excluded from the surrounding3x3 source average.
    surround=(ndi.uniform_filter(x,3)*9-x)/8
    return (x+strength*m*(surround-x)).astype(np.float32)
def edge_agreement(x,z):
    def g(a):
        u=ndi.gaussian_filter(a,1);return np.hypot(ndi.sobel(u,0),ndi.sobel(u,1))
    a=g(x);b=g(z);cut=np.quantile(a,.85);m=a>=cut;n=b>=cut
    return float((m&n).sum()/max((m|n).sum(),1))
def ridge_predict(x,y,q):
    scale=StandardScaler().fit(x);model=Ridge(alpha=100).fit(scale.transform(x),y)
    return model.predict(scale.transform(q)),dict(mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_))
def safety():
    assert (_repo_path(HERE, 'starting-state.json')).exists();d=load_pixels();dev=d['dev'];pix=d['pixels'][dev];m,means,sd,tile=regions(pix)
    X=np.stack([descriptor(p[0]/255.) for p in pix]);features={};edges={}
    for typ,levels in [('photo',[.03,.10,.30]),('boundary',[.05,.15,.40])]:
        for strength in levels:
            key=f'{typ}-{strength:.2f}';rows=[];ee=[]
            for i,p in enumerate(pix):
                x=p[0].astype(np.float32)/255
                z=photo(x[None,None],m[i:i+1],means[i:i+1],sd[i:i+1],tile,np.random.default_rng(SEED+1+int(dev[i])),strength)[0,0] if typ=='photo' else boundary(x,strength)
                rows.append(descriptor(z));ee.append(edge_agreement(x,z))
            features[key]=np.stack(rows);edges[key]=np.array(ee);print('SCREEN_DESCRIPTORS',key,flush=True)
    arrays('safety-features.npz',ids=d['ids'][dev],original=X,**features,**{k+'_edge':v for k,v in edges.items()})
    mapping={int(j):i for i,j in enumerate(dev)};records=[]
    for fold,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            train=np.array(g['A' if stage=='selector' else 'T']);ix=np.array([mapping[int(j)] for j in train]);y=d['y'][train];hi=d['noise'][train]>=12
            splits=list(KFold(3,shuffle=True,random_state=SEED+fold).split(train));orig=np.zeros(len(ix));variant={k:np.zeros(len(ix)) for k in features}
            for a,b in splits:
                orig[b]=ridge_predict(X[ix[a]],y[a],X[ix[b]])[0]
                for key,Fx in features.items():variant[key][b]=ridge_predict(Fx[ix[a]],y[a],Fx[ix[b]])[0]
            rng=np.random.default_rng(SEED+fold);draw=rng.integers(0,len(y),(2000,len(y)));hidx=np.flatnonzero(hi);hdraw=hidx[rng.integers(0,len(hidx),(2000,len(hidx)))];vr={}
            scale=X[ix].std(0);keep=scale>1e-6
            for key,Fx in features.items():
                cor=np.array([np.corrcoef(X[ix,c],Fx[ix,c])[0,1] for c in np.flatnonzero(keep)]);drift=np.sqrt(np.mean(((Fx[ix][:,keep]-X[ix][:,keep])/scale[keep])**2,1))
                harm=np.sqrt(((variant[key]-y)[draw]**2).mean(1))-np.sqrt(((orig-y)[draw]**2).mean(1));hh=np.sqrt(((variant[key]-y)[hdraw]**2).mean(1))-np.sqrt(((orig-y)[hdraw]**2).mean(1))
                checks=dict(correlation_median=float(np.median(cor))>=.98,correlation_p10=float(np.quantile(cor,.1))>=.90,drift_p95=float(np.quantile(drift,.95))<=.25,edge_p05=float(np.quantile(edges[key][ix],.05))>=.95,label_overall_upper90=float(np.quantile(harm,.95))<=.25,label_high_upper90=float(np.quantile(hh,.95))<=.50)
                vr[key]=dict(passed=all(checks.values()),checks=checks,correlation_median=float(np.median(cor)),correlation_p10=float(np.quantile(cor,.1)),drift_p95=float(np.quantile(drift,.95)),edge_p05=float(np.quantile(edges[key][ix],.05)),probe_original_rmse=rmse(orig,y),probe_view_rmse=rmse(variant[key],y),probe_original_high_rmse=rmse(orig[hi],y[hi]),probe_view_high_rmse=rmse(variant[key][hi],y[hi]),RMSE_harm_CI90=np.quantile(harm,[.05,.95]).tolist(),high_RMSE_harm_CI90=np.quantile(hh,[.05,.95]).tolist())
            strength=.03 if vr['photo-0.03']['passed'] else 0.;records.append(dict(fold=fold,stage=stage,train_ids=d['ids'][train].tolist(),selected_strength=strength,variants=vr))
            print('SAFETY',fold,stage,'strength',strength,'weak',vr['photo-0.03']['checks'],flush=True)
    save('safety.json',dict(records=records,continue_training=sum(r['selected_strength']==0 for r in records)<=2,limits='Proxy noninferiority does not prove unknown physical information preserved. Stronger and boundary variants diagnostic only; no boundary invariance forced.',source_sha256=sha(__file__),plan_sha256=sha(_repo_path(HERE, 'PLAN.md'))))

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList();inc=1
        for i,out in enumerate([16,32,48,64]):
            self.blocks.append(nn.Sequential(nn.Conv2d(inc,out,7 if i==0 else 3,stride=4 if i==0 else 2,padding=3 if i==0 else 1),nn.GroupNorm(4,out),nn.SiLU()));inc=out
        self.project=nn.Sequential(nn.Linear(322,64),nn.SiLU(),nn.Dropout(.25),nn.Linear(64,32));self.head=nn.Linear(32,1);self.ssl=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,32));nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        stats=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()];h=x
        for b in self.blocks:h=b(h);stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
        z=self.project(torch.cat(stats,1));return self.head(z).squeeze(1),z,self.ssl(z)
def vicreg(a,b):
    inv=F.mse_loss(a,b);var=.5*(F.relu(1-(a.var(0)+1e-4).sqrt()).mean()+F.relu(1-(b.var(0)+1e-4).sqrt()).mean());cov=0.
    for z in [a,b]:
        z=z-z.mean(0);c=z.T@z/(len(z)-1);cov=cov+(c.square().sum()-c.diag().square().sum())/z.shape[1]/2
    return 25*inv+25*var+cov,torch.stack([inv,var,cov])
@torch.no_grad()
def predict(model,pixels,meta):
    model.eval();pp=[];zz=[]
    for first in range(0,len(pixels),32):
        x=(pixels[first:first+32].astype(np.float32)/255-meta['image_mean'])/meta['image_std'];p=[];z=[]
        for rot,flip in [(0,False),(1,False),(0,True),(1,True)]:
            v=np.rot90(x,rot,axes=(2,3));v=v[:,:,:,::-1] if flip else v
            a,b,_=model(torch.from_numpy(np.ascontiguousarray(v)));p.append(a.numpy());z.append(b.numpy())
        pp.extend(np.mean(p,0).astype(float));zz.extend(np.mean(z,0))
    return np.array(pp)*meta.get('target_std',1)+meta.get('target_mean',0),np.array(zz)
def phase(model,pixels,reg,meta,seed,epochs,strength,targets=None):
    """SSL capability receives no labels; supervised capability gets train labels only."""
    torch.manual_seed(seed);rng=np.random.default_rng(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.0001)
    mask,means,sds,tile=reg;curve=[];avg=None;order_hash=hashlib.sha256();views_hash=hashlib.sha256();start=time.monotonic();steps=0
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(pixels));order_hash.update(order.tobytes());total=0.;parts=np.zeros(3)
        # Split into balanced batches so variance is defined, including tiny probes.
        for ix in np.array_split(order,int(np.ceil(len(order)/32))):
            raw=pixels[ix].astype(np.float32)/255;views=[]
            for _ in range(2):
                v=photo(raw,mask[ix],means[ix],sds[ix],tile,rng,strength)
                rot=int(rng.integers(4));flip=bool(rng.random()<.5);v=np.rot90(v,rot,axes=(2,3));v=v[:,:,:,::-1] if flip else v
                views_hash.update(v.tobytes());views.append(np.ascontiguousarray((v-meta['image_mean'])/meta['image_std']))
            p,z,s=model(torch.from_numpy(np.concatenate(views)))
            if targets is None:loss,terms=vicreg(*s.chunk(2));parts+=terms.detach().numpy()*len(ix)
            else:
                yy=torch.from_numpy(targets[ix]);loss=F.mse_loss(p,torch.cat([yy,yy]))
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);optimizer.step();steps+=1;total+=float(loss.detach())*len(ix)
            limits()
        sched.step();curve.append(dict(epoch=epoch+1,loss=total/len(pixels),ssl_terms=(parts/len(pixels)).tolist()))
        if epoch>=epochs-min(4,epochs):
            state={k:v.detach().clone()/min(4,epochs) for k,v in model.state_dict().items()}
            if avg is None:avg=state
            else:
                for k,v in state.items():avg[k]+=v
    model.load_state_dict(avg);del optimizer,sched,avg;gc.collect()
    return dict(seconds=time.monotonic()-start,updates=steps,epochs=epochs,curve=curve,source_visits=epochs*len(pixels),view_visits=2*epochs*len(pixels),order_sha256=order_hash.hexdigest(),views_sha256=views_hash.hexdigest(),state_sha256=state_sha(model.state_dict()),peak_RSS_bytes=limits(),labels_used=targets is not None)
def rank(z):
    sv=np.linalg.svd(z-z.mean(0),compute_uv=False);mass=sv**2/max(float((sv**2).sum()),1e-12)
    return dict(effective_rank=float(np.exp(-(mass*np.log(np.maximum(mass,1e-30))).sum())),std_mean=float(z.std(0).mean()))
def train():
    screen=json.loads((_repo_path(HERE, 'safety.json')).read_text());assert screen['continue_training'];assert sha(__file__)==screen['source_sha256'];d=load_pixels();elapsed=0.;outputs=[]
    save('code-lock.json',dict(source_sha256=sha(__file__),plan_sha256=sha(_repo_path(HERE, 'PLAN.md')),safety_sha256=sha(_repo_path(HERE, 'safety.json')),seed=SEED,epochs=EPOCHS,arms=ARMS))
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);px=d['pixels'][tr].copy();qp=d['pixels'][q].copy();y=d['y'][tr].copy();ids=d['ids'][tr].tolist();reg=regions(px)
            safe=next(r for r in screen['records'] if r['fold']==k and r['stage']==stage);assert safe['train_ids']==ids;strength=safe['selected_strength'];seed=SEED+100*k
            meta=dict(image_mean=float(px.mean(dtype=np.float64)/255),image_std=float(px.std(dtype=np.float64)/255),target_mean=float(y.mean()),target_std=float(y.std()),train_ids=ids);yz=((y-meta['target_mean'])/meta['target_std']).astype(np.float32)
            for arm in ARMS:
                if elapsed>CAP:raise RuntimeError('Neural budget exhausted')
                folder=_repo_path(HERE, f'models/fold{k}-{stage}-{arm}');folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);model=Model();initial=state_sha(model.state_dict());record=dict(fold=k,stage=stage,arm=arm,seed=seed,strength=strength,initial_sha256=initial,normalization=meta,training_pixels_sha256=hashlib.sha256(px.tobytes()).hexdigest(),training_labels_sha256=hashlib.sha256(y.tobytes()).hexdigest())
                if arm in ['none','ssl']:
                    if arm=='ssl':
                        # Strip target statistics before entering SSL: no label-derived input.
                        sslmeta={n:meta[n] for n in ['image_mean','image_std','train_ids']};r=phase(model,px,reg,sslmeta,seed+10000,EPOCHS,strength);elapsed+=r['seconds'];record['pretraining']=r
                        torch.save(dict(state_dict=model.state_dict(),normalization=sslmeta,record=r),_repo_io(folder/'ssl.pt'))
                    _,zt=predict(model,px,meta);_,zq=predict(model,qp,meta);probe,head=ridge_predict(zt,y,zq);arrays(str((folder/'frozen-probe.npz').relative_to(HERE)),prediction=probe,train=tr,query=q,train_features=zt,query_features=zq);save(str((folder/'frozen-head.json').relative_to(HERE)),head);record['frozen_representation']=rank(zt)
                if arm=='long':
                    r=phase(model,px,reg,meta,seed+10000,EPOCHS,strength,yz);elapsed+=r['seconds'];record['first_supervised']=r
                r=phase(model,px,reg,meta,seed+20000,EPOCHS,strength if arm!='none' else 0,yz);elapsed+=r['seconds'];record['supervised']=r
                pred,zq=predict(model,qp,meta);pt,zt=predict(model,px,meta);record['train_rmse']=rmse(pt,y);record['representation']=rank(zt)
                torch.save(dict(state_dict=model.state_dict(),normalization=meta,arm=arm,seed=seed,source_sha256=sha(__file__)),_repo_io(folder/'model.pt'));arrays(str((folder/'predictions.npz').relative_to(HERE)),prediction=pred,embedding=zq,train=tr,query=q);save(str((folder/'training.json').relative_to(HERE)),record);outputs.append(str(folder.relative_to(HERE)))
                print('FIT_COMPLETE',k,stage,arm,'neural_seconds',round(elapsed,1),'rssMiB',round(limits()/2**20,1),flush=True);del model;gc.collect()
            del px,qp,reg;gc.collect()
            completed=len(outputs)/4;projected=elapsed/completed*10
            if completed==1 and projected>CAP:raise RuntimeError(f'First-partition projected budget {projected:.1f}s exceeds {CAP}s')
    save('training-complete.json',dict(complete=True,models=outputs,neural_seconds=elapsed,peak_RSS_bytes=limits(),source_sha256=sha(__file__)))
def selfcheck():
    a=torch.tensor([[1.,2.],[3.,1.],[2.,4.]],requires_grad=True);b=torch.tensor([[1.,1.],[2.,3.],[4.,2.]],requires_grad=True);v,t=vicreg(a,b)
    aa=a.detach().numpy().astype(float);bb=b.detach().numpy().astype(float);inv=((aa-bb)**2).mean();var=np.mean([np.maximum(0,1-np.sqrt(x.var(0,ddof=1)+1e-4)).mean() for x in [aa,bb]]);cov=np.mean([sum(np.cov(x,rowvar=False)[i,j]**2 for i in range(2) for j in range(2) if i!=j)/2 for x in [aa,bb]])
    assert abs(float(v.detach())-(25*inv+25*var+cov))<1e-5;v.backward();assert a.grad.abs().sum()>0
    x=np.random.default_rng(7).integers(0,256,(4,1,256,256),dtype=np.uint8);reg=regions(x);raw=x.astype(np.float32)/255;out=photo(raw,*reg,np.random.default_rng(1),0);assert np.array_equal(out,raw)
    w=reg[0];out=photo(raw,*reg,np.random.default_rng(1),.03);assert np.array_equal(out[w==0],raw[w==0]);save('selfcheck.json',dict(vicreg_numpy_error=abs(float(v.detach())-(25*inv+25*var+cov)),zero_strength_identity=True,unselected_pixels_unchanged=True,source_sha256=sha(__file__)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['safety','train','selfcheck']);args=ap.parse_args();sys.addaudithook(guard);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):globals()[args.action]()
