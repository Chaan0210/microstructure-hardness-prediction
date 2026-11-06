"""Dense boundary and sampled maximin-affinity learning on source-local masks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import graph as graph
import targets
import numpy as np,json,time,sys,gc,math,hashlib
import torch
from torch import nn
from torch.nn import functional as F
EPOCHS=40;BATCH=32;SEEDS=[2026101793,2026111793];ARMS=['topology','local','hardness'];DEADLINE=None
def state_hash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def seed_all(seed):torch.manual_seed(seed);torch.mps.manual_seed(seed)
def conv(i,o):return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU(),nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU())
def pos():
    y,x=np.mgrid[:16,:16];freq=1/10000**(np.arange(24)/24);p=np.c_[np.sin(x.ravel()[:,None]*freq),np.cos(x.ravel()[:,None]*freq),np.sin(y.ravel()[:,None]*freq),np.cos(y.ravel()[:,None]*freq)]
    return torch.from_numpy(p.astype(np.float32))[None]
class Block(nn.Module):
    def __init__(self):
        super().__init__();self.n1=nn.LayerNorm(96);self.qkv=nn.Linear(96,288);self.out=nn.Linear(96,96);self.n2=nn.LayerNorm(96);self.ff=nn.Sequential(nn.Linear(96,192),nn.GELU(),nn.Dropout(.1),nn.Linear(192,96));self.drop=nn.Dropout(.1)
    def forward(self,x):
        b,n,_=x.shape;q,k,v=self.qkv(self.n1(x)).reshape(b,n,3,4,24).permute(2,0,3,1,4).unbind(0);z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(b,n,96);x=x+self.drop(self.out(z));return x+self.drop(self.ff(self.n2(x)))
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.s1=conv(1,16);self.s2=conv(16,32);self.s3=conv(32,64);self.s4=conv(64,96);self.register_buffer('position',pos());self.attention=nn.Sequential(Block(),Block());self.norm=nn.LayerNorm(96);self.d3=conv(160,64);self.d2=conv(96,32);self.d1=conv(48,16);self.boundary=nn.Conv2d(16,1,1);self.affinity=nn.Conv2d(64,2,3,padding=1);self.head=nn.Sequential(nn.Linear(320,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x):
        a=self.s1(F.avg_pool2d(x,2));b=self.s2(F.avg_pool2d(a,2));d=self.s3(F.avg_pool2d(b,2));h=self.s4(F.avg_pool2d(d,2));z=self.norm(self.attention(h.flatten(2).transpose(1,2)+self.position)).transpose(1,2).reshape(-1,96,16,16)
        d=self.d3(torch.cat([F.interpolate(z,scale_factor=2,mode='bilinear',align_corners=False),d],1));b=self.d2(torch.cat([F.interpolate(d,scale_factor=2,mode='bilinear',align_corners=False),b],1));a=self.d1(torch.cat([F.interpolate(b,scale_factor=2,mode='bilinear',align_corners=False),a],1));f=torch.cat([z.mean((2,3)),(z.var((2,3),correction=0)+1e-6).sqrt(),d.mean((2,3)),(d.var((2,3),correction=0)+1e-6).sqrt()],1);q=self.affinity(d);edges=torch.cat([q[:,0,:,:-1].flatten(1),q[:,1,:-1,:].flatten(1)],1)
        return self.boundary(a)[:,0],edges,f,self.head(f)[:,0]
def init(seed):seed_all(seed);return Net().to('mps')
def transform(x,rot,flip):
    x=np.rot90(x,rot,axes=(-2,-1));return np.ascontiguousarray(x[...,::-1] if flip else x)
def normalize(x,meta):return (x.astype(np.float32)/255-meta['image_mean'])/meta['image_std']
def balanced(logit,y,w):
    loss=F.binary_cross_entropy_with_logits(logit,y,reduction='none');shape=(len(logit),-1);loss=loss.reshape(shape);y=y.reshape(shape);w=w.reshape(shape);total=0.;classes=0.
    for cls in [0,1]:
        ww=w*(y==cls);den=ww.sum(1);total=total+(loss*ww).sum(1)/den.clamp(min=1e-12);classes=classes+(den>0).float()
    return total/classes.clamp(min=1)
def sample(nodes,available,rng):
    ys=[];ws=[];pairs=[]
    for n,v in zip(nodes,available):
        y,w=targets.target_pairs(n,targets.PAIR,int(v));chosen=[]
        for label in [0,1]:
            ix=np.flatnonzero((y==label)&(w>0));chosen.extend(rng.choice(ix,32,replace=len(ix)<32) if len(ix) else np.arange(label*32,(label+1)*32))
        chosen=np.array(chosen);ww=w[chosen].copy()
        for label in [0,1]:
            if not np.any((y==label)&(w>0)):ww[label*32:(label+1)*32]=0
        pairs.append(targets.PAIR[chosen]);ys.append(y[chosen]);ws.append(ww)
    return np.array(pairs),np.array(ys),np.array(ws)
def batch(z,ix,meta,rng):
    rot=int(rng.integers(4));flip=bool(rng.random()<.5);x=transform(normalize(z['pixels'][ix],meta),rot,flip);n=transform(z['nodes'][ix].reshape(-1,4,32,32),rot,flip).reshape(-1,4,1024);by=transform(z['boundary'][ix],rot,flip);bw=transform(z['boundary_weight'][ix],rot,flip);edge=[targets.target_pairs(q,np.c_[graph.U,graph.V],int(v)) for q,v in zip(n,z['available'][ix])];pair,py,pw=sample(n,z['available'][ix],rng)
    return x,by,bw,np.array([q[0] for q in edge]),np.array([q[1] for q in edge]),pair,py,pw
def objective(outputs,pack,arm,y=None):
    b,e,f,p=outputs
    if arm=='hardness':return (p-y).square().mean(),None
    _,by,bw,ey,ew,pair,py,pw=pack;convert=lambda x:torch.from_numpy(x).to('mps');loss=balanced(b,convert(by),convert(bw))+balanced(e,convert(ey),convert(ew));crit=None
    if arm=='topology':
        values=e.detach().cpu().numpy();crit=np.stack([graph.critical_edges(v,q) for v,q in zip(values,pair)]);chosen=torch.gather(e,1,torch.from_numpy(crit).to('mps'));loss=loss+balanced(chosen,convert(py),convert(pw))
    return loss.mean(),crit
def subset(z,ix):return {k:z[k][ix].copy() for k in ['pixels','nodes','boundary','boundary_weight','available']}
def fit(model,z,meta,arm,seed,y=None,epochs=EPOCHS):
    assert len(z['pixels'])==len(meta['train_ids']);assert (y is None)==(arm!='hardness');c.FIT_ACTIVE=True;seed_all(seed+41);rng=np.random.default_rng(seed+41);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02);curve=[];avg=None;steps=0;orderhash=hashlib.sha256();viewhash=hashlib.sha256();start=time.monotonic();peak=c.memory()
    try:
        for epoch in range(epochs):
            if DEADLINE and time.monotonic()>DEADLINE:raise RuntimeError('Frozen computation budget reached')
            lr=.001*(epoch+1)/5 if epoch<5 else .00005+.000475*(1+math.cos(math.pi*(epoch-4)/(epochs-5)))
            for q in opt.param_groups:q['lr']=lr
            order=rng.permutation(len(z['pixels']));orderhash.update(order.tobytes());model.train();total=0.
            for ix in np.array_split(order,int(np.ceil(len(order)/BATCH))):
                pack=batch(z,ix,meta,rng);viewhash.update(pack[0].tobytes());x=torch.from_numpy(pack[0]).to('mps');outputs=model(x);target=None if y is None else torch.from_numpy(y[ix].astype(np.float32)).to('mps');loss,_=objective(outputs,pack,arm,target);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);opt.step();total+=float(loss.detach().cpu())*len(ix);steps+=1
            curve.append(dict(epoch=epoch+1,loss=total/len(order),lr=lr));mem=c.memory();peak={k:max(peak[k],mem[k]) for k in peak}
            if epoch>=epochs-5:
                state={k:v.detach().cpu().clone()/5 for k,v in model.state_dict().items()}
                if avg is None:avg=state
                else:
                    for k,v in state.items():avg[k]+=v
            if (epoch+1)%20==0:print('EPOCH',arm,seed,epoch+1,round(time.monotonic()-start,1),round(total/len(order),5),flush=True)
        model.load_state_dict(avg);torch.mps.synchronize();return dict(seconds=time.monotonic()-start,steps=steps,curve=curve,order_sha256=orderhash.hexdigest(),view_sha256=viewhash.hexdigest(),state_sha256=state_hash(model.state_dict()),peak_memory=peak,hardness_used=arm=='hardness',dataset_reads_blocked=True)
    finally:c.FIT_ACTIVE=False;del opt;gc.collect();torch.mps.empty_cache()
def edge_mapping(rot,flip):
    nodes=transform(np.arange(1024).reshape(32,32),rot,flip).ravel();lookup={tuple(sorted((int(u),int(v)))):j for j,(u,v) in enumerate(zip(graph.U,graph.V))};mapping=np.array([lookup[tuple(sorted((int(nodes[u]),int(nodes[v]))))] for u,v in zip(graph.U,graph.V)]);assert len(np.unique(mapping))==1984;return mapping
@torch.no_grad()
def features(model,px,meta):
    model.eval();latent=[];signature=[];boundaries=[];edges=[];pairs,radii=graph.fixed_pairs()
    for first in range(0,len(px),BATCH):
        x=normalize(px[first:first+BATCH],meta);ff=[];bb=[];ee=[]
        for rot,flip in [(0,False),(1,False),(0,True),(1,True)]:
            b,q,f,_=model(torch.from_numpy(transform(x,rot,flip)).to('mps'));b=b.sigmoid().cpu().numpy();q=q.sigmoid().cpu().numpy();f=f.cpu().numpy();b=b[...,::-1] if flip else b;b=np.rot90(b,-rot,axes=(-2,-1));original=np.empty_like(q);original[:,edge_mapping(rot,flip)]=q;ff.append(f);bb.append(b);ee.append(original)
        b=np.mean(bb,0);q=np.mean(ee,0);latent.extend(np.mean(ff,0));boundaries.extend(b);edges.extend(q);signature.extend([graph.signature(bi,qi,pairs,radii) for bi,qi in zip(b,q)])
    return dict(latent=np.array(latent),signature=np.array(signature),boundary=np.array(boundaries),edges=np.array(edges))
def weak_scores(f,z):
    out=[]
    for b,q,n,available,by,bw in zip(f['boundary'],f['edges'],z['nodes'],z['available'],z['boundary'],z['boundary_weight']):
        ey,ew=targets.target_pairs(n,np.c_[graph.U,graph.V],int(available));py,pw=targets.target_pairs(n,targets.PAIR,int(available));pred=q[graph.critical_edges(q,targets.PAIR)]
        def loss(p,y,w):
            p=np.clip(p,1e-6,1-1e-6);v=-(y*np.log(p)+(1-y)*np.log(1-p));a=[]
            for cls in [0,1]:
                ww=w*(y==cls)
                if ww.sum()>0:a.append(float((v*ww).sum()/ww.sum()))
            return float(np.mean(a)) if a else float(np.log(2))
        out.append([loss(b,by,bw),loss(q,ey,ew),loss(pred,py,pw)])
    return np.array(out)
def load():
    z=dict(np.load(_repo_io(c.HERE/'targets.npz')));d=c.inputs();lookup={int(v):i for i,v in enumerate(z['dev'])};return z,d,lookup
def meta(z,idx,ids):
    px=z['pixels'][idx];return dict(image_mean=float(px.mean(dtype=np.float64)/255),image_std=float(px.std(dtype=np.float64)/255),train_ids=ids.tolist())
def probe():
    z,d,pos=load();tr=np.array(d['graphs'][0]['A'][:BATCH]);ix=np.array([pos[int(j)] for j in tr]);s=subset(z,ix);norm=meta(z,ix,d['ids'][tr]);times={};gradients={};peak={};start=time.monotonic()
    for arm in ARMS:
        m=init(SEEDS[0]);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);rng=np.random.default_rng(71);record=[]
        for step in range(40):
            torch.mps.synchronize();t=time.monotonic();pack=batch(s,np.arange(len(tr)),norm,rng);o=m(torch.from_numpy(pack[0]).to('mps'));y=torch.zeros(len(tr),device='mps') if arm=='hardness' else None;loss,crit=objective(o,pack,arm,y);opt.zero_grad(set_to_none=True);loss.backward();gradient=float(torch.linalg.vector_norm(m.s1[0].weight.grad).cpu());nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();torch.mps.synchronize();record.append(time.monotonic()-t);assert np.isfinite(float(loss.detach().cpu()));peak=c.memory()
        times[arm]=float(np.median(record[20:]));gradients[arm]=gradient;del opt,m;gc.collect();torch.mps.empty_cache()
    matrix=sum(int(np.ceil(len(g['A'])/BATCH))+int(np.ceil(len(g['T'])/BATCH)) for g in d['graphs'])*2*EPOCHS;projection=matrix*sum(times.values())*1.15+600+900;passed=bool(projection<6600 and all(x>0 for x in gradients.values()))
    c.save(_repo_io('resource-probe-b32.json'),dict(passed=passed,seconds_per_batch=times,gradients=gradients,projected_matrix_features_readouts_seconds=projection,batch=BATCH,epochs=EPOCHS,trained_models=60,source_ids=d['ids'][tr].tolist(),true_hardness_read=False,scalar_runtime_target='zeros only in discarded timing probe',parameter_count=sum(p.numel() for p in Net().parameters()),seconds=time.monotonic()-start,peak_memory=peak,source_sha256=c.sha(__file__)));print('RESOURCE PROBE',passed,times,projection,flush=True)
def train():
    global DEADLINE
    pr=json.loads((c.HERE/'resource-probe-b32.json').read_text());account=json.loads((c.HERE/'resource-accounting.json').read_text());assert account['passed'] and account['probe_sha256']==c.sha(c.HERE/'resource-probe-b32.json');assert json.loads((c.HERE/'graph-verification.json').read_text())['all_passed'];z,d,pos=load();yd=c.inputs(labels=True);start=time.monotonic();DEADLINE=start+5700;done=[]
    c.save(_repo_io('encoder-code-lock.json'),dict(source_sha256=c.sha(__file__),target_source_sha256=c.sha(targets.__file__),graph_source_sha256=c.sha(graph.__file__),common_sha256=c.sha(c.__file__),plan_sha256=c.sha(c.HERE/'PLAN.md'),seeds=SEEDS,epochs=EPOCHS,batch=BATCH,torch=str(torch.__version__),device='mps-float32'))
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);ti=np.array([pos[int(j)] for j in tr]);qi=np.array([pos[int(j)] for j in q]);s=subset(z,ti);norm=meta(z,ti,d['ids'][tr]);labels=yd['y'][tr];ym=float(labels.mean());ys=float(labels.std());y=(labels-ym)/ys
            for repeat,seed0 in enumerate(SEEDS):
                seed=seed0+100*k
                for arm in ARMS:
                    path=c.HERE/f'encoders/fold{k}-{stage}-seed{repeat}-{arm}';path.mkdir(parents=True,exist_ok=False);m=init(seed);initial=state_hash(m.state_dict())
                    if arm=='topology':
                        torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=norm),_repo_io(path/'initial.pt'));a=features(m,z['pixels'][ti],norm);b=features(m,z['pixels'][qi],norm);c.arrays(str((path/'random-features.npz').relative_to(c.HERE)),train=tr,query=q,train_latent=a['latent'],train_signature=a['signature'],query_latent=b['latent'],query_signature=b['signature'])
                    record=fit(m,s,norm,arm,seed,y if arm=='hardness' else None);a=features(m,z['pixels'][ti],norm);b=features(m,z['pixels'][qi],norm);weak=weak_scores(b,subset(z,qi));torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=norm,arm=arm,seed=seed,y_mean=ym if arm=='hardness' else None,y_std=ys if arm=='hardness' else None),_repo_io(path/'model.pt'));c.arrays(str((path/'features.npz').relative_to(c.HERE)),train=tr,query=q,train_latent=a['latent'],train_signature=a['signature'],query_latent=b['latent'],query_signature=b['signature'],weak_query_losses=weak,query_boundary=b['boundary'].astype(np.float16),query_edges=b['edges'].astype(np.float16))
                    record.update(initial_sha256=initial,normalization=norm,training_pixels_sha256=c.ah(s['pixels']),training_nodes_sha256=c.ah(s['nodes']),hardness_sha256=c.ah(labels) if arm=='hardness' else None,arm=arm,seed=seed,source_sha256=c.sha(__file__));c.save(_repo_io(str((path/'training.json').relative_to(c.HERE))),record);done.append(str(path.relative_to(c.HERE)));print('ENCODER_COMPLETE',k,stage,repeat,arm,'wall_seconds',round(time.monotonic()-start,1),flush=True);del m,a,b;gc.collect();torch.mps.empty_cache()
    c.save(_repo_io('encoders-complete.json'),dict(complete=True,models=done,seconds=time.monotonic()-start,source_sha256=c.sha(__file__)))
if __name__=='__main__':
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):globals()[sys.argv[1]]()
