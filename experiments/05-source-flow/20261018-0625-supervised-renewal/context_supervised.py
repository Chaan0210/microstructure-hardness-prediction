"""Direct supervised context Transformer; source-local pretraining and inner selection.

The previous context study froze this encoder and compressed features with
PCA/PLS. Here all encoder and head parameters learn original-image hardness,
with a small reconstruction regularizer. Hardness sees full original images,
rotated by C4; token masking occurs only in the auxiliary model computation.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];OLD=_repo_path(ROOT, 'experiments/structure-context-20261017-1415')
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','TORCHINDUCTOR_CACHE_DIR','MPLCONFIGDIR']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
import argparse,copy,hashlib,json,math,time,resource
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from context_components import ContextNet,targets,losses,mask_batch,normalize

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def rmse(p,y):return float(np.mean((p-y)**2)**.5)
def statehash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def source(d,px,fold,stage,arm,repeat):
    folder=_repo_path(OLD, f'encoders/fold{fold}-{stage}-seed{repeat}-spatial');file=folder/('model.pt' if arm=='pretrained' else 'initial.pt');ck=torch.load(_repo_io(file),map_location='cpu',weights_only=True);record=json.loads((folder/'training.json').read_text());meta=ck['normalization'];ids={str(v):i for i,v in enumerate(d['ids'])};tr=np.array([ids[str(i)] for i in meta['train_ids']])
    assert len(set(tr))==len(tr);expected=record['state_sha256' if arm=='pretrained' else 'initial_sha256'];assert statehash(ck['state_dict'])==expected
    assert hashlib.sha256(np.ascontiguousarray(px[tr]).tobytes()).hexdigest()==record['training_pixels_sha256']
    assert hashlib.sha256((_repo_path(OLD, 'encoder.py')).read_bytes()).hexdigest()==record['source_sha256']
    net=ContextNet();net.load_state_dict(ck['state_dict'],strict=True)
    return net,meta,tr,dict(path=str(file.relative_to(ROOT)),state_sha256=expected,original_training_source_sha256=record['source_sha256'])

def input_batch(px,ix,meta,device,rot=0):return torch.tensor(np.ascontiguousarray(np.rot90(normalize(px[ix],meta),rot,axes=(2,3))),device=device)
def perm(x):return torch.arange(256,device=x.device).expand(len(x),-1)
def raw_features(x):return torch.stack([x.pow(k).mean((1,2,3)) for k in [1,2,3,4]],1)

@torch.no_grad()
def extract(net,px,ix,meta,device,batch=16):
    net.eval();rows=[]
    for first in range(0,len(ix),batch):
        ii=ix[first:first+batch];values=[]
        for rot in range(4):
            x=input_batch(px,ii,meta,device,rot);_,f=net(x,perm(x),decode=False);values.append(torch.cat([f,raw_features(x)],1).cpu().numpy())
        rows.append(np.mean(values,0))
    return np.concatenate(rows)

class Model(nn.Module):
    def __init__(self,body,mean,scale,weight,bias):
        super().__init__();self.body=body;self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32));self.head=nn.Linear(len(mean),1)
        with torch.no_grad():self.head.weight.copy_(torch.tensor(weight[None],dtype=torch.float32));self.head.bias.fill_(float(bias))
    def forward(self,x):
        _,f=self.body(x,perm(x),decode=False);f=torch.cat([f,raw_features(x)],1);return self.head(((f-self.mean)/self.scale).clamp(-8,8))[:,0]

@torch.no_grad()
def predict(model,px,ix,meta,device,ym,ys,batch=16):
    model.eval();out=[]
    for first in range(0,len(ix),batch):
        ii=ix[first:first+batch];v=[model(input_batch(px,ii,meta,device,r)).cpu().numpy() for r in range(4)];out.extend(np.mean(v,0)*ys+ym)
    return np.array(out)

def fit_stage(a,d,px,stage,out,epochs,select=False,probe=False):
    out.mkdir(exist_ok=True);body,meta,tr,provenance=source(d,px,a.fold,stage,a.arm,a.repeat);T=d['dev'][d['fold']!=a.fold];V=d['dev'][d['fold']==a.fold];q=np.array(sorted(set(T)-set(tr))) if stage=='selector' else V
    if stage=='refit':assert set(T)==set(tr)
    assert not set(tr)&set(q) and not set(tr)&set(V);body.to(a.device);torch.manual_seed(a.seed)
    if a.device=='mps':torch.mps.manual_seed(a.seed)
    F=extract(body,px,tr,meta,a.device);mean=F.mean(0);scale=np.maximum.reduce([F.std(0),.1*np.sqrt((F*F).mean(0)),np.full(F.shape[1],.001)]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());ridge=Ridge(alpha=100).fit(np.clip((F-mean)/scale,-8,8),(d['y'][tr]-ym)/ys);model=Model(body,mean,scale,ridge.coef_,ridge.intercept_).to(a.device);prior={k:p.detach().clone() for k,p in model.body.named_parameters()};rng=np.random.default_rng(a.seed)
    spec=vars(a)|dict(stage=stage,epochs=epochs,select=select,train=tr.tolist(),query=q.tolist(),train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),initialization=provenance,
        original_pretraining='Source-local100epochmaskedcontext pretraining; all normalization/images exclude thisstagequery andouterV. Reused verified checkpoint, unchanged.',
        objective='Fulloriginal+C4 hardnessMSE +.1originalpixel masked-tokenauxiliaryloss +100/n*linearheadL2 +.001encoderL2SP. No fixedteacherresidualtargets.',
        selection='SelectorA213/B107choose0/20/...200usingEMA0.8perepoch;refitT320withsame200epochLRhorizon. Noouterqueryselection.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),components_sha256=hashlib.sha256((_repo_path(HERE, 'context_components.py')).read_bytes()).hexdigest())
    save(out/'recipe.json',spec);initial=predict(model,px,q,meta,a.device,ym,ys);np.savez(_repo_io(out/'initial.npz'),query=q,prediction=initial);bestloss=rmse(initial,d['y'][q]);bestepoch=0;ema={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best=copy.deepcopy(ema);curve=[dict(epoch=0,query_rmse=bestloss)];first=0;start=time.monotonic()
    opt=torch.optim.AdamW([dict(params=model.body.parameters(),lr=.0001),dict(params=model.head.parameters(),lr=.0005)],weight_decay=.001)
    if (out/'resume.pt').exists() and not probe:
        ck=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert ck['recipe']==spec;model.load_state_dict(ck['raw_state']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);first=ck['epoch'];curve=ck['curve'];ema=ck['ema'];best=ck['best'];bestepoch=ck['bestepoch'];bestloss=ck['bestloss']
        if a.device=='mps':torch.mps.set_rng_state(ck['mps_rng'])
        for s in opt.state.values():
            for k,v in s.items():
                if torch.is_tensor(v) and k!='step':s[k]=v.to(a.device)
    times=[]
    for epoch in range(first,epochs):
        factor=.1+.9*.5*(1+math.cos(math.pi*epoch/a.horizon));opt.param_groups[0]['lr']=.0001*factor;opt.param_groups[1]['lr']=.0005*factor;model.train();order=rng.permutation(tr);total=0
        for first_row in range(0,len(order),a.batch):
            tic=time.monotonic();ix=order[first_row:first_row+a.batch];x=input_batch(px,ix,meta,a.device,int(rng.integers(4)));y=torch.tensor((d['y'][ix]-ym)/ys,dtype=torch.float32,device=a.device);pred=model(x);hard=(pred-y).square().mean()
            ax=x[:min(len(x),8)];mask=torch.tensor(mask_batch(len(ax),rng),device=a.device);reconstruction,_=model.body(ax,perm(ax),mask);aux=losses(reconstruction,targets(ax,perm(ax)),mask)
            anchor=.001*sum((p-prior[k]).square().sum() for k,p in model.body.named_parameters());loss=hard+.1*aux+100/len(tr)*model.head.weight.square().sum()+anchor;assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();grad=float(nn.utils.clip_grad_norm_(model.parameters(),5.));assert np.isfinite(grad)
            if not probe:opt.step()
            if a.device=='mps':torch.mps.synchronize()
            times.append(time.monotonic()-tic);total+=float(hard.detach())*len(ix)
            if probe and len(times)==3:
                r=dict(median_step_seconds=float(np.median(times[1:])),projected_selector_refit_200_seconds=float(np.median(times[1:])*200*(14+20)),gradient_norm=grad,optimizer_updates=0,parameter_count=sum(p.numel() for p in model.parameters()),peak_memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if a.device=='mps' else 0));save(out/'probe.json',r);print('PROBE',json.dumps(r),flush=True);return r
        current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for k,v in current.items():ema[k].mul_(.8).add_(v,alpha=.2)
        row=dict(epoch=epoch+1,online_train_rmse=np.sqrt(total/len(tr))*ys,seconds=time.monotonic()-start)
        if (epoch+1)%20==0 or epoch+1==epochs:
            model.load_state_dict(ema);pr=predict(model,px,q,meta,a.device,ym,ys);score=rmse(pr,d['y'][q]);model.load_state_dict(current);row['query_rmse']=score
            if select and score<bestloss:bestloss=score;bestepoch=epoch+1;best=copy.deepcopy(ema)
            print(stage,json.dumps(row),flush=True)
        curve.append(row)
        if (epoch+1)%20==0 or epoch+1==epochs:
            ck=dict(raw_state=current,optimizer=opt.state_dict(),recipe=spec,epoch=epoch+1,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if a.device=='mps' else None,curve=curve,ema=ema,best=best,bestepoch=bestepoch,bestloss=bestloss);torch.save(ck,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(out/'resume.pt');save(out/'curve.json',curve)
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if a.device=='mps' else 0);assert mem<6*2**30
    final=best if select else ema;model.load_state_dict(final);pr=predict(model,px,q,meta,a.device,ym,ys);torch.save(dict(state=final,recipe=spec,normalization=meta,ym=ym,ys=ys),_repo_io(out/'model.pt'));np.savez(_repo_io(out/'predictions.npz'),train=tr,query=q,prediction=pr,initial=initial);r=dict(rmse=rmse(pr,d['y'][q]),initial_rmse=rmse(initial,d['y'][q]),selected_epoch=bestepoch if select else epochs,seconds=time.monotonic()-start);save(out/'done.json',r);return r

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.device=='mps':assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.35)
    d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');out=_repo_path(HERE, 'runs', a.tag);out.mkdir(exist_ok=True)
    if a.action=='probe':return fit_stage(a,d,px,'selector',out/'probe',1,probe=True)
    sp=out/'selector/done.json';r=json.loads(sp.read_text()) if sp.exists() else fit_stage(a,d,px,'selector',out/'selector',a.horizon,select=True);result=fit_stage(a,d,px,'refit',out/'refit',r['selected_epoch']);z=np.load(_repo_io(out/'refit/predictions.npz'));q=z['query'];ix=d['fold']==a.fold;yy=d['y'][q];base=d['base083'][ix];pred=z['prediction'];ref=np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz'))['C73_cnn'][0,ix];weight=np.where(d['failed'][ix],1.,.16)
    result.update(base083_rmse=rmse(base,yy),quarter_blend_rmse=rmse(.75*base+.25*pred,yy),replace_image_branch_rmse=rmse(base+weight*(pred-ref),yy));np.savez(_repo_io(out/'predictions.npz'),query=q,prediction=pred,base083=base,image_replacement=base+weight*(pred-ref));save(out/'done.json',result);print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['probe','train']);ap.add_argument('--tag',default='context-supervised-f0-pretrained');ap.add_argument('--fold',type=int,default=0);ap.add_argument('--repeat',type=int,default=0);ap.add_argument('--arm',choices=['pretrained','random'],default='pretrained');ap.add_argument('--seed',type=int,default=2026101943);ap.add_argument('--horizon',type=int,default=200);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--device',choices=['cpu','mps'],default='cpu')
    with threadpool_limits(2):main(ap.parse_args())
