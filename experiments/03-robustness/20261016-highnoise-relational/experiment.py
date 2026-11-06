"""Nested raw-image neighborhood learning; no teacher targets in new training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
(_repo_path(HERE, 'tmp')).mkdir(exist_ok=True);os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'));os.environ['MPLCONFIGDIR']=str(_repo_path(HERE, 'mpl-cache'))
import tempfile
tempfile.tempdir=str(_repo_path(HERE, 'tmp'))
import ast,argparse,hashlib,json,time,resource,gc,csv
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
R4=_repo_path(ROOT, 'experiments/rmse-research-20261015/round4-full073')
C76=_repo_path(ROOT, 'experiments/076-complement-20261016')
ARMS=('mse','relational','shuffled')
SEED=2026101651;EPOCHS=40;SIGMA=.5;TEMPERATURE=.2

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def state_sha(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def save(name,obj):
    p=_repo_path(HERE, name);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def arrays(path,**kw):
    p=_repo_path(HERE, path);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(_repo_io(p),**kw)
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:]
        write=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if write and not p.is_relative_to(HERE):raise RuntimeError('Outside experiment write: '+str(p))
        if p.is_relative_to(_repo_path(ROOT, 'data/test')) or (p.is_relative_to(_repo_path(ROOT, 'artifacts')) and p.name.startswith('submission-')):raise RuntimeError('Test prediction/pixel read prohibited in research')
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))

def calibration_functions():
    source=_repo_path(R4, 'noise_calibration.py');tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fit_weights','select']]
    ns=dict(np=np,STRENGTHS=('global',100,30,10,0));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    return ns['select']

def load():
    d=dict(np.load(_repo_path(HERE, 'inputs.npz')));d['graphs']=json.loads((_repo_path(HERE, 'graphs.json')).read_text())
    # The new image learner never trains on historical holdout pixels.
    pix=np.zeros((500,1,256,256),np.uint8)
    for j in d['dev']:
        with Image.open(_repo_path(ROOT, f"data/train/{d['ids'][j]}.png")) as im:
            assert im.mode=='L' and im.size==(256,256);pix[j,0]=np.asarray(im,np.uint8)
    d['pixels']=pix;return d

def prepare():
    assert (_repo_path(HERE, 'starting-state.json')).exists(),'Protection inventory must complete first'
    source=_repo_path(R4.parent, 'inputs.npz');d=dict(np.load(_repo_io(source)));y=d['y'];base=dict(np.load(_repo_path(C76, 'baseline076.npz')))
    graphs=json.loads((_repo_path(R4, 'label-graph.json')).read_text());select=calibration_functions();prior=json.loads((_repo_path(R4, 'noise-calibration/summary.json')).read_text())['aux']['selection']
    used={str(source.relative_to(ROOT)):sha(source),str((_repo_path(C76, 'baseline076.npz')).relative_to(ROOT)):sha(_repo_path(C76, 'baseline076.npz')),str((_repo_path(R4, 'label-graph.json')).relative_to(ROOT)):sha(_repo_path(R4, 'label-graph.json')),str((_repo_path(R4, 'noise_calibration.py')).relative_to(ROOT)):sha(_repo_path(R4, 'noise_calibration.py'))}
    pred=base['p76'].copy();records=[];pos={int(j):i for i,j in enumerate(d['dev'])}
    for k,g in enumerate(graphs):
        A,B,T,V=[np.array(g[n],int) for n in ['A','B','T','V']]
        assert len(A)==213 and len(B)==107 and len(T)==320 and len(V)==80
        assert set(A).isdisjoint(B) and set(A)|set(B)==set(T) and set(T).isdisjoint(V)
        assert np.array_equal(np.sort(np.r_[T,V]),np.sort(d['dev']))
        tpath=_repo_path(C76, f'teachers076/fold{k}-outer0/predictions.npz');t=np.load(_repo_io(tpath))
        assert np.array_equal(t['train'],A) and np.array_equal(t['query'],B)
        apath=_repo_path(R4, f'anchors/fold{k}.npz');a=np.load(_repo_io(apath))
        assert np.array_equal(a['A'],A) and np.array_equal(a['B'],B) and np.array_equal(a['train_ids'],d['ids'][A])
        ipath=_repo_path(R4, f'aux-mean2/fold{k}-selection.npz');inner=np.load(_repo_io(ipath));assert np.array_equal(inner['B'],B)
        good=inner['eligible'];weights,meta=select(inner['residual'][good],(y[B]-inner['base073'])[good],d['noise'][B][good]>=12)
        assert np.array_equal(weights,np.array(prior[k]['addition']['weights']))
        p76B=.16*a['prediction_B']+.84*t['base'];p76B[~good]=a['prediction_B'][~good]
        highB=(d['noise'][B]>=12)&good;p82B=p76B+highB*weights[1]*inner['residual']
        rv=[]
        for seed in [0,1]:
            path=_repo_path(R4, f'aux-seed{seed}/fold{k}-final/predictions.npz');z=np.load(_repo_io(path));assert np.array_equal(z['train_index'],T) and np.array_equal(z['validation_index'],V);rv.append(z['validation_prediction']);used[str(path.relative_to(ROOT))]=sha(path)
            mp=_repo_path(R4, f'aux-seed{seed}/fold{k}-selector/model.pt');ck=torch.load(_repo_io(mp),map_location='cpu',weights_only=True);assert ck['normalization']['train_ids']==d['ids'][A].tolist();used[str(mp.relative_to(ROOT))]=sha(mp)
        ix=np.array([pos[int(j)] for j in V]);eligible=(base['noise'][ix]>=12)&~base['failed'][ix]
        p82V=base['p76'][ix]+eligible*weights[1]*np.mean(rv,axis=0);pred[ix]=p82V
        arrays(f'anchors/fold{k}.npz',A=A,B=B,T=T,V=V,ix=ix,p76B=p76B,p82B=p82B,p76V=base['p76'][ix],p82V=p82V,highB=highB,highV=eligible,old_weight=weights[1])
        records.append(dict(fold=k,old_weights=weights.tolist(),old_selection=meta,training_ids=d['ids'][A].tolist(),calibration_ids=d['ids'][B].tolist(),validation_ids=d['ids'][V].tolist()))
        for path in [tpath,apath,ipath]:used[str(path.relative_to(ROOT))]=sha(path)
    ref=np.load(_repo_path(C76, 'probe-oof.npz'))['old4'];assert np.max(abs(pred-ref))<1e-10
    assert np.array_equal(base['ids'],d['ids'][d['dev']]) and np.array_equal(base['fold'],d['fold'][0])
    arrays('inputs.npz',ids=d['ids'],y=y,dev=d['dev'],fold=d['fold'][0],noise=d['noise'],base076=base['p76'],base082=pred,failed=base['failed'])
    save('graphs.json',graphs);save('baseline-verification.json',dict(baseline076_rmse=rmse(y[d['dev']],base['p76']),baseline082_rmse=rmse(y[d['dev']],pred),baseline082_max_reconstruction_error=float(np.max(abs(pred-ref))),records=records,source_sha256=used,
        limitation='Old base representations fixed. B is calibration-training, not a held-out score for coefficients fitted on B.'))
    save('code-lock.json',dict(source_sha256=sha(__file__),plan_sha256=sha(_repo_path(HERE, 'PLAN.md')),torch_version=str(torch.__version__),seed=SEED,epochs=EPOCHS,arms=ARMS))
    print('PREPARED',rmse(y[d['dev']],pred),flush=True)

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList();inc=1
        for i,out in enumerate([16,32,48,64]):
            self.blocks.append(nn.Sequential(nn.Conv2d(inc,out,7 if i==0 else 3,stride=4 if i==0 else 2,padding=3 if i==0 else 1),nn.GroupNorm(4,out),nn.SiLU()));inc=out
        self.project=nn.Sequential(nn.Linear(322,64),nn.SiLU(),nn.Dropout(.25),nn.Linear(64,32));self.head=nn.Linear(32,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        stats=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()];h=x
        for block in self.blocks:
            h=block(h);stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
        emb=self.project(torch.cat(stats,1));return self.head(emb).squeeze(1),emb

def neighborhood_loss(a,b,y):
    a=F.normalize(a-a.mean(0,keepdim=True),dim=1,eps=1e-6);b=F.normalize(b-b.mean(0,keepdim=True),dim=1,eps=1e-6)
    target_logits=-.5*((y[:,None]-y[None,:])/SIGMA).square();logq=target_logits.log_softmax(1);q=logq.exp()
    logits=a@b.T/TEMPERATURE
    return .5*((q*(logq-logits.log_softmax(1))).sum(1).mean()+(q*(logq-logits.T.log_softmax(1))).sum(1).mean())

def normalize(pixels,meta):return ((pixels.astype(np.float32)/255.-meta['image_mean'])/meta['image_std']).astype(np.float32)

@torch.no_grad()
def predict(model,pixels,meta):
    model.eval();out=[];emb=[]
    for first in range(0,len(pixels),32):
        x=normalize(pixels[first:first+32],meta);ps=[];es=[]
        for rot,flip in [(0,False),(1,False),(0,True),(1,True)]:
            z=np.rot90(x,rot,axes=(2,3));z=z[:,:,:,::-1] if flip else z
            p,e=model(torch.from_numpy(np.ascontiguousarray(z)));ps.append(p.numpy());es.append(e.numpy())
        out.extend(np.mean(ps,axis=0).astype(float)*meta['target_std']+meta['target_mean']);emb.extend(np.mean(es,axis=0))
    return np.asarray(out),np.asarray(emb)

def fit_model(pixels,labels,ids,arm,seed,folder,epochs=EPOCHS):
    """Training capability contains only training pixels, targets and IDs."""
    assert len(pixels)==len(labels)==len(ids) and len(set(ids))==len(ids)
    assert pixels.dtype==np.uint8 and pixels.shape[1:]==(1,256,256) and all(str(i).startswith('TRAIN_') for i in ids)
    folder.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=ImageModel();initial=state_sha(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.0001)
    meta=dict(image_mean=float(pixels.mean(dtype=np.float64)/255.),image_std=float(pixels.std(dtype=np.float64)/255.),target_mean=float(labels.mean()),target_std=float(labels.std()),train_ids=list(ids),training_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),training_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
    labels_z=((labels-meta['target_mean'])/meta['target_std']).astype(np.float32);perm=np.random.default_rng(seed+7919).permutation(len(labels));relations=labels_z[perm] if arm=='shuffled' else labels_z
    context_hash=hashlib.sha256(relations.tobytes()).hexdigest();curve=[];average=None;n_average=min(5,epochs);steps=0;exposures=hashlib.sha256();views_hash=hashlib.sha256();gradients=None
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(labels));exposures.update(order.astype(np.int64).tobytes());main_sum=rel_sum=0.
        for first in range(0,len(order),32):
            ix=order[first:first+32];x=normalize(pixels[ix],meta);views=[];draw=[]
            for _ in range(2):
                rot=int(rng.integers(4));flip=bool(rng.random()<.5);draw.extend([rot,int(flip)]);v=np.rot90(x,rot,axes=(2,3));v=v[:,:,:,::-1] if flip else v;views.append(np.ascontiguousarray(v))
            views_hash.update(np.array(draw,np.int64).tobytes());xb=torch.from_numpy(np.concatenate(views));p,z=model(xb);yy=torch.from_numpy(labels_z[ix]);rel_y=torch.from_numpy(relations[ix]);z1,z2=z.chunk(2)
            mse=F.mse_loss(p,torch.cat([yy,yy]));rel=neighborhood_loss(z1,z2,rel_y);loss=mse+(0. if arm=='mse' else 1.)*rel
            assert torch.isfinite(loss)
            if steps==2:
                params=list(model.blocks.parameters());g0=torch.autograd.grad(mse,params,retain_graph=True);g1=torch.autograd.grad(rel,params,retain_graph=True)
                gradients=dict(mse_encoder_norm=float(torch.sqrt(sum(g.square().sum() for g in g0))),relational_encoder_norm=float(torch.sqrt(sum(g.square().sum() for g in g1))),relational_weight=0 if arm=='mse' else 1)
                assert gradients['relational_encoder_norm']>1e-9
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);optimizer.step();steps+=1
            main_sum+=float(mse.detach())*len(ix);rel_sum+=float(rel.detach())*len(ix)
        schedule.step();curve.append(dict(epoch=epoch+1,online_hardness_rmse=float(np.sqrt(main_sum/len(labels))*meta['target_std']),neighborhood_KL=float(rel_sum/len(labels))))
        if epoch>=epochs-n_average:
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if average is None:average={k:v/n_average for k,v in state.items()}
            else:
                for k,v in state.items():average[k]+=v/n_average
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak>1.5*2**30:raise RuntimeError('Own1.5GiB memory cap exceeded; no other process touched')
        if (epoch+1)%20==0:print(folder.name,arm,'epoch',epoch+1,'seconds',round(time.monotonic()-start,1),flush=True)
    model.load_state_dict(average);trainpred,trainemb=predict(model,pixels,meta);centered=trainemb-trainemb.mean(0);sv=np.linalg.svd(centered,compute_uv=False);mass=sv**2/max(float((sv**2).sum()),1e-12)
    ck=dict(state_dict=average,normalization=meta,arm=arm,seed=seed,epochs=epochs,architecture='pixel-CNN-322-moments-64-32-1',neighborhood_sigma=SIGMA,temperature=TEMPERATURE,neighborhood_weight=0 if arm=='mse' else 1,source_sha256=sha(__file__))
    torch.save(ck,_repo_io(folder/'model.pt'));record=dict(arm=arm,seed=seed,epochs=epochs,updates=steps,source_visits=epochs*len(labels),view_visits=2*epochs*len(labels),initial_sha256=initial,state_sha256=state_sha(average),order_sha256=exposures.hexdigest(),views_sha256=views_hash.hexdigest(),relation_target_sha256=context_hash,parameter_count=sum(p.numel() for p in model.parameters()),gradients=gradients,curve=curve,
        train_rmse=rmse(trainpred,labels),embedding_std_mean=float(trainemb.std(0).mean()),embedding_effective_rank=float(np.exp(-np.sum(mass*np.log(np.maximum(mass,1e-30))))),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,normalization=meta)
    save(str((folder/'training.json').relative_to(HERE)),record);del optimizer,schedule,average;gc.collect();return model,meta,record

def checkpoint(path):
    ck=torch.load(_repo_io(path),map_location='cpu',weights_only=True);model=ImageModel();model.load_state_dict(ck['state_dict']);return model,ck['normalization']

def selfcheck():
    torch.manual_seed(7);a=torch.randn(6,4,requires_grad=True);b=torch.randn(6,4,requires_grad=True);y=torch.tensor([-.9,-.8,-.1,.1,.8,1.])
    value=neighborhood_loss(a,b,y);value.backward();assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    def soft(x):x=x-x.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)
    az=a.detach().numpy().astype(float);bz=b.detach().numpy().astype(float);az-=az.mean(0);bz-=bz.mean(0);az/=np.linalg.norm(az,axis=1,keepdims=True);bz/=np.linalg.norm(bz,axis=1,keepdims=True);yy=y.numpy().astype(float)
    q=soft(-.5*((yy[:,None]-yy[None,:])/SIGMA)**2);l=az@bz.T/TEMPERATURE
    manual=.5*np.mean(np.sum(q*(np.log(q)-np.log(soft(l))),1)+np.sum(q*(np.log(q)-np.log(soft(l.T))),1));assert abs(float(value.detach())-manual)<1e-6
    perm=torch.tensor([3,0,5,2,4,1]);assert torch.allclose(value.detach(),neighborhood_loss(a.detach()[perm],b.detach()[perm],y[perm]),atol=1e-6)
    save('loss-check.json',dict(passed=True,independent_KL_error=abs(float(value.detach())-manual),permutation_equivariance=True,gradients_active=True,source_sha256=sha(__file__)))

def probe():
    d=load();g=d['graphs'][0];A=np.array(g['A']);records=[]
    for arm in ARMS:
        model,meta,rec=fit_model(d['pixels'][A],d['y'][A].copy(),d['ids'][A].tolist(),arm,SEED,_repo_path(HERE, 'resource-probe', arm),epochs=2);records.append(rec);del model;gc.collect()
    for key in ['initial_sha256','order_sha256','views_sha256','updates','view_visits']:assert len({r[key] for r in records})==1,key
    per_step=sum(r['seconds']/r['updates'] for r in records)/3;projection=per_step*40*(7+10)*5*3
    save('resource-probe.json',dict(arms=records,projected_main_seconds=projection,validation_labels_scored=False,passed=projection<5400));assert projection<5400
    print('PROBE',round(projection/60,1),'projected minutes',flush=True)

def run(repeat=0):
    lock=json.loads((_repo_path(HERE, 'code-lock.json')).read_text());assert lock['source_sha256']==sha(__file__) and lock['plan_sha256']==sha(_repo_path(HERE, 'PLAN.md'))
    assert json.loads((_repo_path(HERE, 'resource-probe.json')).read_text())['passed'];d=load();start=time.monotonic();records=[]
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A'] if stage=='selector' else g['T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V'])
            assert set(tr).isdisjoint(q);reports=[]
            for arm in ARMS:
                folder=_repo_path(HERE, f'models/seed{repeat}/fold{k}-{stage}-{arm}');seed=SEED+100*k+100000*repeat
                if (folder/'training.json').exists():
                    model,meta=checkpoint(folder/'model.pt');rec=json.loads((folder/'training.json').read_text());assert rec['seed']==seed and rec['arm']==arm and meta['train_ids']==d['ids'][tr].tolist()
                else:model,meta,rec=fit_model(d['pixels'][tr],d['y'][tr].copy(),d['ids'][tr].tolist(),arm,seed,folder)
                vp,emb=predict(model,d['pixels'][q],meta);loaded,lm=checkpoint(folder/'model.pt');rp,re=predict(loaded,d['pixels'][q],lm);assert np.array_equal(vp,rp) and np.array_equal(emb,re)
                target=folder/'predictions.npz'
                if target.exists():
                    saved=np.load(_repo_io(target));assert np.array_equal(saved['query'],q) and np.array_equal(saved['prediction'],vp)
                else:arrays(str(target.relative_to(HERE)),train=tr,query=q,ids=d['ids'][q],prediction=vp,embedding=emb)
                if not (folder/'verification.json').exists():save(str((folder/'verification.json').relative_to(HERE)),dict(checkpoint_reload_exact=True,query_labels_accepted=False,train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),model_sha256=sha(folder/'model.pt')))
                reports.append(rec);records.append(dict(fold=k,stage=stage,arm=arm,seconds=rec['seconds']));del model,loaded;gc.collect()
                print('COMPLETE',repeat,k,stage,arm,'total_minutes',round((time.monotonic()-start)/60,1),flush=True)
            for key in ['initial_sha256','order_sha256','views_sha256','updates','view_visits']:assert len({r[key] for r in reports})==1,(k,stage,key)
    save(f'training-complete-seed{repeat}.json',dict(complete=True,records=records,seconds=time.monotonic()-start,new_fits=30))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','selfcheck','probe','train']);parser.add_argument('--repeat',type=int,default=0,choices=[0,1]);args=parser.parse_args()
    sys.addaudithook(guard);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if args.action=='train':run(args.repeat)
        else:globals()[args.action]()
