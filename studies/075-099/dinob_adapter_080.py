"""One fixed rank4 attention adapter, with unchanged B heads and009 calibration."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,copy,argparse,resource,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_calibration_079 as previous
import dinob_062 as backbone
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
fv=previous.fv;h=previous.h;prior=previous.prior;SEEDS=previous.SEEDS
OUT=_repo_path(ROOT, 'artifacts/dinob-adapter-080');CFG={'rank':4,'alpha':4,'epochs':8,'microbatch':2,'effective_batch':16,'lr':1e-4,'weight_decay':.1,'anchor_weight':1.,'clip':1.,'seed':20270201}

def inputs(split):
 d=previous.inputs(split);v=json.loads((_repo_path(ROOT, 'reports/dinob-complement-064-validation.json')).read_text());r=prior.OUT/split/'report.json';assert fv.sha(r)==v['results'][split]['report_sha256'];d['head_source']=_repo_path(ROOT, json.loads(r.read_text())['B_model_source']);p=prior.OUT/split/'development_predictions.npz';assert fv.sha(p)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];d['Bmean']=z['Bmean'].copy()
 return d

def prepare():
 previous.check();backbone.check();OUT.mkdir(exist_ok=False);pub=json.loads((_repo_path(ROOT, 'reports/candidate009-public-submission.json')).read_text());assets={}
 for s in ('original','20261031','20261101'):
  d=inputs(s)
  for k in range(5):
   for seed in SEEDS:
    p=d['head_source']/f'fold-{k}/baseB-seed-{seed}.pt';assets[str(p.relative_to(ROOT))]=fv.sha(p)
   for p in [d['head_source']/f'fold-{k}/report.json',previous.base.OUT/s/f'fold-{k}/report.json']:assets[str(p.relative_to(ROOT))]=fv.sha(p)
 p={'hypothesis':'Official ViT-B frozen representation may admit a small regression-specific correction. Fit only low-rank attention updates inside the final B block while holding the same three existing fold-trained MLP heads, their scalers, all original backbone parameters and009 affine coefficients fixed. This isolates representation adaptation and does not add calibration search.',
 'duplicate_audit':{'043':'ViT-S14 lastblock fullparameters,8epochs1e-6 with fixed singlelayerRidge/MSE+anchor.','059':'ViT-S043 adaptedfeatures then unchangedMLP; no jointbackprop or newbackbone fit.','080':'ViT-B14 finalblock qkv/proj rank4 adapters trained through existing3MLPmean; no prior equivalentB experiment found.'},
 'config':CFG,'architecture':'Zero-initialized up projection and Normal(0,.02) down projection rank4, alpha/rank=1, on finalblock attention qkv and outputproj only. 18432trainableparameters; original block, norm, heads frozen in eval. CLS/patchmean/std blocks6/9/12 unchanged except adaptedblock12. 504px and same6912head input.',
 'objective':'SmoothL1 beta1 against outertrain standardized targets +1.0MSE to original frozenBmean standardized training predictions; AdamW weightdecay.1; clipping1;8fixedepochs, no validation curves or epoch selection. Low rank, anchoring, small fixed training budget constrain adaptation. Existing B heads already fit training very closely, so residual signal may be too small; this risk is recorded before results.',
 'primary':'Keep009formula and its previously verified outertrain-only affine coefficients unchanged. Replace only B mean with adaptedB: new009 = reference009 + old_outertrain_slope * .125*(adaptedB-frozenB). No refitted calibrator, ratio or contrast term. Identical frozenhead/scaler control must reproduce Bmean and009.',
 'selection':'Original pooledprimaryRMSE strictly improves009 -> both existingresplits unchanged. Candidate only if all3pooledRMSE strictly improve and independent verification passes. Foldcounts/bootstrapdiagnostics only. If originalfails preserve/reject without resplits. No alternativesweep.',
 'resources':'Existing CPU2/MPS only. Microbenchmark6gradient steps before cache, project original5fold<1800seconds,RSS/MPSdriver<4GB. Cache only development400 frozenprefix400x1297x768float32(~1.59GB). First completefold is reused benchmark and must take<360seconds and RSS/MPSdriver<4GB before remainingfolds. No install/download/cost.',
 'scope':'Development400only;noholdout/testselection;preservebest009. Onlyaftervalidatedrecipe lock may final500fit/testinference proceed. No upload.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinob_adapter_080.py','validate_dinob_adapter_080.py','dinob_062.py','nested_calibration_079.py','nested_calibration_074.py','dinob_mlp_head_063.py']},'inputs':{**assets,**{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/dino-043-validation.json','reports/representation-059-validation.json','reports/dinob-complement-064-validation.json','reports/nested-calibration-074-validation.json','reports/candidate009-public-submission.json']}},'protected':pub['protected_sha256'],'bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'config':CFG,'duplicate_audit':p['duplicate_audit']}),flush=True)

def check():
 previous.check();backbone.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,s in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return p

class LoRALinear(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.base.requires_grad_(False);self.down=nn.Linear(base.in_features,CFG['rank'],bias=False);self.up=nn.Linear(CFG['rank'],base.out_features,bias=False);nn.init.normal_(self.down.weight,std=.02);nn.init.zeros_(self.up.weight)
 def forward(self,x):return self.base(x)+self.up(self.down(x))*(CFG['alpha']/CFG['rank'])

class Tail(nn.Module):
 def __init__(self,m,states,seed):
  super().__init__();torch.manual_seed(seed);self.block=copy.deepcopy(m.blocks[-1]);self.norm=copy.deepcopy(m.norm);self.heads=nn.ModuleList()
  for a in states:
   hh=h.Head(6912);hh.load_state_dict(a['model']);self.heads.append(hh)
  self.requires_grad_(False);self.block.attn.qkv=LoRALinear(self.block.attn.qkv);self.block.attn.proj=LoRALinear(self.block.attn.proj);self.register_buffer('mean',states[0]['scaler_mean'].float());self.register_buffer('scale',states[0]['scaler_scale'].float());self.eval()
 def forward(self,z,early):
  z=self.norm(self.block(z));p=z[:,1:];late=torch.cat((z[:,0],p.mean(1),p.std(1,correction=0)),1);x=(torch.cat((early,late),1)-self.mean)/self.scale;return torch.stack([a(x)[0] for a in self.heads]).mean(0)

def head_states(d,k):
 tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);sc=StandardScaler().fit(d['B'][tr]);states=[];p=[];rec=json.loads((d['head_source']/f'fold-{k}/report.json').read_text())
 for seed in SEEDS:
  path=d['head_source']/f'fold-{k}/baseB-seed-{seed}.pt';assert fv.sha(path)==rec['models'][f'baseB-{seed}']['checkpoint_sha256'];a=torch.load(_repo_io(path),weights_only=True,map_location='cpu');assert a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va] and a['seed']==seed+k and a['config']==h.CONFIG;assert a['target_mean']==float(d['y'][tr].mean()) and a['target_std']==float(d['y'][tr].std())
  for n,v in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[n].numpy(),v)
  mm=h.Head(6912);mm.load_state_dict(a['model']);p.append(h.predict(mm,sc.transform(d['B']),a['target_mean'],a['target_std']));states.append(a)
 baseline=np.mean(p,0);np.testing.assert_array_equal(baseline[va],d['Bmean'][va]);return states,baseline

def prefix(m,x):
 with torch.no_grad():
  z=m.prepare_tokens_with_masks(x)
  for b in m.blocks[:-1]:z=b(z)
 return z.detach()

def tensor(a):return torch.tensor(np.array(a,copy=True),dtype=torch.float32,device='mps')
def frozen_hash(t):return h.state_hash({n:v for n,v in t.state_dict().items() if not (n.endswith('down.weight') or n.endswith('up.weight'))})
def predict(t,d,tokens,ix):
 vals=[]
 with torch.no_grad():
  for start in range(0,len(ix),CFG['microbatch']):
   ii=ix[start:start+CFG['microbatch']];vals.append(t(tensor(tokens[ii]),tensor(d['B'][ii,:4608])).cpu().numpy().astype(float))
 return np.concatenate(vals)

def microbenchmark():
 check();torch.set_num_threads(2);d=inputs('original');tr=np.flatnonzero(d['fid']!=0);sel=tr[:2];states,old=head_states(d,0);m=backbone.model().to('mps');z=prefix(m,torch.tensor(np.concatenate([backbone.ds.inputs(d['ids'][i],'full504') for i in sel]),device='mps'));t=Tail(m,states,CFG['seed']).to('mps');yy=tensor((d['y'][sel]-states[0]['target_mean'])/states[0]['target_std']);anchor=tensor((old[sel]-states[0]['target_mean'])/states[0]['target_std']);early=tensor(d['B'][sel,:4608]);warm=t(z,early).detach().cpu().numpy().astype(float)*states[0]['target_std']+states[0]['target_mean'];np.testing.assert_allclose(warm,old[sel],atol=.003,rtol=0);frozen=frozen_hash(t);params=[p for p in t.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=CFG['weight_decay']);times=[]
 for step in range(6):
  torch.mps.synchronize();now=time.perf_counter();opt.zero_grad(set_to_none=True);p=t(z,early);loss=F.smooth_l1_loss(p,yy,beta=1.)+CFG['anchor_weight']*F.mse_loss(p,anchor);loss.backward();torch.nn.utils.clip_grad_norm_(params,CFG['clip']);opt.step();torch.mps.synchronize()
  if step:times.append(time.perf_counter()-now)
 assert frozen_hash(t)==frozen and all(p.grad is None for p in m.parameters());r={'trainable_parameters':sum(p.numel() for p in params),'median_step_seconds':float(np.median(times)),'projected_original5fold_train_seconds':float(np.median(times)*160*CFG['epochs']*5),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'warm_control_max_error':float(abs(warm-old[sel]).max()),'frozen_parameters_unchanged':True,'training_ids':[d['ids'][i] for i in sel]};r['resource_pass']=r['projected_original5fold_train_seconds']<1800 and r['rss_peak_bytes']<4e9 and r['mps_driver_bytes']<4e9;fv.write(_repo_path(OUT, 'microbenchmark.json'),r);print(json.dumps(r),flush=True);assert r['resource_pass']

def cache():
 check();assert json.loads((_repo_path(OUT, 'microbenchmark.json')).read_text())['resource_pass'];torch.set_num_threads(2);d=inputs('original');m=backbone.model().to('mps');before=h.state_hash(m.state_dict());path=_repo_path(OUT, 'prefix.npy');assert not path.exists();a=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(400,1297,768));start=time.perf_counter();error=0.
 for i,ident in enumerate(d['ids']):
  x=torch.tensor(backbone.ds.inputs(ident,'full504'),device='mps');z=prefix(m,x);a[i]=z.cpu().numpy()[0]
  with torch.no_grad():
   n=m.norm(m.blocks[-1](z));p=n[:,1:];v=torch.cat((n[:,0],p.mean(1),p.std(1,correction=0)),1).cpu().numpy()[0]
  error=max(error,float(abs(v-d['B'][i,-2304:]).max()));np.testing.assert_array_equal(v,d['B'][i,-2304:])
  if (i+1)%80==0:print(json.dumps({'cached_rows':i+1,'seconds':time.perf_counter()-start}),flush=True)
 a.flush();del a;assert h.state_hash(m.state_dict())==before and all(p.grad is None for p in m.parameters());r={'ids':d['ids'],'shape':[400,1297,768],'sha256':fv.sha(path),'frozen_B_tail_descriptor_max_error':error,'all400_frozen_descriptor_exact':True,'seconds':time.perf_counter()-start,'weights_unchanged':True};fv.write(_repo_path(OUT, 'prefix.json'),r);print(json.dumps({k:v for k,v in r.items() if k!='ids'}),flush=True)

def get_cache(d):
 r=json.loads((_repo_path(OUT, 'prefix.json')).read_text());assert r['ids']==d['ids'] and fv.sha(_repo_path(OUT, 'prefix.npy'))==r['sha256'];return np.load(_repo_path(OUT, 'prefix.npy'),mmap_mode='r')

def fit_fold(d,tokens,m,k,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);states,old=head_states(d,k);ym=states[0]['target_mean'];ys=states[0]['target_std'];t=Tail(m,states,CFG['seed']+k).to('mps');fh=frozen_hash(t);initial=h.state_hash(t.state_dict());warm=predict(t,d,tokens,va)*ys+ym;np.testing.assert_allclose(warm,old[va],atol=.003,rtol=0);params=[p for p in t.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=CFG['weight_decay']);orders=np.random.default_rng(CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(CFG['epochs'],1)),axis=1);curves=[]
 for epoch,order in enumerate(orders):
  vals=[]
  for startbatch in range(0,len(tr),CFG['effective_batch']):
   group=tr[order[startbatch:startbatch+CFG['effective_batch']]];opt.zero_grad(set_to_none=True)
   for first in range(0,len(group),CFG['microbatch']):
    ii=group[first:first+CFG['microbatch']];p=t(tensor(tokens[ii]),tensor(d['B'][ii,:4608]));loss=F.smooth_l1_loss(p,tensor((d['y'][ii]-ym)/ys),beta=1.)+CFG['anchor_weight']*F.mse_loss(p,tensor((old[ii]-ym)/ys));assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();vals.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(params,CFG['clip']);assert torch.isfinite(gn);opt.step()
  curves.append({'epoch':epoch+1,'train_objective':float(np.mean(vals)),'seconds':time.perf_counter()-start});print(json.dumps({'fold':k,**curves[-1]}),flush=True)
 assert frozen_hash(t)==fh and all(p.grad is None for p in m.parameters()) and all(not x.training for x in t.modules());p=predict(t,d,tokens,va)*ys+ym;tp=predict(t,d,tokens,tr)*ys+ym;state={n:v.detach().cpu() for n,v in t.state_dict().items()};torch.save({'state':state,'config':CFG,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'target_mean':ym,'target_std':ys,'initial_state_sha256':initial,'frozen_state_sha256':fh},_repo_io(path/'model.pt'));delta=sum(float(state[n].square().sum()) for n in state if n.endswith('up.weight'))**.5
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,adapted=p,frozen=old[va],warm=warm,train_predictions=tp)
 r={'fold':k,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'curve':curves,'config':CFG,'initial_state_sha256':initial,'frozen_state_sha256':fh,'final_state_sha256':h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'adapter_up_l2':delta,'warm_control_max_error':float(abs(warm-old[va]).max()),'train_rmse_before':fv.rmse(d['y'][tr],old[tr]),'train_rmse_after':fv.rmse(d['y'][tr],tp),'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(path/'report.json',r);print(json.dumps({'completed_fold':k,'seconds':r['seconds'],'train_rmse_before':r['train_rmse_before'],'train_rmse_after':r['train_rmse_after']}),flush=True);return p,r

def benchmark_fold():
 check();torch.set_num_threads(2);d=inputs('original');tokens=get_cache(d);m=backbone.model();out=_repo_path(OUT, 'original');out.mkdir(exist_ok=False);p,r=fit_fold(d,tokens,m,0,out/'fold-0');b={'first_fold_seconds':r['seconds'],'projected_original5fold_seconds':r['seconds']*5,'rss_peak_bytes':r['rss_peak_bytes'],'mps_driver_bytes':r['mps_driver_bytes'],'first_fold_reused':True,'fold_report_sha256':fv.sha(out/'fold-0/report.json')};b['resource_pass']=r['seconds']<360 and r['rss_peak_bytes']<4e9 and r['mps_driver_bytes']<4e9;fv.write(_repo_path(OUT, 'benchmark.json'),b);print(json.dumps(b),flush=True);assert b['resource_pass']

def run(split):
 check();b=json.loads((_repo_path(OUT, 'benchmark.json')).read_text());assert b['resource_pass'];torch.set_num_threads(2)
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 d=inputs(split);tokens=get_cache(d);m=backbone.model();out=_repo_path(OUT, split)
 if split=='original':assert out.is_dir() and not (out/'report.json').exists()
 else:out.mkdir(exist_ok=False)
 adapted=np.full(400,np.nan);new=np.full(400,np.nan);slopes=[];start=time.perf_counter()
 for k in range(5):
  path=out/f'fold-{k}';va=np.flatnonzero(d['fid']==k)
  if split=='original' and k==0:
   assert fv.sha(path/'report.json')==b['fold_report_sha256'];r=json.loads((path/'report.json').read_text());assert fv.sha(path/'predictions.npz')==r['predictions_sha256']
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['validation_indices'],va);p=z['adapted'].copy()
  else:p,r=fit_fold(d,tokens,m,k,path)
  adapted[va]=p;fr=json.loads((previous.base.OUT/split/f'fold-{k}/report.json').read_text());s=fr['slope'];slopes.append(s);new[va]=d['reference009'][va]+s*.125*(p-d['Bmean'][va]);torch.mps.empty_cache()
 ps={'reference009':d['reference009'],'frozenB':d['Bmean'],'adaptedB':adapted,'primary':new};comp=prior.old.comparison(d['y'],d['reference009'],new,d['fid'])
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 r={'split':split,'scores':{n:{'rmse':fv.rmse(d['y'],p),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],p[d['fid']==k]) for k in range(5)]} for n,p in ps.items()},'comparison':comp,'unchanged009_slopes':slopes,'fold_report_sha256':{f'fold-{k}/report.json':fv.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds_excluding_reused_benchmark':time.perf_counter()-start,'new_candidate':False};fv.write(out/'report.json',r)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':comp['gain']>0,'comparison':comp})
 check();print(json.dumps(r),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','microbenchmark','cache','benchmark-fold','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'microbenchmark':microbenchmark,'cache':cache,'benchmark-fold':benchmark_fold}.get(a.mode,lambda:run(a.mode))()
