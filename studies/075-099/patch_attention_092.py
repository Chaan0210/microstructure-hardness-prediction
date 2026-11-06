"""Weak exploratory hypothesis: learned local gated weighting versus matched mean."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,math,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import width_control_090 as previous
import frozen_seed_088 as prior
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
fv=prior.fv;e=prior.e;OUT=_repo_path(ROOT, 'artifacts/patch-attention-092');ARMS=('mean','attention');CFG={'hidden':16,'epochs':20,'dropout':.25,'lr':.001,'min_lr':1e-5,'weight_decay':.1,'clip':1.,'microbatch':2,'effective_batch':16,'seed':20270322,'blend_new':.1};inputs=prior.inputs

def prepare():
 previous.check();pub=prior.diag.check_best();OUT.mkdir(exist_ok=False)
 p={'study':'092 learned gated patch pooling versus parameter-matched mean','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Hardness-relevant localpatterns mayneed label-trained importance weighting instead offixedglobalstatistics. Evidence isweak: previous spatial/Fisher/covariancevariants failed, and nolocalgroundtruth establishes the missingcue. User authorized onefixedfeasibility/comparison,not abroadsearch.','duplicate_audit':'No train-label learned gatedattention pooling found.039fixedcrops,042variograms,056covariance,067Fisher,070coarseregionalvariance allfixedaggregation;084 changeslastbackboneblock,notpatchpooling.','architecture':'FrozenofficialB504 lastnormalized1296patchtokens x768,plusfixed(x,y)coordinates in[-1,1]. Foldtrainchannelstandardization. Shared Linear770->16 GELU; g=tanh(Vh)*sigmoid(Uh),V/U16->16. Values=concat(h,g)32. Attention score=sum(g)/sqrt16,softmax across1296patches; mean usesuniformweights. BothsameDropout.25 andLinear32->1. Exactly12913trainableparameters each; allweights are active inbotharms. No unusedattention-only parameters or artificialpadding. The sameg also contributesvaluefeatures.','position_policy':'NoD4/crop/rotation/reflectionaugmentation, no labelinvariance assumed. Fixedabsolutegridcoordinates andfrozenDINOpositionalinformation retained. Pooling iscommutative onlyover token-coordinatepairs; rotatingtheinput isnot assumed to preserveprediction/label.','config':CFG,'pairing':'Identicalwholeinitialstate,folds,trainonlymoments/targetstats,orders,per-microbatchdropoutseedstream,optimizer/schedule. Only aggregationweights differ. Bothgatingbranches remaintrainable/active throughvaluefeatures inmeancontrol. Frozenbackbonenevertrained.','cache':'Derive lastnormalizedfloat32patchtokens oncefromverified080frozenprefix400cache;verify pooledCLS/mean/std against062Bfeatures. No learned fitting duringcache. Sharedcache safe acrossfolds; channelnormalizers fitonlyoutertrain.','formula':'Eacharmblend=.9*verifiedsamefold011+.1*armrawtargetprediction. No newcalibration,ratios orfallbackmean-onlycandidate.','screen':'Resourcepass first. Original5fold pilot usesfixed20epochendpoint. Advance onlyif attentionstandalone beatsmean,attentionblend beatsmeanblend,andattentionblend beats011 with>=3/5foldgains. No firstfoldperformance selection. Onfailure terminate thisdirection, noarchitecture/rank/temperature/epoch/ratio changes.','confirmation':'Samecomparison inbothknownresplits. Promotionrequiresall3screenpasses and materialgain>=.03vs011,conditionalpaired95%lower>0,andgainpositiveafterremovingtop5improvementrows inall3;independentvalidationpasses. No automaticupload.','resource':'Cachefloat32about1.59GBadditionaldisk;available78GiBbeforestart. Microbenchmark6stepsperarm,projectedpaired5fold<=600seconds with25%overhead,RSS/MPS<4GB. Firstcompletefoldpair<=120seconds,RSS/MPS<4GB;reuseitonce. Existinguv CPU2/MPS only,noinstall/download/cost. Ifresourcegatefails stopwithoutconfigchanges.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['patch_attention_092.py','validate_patch_attention_092.py','width_control_090.py','dinob_062.py','dinob_adapter_080.py']},'evidence':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/research-synthesis-091.json','reports/width-control-090-validation.json','artifacts/dinob-adapter-080/prefix.json','reports/dinob-062-validation.json','reports/candidate011-public-submission.json']},'protected011':pub['protected_sha256'],'limits':'Weak exploratory hypothesis and historicallyreused400. Resplits notindependent. Fixedarchitecture/seed andtinyset mayunderfit/overfit. Attentionweights aremodelweights,notprovenphysicalimportance. ConditionalCIs omit fitting/adaptiveselection.','scope':'Development400only;noholdout/testselection,externaldata,install,cost orupload.'};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'parameters_each':12913,'resources':p['resource']}),flush=True)

def check():
 previous.check();prior.diag.check_best();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,h in {**p['sources'],**p['evidence']}.items():assert fv.sha(_repo_path(ROOT, n))==h,n
 return p

def cache():
 check();torch.set_num_threads(2);d=inputs('original');src=e.get_cache(d);source_offset=src.offset;del src;m=e.backbone.model().to('mps');initial=e.h.state_hash(m.state_dict());path=_repo_path(OUT, 'patch_tokens.npy');dest=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(400,1296,768));mom=[];start=time.perf_counter();maxerr=0.
 with (e.base.OUT/'prefix.npy').open('rb') as stream:
  stream.seek(source_offset)
  for i in range(400):
   raw=stream.read(1297*768*4);assert len(raw)==1297*768*4;z=torch.tensor(np.frombuffer(raw,dtype=np.float32).reshape(1,1297,768).copy(),device='mps')
   with torch.no_grad():zz=m.norm(m.blocks[-1](z));pp=zz[:,1:];desc=torch.cat((zz[:,0],pp.mean(1),pp.std(1,correction=0)),1).cpu().numpy()[0]
   np.testing.assert_array_equal(desc,d['B'][i,-2304:]);arr=pp.cpu().numpy()[0];dest[i]=arr;aa=arr.astype(np.float64);mom.append(np.stack((aa.mean(0),np.square(aa).mean(0))))
   if (i+1)%80==0:
    dest.flush();del dest;dest=np.load(_repo_io(path),mmap_mode='r+');print(json.dumps({'cache_rows':i+1,'seconds':time.perf_counter()-start}),flush=True)
 dest.flush();del dest;assert initial==e.h.state_hash(m.state_dict()) and all(p.grad is None for p in m.parameters())
 with (_repo_path(OUT, 'token_moments.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),moments=np.stack(mom))
 r={'ids':d['ids'],'shape':[400,1296,768],'dtype':'float32','token_sha256':fv.sha(path),'moments_sha256':fv.sha(_repo_path(OUT, 'token_moments.npz')),'source_prefix_sha256':fv.sha(e.base.OUT/'prefix.npy'),'all400_pooled_descriptor_exact':True,'frozen_weights_unchanged':True,'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(_repo_path(OUT, 'cache.json'),r);print(json.dumps({k:v for k,v in r.items() if k!='ids'}),flush=True)

def get_cache(d):
 r=json.loads((_repo_path(OUT, 'cache.json')).read_text());assert r['ids']==d['ids'] and fv.sha(_repo_path(OUT, 'patch_tokens.npy'))==r['token_sha256'] and fv.sha(_repo_path(OUT, 'token_moments.npz'))==r['moments_sha256'];tokens=np.load(_repo_path(OUT, 'patch_tokens.npy'),mmap_mode='r');assert tokens.shape==(400,1296,768) and tokens.dtype==np.float32
 with np.load(_repo_path(OUT, 'token_moments.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];mom=z['moments'].copy()
 return tokens,mom

def normalize(mom,tr):
 mean=mom[tr,0].mean(0);var=np.maximum(mom[tr,1].mean(0)-mean**2,0.);scale=np.maximum(np.sqrt(var),1e-6);return mean,scale,var

def coordinates():
 yy,xx=np.meshgrid(np.linspace(-1,1,36,dtype=np.float32),np.linspace(-1,1,36,dtype=np.float32),indexing='ij');return np.stack((xx.ravel(),yy.ravel()),1)

class Pool(nn.Module):
 def __init__(self,mean,scale,seed,arm):
  super().__init__();torch.manual_seed(seed);h=CFG['hidden'];self.project=nn.Linear(770,h);self.gate_v=nn.Linear(h,h);self.gate_u=nn.Linear(h,h);self.drop=nn.Dropout(CFG['dropout']);self.readout=nn.Linear(2*h,1);self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32));self.register_buffer('coords',torch.tensor(coordinates()));self.arm=arm;self.eval()
 def forward(self,x,return_weights=False):
  x=torch.cat(((x-self.mean)/self.scale,self.coords.expand(len(x),-1,-1)),2);h=F.gelu(self.project(x));g=torch.tanh(self.gate_v(h))*torch.sigmoid(self.gate_u(h));values=torch.cat((h,g),2);score=g.sum(-1)/math.sqrt(CFG['hidden']);weights=score.softmax(1) if self.arm=='attention' else torch.full_like(score,1/1296);pooled=(weights[:,:,None]*values).sum(1);p=self.readout(self.drop(pooled)).squeeze(-1);return (p,weights) if return_weights else p

def tensor(tokens,ii):return torch.tensor(np.array(tokens[ii],copy=True),dtype=torch.float32,device='mps')
def seed_for(k,ep,j):return CFG['seed']+k+10000*(ep+1)+j
def schedule(opt,ep):
 lr=CFG['min_lr']+.5*(CFG['lr']-CFG['min_lr'])*(1+math.cos(math.pi*ep/CFG['epochs']))
 for group in opt.param_groups:group['lr']=lr

def predict(model,tokens,ix):
 model.eval();p=[];ess=[];mx=[]
 with torch.no_grad():
  for j in range(0,len(ix),2):
   v,w=model(tensor(tokens,ix[j:j+2]),True);p.append(v.cpu().numpy().astype(float));ess.extend((1/(w*w).sum(1)).cpu().numpy().astype(float));mx.extend(w.max(1).values.cpu().numpy().astype(float))
 return np.concatenate(p),np.array(ess),np.array(mx)

def microbenchmark():
 check();torch.set_num_threads(2);d=inputs('original');tokens,mom=get_cache(d);tr=np.flatnonzero(d['fid']!=0);mean,scale,var=normalize(mom,tr);yy=e.tensor((d['y'][tr[:2]]-d['y'][tr].mean())/d['y'][tr].std());records={}
 for arm in ARMS:
  model=Pool(mean,scale,CFG['seed'],arm).to('mps');initial=e.h.state_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);times=[];model.train();grads=None
  for step in range(6):
   torch.manual_seed(seed_for(0,0,2*step));torch.mps.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(model(tensor(tokens,tr[:2])),yy,beta=1.);loss.backward()
   if step==0:grads={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert all(v>0 and np.isfinite(v) for v in grads.values())
   torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['clip']);opt.step();torch.mps.synchronize()
   if step:times.append(time.perf_counter()-start)
  records[arm]={'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial,'median_microstep_seconds':float(np.median(times)),'first_parameter_gradient_norms':grads,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};del model,opt;torch.mps.empty_cache()
 assert records['mean']['parameters']==records['attention']['parameters']==12913 and records['mean']['initial_state_sha256']==records['attention']['initial_state_sha256'];projected=sum(a['median_microstep_seconds'] for a in records.values())*160*CFG['epochs']*5*1.25;r={'arms':records,'projected_paired5fold_seconds_with25pct_overhead':projected,'resource_pass':projected<=600 and max(a['rss_peak_bytes'] for a in records.values())<4e9 and max(a['mps_driver_bytes'] for a in records.values())<4e9};fv.write(_repo_path(OUT, 'microbenchmark.json'),r);print(json.dumps(r),flush=True);assert r['resource_pass']

def fit_arm(d,tokens,mom,k,arm,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);mean,scale,var=normalize(mom,tr);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());model=Pool(mean,scale,CFG['seed']+k,arm).to('mps');initial=e.h.state_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);orders=np.random.default_rng(CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(CFG['epochs'],1)),axis=1);curve=[];seeds=[];firstgrads=None
 for ep,order in enumerate(orders):
  schedule(opt,ep);model.train();losses=[]
  for j in range(0,len(tr),16):
   group=tr[order[j:j+16]];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),2):
    ii=group[offset:offset+2];rng=seed_for(k,ep,j+offset);torch.manual_seed(rng);seeds.append(rng);p=model(tensor(tokens,ii));loss=F.smooth_l1_loss(p,e.tensor((d['y'][ii]-ym)/ys),beta=1.);assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   if firstgrads is None:firstgrads={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert all(v>0 and np.isfinite(v) for v in firstgrads.values())
   gn=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['clip']);assert torch.isfinite(gn);opt.step()
  rec={'epoch':ep+1,'train_loss_online':float(np.mean(losses)),'lr':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-start};curve.append(rec)
  if (ep+1)%5==0:print(json.dumps({'fold':k,'arm':arm,**rec}),flush=True)
 p,ess,mx=predict(model,tokens,va);p=p*ys+ym;tp,tess,tmx=predict(model,tokens,tr);tp=tp*ys+ym;state={n:v.detach().cpu() for n,v in model.state_dict().items()};torch.save({'state':state,'arm':arm,'config':CFG,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'scaler_mean':torch.tensor(mean),'scaler_scale':torch.tensor(scale),'scaler_var':torch.tensor(var),'target_mean':ym,'target_std':ys,'seed':CFG['seed']+k},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,predictions=p,train_predictions=tp,effective_patch_count=ess,max_weight=mx,train_effective_patch_count=tess,train_max_weight=tmx)
 r={'arm':arm,'fold':k,'config':CFG,'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial,'final_state_sha256':e.h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'first_parameter_gradient_norms':firstgrads,'train_RMSE':fv.rmse(d['y'][tr],tp),'curve':curve,'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'predictions.npz'),'effective_patch_count_quantiles':np.quantile(ess,[0,.25,.5,.75,1]).tolist(),'max_weight_quantiles':np.quantile(mx,[0,.25,.5,.75,1]).tolist(),'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(path/'report.json',r);return p,r

def fit_pair(d,tokens,mom,k,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();pred={};records={}
 for arm in ARMS:pred[arm],records[arm]=fit_arm(d,tokens,mom,k,arm,path/arm);torch.mps.empty_cache()
 for n in ['initial_state_sha256','batch_order_sha256','dropout_seeds_sha256','parameters']:assert records['mean'][n]==records['attention'][n]
 r={'fold':k,'arms':records,'seconds':time.perf_counter()-start};fv.write(path/'report.json',r);return pred,r

def pilot_fold():
 check();assert json.loads((_repo_path(OUT, 'microbenchmark.json')).read_text())['resource_pass'];torch.set_num_threads(2);d=inputs('original');tokens,mom=get_cache(d);out=_repo_path(OUT, 'original');out.mkdir(exist_ok=False);p,r=fit_pair(d,tokens,mom,0,out/'fold-0');rss=max(x['rss_peak_bytes'] for x in r['arms'].values());mps=max(x['mps_driver_bytes'] for x in r['arms'].values());b={'first_pair_seconds':r['seconds'],'projected5fold_seconds':r['seconds']*5,'rss_peak_bytes':rss,'mps_driver_bytes':mps,'resource_pass':r['seconds']<=120 and rss<4e9 and mps<4e9,'pair_reused':True,'fold_report_sha256':fv.sha(out/'fold-0/report.json')};fv.write(_repo_path(OUT, 'pilot.json'),b);print(json.dumps(b),flush=True);assert b['resource_pass']

def evaluate(d,out,preds,seconds):
 y=d['y'];fid=d['fid'];ps={**preds,'reference011':d['reference011']};ps.update({a+'_blend':.9*d['reference011']+.1*preds[a] for a in ARMS});pairs={'attention_vs_mean':('mean','attention'),'attentionblend_vs_meanblend':('mean_blend','attention_blend'),'attentionblend_vs011':('reference011','attention_blend')};comp={n:prior.e.base.prior.old.comparison(y,ps[a],ps[b],fid) for n,(a,b) in pairs.items()};advance=all(c['gain']>0 for c in comp.values()) and comp['attentionblend_vs011']['positive_folds']>=3
 from dropout_control_089 import statistics
 dd=dict(d);cc,sd,material,sq,ab,change=statistics(dd,ps['attention_blend']);assert cc==comp['attentionblend_vs011']
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=y,fold=fid,squared_error_gain=sq,absolute_error_gain=ab,prediction_change=change,**ps)
 r={'scores':{n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in ps.items()},'comparisons':comp,'advance':advance,'material_gate':advance and material,'sample_diagnostics':sd,'fold_report_sha256':{f'fold-{k}/report.json':fv.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds_excluding_reused_pair':seconds};fv.write(out/'report.json',r);return r

def run(split):
 check();assert json.loads((_repo_path(OUT, 'pilot.json')).read_text())['resource_pass'];torch.set_num_threads(2)
 if split!='original':assert json.loads((_repo_path(OUT, 'original/report.json')).read_text())['advance']
 d=inputs(split);tokens,mom=get_cache(d);out=_repo_path(OUT, split)
 if split=='original':assert out.is_dir() and not (out/'report.json').exists()
 else:out.mkdir(exist_ok=False)
 preds={a:np.full(400,np.nan) for a in ARMS};start=time.perf_counter()
 for k in range(5):
  path=out/f'fold-{k}';va=np.flatnonzero(d['fid']==k)
  if split=='original' and k==0:
   b=json.loads((_repo_path(OUT, 'pilot.json')).read_text());assert fv.sha(path/'report.json')==b['fold_report_sha256'];r=json.loads((path/'report.json').read_text());p={}
   for arm in ARMS:
    assert fv.sha(path/arm/'predictions.npz')==r['arms'][arm]['predictions_sha256']
    with np.load(_repo_io(path/arm/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['validation_indices'],va);p[arm]=z['predictions'].copy()
  else:p,r=fit_pair(d,tokens,mom,k,path)
  for arm in ARMS:preds[arm][va]=p[arm]
 r=evaluate(d,out,preds,time.perf_counter()-start);check();print(json.dumps({'split':split,**r}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','cache','microbenchmark','pilot',*prior.s.SPLITS]);a=ap.parse_args();{'prepare':prepare,'cache':cache,'microbenchmark':microbenchmark,'pilot':pilot_fold}.get(a.mode,lambda:run(a.mode))()
