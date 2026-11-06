"""Execute the one fixed095 protocol after its resource gate, without changing it."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,gc,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import early_adapter_095 as s
import numpy as np,torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
fv=s.fv;OUT=s.OUT;CFG=s.CFG

def prepare():
 s.check();b=json.loads((OUT/'benchmark.json').read_text());assert b['resource_pass'];p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_plan_sha256':fv.sha(OUT/'plan.json'),'benchmark_sha256':fv.sha(OUT/'benchmark.json'),'repair_manifest_sha256':fv.sha(OUT/'benchmark-repair.json'),'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['train_early_adapter_095.py','validate_early_adapter_095.py','run_early_adapter_095.py','early_adapter_095.py','early_adapter_095_benchmark_repair.py']},'unchanged_model_config':CFG,'execution_details':'All3arms retrained micro1/effective16; original084headinit/orderseed. Sameper-image dropoutseedinall3arms. Noouterepochpredictions; endpointtrain320+validation80only. Earlyallrawimagefullgraph,lateverified080cache,frozenBfeatures. No firstfoldperformancegate. Afterfirstcompletefold resourceprojectiononly maystop if>3hours.','runtime_monitor':'Check conservativeRSSpeak+MPSdriver<8GB aftereachoptimizerstep; reportactualfirstfold/rollingprojected5fold. Resourcefailure stopswithout tuning. No checkpointing needed perpassedbenchmark.','validation':'Allsavedmodels strictnonbackbone-state reload andpredictions; scalers/targets/IDs inspected. Originalfold0 eacharm first2updates replay withouterlabelsperturbed, equalstate tosavedprefix. Earlyfullbackbonefrozenhash before/after. No full12epochreplay claimed.','scope':'Noholdout/test/full500/newcandidate/upload. Ifbothresplits qualify,reporteligible researchresult toparent,do notautomatically produce orsubmitCSV.'};fv.write(OUT/'execution-plan.json',p);print(json.dumps(p),flush=True)
def check():
 s.check();p=json.loads((OUT/'execution-plan.json').read_text());assert p['base_plan_sha256']==fv.sha(OUT/'plan.json') and p['benchmark_sha256']==fv.sha(OUT/'benchmark.json')
 for n,hh in p['sources'].items():assert fv.sha(_repo_path(ROOT, n))==hh,n
 return p

def predict(model,d,tokens,ix):
 model.eval();ps=[]
 with torch.no_grad():
  for i in ix:ps.append(s.batch(model,d,tokens,int(i)).cpu().numpy().astype(float)[0])
 return np.array(ps)

def orders_for(k,n):return np.random.default_rng(CFG['seed']+k).permuted(np.tile(np.arange(n),(12,1)),axis=1)
def fit(d,m,k,arm,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert len(tr)==320 and len(va)==80 and not set(d['g'][tr])&set(d['g'][va]);sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());model=s.Model(m,sc,CFG['seed']+k,arm,False).to('mps');opt=s.optimizer(model);tokens=s.e.get_cache(d) if arm=='late' else None;initial=s.h.state_hash(s.checkpoint_state(model));headinit=s.h.state_hash(model.head.state_dict());adapterinit=s.h.state_hash(model.adapter.state_dict());backbonehash=s.h.state_hash(m.state_dict());warm=predict(model,d,tokens,tr[:3]);expected=[]
 with torch.no_grad():
  for i in tr[:3]:expected.append(model.head((s.tensor(d['B'][i:i+1])-model.mean)/model.scale).cpu().numpy()[0])
 np.testing.assert_allclose(warm,expected,atol=1e-5,rtol=0);orders=orders_for(k,len(tr));curve=[];seeds=[];firstgrad=[];peak=s.memory();first32state=None
 for ep,order in enumerate(orders):
  s.schedule(opt,ep);model.training_mode();losses=[];epstart=time.perf_counter()
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];opt.zero_grad(set_to_none=True)
   for off,i in enumerate(group):
    seed=s.seed_for(k,ep,first+off);torch.manual_seed(seed);seeds.append(seed);p=s.batch(model,d,tokens,int(i));loss=F.smooth_l1_loss(p,s.tensor([(d['y'][i]-ym)/ys]),beta=1.);assert torch.isfinite(loss);(loss/len(group)).backward();losses.append(float(loss.detach()))
   if ep==0 and first<32:firstgrad.append({n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.adapter.named_parameters()})
   hn=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(hn)
   if arm!='frozen':an=torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.);assert torch.isfinite(an)
   opt.step();torch.mps.synchronize();mem=s.memory();peak={n:max(peak[n],mem[n]) for n in mem};assert mem['conservative_sum_bytes']<8e9,'RESOURCE_MEMORY_8GB'
   if ep==0 and first==16:
    first32state=s.checkpoint_state(model);torch.save(first32state,_repo_io(path/'first-two-updates.pt'))
   if arm=='early' and ((first//16+1)%5==0):print(json.dumps({'fold':k,'arm':arm,'epoch':ep+1,'effective_step':first//16+1,'seconds':time.perf_counter()-start,'memory':mem}),flush=True)
  rec={'epoch':ep+1,'online_train_loss':float(np.mean(losses)),'head_lr':opt.param_groups[0]['lr'],'adapter_lr':opt.param_groups[1]['lr'] if arm!='frozen' else None,'epoch_seconds':time.perf_counter()-epstart,'seconds':time.perf_counter()-start};curve.append(rec);print(json.dumps({'fold':k,'arm':arm,**rec}),flush=True)
 tp=predict(model,d,tokens,tr)*ys+ym;p=predict(model,d,tokens,va)*ys+ym;assert s.h.state_hash(m.state_dict())==backbonehash and all(v.grad is None for v in m.parameters());state=s.checkpoint_state(model)
 if arm!='frozen':assert firstgrad[0]['U.weight']>0 and firstgrad[0]['V.weight']==0 and firstgrad[1]['V.weight']>0 and float(model.adapter.U.weight.norm())>0
 ck={'state':state,'config':CFG,'arm':arm,'fold':k,'seed':CFG['seed']+k,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'backbone_sha256':backbonehash,'initial_state_sha256':initial};torch.save(ck,_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,warm_train3=warm,train_predictions=tp,predictions=p)
 r={'fold':k,'arm':arm,'initial_state_sha256':initial,'initial_head_sha256':headinit,'initial_adapter_sha256':adapterinit,'final_state_sha256':s.h.state_hash(state),'backbone_sha256':backbonehash,'backbone_unchanged':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'warm_prediction_max_error':float(abs(warm-np.array(expected)).max()),'first_two_update_adapter_gradients':firstgrad,'adapter_up_l2':float(model.adapter.U.weight.norm().detach()),'train_RMSE':fv.rmse(d['y'][tr],tp),'curve':curve,'memory':peak,'seconds':time.perf_counter()-start,'checkpoint_sha256':fv.sha(path/'model.pt'),'first_two_updates_sha256':fv.sha(path/'first-two-updates.pt'),'predictions_sha256':fv.sha(path/'predictions.npz')};fv.write(path/'report.json',r);del model,opt,tokens,first32state,state,ck;gc.collect();torch.mps.empty_cache();print(json.dumps({'completed_fold':k,'arm':arm,'seconds':r['seconds'],'train_RMSE':r['train_RMSE']}),flush=True);return p,r

def evaluate(d,preds,out,elapsed):
 ps={**preds,'reference009':d['reference009'],'reference011':d['reference011']};ps.update({a+'_blend':.9*d['reference009']+.1*preds[a] for a in s.ARMS});pairs={'early_vs_late':('late','early'),'early_vs_frozen':('frozen','early'),'earlyblend_vs_lateblend':('late_blend','early_blend'),'earlyblend_vs_frozenblend':('frozen_blend','early_blend'),'earlyblend_vs011':('reference011','early_blend'),'frozenblend_vs011':('reference011','frozen_blend')};comp={n:s.e.base.prior.old.comparison(d['y'],ps[a],ps[b],d['fid']) for n,(a,b) in pairs.items()};c=comp['earlyblend_vs011'];gate=all(comp[n]['gain']>0 for n in ['early_vs_late','early_vs_frozen','earlyblend_vs_lateblend','earlyblend_vs_frozenblend']) and c['gain']>=.1 and c['positive_folds']>=4;rest=np.array([i for i,x in enumerate(d['ids']) if x not in s.TOP]);rg=fv.rmse(d['y'][rest],d['reference011'][rest])-fv.rmse(d['y'][rest],ps['early_blend'][rest]);sq=(d['y']-d['reference011'])**2-(d['y']-ps['early_blend'])**2
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],squared_error_gain=sq,**ps)
 r={'scores':{n:{'rmse':fv.rmse(d['y'],p),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],p[d['fid']==k]) for k in range(5)]} for n,p in ps.items()},'comparisons':comp,'advance':bool(gate),'fixed395_gain_vs011':float(rg),'fixed5_ids':list(s.TOP),'candidate_policy':'No late/frozen fallback; no candidateCSV/full500/test/upload in095.','predictions_sha256':fv.sha(out/'development_predictions.npz'),'elapsed_seconds':elapsed,'limits':'Onefixedseed,adaptivehistory,same400repartitions.0.10/4foldruleispracticalgate,notsignificance. Fixed395excludes same093IDs,notnewtop5.'};fv.write(out/'report.json',r);return r

def run(split):
 check();assert json.loads((OUT/'benchmark.json').read_text())['resource_pass']
 if split!='original':assert json.loads((OUT/'original/report.json').read_text())['advance'] and json.loads((_repo_path(ROOT, 'reports/early-adapter-095-original-validation.json')).read_text())['passed']
 torch.set_num_threads(2);d=s.inputs(split);m=s.backbone.model().to('mps');out=OUT/split;out.mkdir(exist_ok=False);start=time.perf_counter();preds={a:np.full(400,np.nan) for a in s.ARMS};foldtimes=[]
 for k in range(5):
  fstart=time.perf_counter();fd=out/f'fold-{k}';fd.mkdir(exist_ok=False);recs={};va=np.flatnonzero(d['fid']==k)
  for arm in s.ARMS:p,recs[arm]=fit(d,m,k,arm,fd/arm);preds[arm][va]=p
  for key in ['initial_state_sha256','initial_head_sha256','initial_adapter_sha256','batch_order_sha256','dropout_seeds_sha256']:assert len(set(v[key] for v in recs.values()))==1,key
  foldtimes.append(time.perf_counter()-fstart);rr={'fold':k,'arms':recs,'seconds':foldtimes[-1],'projected5fold_seconds':float(np.mean(foldtimes)*5)};fv.write(fd/'report.json',rr);print(json.dumps({'completed_three_arm_fold':k,'seconds':foldtimes[-1],'projected5fold_seconds':rr['projected5fold_seconds']}),flush=True)
  if rr['projected5fold_seconds']>10800:fv.write(out/'resource-stop.json',rr);raise RuntimeError('RESOURCE_PROJECTED_5FOLD_3H')
 r=evaluate(d,preds,out,time.perf_counter()-start);check();print(json.dumps({'split':split,**r}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','original','20261031','20261101']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
