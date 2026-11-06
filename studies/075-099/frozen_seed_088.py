"""One fixed3seed test of the new regularized head; no seed/weight selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,argparse,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import diagnose_frozen_head_087 as diag
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
s=diag.study;e=s.e;fv=s.fv;OUT=_repo_path(ROOT, 'artifacts/frozen-seed-088');SEEDS=(20270215,20270222,20270308);CFG=e.CFG

def prepare():
 s.check();pub=diag.check_best();r=json.loads((diag.OUT/'report.json').read_text());assert r['passed'];OUT.mkdir(exist_ok=False)
 plan={'study':'088 fixed three-seed average of084/085 fresh frozen head','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Same new regularized12epoch B head has unmeasured initialization/order/dropout variability. Average exactly3fixed seeds to reduce this algorithmic variance without changing representation, optimizer,regularization,training length,normalization,calibration or total10%weight.','diagnostic_basis':'087 rules out changed features/input scaling and shows old/new interpolation differs, but cannot causally separate regularization from length/capacity/batch. Existing old3seed averaging reduces MSE by1.53..1.66; transfer to new recipe is untested. This is a controlled seed-only test, not attribution of prior improvement to one regularizer.','seeds':list(SEEDS),'config':CFG,'seed_rule':'Reuse20270215 per fold from084/085; add20270222 and20270308. Each actualseed=base+fold,full500=base. All3 equally weighted; never drop/rank/select a seed. Same12epochs and data-order/dropout algorithm, with seed as sole changed variable.','formula':'.9*samefold009+.1*mean(new_head_seed20270215,new_head_seed20270222,new_head_seed20270308)','gate':'Run allthree existing partitions unchanged. Require pooledRMSE below011 and >=3/5 improvedfolds in EACH partition, plus independent verification and originalbase-seed replay exact. Otherwise reject wholebundle. No alternate seeds/ratios/epochs/head settings; no fallbackcandidate.','full500':'If eligible, lock012; retain existing full500011 head as firstseed, fit onlytwo additional full500 heads withsameCFG andfixedseeds. Keep009 unchanged. Infer1000onlyafterallmodelsarefixed. Parent controls submission.','duplicate_audit':r['duplicate_audit'],'selection_bias':r['selection_bias']+' The new seeds are fixed before seeing any of their development outcomes. Same400 historical reuse remains; no Public score informs seeds or weights.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['frozen_seed_088.py','validate_frozen_seed_088.py','diagnose_frozen_head_087.py','dinob_joint_084.py']},'evidence':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/frozen-head-diagnostics-087/report.json','reports/candidate011-public-submission.json','reports/frozen-complement-085-validation.json']},'protected011':pub['protected_sha256'],'cost':0,'scope':'Development400onlybeforelock;noholdout/testselection/install/download;009/010/011preserved,010unsubmitted,no upload.','limits':r['selection_bias']};fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps({'prepared':True,'seeds':SEEDS,'gate':plan['gate']}),flush=True)

def check():
 s.check();diag.check_best();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['seeds']==list(SEEDS) and p['config']==CFG
 for n,h in {**p['sources'],**p['evidence']}.items():assert fv.sha(_repo_path(ROOT, n))==h,n
 return p

def inputs(split):
 d=e.inputs(split);v=json.loads((_repo_path(ROOT, 'reports/frozen-complement-085-validation.json')).read_text());p=s.OUT/split/'development_predictions.npz';assert fv.sha(p)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);d['reference011']=z['primary'].copy();d['basehead']=z['frozen'].copy()
 return d

def path_for(split,seed,k):return s.foldpath(split,k) if seed==SEEDS[0] else _repo_path(OUT, split, f'seed-{seed}', f'fold-{k}')

def fit(d,m,k,seed,path,full=False):
 path.mkdir(parents=True,exist_ok=False);start=time.perf_counter();tr=np.arange(len(d['y'])) if full else np.flatnonzero(d['fid']!=k);va=np.array([],dtype=int) if full else np.flatnonzero(d['fid']==k)
 if not full:assert not set(d['g'][tr])&set(d['g'][va])
 sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());actual=seed+k;model=e.Model(m,sc,actual,False).to('mps');initial=e.h.state_hash(model.state_dict());fixed=e.fixed_hash(model);opt=e.optimizer(model);orders=np.random.default_rng(actual).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);curve=[];seeds=[]
 for ep,order in enumerate(orders):
  e.schedule(opt,ep);model.training_mode();losses=[]
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),2):
    ii=group[offset:offset+2];rng=actual+10000*(ep+1)+first+offset;torch.manual_seed(rng);seeds.append(rng);p=model(e.tensor(d['B'][ii]));loss=F.smooth_l1_loss(p,e.tensor((d['y'][ii]-ym)/ys),beta=1.);assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(gn);opt.step()
  tp=e.predict(model,d,None,tr)*ys+ym;curve.append({'epoch':ep+1,'online_train_loss':float(np.mean(losses)),'train_RMSE':fv.rmse(d['y'][tr],tp),'head_lr':opt.param_groups[0]['lr']})
 p=e.predict(model,d,None,va)*ys+ym if len(va) else np.array([]);assert e.fixed_hash(model)==fixed;state={n:v.detach().cpu() for n,v in model.state_dict().items()};torch.save({'state':state,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'config':CFG,'base_seed':seed,'seed':actual,'target_mean':ym,'target_std':ys,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'initial_state_sha256':initial,'fixed_state_sha256':fixed},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,train_predictions=tp,predictions=p)
 r={'base_seed':seed,'seed':actual,'config':CFG,'initial_state_sha256':initial,'fixed_state_sha256':fixed,'final_state_sha256':e.h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'curve':curve,'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};fv.write(path/'report.json',r);print(json.dumps({'fold':k,'seed':seed,'full500':full,'seconds':r['seconds']}),flush=True);return p,r

def run(split):
 check();torch.set_num_threads(2);d=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);m=e.backbone.model();pred=np.full((3,400),np.nan);records={};start=time.perf_counter()
 for si,seed in enumerate(SEEDS):
  for k in range(5):
   path=path_for(split,seed,k);va=np.flatnonzero(d['fid']==k)
   if si==0:
    rec=json.loads((path/'report.json').read_text());assert fv.sha(path/'predictions.npz')==rec['predictions_sha256']
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['validation_indices'],va);p=z['predictions'].copy()
   else:p,rec=fit(d,m,k,seed,path);torch.mps.empty_cache()
   pred[si,va]=p;records[str((path/'report.json').relative_to(ROOT))]=fv.sha(path/'report.json')
 np.testing.assert_array_equal(pred[0],d['basehead']);mean=pred.mean(0);primary=.9*d['reference009']+.1*mean;comp=e.base.prior.old.comparison(d['y'],d['reference011'],primary,d['fid']);assert np.isfinite(pred).all()
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],seed_predictions=pred,mean3=mean,primary=primary,reference009=d['reference009'],reference011=d['reference011'])
 r={'scores':{n:fv.rmse(d['y'],v) for n,v in [('reference011',d['reference011']),('primary',primary),('head_mean3',mean)]},'comparison':comp,'advance':comp['gain']>0 and comp['positive_folds']>=3,'individual_head_scores_diagnostic_only':[fv.rmse(d['y'],v) for v in pred],'seed_prediction_variance_mean':float(np.mean(np.var(pred,axis=0))),'head_average_individual_MSE':float(np.mean((pred-d['y'])**2)),'head_mean3_MSE':float(np.mean((mean-d['y'])**2)),'fold_report_sha256':records,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':time.perf_counter()-start};fv.write(out/'report.json',r);check();print(json.dumps({'split':split,**r}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare',*s.SPLITS]);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
