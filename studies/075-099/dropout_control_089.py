"""Single matched ablation: dropout.50 to.25, all other011 settings fixed."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,datetime,argparse,inspect
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import frozen_seed_088 as old
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
e=old.e;fv=old.fv;OUT=_repo_path(ROOT, 'artifacts/dropout-control-089');CFG={**e.CFG,'dropout':.25};SEED=20270215

def source_control():
 a=inspect.getsource(old.fit).replace('def fit(d,m,k,seed,path,full=False):','def fit(d,m,k,seed,path,full=False,dropout=.25):').replace("path.mkdir(parents=True,exist_ok=False);start=", "cfg={**CFG,'dropout':dropout};path.mkdir(parents=True,exist_ok=False);start=").replace("model=e.Model(m,sc,actual,False).to('mps');initial=", "model=e.Model(m,sc,actual,False).to('mps');model.head.net[2].p=dropout;initial=").replace("'config':CFG", "'config':cfg")
 assert inspect.getsource(fit)==a

def prepare():
 old.check();source_control();OUT.mkdir(exist_ok=False);pub=old.diag.check_best();p={'study':'089 one-factor dropout ablation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'New32wide12epoch head has trainRMSE~5.6 and088 seed dispersion5.13..5.82 MSE, versus old near-interpolating recipe. Reducing only stochastic dropout from.5 to established.25 may reduce underfitting/noisy training while retaining smallwidth,shortschedule,weightdecay. This is a hypothesis, not a known cause.','factor':'Dropout probability only:.50 control versus.25 treatment. Same initialized tensors/seed20270215+fold,32width,12epochs,per-microbatch random-seed stream,permutations,micro2/effective16,AdamWwd.1,cosine lr3e-4..1e-5,SmoothL1beta1,clip1,features,scalers,targets and.9/.1blend. No extra seeds or ratio/epoch search. Masks differ as intended because drop probability differs; RNG seed stream is paired.','config':CFG,'duplicate_audit':'051/052/063/064 use.25 with64wide100epochs;084/085/088 use.5 with32wide12epochs. No dropout-only comparison found.057 tested inner-selected5/15/40/100epochs for old S64wide recipe and failed; do not repeat. Width change would change tensor shapes and initialization; dropout offers the cleanest one-factor comparison.','formula':'.9*samefold009+.1*dropout25_head versus unchanged011','candidate_gate':'Allthree partitions must have gain>=.03RMSE vs011,>=3/5improvingfolds,conditionalpaired95%gainCI lower>0,and positiveRMSEgain after removing5largest positive squared-error contributors. Independentvalidation/controlreplay mustpass. No weaker fallback or thresholds changed after outcomes.','threshold_rationale':'.03 is a pre-result operational minimum, comparable to the smallest085 gain. CI and deletion checks exclude tiny or few-row-driven candidates; neither accounts for adaptive selection or correlated observations. They are not a calibrated significance test or Public improvement promise.','sample_diagnostics':'Save per-ID squared/absolute error gains and prediction changes ineachpartition. Report counts improving in0/1/2/3partitions,mean perID gain,trimmed and per-fold gains,error-gain quantiles,conditional intervals,and RMS prediction change. SameIDs acrosssplits are not independent samples. No subgroup or checkpoint choice.','execution':'Train one new model per fold on all3partitions,15fits. Reuse011controlmodels. Replay control.5 and treatment.25 onfold0 eachpartition with heldoutlabelsperturbed; controlstate must match saved011. Existinguv,CPU2/MPS,~2minutestrain plusverification; noinstall/cost.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dropout_control_089.py','validate_dropout_control_089.py','frozen_seed_088.py','dinob_joint_084.py']},'evidence':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/frozen-head-087-seed-088-summary.json','reports/frozen-seed-088-validation.json','artifacts/frozen-head-diagnostics-087/report.json','reports/candidate011-public-submission.json','artifacts/earlystop-057/plan.json'] if (_repo_path(ROOT, n)).exists()},'protected011':pub['protected_sha256'],'limits':'Same400 observations repeatedly reused and011 originated as posthoc084control. This ablation estimates one-factor effect conditional on a historicallyselected recipe and fixedseed. Doesnotidentify allcauses of011gain or generalize across allseeds. ConditionalCIs omit model-fit/adaptive-selection uncertainty. No new independent validation.','scope':'No additionalholdout evaluation,testselection,install/download/cost/upload. Candidateonlyafterstronggate;parentcontrolsfinalsubmission.'};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'factor':p['factor'],'gate':p['candidate_gate']}),flush=True)

def check():
 old.check();source_control();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,h in {**p['sources'],**p['evidence']}.items():assert fv.sha(_repo_path(ROOT, n))==h,n
 return p

def statistics(d,p):
 y=d['y'];ref=d['reference011'];fid=d['fid'];comp=e.base.prior.old.comparison(y,ref,p,fid);sq=(y-ref)**2-(y-p)**2;ab=abs(y-ref)-abs(y-p);change=p-ref;keep=np.ones(len(y),dtype=bool);keep[np.argsort(-sq,kind='stable')[:5]]=False;trim=fv.rmse(y[keep],ref[keep])-fv.rmse(y[keep],p[keep]);diag={'improved_squared_error_rows':int((sq>0).sum()),'worse_rows':int((sq<0).sum()),'squared_error_gain_quantiles':np.quantile(sq,[0,.1,.25,.5,.75,.9,1]).tolist(),'absolute_error_gain_mean':float(ab.mean()),'prediction_difference_RMS':float(np.sqrt(np.mean(change**2))),'prediction_abschange_quantiles':np.quantile(abs(change),[0,.25,.5,.75,.95,1]).tolist(),'gain_without_top5_improvement_rows':trim,'top5_row_IDs_diagnostic_only':[d['ids'][i] for i in np.argsort(-sq,kind='stable')[:5]],'residual_correlation':float(np.corrcoef(y-ref,y-p)[0,1])};passed=comp['gain']>=.03 and comp['positive_folds']>=3 and comp['conditional_ci95'][0]>0 and trim>0;return comp,diag,bool(passed),sq,ab,change

def run(split):
 check();torch.set_num_threads(2);d=old.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);m=e.backbone.model();pred=np.full(400,np.nan);records={};train=[];start=time.perf_counter()
 for k in range(5):
  va=np.flatnonzero(d['fid']==k);path=out/f'fold-{k}';p,r=fit(d,m,k,SEED,path);pred[va]=p;train.append(r['curve'][-1]['train_RMSE']);records[f'fold-{k}/report.json']=fv.sha(path/'report.json');torch.mps.empty_cache()
 primary=.9*d['reference009']+.1*pred;comp,diag,advance,sq,ab,change=statistics(d,primary)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],new_head=pred,control_head=d['basehead'],primary=primary,reference009=d['reference009'],reference011=d['reference011'],squared_error_gain=sq,absolute_error_gain=ab,prediction_change=change)
 r={'scores':{'reference011':fv.rmse(d['y'],d['reference011']),'primary':fv.rmse(d['y'],primary),'head':fv.rmse(d['y'],pred),'control_head':fv.rmse(d['y'],d['basehead'])},'comparison':comp,'sample_diagnostics':diag,'advance':advance,'pooled_train_RMSE':float(np.sqrt(np.mean(np.array(train)**2))),'fold_report_sha256':records,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':time.perf_counter()-start};fv.write(out/'report.json',r);check();print(json.dumps({'split':split,**r}),flush=True)

def fit(d,m,k,seed,path,full=False,dropout=.25):
 cfg={**CFG,'dropout':dropout};path.mkdir(parents=True,exist_ok=False);start=time.perf_counter();tr=np.arange(len(d['y'])) if full else np.flatnonzero(d['fid']!=k);va=np.array([],dtype=int) if full else np.flatnonzero(d['fid']==k)
 if not full:assert not set(d['g'][tr])&set(d['g'][va])
 sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());actual=seed+k;model=e.Model(m,sc,actual,False).to('mps');model.head.net[2].p=dropout;initial=e.h.state_hash(model.state_dict());fixed=e.fixed_hash(model);opt=e.optimizer(model);orders=np.random.default_rng(actual).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);curve=[];seeds=[]
 for ep,order in enumerate(orders):
  e.schedule(opt,ep);model.training_mode();losses=[]
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),2):
    ii=group[offset:offset+2];rng=actual+10000*(ep+1)+first+offset;torch.manual_seed(rng);seeds.append(rng);p=model(e.tensor(d['B'][ii]));loss=F.smooth_l1_loss(p,e.tensor((d['y'][ii]-ym)/ys),beta=1.);assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(gn);opt.step()
  tp=e.predict(model,d,None,tr)*ys+ym;curve.append({'epoch':ep+1,'online_train_loss':float(np.mean(losses)),'train_RMSE':fv.rmse(d['y'][tr],tp),'head_lr':opt.param_groups[0]['lr']})
 p=e.predict(model,d,None,va)*ys+ym if len(va) else np.array([]);assert e.fixed_hash(model)==fixed;state={n:v.detach().cpu() for n,v in model.state_dict().items()};torch.save({'state':state,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'config':cfg,'base_seed':seed,'seed':actual,'target_mean':ym,'target_std':ys,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'initial_state_sha256':initial,'fixed_state_sha256':fixed},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,train_predictions=tp,predictions=p)
 r={'base_seed':seed,'seed':actual,'config':cfg,'initial_state_sha256':initial,'fixed_state_sha256':fixed,'final_state_sha256':e.h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'curve':curve,'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};fv.write(path/'report.json',r);print(json.dumps({'fold':k,'seed':seed,'full500':full,'seconds':r['seconds']}),flush=True);return p,r

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare',*old.s.SPLITS]);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
