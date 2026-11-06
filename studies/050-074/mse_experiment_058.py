"""Paired objective alignment study: half-MSE vs007 SmoothL1, no other changes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import earlystop_experiment_057 as base
import mse_head_058 as head
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/mse-058')
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,v in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==v,n
 return p
def prepare():
 OUT.mkdir(exist_ok=False);old=json.loads((base.OUT/'plan.json').read_text());p={'hypothesis':'SmoothL1 minimizes a robust objective different from squared-error scoring. At standardized errors>1 gradients saturate; fixedhalfMSE tests objective alignment without manual tailweight or range expansion. Could worsen generalization; not assumed better.','evidence':'057inner-epoch selection reduced fitgap but worsened mixture;3/5fold selected100. Samefeature MLP loss comparison not found in051/052. This is another optimization control, not new representation.','loss':'0.5*mean((prediction-standardized_target)^2). Matches SmoothL1beta1 curvature and gradients insideabs(error)<1; removes linear tails. Gradientclip1 unchanged; no per-sample weights.','config':base.h.CONFIG,'effective_aux_weight':0.,'epochs':100,'seeds':list(head.SEEDS),'paired':'Same architecture/dropout/wholemodelinitialization/dataorder/optimizer/scheduler/seed. Only loss differs. 100endpoint fixed;5/15/40 snapshottrain curves diagnostic only. Existing007samefold/seed verifiedstate and OOF servecontrol.','primary':'.75samefold003+.25mean3MSEhead. Compare007. No057earlystop applied; no weight changes.','screen':'Original gain>conditionalbootstrapSD and>=3/5fold positive; only then repeatfixedrecipe both20261031/20261101. No fallback epoch/learning-rate/seed/weight tuning.','scope':'Development400only; noholdout/test/candidateCSV/upload/newdata/install/cost; CPU2threads.','resource':'Prior identicalshape100epochbenchmark1.2sfit;15fitsoriginal expected<30s; stop if benchmarkprojected>300s orRSS>2GB.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['mse_head_058.py','mse_experiment_058.py','earlystop_experiment_057.py','auxiliary_head_051.py','auxiliary_experiment_051.py']},'inputs':old['input_hashes'],'protected':old['protected'],'bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'fits_per_split':15}),flush=True)
def benchmark():
 check();torch.set_num_threads(2);r=torch.tensor([-.8,.2,.7],requires_grad=True);a=.5*(r*r).mean();b=torch.nn.functional.smooth_l1_loss(r,torch.zeros_like(r),beta=1.);torch.testing.assert_close(a,b,atol=1e-8,rtol=0);ga=torch.autograd.grad(a,r,retain_graph=True)[0];gb=torch.autograd.grad(b,r)[0];torch.testing.assert_close(ga,gb,atol=1e-8,rtol=0)
 ids,X,y,g,fid,*_=base.inputs('original');tr=np.flatnonzero(fid!=0);sc,ym,ys,xt,yt=base.prep(X[tr],y[tr]);t=time.perf_counter();states,rec=head.fit(xt,yt,head.SEEDS[0]);seconds=time.perf_counter()-t;old=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052/original/fold-0/report.json')).read_text())['models'][str(head.SEEDS[0])];assert rec['initial_hash']==old['initial_state_sha256'] and rec['order_hash']==old['batch_order_sha256'];assert rec['final_hash']!=old['final_state_sha256'];p=head.predict(states[100],xt,ym,ys);assert np.isfinite(p).all();report={'seconds':seconds,'projected15fit_seconds':seconds*15,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'central_curvature_gradient_fixture_passed':True,'same_initialization_batchorder':True,'loss_changed_final_state':True};report['resource_pass']=bool(seconds*15<300 and report['rss_peak_bytes']<2e9);fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)
def run(split):
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,groups,fid,r003,r007,oldmean=base.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((3,400),np.nan);trainerrors=[];logs=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);sc,ym,ys,xt,yt=base.prep(X[tr],y[tr]);xv=sc.transform(X[va]);fd=out/f'fold-{k}';fd.mkdir();oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}', 'report.json')).read_text());record={'fold':k,'models':{}};tp=[]
  for si,seed in enumerate(head.SEEDS):
   states,r=head.fit(xt,yt,seed+k);old=oldlog['models'][str(seed)];assert r['initial_hash']==old['initial_state_sha256'] and r['order_hash']==old['batch_order_sha256'];path=fd/f'seed-{seed}.pt';torch.save({'state':states[100],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'seed':seed+k,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va]},_repo_io(path));pred[si,va]=head.predict(states[100],xv,ym,ys);tp.append(head.predict(states[100],xt,ym,ys));r.update(checkpoint_sha256=fv.sha(path),validation_rmse=fv.rmse(y[va],pred[si,va]));record['models'][str(seed)]=r
  avg=np.mean(tp,axis=0);trainerrors.extend((avg-y[tr]).tolist());record['mean3_train_rmse']=fv.rmse(y[tr],avg);fv.write(fd/'report.json',record);logs.append(record);print(json.dumps({'split':split,'fold':k,'train_rmse':record['mean3_train_rmse'],'seconds':time.perf_counter()-t}),flush=True)
 mean=pred.mean(0);new=.75*r003+.25*mean;ps={'reference007':r007,'mse_mlp':mean,'primary':new,'reference003':r003};scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in ps.items()};gain=fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261229).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain':gain,'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'mse_control_mlp_residual_correlation':float(np.corrcoef(y-mean,y-oldmean)[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'mean3_train_rmse_pooled':float(np.sqrt(np.mean(np.array(trainerrors)**2))),'seconds':time.perf_counter()-t,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(a>0 for a in fg)>=3),'comparison':comp})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
