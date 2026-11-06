"""Nested epoch selection isolates optimization from frozen representation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import auxiliary_head_051 as h
import auxiliary_experiment_051 as prior
import earlystop_head_057 as head
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/earlystop-057')
def inputs(split):
 ids,X,y,g,fid,r003,r006=prior.inputs(split);assert ids==sorted(json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())['development_folds']);src=_repo_path(ROOT, 'artifacts/seed-stability-052', split, 'development_oof.csv');assert fv.sha(src)==json.loads((_repo_path(ROOT, 'reports/seed-052-validation.json')).read_text())['results'][split]['oof_sha256']
 with src.open() as file:rows=list(csv.DictReader(file))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(fid,[int(r['fold']) for r in rows]);np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);np.testing.assert_array_equal(r003,[float(r['reference003']) for r in rows]);return ids,X,y,g,fid,r003,np.array([float(r['candidate007_design']) for r in rows]),np.array([float(r['mean3']) for r in rows])
def prepare():
 OUT.mkdir(exist_ok=False);protected=json.loads((_repo_path(ROOT, 'artifacts/token-cov-056/plan.json')).read_text())['protected'];p={'hypothesis':'007 MLP trainRMSE~.6 versusOOF~14 may reflect excessive optimization. Select training duration only using innerOOF; hold frozen3456DINO,64hidden,dropout.25,optimizer/loss/seeds/ensemble fixed. Positive result implicates optimization; failure does not prove representation ceiling.','duplicate_audit':'051/052 used fixed100epochs and diagnostic curves only; no inner epoch selection found. No054-056 features reused.','epochs':list(head.EPOCHS),'seeds':list(head.SEEDS),'config':h.CONFIG,'effective_aux_weight':0.,'trajectory':'100epoch cosine schedule remains unchanged, snapshots at5/15/40/100. No schedule compression or changing batch/dropout order. Outer100 endpoint must reproduce existing007 components to1e-8.','selection':'For each outerfold, innerGroupKFold3 shuffle seed20261227+fold. Fit scalers/targetstats within eachinnertrain. At each epoch average3fixedseed predictions, pooled innerOOF RMSE picks one common epoch (tie shorter). No outer labels/checkpoints affect epoch choice. Innerseed=base+outerfold+100*(innerfold+1); outerseed=base+outerfold. Inner~213 versusouter320 rows is a known training-size mismatch.','primary':'.75samefold003+.25mean3 MLP at selectedepoch. Matched100epochcontrol must equal007. All3seeds retained. No ensemble weights changed.','screen':'Original improves007 and>=3/5fold gains; gain>conditionalpairedbootstrapSD. Rule allocates compute,not statistical significance. Ifpasses repeat identicalrecipe on20261031 and20261101, usingfreshinner/outerfit for eachfold. Otherwise stop thishypothesis with no alternatecheckpoint selected.','diagnostics':'Fixedcheckpoint train/outervalidation curves diagnostic only. Compare innerselectedepoch, traininggap, seed/007 residualcorrelation, extrapolation. No tuning after outerresults.','resource':'CPU2threads cachedfrozenfeatures. Benchmark one100epochtrajectory plus exact051control. Projected60trajectories original<300s,RSS<2GB; noGPU needed. Eachsplit45inner+15outertrajectories.','scope':'Development400 only,existinguv/packages/localCPU; noholdout/test/newdata/install/cost/candidateCSV/upload.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['earlystop_head_057.py','earlystop_experiment_057.py','auxiliary_head_051.py','auxiliary_experiment_051.py']},'input_hashes':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/dino-multilayer-044/development_features.npz','artifacts/run-012/development_features.npz']},'protected':protected,'bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'epochs':head.EPOCHS,'trajectories_per_split':60}),flush=True)
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,s in {**p['sources'],**p['input_hashes'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return p
def prep(X,y):
 sc=StandardScaler().fit(X);mean=float(y.mean());std=float(y.std());assert std>0;return sc,mean,std,torch.tensor(sc.transform(X),dtype=torch.float32),torch.tensor((y-mean)/std,dtype=torch.float32)
def benchmark():
 check();torch.set_num_threads(2);ids,X,y,g,fid,*_=inputs('original');tr=np.flatnonzero(fid!=0);sc,mean,std,xt,yt=prep(X[tr],y[tr]);t=time.perf_counter();states,r=head.fit(xt,yt,head.SEEDS[0]);seconds=time.perf_counter()-t;_,_,_,edges,bins=h.preprocess(X[tr],y[tr]);control,cr=h.fit(xt,yt,torch.tensor(bins),head.SEEDS[0],0.);assert h.state_hash(states[100])==h.state_hash(control.state_dict());assert r['order_hash']==cr['batch_order_sha256'];errors=[]
 for epoch,state in states.items():
  p=head.predict(state,xt,mean,std);assert np.isfinite(p).all();errors.append(float(fv.rmse(y[tr],p)))
 report={'trajectory_seconds':seconds,'projected_original_seconds':seconds*60,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'exact051_final_state_match':True,'matching_batch_order':True,'checkpoint_train_rmse':dict(zip(map(str,head.EPOCHS),errors))};report['resource_pass']=bool(report['projected_original_seconds']<300 and report['rss_peak_bytes']<2e9);fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)
def run(split):
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,groups,fid,r003,r007,oldmean=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((len(head.EPOCHS),3,400),np.nan);selected=np.full(400,np.nan);logs=[];train_error=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);fd=out/f'fold-{k}';fd.mkdir();inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261227+k).split(X[tr],y[tr],groups[tr]));ip=np.full((len(head.EPOCHS),3,len(tr)),np.nan);innerlogs=[]
  for j,(it,iv) in enumerate(inner):
   assert not set(groups[tr[it]])&set(groups[tr[iv]]);sc,ym,ys,xt,yt=prep(X[tr[it]],y[tr[it]]);xv=sc.transform(X[tr[iv]]);ir={'train_indices':tr[it].tolist(),'validation_indices':tr[iv].tolist(),'target_mean':ym,'target_std':ys,'seed_logs':{}}
   with (fd/f'inner-{j}-scaler.npz').open('xb') as file:np.savez_compressed(_repo_io(file),mean=sc.mean_,scale=sc.scale_,var=sc.var_)
   for si,base in enumerate(head.SEEDS):
    states,r=head.fit(xt,yt,base+k+100*(j+1))
    for ei,epoch in enumerate(head.EPOCHS):ip[ei,si,iv]=head.predict(states[epoch],xv,ym,ys)
    ir['seed_logs'][str(base)]=r
   innerlogs.append(ir)
  innermean=ip.mean(1);scores=[{'epoch':epoch,'rmse':fv.rmse(y[tr],innermean[ei])} for ei,epoch in enumerate(head.EPOCHS)];chosen=min(scores,key=lambda a:(a['rmse'],a['epoch']))['epoch'];ei_chosen=head.EPOCHS.index(chosen)
  with (fd/'inner_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),train_indices=tr,predictions=ip)
  sc,ym,ys,xt,yt=prep(X[tr],y[tr]);xv=sc.transform(X[va]);seedlogs={};trainpred=[]
  for si,base in enumerate(head.SEEDS):
   states,r=head.fit(xt,yt,base+k)
   for ei,epoch in enumerate(head.EPOCHS):pred[ei,si,va]=head.predict(states[epoch],xv,ym,ys)
   trainpred.append(head.predict(states[chosen],xt,ym,ys));path=fd/f'seed-{base}.pt';torch.save({'states':{str(epoch):states[epoch] for epoch in sorted({chosen,100})},'selected_epoch':chosen,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'seed':base+k,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va]},_repo_io(path));r['checkpoint_sha256']=fv.sha(path);seedlogs[str(base)]=r
  meanpred=pred[:,:,va].mean(1);np.testing.assert_allclose(meanpred[-1],oldmean[va],atol=1e-8,rtol=0);selected[va]=meanpred[ei_chosen];tp=np.mean(trainpred,axis=0);train_error.extend((tp-y[tr]).tolist());record={'fold':k,'selected_epoch':chosen,'inner_scores':scores,'inner_logs':innerlogs,'inner_predictions_sha256':fv.sha(fd/'inner_predictions.npz'),'models':seedlogs,'selected_mean3_train_rmse':fv.rmse(y[tr],tp),'diagnostic_outer_checkpoint_rmse':{str(epoch):fv.rmse(y[va],meanpred[ei]) for ei,epoch in enumerate(head.EPOCHS)},'control007_component_max_error':float(abs(meanpred[-1]-oldmean[va]).max())};fv.write(fd/'report.json',record);logs.append(record);print(json.dumps({'split':split,'fold':k,'selected_epoch':chosen,'selected_train_rmse':record['selected_mean3_train_rmse'],'elapsed_seconds':time.perf_counter()-t}),flush=True)
 new=.75*r003+.25*selected;control=.75*r003+.25*pred[-1].mean(0);np.testing.assert_allclose(control,r007,atol=1e-8,rtol=0);ps={'reference007':r007,'selected_mlp':selected,'primary':new,'reference003':r003,'control100':control};scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in ps.items()};gain=fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261228).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain':gain,'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'selected_control_mlp_residual_correlation':float(np.corrcoef(y-selected,y-oldmean)[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,all_checkpoint_seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'selected_epochs':[a['selected_epoch'] for a in logs],'selected_train_rmse_pooled':float(np.sqrt(np.mean(np.array(train_error)**2))),'seconds':time.perf_counter()-t,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'control007_max_error':float(abs(control-r007).max()),'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(a>0 for a in fg)>=3),'comparison':comp})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
