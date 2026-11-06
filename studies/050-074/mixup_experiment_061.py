"""Fixed featuremixup regularization versus007; samefrozenrepresentation andhead."""
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
import mixup_head_061 as head
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/mixup-061')
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,v in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==v,n
 return p
def prepare():
 OUT.mkdir(exist_ok=False);old=json.loads((base.OUT/'plan.json').read_text());p={'hypothesis':'Encourage a smoother head between training feature vectors without dropping lowvariance directions. Fixedfeature mixup can regularize interpolation but may violate nonlinear hardness relations; benefit unknown.','evidence':'057earlystop and060ranktruncation reducedtrainfitbutworsenedensemble.059weakRidgeadaptation+MLP gave nearlysame residuals. Noearliermixup implementationfound.','reference':'https://arxiv.org/abs/1710.09412','reference_access':'Primaryabstractread. Ownimplementation of convexinput/targetmixing on cachedDINOfeatures, notclaimofphysicalintermediateimage or paperperformance. Noexternalcode/data used.','mixup':'Only standardized outertrain features andtargets. Perbatch independentPCG64seed(base+fold+2000),partnerpermutation ofsamebatchindices,perexamplelambdaBeta(.4,.4). Same lambda mixesfeaturesandlabels. Rawpairing may includeself. No rank/alpha/mixratio search.','config':base.h.CONFIG,'effective_aux_weight':0.,'epochs':100,'seeds':list(head.SEEDS),'paired':'Samehead/dropout/globaltorchstreams/baseorder/AdamW/gradientclip100cosineschedule as007. IndependentNumPy mixingRNG doesnot altertorchdropout. Only trainminibatchinterpolation added. Evaluate original unmixed train/validation examples.','primary':'.75samefold003+.25mean3mixupMLP versus007.100endpointfixed,no057earlystop,058MSEor060PCA.','screen':'Originalgain>conditionalbootstrapSD and>=3/5positivefolds, thenunchangedbothresplits. Noalphatuning,lambdaadaptation,weightsearch orfallback.','scope':'Development400only,localuvCPU2threads,existingpackages,cost0; noexternaldata/holdout/test/candidateCSV/upload/install. Trainingaugmentation derivesonlyauthorizedfoldtrainrows.','resource':'15fitsperpartitionexpected<30s. Benchmarkonefit,projected15<300s,RSS<2GB.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['mixup_head_061.py','mixup_experiment_061.py','earlystop_experiment_057.py','auxiliary_head_051.py','auxiliary_experiment_051.py']},'inputs':old['input_hashes'],'protected':old['protected'],'bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'alpha':.4}),flush=True)
def benchmark():
 check();torch.set_num_threads(2);X=torch.arange(12,dtype=torch.float32).reshape(4,3);y=torch.arange(4,dtype=torch.float32);ix=np.arange(4);partner=ix[::-1].copy()
 for w in (0.,1.):
  xm,ym=head.mix(X,y,ix,partner,np.full(4,w));torch.testing.assert_close(xm,X[ix] if w else X[partner],atol=0,rtol=0);torch.testing.assert_close(ym,y[ix] if w else y[partner],atol=0,rtol=0)
 xm,ym=head.mix(X,y,ix,partner,np.full(4,.5));torch.testing.assert_close(xm,(X+X[partner])/2,atol=0,rtol=0);torch.testing.assert_close(ym,(y+y[partner])/2,atol=0,rtol=0)
 ids,X,y,g,fid,*_=base.inputs('original');tr=np.flatnonzero(fid!=0);sc,ym,ys,xt,yt=base.prep(X[tr],y[tr]);t=time.perf_counter();states,rec=head.fit(xt,yt,head.SEEDS[0]);seconds=time.perf_counter()-t;old=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052/original/fold-0/report.json')).read_text())['models'][str(head.SEEDS[0])];assert rec['initial_hash']==old['initial_state_sha256'] and rec['order_hash']==old['batch_order_sha256'];assert rec['final_hash']!=old['final_state_sha256'];report={'seconds':seconds,'projected15fit_seconds':seconds*15,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'convex_input_target_endpoint_midpoint_fixture_passed':True,'same_initialization_baseorder':True};report['resource_pass']=bool(seconds*15<300 and report['rss_peak_bytes']<2e9);fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)
def run(split):
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,groups,fid,r003,r007,oldmean=base.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((3,400),np.nan);trainerrors=[];logs=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);sc,ym,ys,xt,yt=base.prep(X[tr],y[tr]);xv=sc.transform(X[va]);fd=out/f'fold-{k}';fd.mkdir();oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}', 'report.json')).read_text());record={'fold':k,'models':{}};tp=[]
  for si,seed in enumerate(head.SEEDS):
   states,r=head.fit(xt,yt,seed+k);old=oldlog['models'][str(seed)];assert r['initial_hash']==old['initial_state_sha256'] and r['order_hash']==old['batch_order_sha256'];path=fd/f'seed-{seed}.pt';torch.save({'state':states[100],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'seed':seed+k,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va]},_repo_io(path));pred[si,va]=head.predict(states[100],xv,ym,ys);tp.append(head.predict(states[100],xt,ym,ys));r.update(checkpoint_sha256=fv.sha(path),validation_rmse=fv.rmse(y[va],pred[si,va]));record['models'][str(seed)]=r
  avg=np.mean(tp,axis=0);trainerrors.extend((avg-y[tr]).tolist());record['mean3_train_rmse']=fv.rmse(y[tr],avg);fv.write(fd/'report.json',record);logs.append(record);print(json.dumps({'split':split,'fold':k,'train_rmse':record['mean3_train_rmse'],'seconds':time.perf_counter()-t}),flush=True)
 mean=pred.mean(0);new=.75*r003+.25*mean;ps={'reference007':r007,'mixup_mlp':mean,'primary':new,'reference003':r003};scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in ps.items()};gain=fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20270101).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain':gain,'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'mixup_control_mlp_residual_correlation':float(np.corrcoef(y-mean,y-oldmean)[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'mean3_train_rmse_pooled':float(np.sqrt(np.mean(np.array(trainerrors)**2))),'seconds':time.perf_counter()-t,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(a>0 for a in fg)>=3),'comparison':comp})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
