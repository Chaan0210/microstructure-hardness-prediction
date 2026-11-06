"""Fixed rank64 input projection, otherwise matched007 MLP."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import earlystop_experiment_057 as base
import auxiliary_head_051 as h
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/pca-mlp-060');RANK=64;SEEDS=(20261128,20261205,20261219)
def prepare():
 OUT.mkdir(exist_ok=False);prior=json.loads((base.OUT/'plan.json').read_text());plan={'hypothesis':'Raw3456 frozenDINO coordinates allow easy interpolation. Unsupervised rank64 projection may remove poorlyestimated directions while retaining useful nonlinear signal;may also discard targetinformation.','difference':'037PCA64usedRBFkernelonfinal1152;048usedsupervised4D bottleneck+kernel. NoequivalentPCA64projectback3456+same007MLP found.','preprocessing':'Fit StandardScaler3456 onoutertrain, thenPCA64fullSVD,whitenFalse onscaledoutertrain. Transform→inverse_transform reconstructs inoriginalstandardized3456space. No second rescaling/whitening. Sameheadshape/init, onlyinput rankreduces. PCAtrainedwithoutlabels.','why_projectback':'Preserve3456inputcoordinates/same64hiddenweights andseed initialization. Functiononrank64subspace has lower effectiveinputdimension, thoughrawparametercountunchanged. Discardednullspaceweights remain butdo not see data except numericalroundoff;AdamWunchanged.','fixed':'Rank64predeclared,no ranksearch/innerselection;100epochSmoothL1/3seeds/dropout.25/AdamW/cosine unchanged. Same007architecture anddataorder.','primary':'.75samefold003+.25mean3PCAinputMLP versus007.','screen':'Originalgain>pairedbootstrapSD and>=3/5fold positive. Onlythenbothpredefinedresplits withPCA/scalers/heads refitperouterfold. Nootherweights/ranks/epochs tried.','resource':'CPU2threads,5PCAfits+15MLPfits perpartition; benchmarkonePCA+oneMLP,projected<300s andRSS<2GB.','scope':'Development400only,existinguvpackages,cost0;noholdout/test/candidateCSV/upload/newinstall/externaldata','bias':fv.CAVEAT,'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['pca_mlp_060.py','auxiliary_head_051.py','earlystop_experiment_057.py','auxiliary_experiment_051.py']},'inputs':prior['input_hashes'],'protected':prior['protected']};fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps({'prepared':True,'rank':64}),flush=True)
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,d in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==d,n
 return p
def projected(sc,pca,X):return pca.inverse_transform(pca.transform(sc.transform(X)))
def prepare_fold(X,y,tr):
 sc=StandardScaler().fit(X[tr]);pca=PCA(n_components=RANK,svd_solver='full',whiten=False).fit(sc.transform(X[tr]));ym=float(y[tr].mean());ys=float(y[tr].std());return sc,pca,ym,ys,torch.tensor(projected(sc,pca,X[tr]),dtype=torch.float32),torch.tensor((y[tr]-ym)/ys,dtype=torch.float32)
def benchmark():
 check();torch.set_num_threads(2);ids,X,y,g,fid,*_=base.inputs('original');tr=np.flatnonzero(fid!=0);t=time.perf_counter();sc,pca,ym,ys,xt,yt=prepare_fold(X,y,tr);pcasec=time.perf_counter()-t;_,_,_,_,bins=h.preprocess(X[tr],y[tr]);t=time.perf_counter();m,r=h.fit(xt,yt,torch.tensor(bins),SEEDS[0],0.);fitsec=time.perf_counter()-t
 C=pca.components_;np.testing.assert_allclose(C@C.T,np.eye(64),atol=1e-10,rtol=0);Z=projected(sc,pca,X[tr]);Z2=pca.inverse_transform(pca.transform(Z));np.testing.assert_allclose(Z,Z2,atol=1e-10,rtol=0);assert np.linalg.matrix_rank(Z-Z.mean(0),tol=1e-8)<=64;report={'pca_seconds':pcasec,'fit_seconds':fitsec,'projected_original_seconds':5*pcasec+15*fitsec,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'orthonormal_projection_idempotence_rank_passed':True,'train_explained_variance':float(pca.explained_variance_ratio_.sum())};report['resource_pass']=bool(report['projected_original_seconds']<300 and report['rss_peak_bytes']<2e9);fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)
def run(split):
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,g,fid,r003,r007,oldmean=base.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((3,400),np.nan);trainerr=[];logs=[];start=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(g[tr])&set(g[va]);fd=out/f'fold-{k}';fd.mkdir();sc,pca,ym,ys,xt,yt=prepare_fold(X,y,tr);xv=projected(sc,pca,X[va]);_,_,_,_,bins=h.preprocess(X[tr],y[tr]);joblib.dump({'scaler':sc,'pca':pca,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va]},_repo_io(fd/'transform.joblib'));old=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}', 'report.json')).read_text());log={'fold':k,'explained_variance':float(pca.explained_variance_ratio_.sum()),'transform_sha256':fv.sha(fd/'transform.joblib'),'models':{}};tp=[]
  for si,seed in enumerate(SEEDS):
   model,r=h.fit(xt,yt,torch.tensor(bins),seed+k,0.);ref=old['models'][str(seed)];assert r['initial_state_sha256']==ref['initial_state_sha256'] and r['batch_order_sha256']==ref['batch_order_sha256'];pred[si,va]=h.predict(model,xv,ym,ys);tp.append(h.predict(model,xt,ym,ys));path=fd/f'seed-{seed}.pt';torch.save({'model':model.state_dict(),'target_mean':ym,'target_std':ys,'seed':seed+k,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va],'transform_sha256':log['transform_sha256']},_repo_io(path));r['checkpoint_sha256']=fv.sha(path);log['models'][str(seed)]=r
  avg=np.mean(tp,axis=0);trainerr.extend((avg-y[tr]).tolist());log['mean3_train_rmse']=fv.rmse(y[tr],avg);log['validation_rmse']=fv.rmse(y[va],pred[:,va].mean(0));logs.append(log);fv.write(fd/'report.json',log);print(json.dumps({'split':split,'fold':k,'train_rmse':log['mean3_train_rmse'],'validation_rmse':log['validation_rmse'],'seconds':time.perf_counter()-start}),flush=True)
 mean=pred.mean(0);new=.75*r003+.25*mean;ps={'reference007':r007,'pca_mlp':mean,'primary':new,'reference003':r003};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in ps.items()};gain=fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261231).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain':gain,'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'pca_original_mlp_residual_correlation':float(np.corrcoef(y-mean,y-oldmean)[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'train_rmse_pooled':float(np.sqrt(np.mean(np.array(trainerr)**2))),'explained_variance_by_fold':[r['explained_variance'] for r in logs],'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(a>0 for a in fg)>=3),'comparison':comp})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
