"""Fixed combination: verified043 fold-specific adapted block12 +007 MLP."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch,joblib
import dino_lastblock_043 as adapt
import earlystop_experiment_057 as base
import auxiliary_head_051 as head
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/adapted-mlp-059');SEEDS=(20261128,20261205,20261219)
def prepare():
 OUT.mkdir(exist_ok=False);p={'hypothesis':'Weak lastblockadaptation043 gave small consistent gain withfrozenRidge; nonlinearhead007 improved frozenfeatures. Their combination may reveal useful nonlinearity in hardness-adapted features or overfit a twice-supervised representation. Fixedcombination untested; no jointbackprop or newbackbone fit.','architecture':'Useunchanged frozenblocks6/9 CLS+mean/std from044; replacefinalblock12 descriptor withsameouterfold043 weakadaptedblock descriptor. 3456dimensions,same00764hidden/dropout.25/SmoothL1/100epochcosine/3seedmean.','paired':'Onlylastblock representation changes; samewholeMLPinitialization,dataorder,optimizer/seed/epochs. Foldtrain StandardScaler/targetmeanstd refit onadaptedfeatures.','leakage':'Reuse only043 checkpoint whose trainIDs/valIDs exactlymatch currentouterfold; verifyoriginalsource/head/checkpointhashes, frozenprefix provenance, norm/headunchanged. Adapting onoutertrain andfittingMLP onsameoutertrain is validtwo-stage training; outervalidation labels never enter eitherstage. Repeatpartitions use their own15previouslyvalidated043backbones, not originalfoldbackbones.','primary':'.75samefold003+.25mean3adaptedfeatureMLP;compare verifiedsamefold007. Nootherblend/epoch/projection/layer choices.','screen':'Originalgain>pairedconditionalbootstrapSD and>=3/5positivefolds. Onlythenbothfixedrepartitions. Stoponfailure, no interpolationweights betweenrepresentations.','reference':'043recipe isfixed8epochs,LR1e-6,onlyblock12,MSE+anchor. Its adaptation is supervised but modelselection here uses currentpairedOOF,notPublic.','resource':'Frozen development400 prefix memmap400x1297x384;imagebatch4 lastblock inference. Benchmark8rows, projected5fold400rows<300s,RSS<2.5GB,MPSdriver<3GB. ThenCPU2threads15MLP fits~7s based058.','scope':'Development400only;existinguv/torch/DINO/numpy/sklearn; noinstall,cost,externaldata,holdout/test,candidateCSV/upload.','bias':fv.CAVEAT,'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['adapted_mlp_059.py','dino_lastblock_043.py','earlystop_experiment_057.py','auxiliary_head_051.py','dino_frozen_037.py']},'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/dino-multilayer-044/development_features.npz','artifacts/run-012/development_features.npz','artifacts/dino-lastblock-043/prefix.json','reports/dino-043-validation.json']},'protected':json.loads((_repo_path(ROOT, 'artifacts/token-cov-056/plan.json')).read_text())['protected']};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'fixed_dimensions':3456}),flush=True)
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,d in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==d,n
 official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,d in official['official_source_sha256'].items():assert fv.sha(adapt.ds.base.SOURCE/n)==d,n
 return p
def cache(ids):
 meta=json.loads((adapt.OUT/'prefix.json').read_text());assert meta['ids']==ids and fv.sha(adapt.OUT/'prefix.npy')==meta['sha256'];return np.load(_repo_io(adapt.OUT/'prefix.npy'),mmap_mode='r')
def tail_for(m,split,k,ids,X,y,fid):
 report=json.loads((adapt.OUT/split/'report.json').read_text());rec=report['folds'][k];tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert rec['train_ids']==[ids[j] for j in tr] and rec['validation_ids']==[ids[j] for j in va];cp=adapt.OUT/split/f'fold-{k}'/'model.pt';hp=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, f'fold-{k}', 'full504.joblib');assert fv.sha(cp)==rec['checkpoint_sha256'] and fv.sha(hp)==rec['head_source_sha256'];obj=joblib.load(_repo_io(hp));r=obj['model'];assert obj['train_ids']==rec['train_ids'] and obj['validation_ids']==rec['validation_ids'];np.testing.assert_allclose(r.scaler.mean_,X[tr,-1152:].mean(0),atol=1e-10,rtol=0);assert r.ymean==float(y[tr].mean()) and r.ystd==float(y[tr].std());a=torch.load(_repo_io(cp),map_location='cpu',weights_only=True);assert a['train_ids']==rec['train_ids'] and a['validation_ids']==rec['validation_ids'] and a['epoch']==8
 tail=adapt.Tail(m,r);initial={n:v.detach().cpu().clone() for n,v in tail.state_dict().items()}
 for n,v in a['state'].items():
  if not n.startswith('block.'):assert torch.equal(v,initial[n])
 tail.load_state_dict(a['state']);tail.requires_grad_(False);tail.eval();return tail.to('mps'),r,rec

def features(tail,tokens,indices):
 vals=[]
 with torch.inference_mode():
  for j in range(0,len(indices),4):
   z=torch.from_numpy(np.array(tokens[indices[j:j+4]],copy=True)).to('mps');z=tail.norm(tail.block(z));p=z[:,1:];v=torch.cat((z[:,0],p.mean(1),p.std(1,correction=0)),1);assert torch.isfinite(v).all();vals.append(v.cpu().numpy())
 return np.concatenate(vals)
def benchmark():
 check();torch.set_num_threads(2);ids,X,y,g,fid,*_=base.inputs('original');tokens=cache(ids);m=adapt.ds.base.model().to('mps');tail,r,rec=tail_for(m,'original',0,ids,X,y,fid);va=np.flatnonzero(fid==0);idx=va[:8];t=time.perf_counter();F=features(tail,tokens,idx);seconds=time.perf_counter()-t;v=r.predict(F.astype(float));expected=np.array(rec['predictions'])[:8];np.testing.assert_allclose(v,expected,atol=2e-3,rtol=0);np.testing.assert_array_equal(F,features(tail,tokens,idx));assert not any(p.requires_grad or p.grad is not None for p in tail.parameters());report={'rows':8,'seconds':seconds,'projected_2000row_seconds':seconds*250,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'adapted043_prediction_max_error':float(abs(v-expected).max()),'repeat_exact':True,'tail_frozen':True};report['resource_pass']=bool(report['projected_2000row_seconds']<300 and report['rss_peak_bytes']<2.5e9 and report['mps_driver_bytes']<3e9);fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)
def run(split):
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,g,fid,r003,r007,oldmean=base.inputs(split);tokens=cache(ids);m=adapt.ds.base.model().to('mps');out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((3,400),np.nan);trainerr=[];logs=[];t=time.perf_counter();extract_seconds=0.;fit_seconds=0.
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(g[tr])&set(g[va]);fd=out/f'fold-{k}';fd.mkdir();tail,r,rec=tail_for(m,split,k,ids,X,y,fid);before=head.state_hash(tail.state_dict());tf=time.perf_counter();F=features(tail,tokens,np.arange(400));extract_seconds+=time.perf_counter()-tf;assert head.state_hash(tail.state_dict())==before;checkpred=r.predict(F[va].astype(float));np.testing.assert_allclose(checkpred,rec['predictions'],atol=2e-3,rtol=0);A=np.concatenate((X[:,:2304],F.astype(float)),axis=1);assert A.shape==(400,3456)
  with (fd/'features.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=A)
  del tail;torch.mps.empty_cache();sc,ym,ys,xt,yt=base.prep(A[tr],y[tr]);_,_,_,edges,bins=head.preprocess(A[tr],y[tr]);xv=sc.transform(A[va]);oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}', 'report.json')).read_text());log={'fold':k,'adapted_backbone_sha256':rec['checkpoint_sha256'],'feature_sha256':fv.sha(fd/'features.npz'),'adapted043_prediction_max_error':float(abs(checkpred-rec['predictions']).max()),'frozen_blocks69_unchanged':True,'adapted_feature_change_rms':float(np.sqrt(np.mean((F-X[:,-1152:])**2))),'models':{}};tp=[]
  for si,seed in enumerate(SEEDS):
   ts=time.perf_counter();model,rh=head.fit(xt,yt,torch.tensor(bins),seed+k,0.);fit_seconds+=time.perf_counter()-ts;old=oldlog['models'][str(seed)];assert rh['initial_state_sha256']==old['initial_state_sha256'] and rh['batch_order_sha256']==old['batch_order_sha256'];pred[si,va]=head.predict(model,xv,ym,ys);tp.append(head.predict(model,xt,ym,ys));path=fd/f'seed-{seed}.pt';torch.save({'model':model.state_dict(),'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'target_mean':ym,'target_std':ys,'seed':seed+k,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va],'backbone_sha256':rec['checkpoint_sha256']},_repo_io(path));rh['checkpoint_sha256']=fv.sha(path);log['models'][str(seed)]=rh
  avg=np.mean(tp,axis=0);trainerr.extend((avg-y[tr]).tolist());log['mean3_train_rmse']=fv.rmse(y[tr],avg);log['mean3_validation_rmse']=fv.rmse(y[va],pred[:,va].mean(0));fv.write(fd/'report.json',log);logs.append(log);print(json.dumps({'split':split,'fold':k,'train_rmse':log['mean3_train_rmse'],'validation_rmse':log['mean3_validation_rmse'],'elapsed_seconds':time.perf_counter()-t}),flush=True)
 mean=pred.mean(0);new=.75*r003+.25*mean;ps={'reference007':r007,'adapted_mlp':mean,'primary':new,'reference003':r003};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in ps.items()};gain=fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261230).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain':gain,'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'adapted_frozen_mlp_residual_correlation':float(np.corrcoef(y-mean,y-oldmean)[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'train_rmse_pooled':float(np.sqrt(np.mean(np.array(trainerr)**2))),'seconds':time.perf_counter()-t,'extraction_seconds':extract_seconds,'MLP_fit_seconds':fit_seconds,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(a>0 for a in fg)>=3),'comparison':comp})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
