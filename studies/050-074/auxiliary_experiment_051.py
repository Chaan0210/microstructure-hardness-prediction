"""Fixed regression-only versus quantile-classification auxiliary head study."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import auxiliary_head_051 as h
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/auxiliary-head-051')

def inputs(split):
 with np.load(_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].astype(float).reshape(400,3456)
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 assert X.shape==(400,3456) and len(set(ids))==400
 source=_repo_path(ROOT, 'artifacts/dino-multilayer-044', split, 'development_oof.csv');verified=json.loads((_repo_path(ROOT, 'reports/dino-044-validation.json')).read_text());assert fv.sha(source)==verified['results'][split]['oof_sha256']
 with source.open() as f:rows=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids;np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows]);fid=np.array([int(a['fold']) for a in rows]);ref003=np.array([float(a['reference003']) for a in rows]);ref006=np.array([float(a['multi3_blend']) for a in rows]);assert np.isfinite(X).all()
 return ids,X,y,groups,fid,ref003,ref006

def prepare():
 OUT.mkdir(exist_ok=False)
 protected={n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/submission-candidate-006.csv','artifacts/dino-multilayer-044/candidate-design.json','artifacts/final-model-045/dino-regressor.joblib']}
 plan={'hypothesis':'Coarse hardness classification supplies an additional supervised gradient to the same64-dimensional hidden representation; this can regularize or interfere with regression. Classifier is not independent of regression sharedMLP.','reference_url':'https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution','source_access':'Official writeup opened but no body exposed by browser. Indexed primary-source snippet confirms SmoothL1 regression plus CrossEntropy classification; seven bins/0.3 and published scores supplied by parent, not independently fully re-read here. No external code, data or weights downloaded.','transfer_limits':'Own small frozenDINOv2 head, one hardness target, foldtrain quantile bins. Not a reproduction of DINOv3Large five-target trainable-backbone solution or UEP; no pseudo labels/online learning/test adaptation.','config':h.CONFIG,'arms':{'regression_only':0.,'auxiliary':.3},'architecture':'StandardScaler3456 -> Linear64 -> GELU -> Dropout.25 -> Linear1 regression and Linear7 logits, shared bottleneck; backbone frozen cached.','target':'Foldtrain mean/std, SmoothL1 beta1 on standardized hardness, mean CE coefficient.3. Six quantile boundaries j/7,j1..6 in original hardness, foldtrain only, searchsorted right. Duplicated cutpoints cause error, not silently changed bins.','paired_controls':'Exactly identical initialized whole model, data permutations, dropout seed, optimizer/scheduler/epochs/regularization and both heads instantiated. Only auxiliary coefficient differs. Regression-only classifier gets no gradient and stays unchanged.','epoch_rule':'100 fixed epochs,last endpoint,cosine scheduler. Diagnostics epoch0/20/40/60/80/100 do not choose checkpoint. No loss/width/dropout/seed/epoch sweep.','primary':'auxiliary_blend=.75samefold003+.25auxiliary. Matchedcontrol=.75samefold003+.25regression_only. Compare both to verified samefold006. Weights fixed, no Public adjustment.','screen':'Advance auxiliary only if original pooled primary RMSE improves both006 and matchedcontrol, and >=3/5folds improve against each. Gate controls compute, not statistical significance. Otherwise stop, no wider search or fallback winner.','confirmation':'If screen passes, identical recipe on both existing partitions20261031/20261101, refitting all heads/scalers/targetstats/bins within each fold. Only consistency across3partitions supports candidate consideration; no candidate created by this script.','resource':'CPU2threads. Benchmark originalfold0 training only 10epochs eacharm, same100epoch scheduler; discard benchmark models. Stop if projected original10fits>5minutes or RSS>2GB.','scope':'Development400 cached features/labels only. No image extraction, holdout/test reads, installs/cost/uploads. Historical repeated-selection bias remains.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['auxiliary_head_051.py','auxiliary_experiment_051.py']},'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/dino-multilayer-044/development_features.npz','artifacts/run-012/development_features.npz']},'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps({'stage':'prepared','config':h.CONFIG}),flush=True)

def verify():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,s in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return p

def benchmark():
 verify();torch.set_num_threads(2);ids,X,y,g,f,_,_=inputs('original');tr=np.flatnonzero(f!=0);sc,ym,ys,edges,bins=h.preprocess(X[tr],y[tr]);xt=torch.tensor(sc.transform(X[tr]),dtype=torch.float32);yt=torch.tensor((y[tr]-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins,dtype=torch.long)
 gradient=h.shared_gradient_check(xt[:64],yt[:64],bt[:64],h.CONFIG['seed']);start=time.perf_counter();records=[]
 for a in (0.,.3):m,r=h.fit(xt,yt,bt,h.CONFIG['seed'],a,epochs=10);records.append(r)
 elapsed=time.perf_counter()-start;assert records[0]['initial_state_sha256']==records[1]['initial_state_sha256'];assert records[0]['batch_order_sha256']==records[1]['batch_order_sha256'];report={'device':'cpu','threads':2,'ten_epochs_each_arm_seconds':elapsed,'projected_original_fivefold_seconds':elapsed*50,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'parameters':sum(p.numel() for p in m.parameters()),'gradient_check':gradient,'paired_initialization_and_batches_identical':True,'train_only':True,'classification_parameter_delta':[r['classifier_max_parameter_change'] for r in records]};report['resource_pass']=report['projected_original_fivefold_seconds']<300 and report['peak_rss_bytes']<2e9;fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)

def run(split):
 verify();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,groups,fid,ref003,ref006=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred={k:np.full(400,np.nan) for k in ('regression_only','auxiliary')};logs=[];start=time.perf_counter()
 for fold in range(5):
  tr=np.flatnonzero(fid!=fold);va=np.flatnonzero(fid==fold);assert len(tr)==320 and len(va)==80;assert not set(groups[tr])&set(groups[va]);sc,ym,ys,edges,bins=h.preprocess(X[tr],y[tr]);xt=torch.tensor(sc.transform(X[tr]),dtype=torch.float32);yt=torch.tensor((y[tr]-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins,dtype=torch.long);xv=torch.tensor(sc.transform(X[va]),dtype=torch.float32);yv=torch.tensor((y[va]-ym)/ys,dtype=torch.float32);bv=torch.tensor(np.searchsorted(edges,y[va],side='right'),dtype=torch.long);seed=h.CONFIG['seed']+fold;rec={'fold':fold,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'bin_edges':edges.tolist(),'bin_train_counts':np.bincount(bins,minlength=7).tolist(),'target_mean':ym,'target_std':ys,'models':{}};fd=out/f'fold-{fold}';fd.mkdir();t=time.perf_counter()
  for arm,weight in [('regression_only',0.),('auxiliary',.3)]:
   model,r=h.fit(xt,yt,bt,seed,weight,validation=(xv,yv,bv));pred[arm][va]=h.predict(model,xv,ym,ys);path=fd/f'{arm}.pt';state={'model':model.state_dict(),'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'target_mean':ym,'target_std':ys,'bin_edges':torch.tensor(edges),'train_ids':rec['train_ids'],'validation_ids':rec['validation_ids'],'config':h.CONFIG,'seed':seed,'aux_weight':weight};torch.save(state,_repo_io(path))
   fresh=h.Head();a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);fresh.load_state_dict(a['model']);reload=h.predict(fresh,(X[va]-a['scaler_mean'].numpy())/a['scaler_scale'].numpy(),ym,ys);error=float(abs(reload-pred[arm][va]).max());assert error<1e-5
   r.update(train_rmse=fv.rmse(y[tr],h.predict(model,xt,ym,ys)),validation_rmse=fv.rmse(y[va],pred[arm][va]),reload_max_error=error,checkpoint_sha256=fv.sha(path));rec['models'][arm]=r
  assert rec['models']['regression_only']['initial_state_sha256']==rec['models']['auxiliary']['initial_state_sha256'];assert rec['models']['regression_only']['batch_order_sha256']==rec['models']['auxiliary']['batch_order_sha256'];rec['seconds']=time.perf_counter()-t;logs.append(rec);fv.write(fd/'report.json',rec);print(json.dumps({'split':split,'fold':fold,'seconds':rec['seconds'],'validation_rmse':{a:q['validation_rmse'] for a,q in rec['models'].items()}}),flush=True)
 pred.update({a+'_blend':.75*ref003+.25*v for a,v in list(pred.items())});pred['reference003']=ref003;pred['reference006']=ref006;scores={a:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for a,v in pred.items()};comparisons={};ix=np.random.default_rng(20261128).integers(0,400,(3000,400));primary=pred['auxiliary_blend']
 for reference in ('reference006','regression_only_blend'):
  v=pred[reference];gain=fv.rmse(y,v)-fv.rmse(y,primary);fg=[fv.rmse(y[fid==k],v[fid==k])-fv.rmse(y[fid==k],primary[fid==k]) for k in range(5)];boot=np.sqrt(np.mean((y-v)[ix]**2,axis=1))-np.sqrt(np.mean((y-primary)[ix]**2,axis=1));comparisons[reference]={'gain':gain,'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'conditional_bootstrap_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'bootstrap_sd':float(boot.std())}
 with (out/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[v[j] for v in pred.values()]] for j,i in enumerate(ids))
 report={'split':split,'scores':scores,'comparisons':comparisons,'folds':logs,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'selection_bias':'Repeatedly inspected development400, one fixed seed; conditional bootstrap is not full procedure uncertainty.','new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':all(v['gain']>0 and v['positive_folds']>=3 for v in comparisons.values()),'reason':'Both baseline006 and matchedregressioncontrol must improve; no fallback winner or sweep','comparisons':comparisons})
 verify();print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.action,lambda:run(a.action))()
