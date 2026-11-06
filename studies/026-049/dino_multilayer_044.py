"""Predeclared frozen blocks6/9/12 texture representation comparison."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import dino_spatial_039 as ds
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/dino-multilayer-044');LAYERS=(5,8,11);LAMBDAS=(.1,1.,10.)

def feature(m,x):
 assert not m.training and not any(p.requires_grad for p in m.parameters())
 with torch.inference_mode():
  layers=m.get_intermediate_layers(x,n=list(LAYERS),return_class_token=True,norm=True)
  a=[torch.cat((c,p.mean(1),p.std(1,correction=0)),1) for p,c in layers]
  assert all(v.shape==(len(x),1152) and torch.isfinite(v).all() for v in a)
  return torch.stack(a,1)

def load():
 ids,_=ds.base.scope()
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 return ids,y,groups

def prepare():
 OUT.mkdir(exist_ok=False)
 protected={str(p.relative_to(ROOT)):fv.sha(p) for p in [*ROOT.glob('artifacts/submission-candidate-005*'),_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'),_repo_path(ROOT, 'artifacts/dino-spatial-039/candidate-design.json'),*(_repo_path(ROOT, 'artifacts/final-model-041')).glob('*'),*(_repo_path(ROOT, 'artifacts/dino-lastblock-043')).rglob('model.pt'),*(_repo_path(ROOT, 'artifacts/dino-lastblock-043')).rglob('*.json'),_repo_path(ROOT, 'reports/dino-043-validation.json'),_repo_path(ROOT, 'reports/dino-043-summary.json')] if p.is_file()}
 plan={'hypothesis':'Intermediate pretrained transformer blocks may retain local grain/texture information discarded by final semantic representation. Combine block6,9,12 descriptors without tuning layer indices.','layers_zero_indexed':list(LAYERS),'primary':'multi3: concatenate each block CLS384+patchmean384+populationstd384 =3456features','diagnostic':'mid6: same1152-dimensional descriptor atblock6; diagnostic only, never promoted based on its outer score','control':'block12 descriptor1152,exactly matched005;all504px fullimagebicubicRGB/ImageNetnorm,no augmentation,official frozen model','preprocessing':'Each coordinate StandardScaler and targetmean/std fitted inside innertrain/outertrain only. Each layer same1152features/equal normalized block energy. No PCA or learned layer selection.','regularization':'Ridgealpha=lambda*p with lambda(.1,1,10),same inner GroupKFold3shuffle seed20261120+outerfold. Proportional alpha makes duplication of identical feature blocks prediction-invariant; does not equalize correlation/effective rank.','screen':'Only primarymulti3 may advance. OriginalCV gain over005>pairedbootstrapSD and at least3/5fold improvements. Then sameprimary+matchedcontrol on both existing repartitions. Mid6 is diagnostic and never an alternative selected winner.','secondary043':'Only if primary has positive gain over005 on all3partitions, report predetermined .5*043blend+.5*multi3blend =.75samefold003+.125adapted043+.125frozenmulti3. No weight search; label this exploratory combination after historical selection.','scope':'Development400 only; no holdout/test/Public-score model selection, new data, installations, paid service, candidateCSV or upload.','bias':'Already reused development400 and partitions; repeated-selection optimism persists. Bootstrap gate is a compute rule,not significance.','source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('dino_multilayer_044.py','dino_regression_037.py','dino_spatial_039.py','dino_frozen_037.py')},'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps(plan),flush=True)

def extract():
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in plan['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
 official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,h in official['official_source_sha256'].items():assert fv.sha(ds.base.SOURCE/n)==h
 # Regression-only synthetic fixture checks dimension scaling; never added to model data.
 rng=np.random.default_rng(20261129);x=rng.normal(size=(40,8));y=rng.normal(size=40);q=rng.normal(size=(7,8));a=FrozenRegressor('all_ridge',8).fit(x,y).predict(q);b=FrozenRegressor('all_ridge',24).fit(np.tile(x,(1,3)),y).predict(np.tile(q,(1,3)));np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
 torch.set_num_threads(2);ids,_,_=load();m=ds.base.model();cpu=feature(m,torch.from_numpy(ds.inputs(ids[0],'full504'))).numpy();m=m.to('mps');before={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};start=time.perf_counter();values=[]
 for j,i in enumerate(ids):
  values.append(feature(m,torch.from_numpy(ds.inputs(i,'full504')).to('mps')).cpu().numpy()[0])
  if (j+1)%80==0:print(json.dumps({'rows':j+1,'seconds':time.perf_counter()-start}),flush=True)
 X=np.stack(values);assert X.shape==(400,3,1152) and np.isfinite(X).all();np.testing.assert_allclose(cpu[0],X[0],atol=2e-3,rtol=2e-3)
 assert all(torch.equal(before[n],v.detach().cpu()) for n,v in m.state_dict().items()) and all(p.grad is None for p in m.parameters())
 with np.load(_repo_path(ROOT, 'artifacts/dino-spatial-039/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(X[:,2],z['full504'])
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 report={'shape':list(X.shape),'seconds':time.perf_counter()-start,'feature_sha256':fv.sha(_repo_path(OUT, 'development_features.npz')),'cpu_mps_feature_max_error':float(abs(cpu[0]-X[0]).max()),'duplication_unit_prediction_error':float(abs(a-b).max()),'finalblock_exact005':True,'weights_unchanged':True,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'image_sha256':{i:fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids}}
 fv.write(_repo_path(OUT, 'extraction.json'),report);print(json.dumps(report|{'image_sha256':'saved'}),flush=True)

def run(split):
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ids,y,groups=load();ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_features.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();X=z['X'].astype(np.float64)
 panels={'control005':X[:,2],'multi3':X.reshape(400,3456)}
 if split=='original':panels['mid6']=X[:,0]
 source=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, 'development_oof.csv');checked=json.loads((_repo_path(ROOT, 'reports/dino-039-validation.json')).read_text());assert fv.sha(source)==checked['results'][split]['oof_sha256']
 with source.open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref003=np.array([float(r['reference003']) for r in rows]);ref005=np.array([float(r['full504_blend']) for r in rows]);pred={a:np.full(400,np.nan) for a in panels};logs=[];t=time.perf_counter();d=_repo_path(OUT, split);d.mkdir(exist_ok=False)
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(X[tr],y[tr],groups[tr]));fd=d/f'fold-{k}';fd.mkdir();log={'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'models':{}}
  for arm,A in panels.items():
   scores=[]
   for lam in LAMBDAS:
    p=np.full(len(tr),np.nan)
    for it,iv in inner:
     assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',lam*A.shape[1]).fit(A[tr[it]],y[tr[it]]);p[iv]=m.predict(A[tr[iv]])
    scores.append({'lambda':lam,'rmse':fv.rmse(y[tr],p)})
   chosen=min(scores,key=lambda a:a['rmse'])['lambda'];m=FrozenRegressor('all_ridge',chosen*A.shape[1]).fit(A[tr],y[tr]);pred[arm][va]=m.predict(A[va]);path=fd/f'{arm}.joblib';joblib.dump({'model':m,'train_ids':log['train_ids'],'validation_ids':log['validation_ids']},_repo_io(path));np.testing.assert_allclose(joblib.load(_repo_io(path))['model'].predict(A[va]),pred[arm][va],atol=1e-10,rtol=0);log['models'][arm]={'chosen_lambda':chosen,'inner_scores':scores,'train_rmse':fv.rmse(y[tr],m.predict(A[tr])),'validation_rmse':fv.rmse(y[va],pred[arm][va]),'sha256':fv.sha(path)}
  logs.append(log);fv.write(d/f'fold-{k}-report.json',log)
 pred.update({a+'_blend':.75*ref003+.25*p for a,p in list(pred.items())});pred['reference003']=ref003;np.testing.assert_allclose(pred['control005_blend'],ref005,rtol=0,atol=1e-10)
 scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};new=pred['multi3_blend'];ix=np.random.default_rng(20261130).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref005)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));gain=fv.rmse(y,ref005)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref005[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];comparison={'gain':gain,'fold_gains':fg,'bootstrap_sd':float(delta.std()),'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist()}
 with (d/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[a[j] for a in pred.values()]] for j,i in enumerate(ids))
 report={'scores':scores,'comparison':comparison,'folds':logs,'seconds':time.perf_counter()-t,'reference_oof_sha256':fv.sha(source)};fv.write(d/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>delta.std() and sum(v>0 for v in fg)>=3),'primary':'multi3','gain':gain,'bootstrap_sd':float(delta.std()),'positive_folds':sum(v>0 for v in fg),'diagnostic_not_eligible':'mid6'})
 for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
 print(json.dumps({'split':split,'scores':{n:a['rmse'] for n,a in scores.items()},'comparison':comparison,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','extract','original','20261031','20261101'));a=p.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
