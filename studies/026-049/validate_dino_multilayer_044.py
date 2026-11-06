"""Independent feature/model reload and fixed secondary OOF combination checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dino_multilayer_044 as e
import numpy as np,torch,joblib
from dino_regression_037 import FrozenRegressor
fv=e.fv;out=e.OUT;plan=json.loads((out/'plan.json').read_text());ex=json.loads((out/'extraction.json').read_text());sel=json.loads((out/'selection.json').read_text());required=('original','20261031','20261101') if sel['advance'] else ('original',);assert all((out/s/'report.json').exists() for s in required)
for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
assert fv.sha(out/'development_features.npz')==ex['feature_sha256'];ids,y,groups=e.load()
with np.load(_repo_io(out/'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].astype(np.float64)
assert X.shape==(400,3,1152) and len(set(ids))==400 and np.isfinite(X).all()
with np.load(_repo_path(ROOT, 'artifacts/dino-spatial-039/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(X[:,2],z['full504'])
for i,h in ex['image_sha256'].items():assert i in ids and fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png'))==h
assert len(ex['image_sha256'])==400
m=e.ds.base.model();torch.set_num_threads(2);cpu=np.concatenate([e.feature(m,torch.from_numpy(e.ds.inputs(i,'full504'))).numpy() for i in ids[:2]]);np.testing.assert_allclose(cpu,X[:2],atol=2e-3,rtol=2e-3);cpuerror=float(abs(cpu-X[:2]).max());del m
panels={'control005':X[:,2],'multi3':X.reshape(400,3456),'mid6':X[:,0]};results={};data={};count=0;maxerror=0.
verified039=json.loads((_repo_path(ROOT, 'reports/dino-039-validation.json')).read_text())
for split in required:
 d=out/split;r=json.loads((d/'report.json').read_text());assert len(r['folds'])==5 and [a['fold'] for a in r['folds']]==list(range(5));source=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, 'development_oof.csv');assert fv.sha(source)==r['reference_oof_sha256']==verified039['results'][split]['oof_sha256']
 with source.open() as f:old=list(csv.DictReader(f))
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids==[a['ID'] for a in old];np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows]);fid=np.array([int(a['fold']) for a in rows]);np.testing.assert_array_equal(fid,[int(a['fold']) for a in old]);p={n:np.array([float(a[n]) for a in rows]) for n in r['scores']}
 np.testing.assert_array_equal(p['reference003'],[float(a['reference003']) for a in old]);np.testing.assert_allclose(p['control005_blend'],[float(a['full504_blend']) for a in old],atol=1e-10,rtol=0)
 arms=('control005','multi3','mid6') if split=='original' else ('control005','multi3')
 for arm in arms:np.testing.assert_allclose(p[arm+'_blend'],.75*p['reference003']+.25*p[arm],atol=1e-12,rtol=0)
 for n,v in p.items():
  assert np.isfinite(v).all() and abs(fv.rmse(y,v)-r['scores'][n]['rmse'])<1e-10
  np.testing.assert_allclose([fv.rmse(y[fid==k],v[fid==k]) for k in range(5)],r['scores'][n]['fold_rmse'],atol=1e-10,rtol=0)
 for k,f in enumerate(r['folds']):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);assert f['train_ids']==[ids[j] for j in tr] and f['validation_ids']==[ids[j] for j in va];held=[]
  for it,iv in f['inner_splits']:assert not set(it)&set(iv) and set(it)|set(iv)==set(tr) and not set(groups[it])&set(groups[iv]);held.extend(iv)
  assert sorted(held)==sorted(tr) and set(f['models'])==set(arms)
  for arm in arms:
   rec=f['models'][arm];path=d/f'fold-{k}'/f'{arm}.joblib';assert fv.sha(path)==rec['sha256'];obj=joblib.load(_repo_io(path));m=obj['model'];A=panels[arm];assert obj['train_ids']==f['train_ids'] and obj['validation_ids']==f['validation_ids'];np.testing.assert_allclose(m.scaler.mean_,A[tr].mean(0),atol=1e-10,rtol=0);assert m.scaler.n_samples_seen_==320 and abs(m.ymean-y[tr].mean())<1e-10 and abs(m.ystd-y[tr].std())<1e-10
   chosen=min(rec['inner_scores'],key=lambda q:q['rmse'])['lambda'];assert rec['chosen_lambda']==chosen and m.alpha==chosen*A.shape[1];v=m.predict(A[va]);np.testing.assert_allclose(v,p[arm][va],atol=1e-10,rtol=0);maxerror=max(maxerror,float(abs(v-p[arm][va]).max()));assert abs(fv.rmse(y[tr],m.predict(A[tr]))-rec['train_rmse'])<1e-10;count+=1
 new=p['multi3_blend'];ref=p['control005_blend'];gain=fv.rmse(y,ref)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261130).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));assert abs(gain-r['comparison']['gain'])<1e-10 and abs(delta.std()-r['comparison']['bootstrap_sd'])<1e-12;np.testing.assert_allclose(fg,r['comparison']['fold_gains'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.percentile(delta,[2.5,97.5]),r['comparison']['conditional_ci95'],atol=1e-12,rtol=0)
 if split=='original':assert sel['advance']==bool(gain>delta.std() and sum(v>0 for v in fg)>=3)
 results[split]={'scores':{n:v['rmse'] for n,v in r['scores'].items()},'comparison':r['comparison'],'chosen_lambdas':{a:[f['models'][a]['chosen_lambda'] for f in r['folds']] for a in arms},'mean_train_rmse':{a:float(np.mean([f['models'][a]['train_rmse'] for f in r['folds']])) for a in arms},'oof_sha256':fv.sha(d/'development_oof.csv'),'seconds':r['seconds']};data[split]=(fid,p)
assert count==(35 if sel['advance'] else 15)
secondary={}
if len(results)==3 and all(r['comparison']['gain']>0 for r in results.values()):
 checked043=json.loads((_repo_path(ROOT, 'reports/dino-043-validation.json')).read_text());assert checked043['models_reloaded']==15
 for split,(fid,p) in data.items():
  r43=json.loads((_repo_path(ROOT, 'artifacts/dino-lastblock-043', split, 'report.json')).read_text())
  with (_repo_path(ROOT, 'artifacts/dino-lastblock-043', split, 'development_oof.csv')).open() as f:rows=list(csv.DictReader(f))
  assert [q['ID'] for q in rows]==ids;np.testing.assert_array_equal(y,[float(q['hardness']) for q in rows]);np.testing.assert_array_equal(fid,[int(q['fold']) for q in rows]);np.testing.assert_allclose(p['reference003'],[float(q['reference003']) for q in rows],atol=1e-12,rtol=0);adapted=np.array([float(q['adapted_blend']) for q in rows])
  for k,f in enumerate(r43['folds']):np.testing.assert_allclose([float(rows[j]['adapted']) for j in np.flatnonzero(fid==k)],f['predictions'],atol=1e-10,rtol=0)
  assert abs(fv.rmse(y,adapted)-checked043['results'][split]['blend_rmse'])<1e-10
  combo=.5*adapted+.5*p['multi3_blend'];assert np.isfinite(combo).all();combo_rmse=fv.rmse(y,combo);ix=np.random.default_rng(20261201).integers(0,400,(3000,400));ci={}
  for name,ref in {'005':p['control005_blend'],'043':adapted,'044':p['multi3_blend']}.items():
   delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-combo)[ix]**2,axis=1));ci[name]={'gain':fv.rmse(y,ref)-combo_rmse,'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist(),'fold_gains':[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],combo[fid==k]) for k in range(5)]}
  secondary[split]={'formula':'.5*043blend+.5*044blend,predeclared,no fitting','combo_rmse':combo_rmse,'adapted043_rmse':fv.rmse(y,adapted),'comparison':ci,'blend_residual_correlation043_044':float(np.corrcoef(y-adapted,y-p['multi3_blend'])[0,1]),'prediction_changes_correlation':float(np.corrcoef(adapted-p['control005_blend'],p['multi3_blend']-p['control005_blend'])[0,1])}
for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
summary={'results':results,'secondary_fixed_combination':secondary,'models_reloaded':count,'regression_reload_max_error':maxerror,'cpu_feature_max_error':cpuerror,'finalblock005_exact':True,'fold_preprocessing_groups_inner_splits_ids_metrics_gate_verified':True,'best005_and043_protected':True,'extraction_seconds':ex['seconds'],'regression_seconds':sum(r['seconds'] for r in results.values()),'scope':'Only development400; no holdout/test, new packages, cost, submissionCSV or upload.','limits':['Historical repeated selection on400andthe samepartitions remains optimistic.','Proportional Ridge alpha controls exactduplication but not effective rank/correlation.','Secondary043combination is predeclared but evaluated after main screen; not an independent validation or guaranteed Public improvement.']}
path=_repo_path(ROOT, 'reports/dino-044-validation.json')
if path.exists():assert json.loads(path.read_text())==summary
else:fv.write(path,summary)
print(json.dumps(summary,indent=2))
