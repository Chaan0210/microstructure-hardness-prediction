"""Completed-study reload/invariance checks; no fitting or model selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dino_d4_046 as e
import numpy as np,torch,joblib
from dino_regression_037 import FrozenRegressor
fv=e.fv;out=e.OUT;plan=json.loads((out/'plan.json').read_text());ex=json.loads((out/'extraction.json').read_text());sel=json.loads((out/'selection.json').read_text());required=('original','20261031','20261101') if sel['advance'] else ('original',);assert all((out/s/'report.json').exists() for s in required)
for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
assert fv.sha(out/'development_features.npz')==ex['feature_sha256'];ids,y,groups=e.multi.load()
with np.load(_repo_io(out/'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;V=z['views'].astype(np.float64)
assert V.shape==(400,8,3456) and np.isfinite(V).all() and len(set(ids))==400
with np.load(_repo_io(e.multi.OUT/'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(V[:,0],z['X'].reshape(400,3456))
assert set(ex['image_sha256'])==set(ids)
for i,h in ex['image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png'))==h
# IndependentCPU whole8-view firstimage recomputation; no labels choose image.
torch.set_num_threads(2);m=e.multi.ds.base.model();raw=e.raw(ids[0]);cpu=np.stack([e.multi.feature(m,torch.from_numpy(e.image_input(e.transform(raw,*v)))).numpy().reshape(3456) for v in e.VIEWS]).astype(np.float64);np.testing.assert_allclose(cpu,V[0],atol=2e-3,rtol=2e-3);cpuerr=float(abs(cpu-V[0]).max());del m
# Every reorientation maps the same pixelorbit to a permutation; exactperimage aggregation invariance.
pixels=[e.transform(raw,*v).tobytes() for v in e.VIEWS];invariance=0.
for view in e.VIEWS:
 transformed=e.transform(raw,*view);mapping=[pixels.index(e.transform(transformed,*v).tobytes()) for v in e.VIEWS];assert sorted(mapping)==list(range(8));error=float(abs(cpu[mapping].mean(0)-cpu.mean(0)).max());invariance=max(invariance,error);assert error<1e-12
panels={'control006':V[:,0],'d4mean':V.mean(1)};results={};count=0;maxerr=0.;ttaerr=0.
checked=json.loads((_repo_path(ROOT, 'reports/dino-044-validation.json')).read_text())
for split in required:
 d=out/split;r=json.loads((d/'report.json').read_text());assert len(r['folds'])==5 and [f['fold'] for f in r['folds']]==list(range(5));src=e.multi.OUT/split/'development_oof.csv';assert fv.sha(src)==r['reference_oof_sha256']==checked['results'][split]['oof_sha256'];assert fv.sha(d/'orientation_diagnostic.npz')==r['orientation_sha256']
 with src.open() as f:old=list(csv.DictReader(f))
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids==[a['ID'] for a in old];np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows]);fid=np.array([int(a['fold']) for a in rows]);np.testing.assert_array_equal(fid,[int(a['fold']) for a in old]);p={n:np.array([float(a[n]) for a in rows]) for n in r['scores']};np.testing.assert_array_equal(p['reference003'],[float(a['reference003']) for a in old]);np.testing.assert_allclose(p['control006_blend'],[float(a['multi3_blend']) for a in old],atol=1e-10,rtol=0)
 with np.load(_repo_io(d/'orientation_diagnostic.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;vp=z['control_view_predictions'];bins=z['target_bins']
 assert vp.shape==(400,8) and np.isfinite(vp).all()
 for a in (*panels,'tta_only'):np.testing.assert_allclose(p[a+'_blend'],.75*p['reference003']+.25*p[a],atol=1e-12,rtol=0)
 for n,a in p.items():
  assert np.isfinite(a).all() and abs(fv.rmse(y,a)-r['scores'][n]['rmse'])<1e-10
  np.testing.assert_allclose([fv.rmse(y[fid==k],a[fid==k]) for k in range(5)],r['scores'][n]['fold_rmse'],atol=1e-10,rtol=0)
 for k,f in enumerate(r['folds']):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);assert f['train_ids']==[ids[j] for j in tr] and f['validation_ids']==[ids[j] for j in va];held=[]
  for it,iv in f['inner_splits']:assert not set(it)&set(iv) and set(it)|set(iv)==set(tr) and not set(groups[it])&set(groups[iv]);held.extend(iv)
  assert sorted(held)==sorted(tr);q=np.quantile(y[tr],[.2,.8]);np.testing.assert_array_equal(q,f['training_target_quantiles20_80']);np.testing.assert_array_equal(np.digitize(y[va],q),bins[va])
  for arm,A in panels.items():
   rec=f['models'][arm];path=d/f'fold-{k}'/f'{arm}.joblib';assert fv.sha(path)==rec['sha256'];obj=joblib.load(_repo_io(path));m=obj['model'];assert obj['train_ids']==f['train_ids'] and obj['validation_ids']==f['validation_ids'];np.testing.assert_allclose(m.scaler.mean_,A[tr].mean(0),atol=1e-10,rtol=0);assert m.scaler.n_samples_seen_==320 and abs(m.ymean-y[tr].mean())<1e-10 and abs(m.ystd-y[tr].std())<1e-10
   chosen=min(rec['inner_scores'],key=lambda s:s['rmse'])['lambda'];assert m.alpha==chosen*3456 and rec['chosen_lambda']==chosen;prediction=m.predict(A[va]);np.testing.assert_allclose(prediction,p[arm][va],atol=1e-10,rtol=0);maxerr=max(maxerr,float(abs(prediction-p[arm][va]).max()));assert abs(fv.rmse(y[tr],m.predict(A[tr]))-rec['train_rmse'])<1e-10;count+=1
   if arm=='control006':
    viewp=m.predict(V[va].reshape(-1,3456)).reshape(len(va),8);np.testing.assert_allclose(viewp,vp[va],atol=1e-10,rtol=0);np.testing.assert_allclose(viewp[:,0],p[arm][va],atol=1e-10,rtol=0);av=m.predict(panels['d4mean'][va]);np.testing.assert_allclose(av,p['tta_only'][va],atol=1e-10,rtol=0);np.testing.assert_allclose(viewp.mean(1),av,atol=1e-10,rtol=0);ttaerr=max(ttaerr,float(abs(viewp.mean(1)-av).max()))
 new=p['d4mean_blend'];ref=p['control006_blend'];fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];gain=fv.rmse(y,ref)-fv.rmse(y,new);ix=np.random.default_rng(20261202).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));assert abs(gain-r['comparison']['gain'])<1e-10 and abs(delta.std()-r['comparison']['bootstrap_sd'])<1e-12;np.testing.assert_allclose(fg,r['comparison']['fold_gains'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.percentile(delta,[2.5,97.5]),r['comparison']['conditional_ci95'],atol=1e-12,rtol=0)
 if split=='original':assert sel['advance']==bool(gain>delta.std() and sum(a>0 for a in fg)>=3)
 diag=r['diagnostic'];std=vp.std(1);np.testing.assert_allclose(np.percentile(std,[0,25,50,75,90,100]),diag['control_DINO_view_std_quantiles'],atol=1e-12,rtol=0);np.testing.assert_allclose([fv.rmse(y,vp[:,j]) for j in range(8)],diag['control_DINO_each_view_rmse'],atol=1e-12,rtol=0)
 for b,name in enumerate(('low20','middle60','high20')):
  a=diag['fold_train_target_bins'][name];mask=bins==b;assert a['rows']==int(mask.sum()) and abs(fv.rmse(y[mask],ref[mask])-a['006_rmse'])<1e-10 and abs(fv.rmse(y[mask],new[mask])-a['d4_rmse'])<1e-10
 results[split]={'scores':{n:a['rmse'] for n,a in r['scores'].items()},'comparison':r['comparison'],'diagnostic':diag,'chosen_lambdas':{a:[f['models'][a]['chosen_lambda'] for f in r['folds']] for a in panels},'oof_sha256':fv.sha(d/'development_oof.csv'),'seconds':r['seconds']}
assert count==10*len(required)
for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
summary={'results':results,'models_reloaded':count,'max_reload_prediction_error':maxerr,'CPU8view_feature_max_error':cpuerr,'D4_orbit_mean_permutation_error':invariance,'linear_TTA_feature_vs_prediction_mean_error':ttaerr,'identity_descriptor_exact006':True,'fold_preprocessing_groups_inner_splits_metrics_gate_checked':True,'best006_protected':True,'screen_advanced':sel['advance'],'extraction_seconds':ex['seconds'],'regression_seconds':sum(a['seconds'] for a in results.values()),'limits':['D4 preserves relative imagegeometry but discardsabsoluteorientation/handedness; empiricalCVdoesnot prove labelinvariance.','Only DINO branch usesD4mean; unchanged003branchprevents claiming wholeensembleD4invariance.','Development400andpartitionsarehistoricallyreused;conditionalbootstrapomitsfittingandselectionuncertainty.','Orientationandtargetrangeerrorsareposthocdiagnostics,not causalproof or newtuning.'],'scope':'No holdout/testimagesorlabels,newpackages,cost,candidateCSV,upload. Existing006onlyhashedforpreservation.'}
p=_repo_path(ROOT, 'reports/dino-046-validation.json')
if p.exists():assert json.loads(p.read_text())==summary
else:fv.write(p,summary)
print(json.dumps(summary,indent=2))
