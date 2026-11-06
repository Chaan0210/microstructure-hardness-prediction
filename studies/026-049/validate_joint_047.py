"""Independent reload and numerical audit of bounded jointkernel study047."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joint_experiment_047 as e
import numpy as np,joblib
from scipy.spatial.distance import pdist
fv=e.fv;out=e.OUT;plan=json.loads((out/'plan.json').read_text());sel=json.loads((out/'selection.json').read_text());required=('original','20261031','20261101') if sel['advance'] else ('original',);assert all((out/s/'report.json').exists() for s in required)
for n,h in {**plan['source_sha256'],**plan['input_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
ids,D,M,y,groups,blocks=e.inputs();count=0;error=0.;batcherror=0.;mineig=1.;results={}
for split in required:
 d=out/split;r=json.loads((d/'report.json').read_text());assert len(r['folds'])==5 and [a['fold'] for a in r['folds']]==list(range(5));source=_repo_path(ROOT, 'artifacts/dino-multilayer-044', split, 'development_oof.csv');assert fv.sha(source)==r['reference_oof_sha256']
 with source.open() as f:old=list(csv.DictReader(f))
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 with (_repo_path(ROOT, 'artifacts/dino-frozen-037/regression', split, 'development_oof.csv')).open() as f:prior=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids==[a['ID'] for a in old]==[a['ID'] for a in prior];np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows]);fid=np.array([int(a['fold']) for a in rows]);np.testing.assert_array_equal(fid,[int(a['fold']) for a in old]);p={n:np.array([float(a[n]) for a in rows]) for n in r['scores']}
 np.testing.assert_allclose(p['reference003'],[float(a['fresh003']) for a in prior],atol=1e-10,rtol=0);np.testing.assert_allclose(p['control006'],[float(a['multi3_blend']) for a in old],atol=1e-10,rtol=0)
 np.testing.assert_allclose(p['primary'],.75*p['reference003']-.1875*p['old_kernel']+.4375*p['joint'],atol=1e-12,rtol=0);np.testing.assert_allclose(p['diagnostic_dino_replace'],.75*p['reference003']+.25*p['joint'],atol=1e-12,rtol=0)
 bins=np.full(400,-1)
 for k,f in enumerate(r['folds']):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);assert f['train_ids']==[ids[j] for j in tr] and f['validation_ids']==[ids[j] for j in va];held=[]
  for it,iv in f['inner_splits']:assert not set(it)&set(iv) and set(it)|set(iv)==set(tr) and not set(groups[it])&set(groups[iv]);held.extend(iv)
  assert sorted(held)==sorted(tr);bins[va]=np.digitize(y[va],np.quantile(y[tr],[.2,.8]));assert f['joint_selection']==min(f['joint_inner_scores'],key=lambda q:q['rmse']) and f['control_selection']==min(f['control_inner_scores'],key=lambda q:q['rmse'])
  for kind in ('control','joint'):
   path=d/f'fold-{k}'/f'{kind}.joblib';assert fv.sha(path)==f['model_sha256'][kind];obj=joblib.load(_repo_io(path));m=obj['model'];assert obj['train_ids']==f['train_ids'] and obj['validation_ids']==f['validation_ids']
   if kind=='control':
    np.testing.assert_allclose(m.scaler.mean_,D[tr].mean(0),atol=1e-10,rtol=0);assert m.scaler.n_samples_seen_==320 and m.alpha==f['control_selection']['alpha']*3456;prediction=m.predict(D[va]);target=p['control_dino'][va]
   else:
    assert m.n_train==320 and m.kind==f['joint_selection']['kind'] and m.alpha==f['joint_selection']['alpha'];np.testing.assert_allclose(m.dscale.mean_,D[tr].mean(0),atol=1e-10,rtol=0);np.testing.assert_array_equal(m.mbound.data_min_,M[tr].min(0));np.testing.assert_array_equal(m.mbound.data_max_,M[tr].max(0));np.testing.assert_allclose(m.mscale.mean_,m.mbound.transform(M[tr]).mean(0),atol=1e-10,rtol=0);np.testing.assert_allclose(m.M,m.mscale.transform(m.mbound.transform(M[tr]))*m.weights,atol=1e-12,rtol=0);dist=pdist(m.M,'sqeuclidean');assert abs(m.gamma-1/np.median(dist[dist>1e-12]))<1e-12
    K=m.kernel(D[tr],M[tr]);np.testing.assert_allclose(K,m.kernel_train,atol=1e-10,rtol=0);np.testing.assert_allclose(K,K.T,atol=1e-12,rtol=0);ev=float(np.linalg.eigvalsh(K).min());mineig=min(mineig,ev);assert ev>-1e-9 and abs(np.trace(K)/320-1)<1e-12
    prediction=m.predict(D[va],M[va]);target=p['joint'][va];single=m.predict(D[va[:1]],M[va[:1]]);batched=m.predict(np.concatenate([D[va[:1]],D[tr[:1]]*100]),np.concatenate([M[va[:1]],M[tr[:1]]*100]));np.testing.assert_allclose(single,batched[:1],atol=1e-10,rtol=0);batcherror=max(batcherror,float(abs(single-batched[:1]).max()));assert abs(fv.rmse(y[tr],m.predict(D[tr],M[tr]))-f['joint_train_rmse'])<1e-10
   assert abs(m.ymean-y[tr].mean())<1e-10 and abs(m.ystd-y[tr].std())<1e-10;np.testing.assert_allclose(prediction,target,atol=1e-10,rtol=0);error=max(error,float(abs(prediction-target).max()));count+=1
  kp=_repo_path(ROOT, 'artifacts/dino-frozen-037/regression', split, f'fold-{k}', 'compact-kernel.joblib');assert fv.sha(kp)==f['source_morph_kernel_sha256'];b=joblib.load(_repo_io(kp));assert b['train_ids']==f['train_ids'] and b['validation_ids']==f['validation_ids'];km=b['model'];np.testing.assert_array_equal(km.bound_.data_min_,M[tr].min(0));np.testing.assert_array_equal(km.bound_.data_max_,M[tr].max(0));assert abs(km.ym_-y[tr].mean())<1e-10 and abs(km.ys_-y[tr].std())<1e-10;np.testing.assert_allclose(km.predict(M[va]),p['old_kernel'][va],atol=1e-10,rtol=0)
 for n,q in p.items():
  assert np.isfinite(q).all() and abs(fv.rmse(y,q)-r['scores'][n]['rmse'])<1e-10;np.testing.assert_allclose([fv.rmse(y[fid==k],q[fid==k]) for k in range(5)],r['scores'][n]['fold_rmse'],atol=1e-10,rtol=0)
 new=p['primary'];ref=p['control006'];gain=fv.rmse(y,ref)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(2026100201).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));assert abs(gain-r['comparison']['gain'])<1e-12 and abs(delta.std()-r['comparison']['bootstrap_sd'])<1e-12;np.testing.assert_allclose(fg,r['comparison']['fold_gains'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.percentile(delta,[2.5,97.5]),r['comparison']['conditional_ci95'],atol=1e-12,rtol=0)
 if split=='original':assert sel['advance']==bool(gain>delta.std() and sum(q>0 for q in fg)>=3)
 tails={name:{'rows':int((bins==b).sum()),'006_rmse':fv.rmse(y[bins==b],ref[bins==b]),'joint_blend_rmse':fv.rmse(y[bins==b],new[bins==b])} for b,name in enumerate(('low20','middle60','high20'))}
 results[split]={'scores':{n:q['rmse'] for n,q in r['scores'].items()},'comparison':r['comparison'],'selected_models':[f['joint_selection'] for f in r['folds']],'mean_train_rmse':float(np.mean([f['joint_train_rmse'] for f in r['folds']])),'fold_training_target_range_diagnostic':tails,'seconds':r['seconds'],'rss_peak_bytes':r['rss_peak_bytes'],'oof_sha256':fv.sha(d/'development_oof.csv')}
assert count==10*len(required)
for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
summary={'results':results,'models_reloaded':count,'reload_prediction_max_error':error,'query_batch_independence_error':batcherror,'minimum_train_kernel_eigenvalue':mineig,'mean_train_kernel_diagonal':1,'samefold_reference003_and006_reproduced':True,'train_only_bounds_scalers_bandwidth_centers_target_and_inner_groups_verified':True,'best006_protected':True,'screen_advanced':sel['advance'],'scope':'Development400 only; no test/holdoutmodelselection, no costs/packages/newcandidate/upload.','limits':'Historically reused400 and folds; current nested selection does not remove historical selection bias. Negative result applies to this fixed jointfamily,not a performance ceiling.'}
p=_repo_path(ROOT, 'reports/joint-047-validation.json')
if p.exists():assert json.loads(p.read_text())==summary
else:fv.write(p,summary)
print(json.dumps(summary,indent=2))
