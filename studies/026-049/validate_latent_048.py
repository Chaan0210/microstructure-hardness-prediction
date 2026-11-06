"""Separate-process validation of fixed supervised4D readout."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import latent_experiment_048 as e
import numpy as np,joblib
from scipy.spatial.distance import pdist
fv=e.fv;out=e.OUT;plan=json.loads((out/'plan.json').read_text());sel=json.loads((out/'selection.json').read_text());required=('original','20261031','20261101') if sel['advance'] else ('original',);assert all((out/s/'report.json').exists() for s in required)
for n,h in {**plan['source_sha256'],**plan['input_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
ids,D,_,y,groups,_=e.inp.inputs();results={};count=0;maxerror=0.;mineig=1.;batcherror=0.
for split in required:
 d=out/split;r=json.loads((d/'report.json').read_text());assert len(r['folds'])==5 and [f['fold'] for f in r['folds']]==list(range(5));src=_repo_path(ROOT, 'artifacts/dino-multilayer-044', split, 'development_oof.csv');assert fv.sha(src)==r['reference_oof_sha256']
 with src.open() as f:old=list(csv.DictReader(f))
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids==[a['ID'] for a in old];np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows]);fid=np.array([int(a['fold']) for a in rows]);np.testing.assert_array_equal(fid,[int(a['fold']) for a in old]);p={n:np.array([float(a[n]) for a in rows]) for n in r['scores']};np.testing.assert_allclose(p['control_blend'],[float(a['multi3_blend']) for a in old],atol=1e-10,rtol=0);np.testing.assert_array_equal(p['reference003'],[float(a['reference003']) for a in old])
 for n in ('control','latent','linear_pls_diagnostic'):np.testing.assert_allclose(p[n+'_blend'],.75*p['reference003']+.25*p[n],atol=1e-12,rtol=0)
 for k,f in enumerate(r['folds']):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);assert f['train_ids']==[ids[j] for j in tr] and f['validation_ids']==[ids[j] for j in va];held=[]
  for it,iv in f['inner_splits']:assert not set(it)&set(iv) and set(it)|set(iv)==set(tr) and not set(groups[it])&set(groups[iv]);held.extend(iv)
  assert sorted(held)==sorted(tr);assert f['selected']==min(f['inner_scores'],key=lambda a:a['rmse']) and f['control_selected']==min(f['control_inner_scores'],key=lambda a:a['rmse'])
  for name in ('control','latent'):
   path=d/f'fold-{k}'/f'{name}.joblib';assert fv.sha(path)==f['model_sha256'][name];obj=joblib.load(_repo_io(path));m=obj['model'];assert obj['train_ids']==f['train_ids'] and obj['validation_ids']==f['validation_ids'];scaler=m.scaler if name=='control' else m.scale;np.testing.assert_allclose(scaler.mean_,D[tr].mean(0),atol=1e-10,rtol=0);assert scaler.n_samples_seen_==320 and abs(m.ymean-y[tr].mean())<1e-10 and abs(m.ystd-y[tr].std())<1e-10
   if name=='control':assert m.alpha==f['control_selected']['alpha']*3456
   else:
    assert m.alpha==f['selected']['alpha'] and m.pls.n_components==4 and not m.pls.scale and m.n_train==320;A=m.scale.transform(D[tr]);np.testing.assert_allclose(m.pls._x_mean,A.mean(0),atol=1e-10,rtol=0);np.testing.assert_allclose(m.pls._y_mean,((y[tr]-m.ymean)/m.ystd).mean(),atol=1e-10,rtol=0);np.testing.assert_allclose(m.pls.transform(A),m.Z,atol=1e-10,rtol=0);np.testing.assert_allclose(m.pls.x_scores_,m.Z,atol=1e-8,rtol=0)
    dist=pdist(m.Z,'sqeuclidean');assert abs(m.gamma-1/np.median(dist[dist>1e-12]))<1e-12;K=m.kernel_train;np.testing.assert_allclose(K,K.T,atol=1e-12,rtol=0);ev=float(np.linalg.eigvalsh(K).min());assert ev>-1e-9 and abs(np.trace(K)/320-1)<1e-12;mineig=min(mineig,ev);one=m.predict(D[va[:1]]);mixed=m.predict(np.concatenate([D[va[:1]],D[tr[:1]]*100]));np.testing.assert_allclose(one,mixed[:1],atol=1e-10,rtol=0);batcherror=max(batcherror,float(abs(one-mixed[:1]).max()));np.testing.assert_allclose(m.linear_predict(D[va]),p['linear_pls_diagnostic'][va],atol=1e-10,rtol=0);assert abs(fv.rmse(y[tr],m.predict(D[tr]))-f['latent_train_rmse'])<1e-10
   prediction=m.predict(D[va]);np.testing.assert_allclose(prediction,p[name][va],atol=1e-10,rtol=0);maxerror=max(maxerror,float(abs(prediction-p[name][va]).max()));count+=1
 for n,q in p.items():
  assert np.isfinite(q).all() and abs(fv.rmse(y,q)-r['scores'][n]['rmse'])<1e-10;np.testing.assert_allclose([fv.rmse(y[fid==k],q[fid==k]) for k in range(5)],r['scores'][n]['fold_rmse'],atol=1e-10,rtol=0)
 new=p['latent_blend'];ref=p['control_blend'];gain=fv.rmse(y,ref)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(2026100202).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));assert abs(gain-r['comparison']['gain'])<1e-12 and abs(delta.std()-r['comparison']['bootstrap_sd'])<1e-12;np.testing.assert_allclose(fg,r['comparison']['fold_gains'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.percentile(delta,[2.5,97.5]),r['comparison']['conditional_ci95'],atol=1e-12,rtol=0)
 if split=='original':assert sel['advance']==bool(gain>delta.std() and sum(q>0 for q in fg)>=3)
 results[split]={'scores':{n:q['rmse'] for n,q in r['scores'].items()},'comparison':r['comparison'],'selected_alpha':[f['selected']['alpha'] for f in r['folds']],'mean_train_rmse':float(np.mean([f['latent_train_rmse'] for f in r['folds']])),'mean_linear_pls_train_rmse':float(np.mean([f['linear_pls_train_rmse'] for f in r['folds']])),'pls_n_iter':[f['pls_n_iter'] for f in r['folds']],'seconds':r['seconds'],'rss_peak_bytes':r['rss_peak_bytes'],'oof_sha256':fv.sha(d/'development_oof.csv')}
assert count==10*len(required)
for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
summary={'results':results,'models_reloaded':count,'max_reload_error':maxerror,'batch_independence_error':batcherror,'minimum_kernel_eigenvalue':mineig,'same006_control_reproduced':True,'fold_only_scaler_target_supervised_PLStraining_scores_bandwidth_inner_groups_verified':True,'best006_protected':True,'screen_advanced':sel['advance'],'scope':'Development400only;noholdout/testfit,externaldata,installation,cost,candidateCSVorupload','limits':'CurrentnestedCV controlsalpha,not historicaldevelopment/modelselection. Fixed4Dfailureisnot evidenceofabsoluteperformanceceiling.'}
p=_repo_path(ROOT, 'reports/latent-048-validation.json')
if p.exists():assert json.loads(p.read_text())==summary
else:fv.write(p,summary)
print(json.dumps(summary,indent=2))
