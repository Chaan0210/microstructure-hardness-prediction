"""Independent replay of study062 cache and nested Ridge reports; development only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_062 as e
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
fv=e.fv

def main():
 t=time.perf_counter();plan=e.check();torch.set_num_threads(2)
 ids,S,y,g,fid,r003,r007,_=e.baseline.inputs('original');allowed,split=e.ds.base.scope();assert ids==allowed
 ex=json.loads((e.OUT/'extraction.json').read_text());assert fv.sha(e.OUT/'development_features.npz')==ex['feature_sha256']
 with np.load(_repo_io(e.OUT/'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;B=z['X'].copy()
 assert B.shape==(400,6912) and B.dtype==np.float32 and np.isfinite(B).all()
 assert set(ex['image_sha256'])==set(ids)
 for ident,digest in ex['image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==digest
 m=e.model().to('mps');initial=e.baseline.h.state_hash(m.state_dict());maxdiff=0.
 for j,ident in enumerate(ids):
  x=torch.from_numpy(e.ds.inputs(ident,'full504')).to('mps')
  with torch.inference_mode():
   layers=m.get_intermediate_layers(x,n=[5,8,11],return_class_token=True,norm=True)
   parts=[]
   for patches,cls in layers:
    assert patches.shape==(1,1296,768) and cls.shape==(1,768)
    parts.extend([cls,patches.mean(dim=1),patches.std(dim=1,correction=0)])
   actual=torch.cat(parts,dim=1).cpu().numpy()[0]
  np.testing.assert_array_equal(actual,B[j]);maxdiff=max(maxdiff,float(abs(actual-B[j]).max()))
  if (j+1)%100==0:print(json.dumps({'cache_replayed':j+1,'max_error':maxdiff,'seconds':time.perf_counter()-t}),flush=True)
 assert initial==e.baseline.h.state_hash(m.state_dict()) and all(p.grad is None for p in m.parameters())
 del m;torch.mps.empty_cache()
 report=json.loads((e.OUT/'original/report.json').read_text());path=e.OUT/'original/development_predictions.npz';assert fv.sha(path)==report['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 assert data['ids'].tolist()==ids;np.testing.assert_array_equal(data['y'],y);np.testing.assert_array_equal(data['fold'],fid)
 checks=[]
 for rec in report['folds']:
  arm=rec['arm'];k=rec['fold'];X=S if arm=='controlS' else B.astype(float);tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(g[tr])&set(g[va]);p=e.OUT/'original'/f'{arm}-fold-{k}.joblib';assert fv.sha(p)==rec['model_sha256'];a=joblib.load(_repo_io(p));assert a['train_ids']==[ids[i] for i in tr] and a['validation_ids']==[ids[i] for i in va];fitted=a['model'];sc=StandardScaler().fit(X[tr]);np.testing.assert_array_equal(sc.mean_,fitted.scaler.mean_);np.testing.assert_array_equal(sc.scale_,fitted.scaler.scale_);assert fitted.ymean==float(y[tr].mean()) and fitted.ystd==float(y[tr].std()) and fitted.pca is None
  pv=fitted.predict(X[va]);np.testing.assert_array_equal(pv,data[arm][va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(X[tr],y[tr],g[tr]));assert [[tr[it].tolist(),tr[iv].tolist()] for it,iv in inner]==rec['inner_splits'];scores=[]
  for lam in e.LAMBDAS:
   ip=np.full(len(tr),np.nan)
   for it,iv in inner:
    assert not set(g[tr[it]])&set(g[tr[iv]])
    f=e.FrozenRegressor('all_ridge',lam*X.shape[1]).fit(X[tr[it]],y[tr[it]]);ip[iv]=f.predict(X[tr[iv]])
   scores.append({'lambda':lam,'rmse':fv.rmse(y[tr],ip)})
  assert scores==rec['inner_scores'];chosen=min(scores,key=lambda r:r['rmse'])['lambda'];assert chosen==rec['chosen_lambda'];replay=e.FrozenRegressor('all_ridge',chosen*X.shape[1]).fit(X[tr],y[tr]);np.testing.assert_array_equal(replay.predict(X[va]),pv);assert rec['train_rmse']==fv.rmse(y[tr],fitted.predict(X[tr]));assert rec['validation_rmse']==fv.rmse(y[va],pv);checks.append({'arm':arm,'fold':k,'reload_error':0.,'nested_replay_error':0.})
 for arm in ('controlS','baseB'):np.testing.assert_array_equal(data[arm+'_blend'],.75*r003+.25*data[arm])
 np.testing.assert_array_equal(data['reference007'],r007);np.testing.assert_array_equal(data['reference003'],r003);np.testing.assert_array_equal(data['controlS_blend'],data['reference006'])
 for key,score in report['scores'].items():assert score=={'rmse':fv.rmse(y,data[key]),'fold_rmse':[fv.rmse(y[fid==k],data[key][fid==k]) for k in range(5)]}
 c=report['comparison'];new=data['baseB_blend'];fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];assert c['gain_vs007']==fv.rmse(y,r007)-fv.rmse(y,new) and c['fold_gains_vs007']==fg and c['positive_folds_vs007']==sum(a>0 for a in fg);ix=np.random.default_rng(20270103).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));assert float(boot.std())==c['bootstrap_sd'];np.testing.assert_array_equal(np.percentile(boot,[2.5,97.5]),c['conditional_ci95']);assert json.loads((e.OUT/'selection.json').read_text())['advance']==bool(c['gain_vs007']>boot.std() and sum(a>0 for a in fg)>=3)
 e.check();result={'passed':True,'cache_rows_reextracted':400,'cache_max_error':maxdiff,'weight_sha256':plan['weights']['sha256'],'models_verified':checks,'scores':report['scores'],'comparison':c,'seconds':time.perf_counter()-t,'source_sha256':fv.sha(Path(__file__)),'scope':'Development400 only; no holdout images/labels or test predictions inspected. Existing protected artifacts hash-only checked.'};fv.write(_repo_path(ROOT, 'reports/dinob-062-validation.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('scores','models_verified')}),flush=True)
if __name__=='__main__':main()
