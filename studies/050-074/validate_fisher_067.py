"""Independent Fisher formula, dictionary provenance, fold fits and OOF audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import fisher_experiment_067 as e
import numpy as np,torch,joblib
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
fv=e.fv

def independent_encode(dic,P):
 Z=(np.asarray(P,dtype=float)-dic.pca.mean_)@dic.pca.components_.T;w=dic.gmm.weights_;mu=dic.gmm.means_;var=dic.gmm.covariances_;logp=-.5*np.sum(np.log(2*np.pi*var)[None]+(Z[:,None]-mu[None])**2/var[None],axis=2)+np.log(w)[None];Q=np.exp(logp-logsumexp(logp,axis=1)[:,None]);q0=Q.sum(0);q1=Q.T@Z;q2=Q.T@(Z*Z);U=(q1-q0[:,None]*mu)/(len(Z)*np.sqrt(w)[:,None]*np.sqrt(var));V=(q2-2*mu*q1+q0[:,None]*(mu*mu-var))/(len(Z)*np.sqrt(2*w)[:,None]*var);raw=np.r_[U.ravel(),V.ravel()];power=np.sign(raw)*np.sqrt(abs(raw));return power/max(np.linalg.norm(power),1e-12)

def main():
 plan=e.check();torch.set_num_threads(2);start=time.perf_counter();ids,_=e.ds.base.scope();ex=json.loads((e.OUT/'extraction.json').read_text());assert ex['ids']==ids;T=np.load(_repo_io(e.OUT/'development_tokens.npy'),mmap_mode='r');assert T.shape==(400,1296,384) and T.dtype==np.float32;assert e.streamsha(e.OUT/'development_tokens.npy')==ex['tokens_sha256'];assert set(ex['image_sha256'])==set(ids)
 for ident,digest in ex['image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==digest
 dm=e.ds.base.model().to('mps');max_token=0.
 for j,ident in enumerate(ids):
  actual,_=e.tokens(dm,ident,'mps');np.testing.assert_array_equal(actual,T[j]);max_token=max(max_token,float(abs(actual-T[j]).max()))
  if (j+1)%100==0:print(json.dumps({'tokens_replayed':j+1,'seconds':time.perf_counter()-start}),flush=True)
 del dm;torch.mps.empty_cache();fixture=e.fixtures.run();results={};models=0;formula_max=0.;reload_max=0.;replay=[]
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not out.exists():continue
  ids,y,g,fid,ref,T,ix=e.inputs(split);r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  assert d['ids'].tolist()==ids;np.testing.assert_array_equal(d['y'],y);np.testing.assert_array_equal(d['fold'],fid);np.testing.assert_array_equal(d['reference008'],ref);p=np.full(400,np.nan);fold_audits=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);fd=out/f'fold-{k}';rec=json.loads((fd/'report.json').read_text());assert rec==r['folds'][k];path=fd/'model.joblib';assert fv.sha(path)==rec['model_sha256'];a=joblib.load(_repo_io(path));dic=a['dictionary'];head=a['head'];train_ids=[ids[j] for j in tr];assert a['train_ids']==dic.train_ids==rec['dictionary_train_ids']==train_ids and a['validation_ids']==[ids[j] for j in va];assert not set(dic.train_ids)&set(a['validation_ids']);np.testing.assert_array_equal(a['train_patch_indices'],ix[tr]);samples=e.sample(T,ix,tr);assert samples.shape==(20480,384) and dic.sample_sha256==hashlib.sha256(samples.tobytes()).hexdigest()==rec['sample_sha256'];np.testing.assert_allclose(dic.pca.mean_,samples.mean(0),rtol=0,atol=1e-12);assert dic.pca.n_samples_==20480 and dic.pca.n_components_==32 and dic.pca.whiten is False;assert dic.gmm.covariance_type=='diag' and dic.gmm.n_components==8 and dic.gmm.converged_ and dic.gmm.random_state==e.f.SEED and dic.gmm.n_init==1 and dic.gmm.max_iter==200;assert dic.gmm.covariances_.min()>0 and abs(dic.gmm.weights_.sum()-1)<1e-12
   assert fv.sha(fd/'features.npz')==rec['features_sha256']
   with np.load(_repo_io(fd/'features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;A=z['X'].copy()
   assert A.shape==(400,512) and np.isfinite(A).all();np.testing.assert_allclose(np.linalg.norm(A,axis=1),1.,rtol=0,atol=1e-12)
   # Recompute everyencoding via separate posterior and sufficient-statistic algebra.
   independent=np.stack([independent_encode(dic,T[j]) for j in range(400)]);delta=float(abs(A-independent).max());formula_max=max(formula_max,delta);np.testing.assert_allclose(A,independent,atol=1e-9,rtol=1e-8)
   sc=StandardScaler().fit(A[tr]);np.testing.assert_array_equal(head.scaler.mean_,sc.mean_);np.testing.assert_array_equal(head.scaler.scale_,sc.scale_);assert head.scaler.n_samples_seen_==320 and head.ym==float(y[tr].mean()) and head.ys==float(y[tr].std()) and head.ridge.alpha==512.;rp=head.predict(A[va]);np.testing.assert_array_equal(rp,d['fisher512'][va]);p[va]=rp;reload_max=max(reload_max,float(abs(rp-d['fisher512'][va]).max()));rr=Ridge(alpha=512.).fit(sc.transform(A[tr]),(y[tr]-head.ym)/head.ys);np.testing.assert_array_equal(rr.coef_,head.ridge.coef_);np.testing.assert_array_equal(rr.predict(sc.transform(A[va]))*head.ys+head.ym,rp);assert rec['train_rmse']==fv.rmse(y[tr],head.predict(A[tr])) and rec['validation_rmse']==fv.rmse(y[va],rp)
   if k==0:
    repeat=e.f.FisherDictionary().fit(samples,train_ids)
    for key in ('mean_','components_','explained_variance_'):np.testing.assert_array_equal(getattr(repeat.pca,key),getattr(dic.pca,key))
    for key in ('weights_','means_','covariances_'):np.testing.assert_array_equal(getattr(repeat.gmm,key),getattr(dic.gmm,key))
    subset=np.array([va[0],va[-1],tr[0]]);encoded=e.encode(repeat,T,subset);np.testing.assert_array_equal(encoded,A[subset]);replay.append({'split':split,'fold':0,'PCA_GMM_full_train_refit_exact':True,'encoding_error':0.})
   models+=1;fold_audits.append({'fold':k,'train_images':320,'dictionary_patches':20480,'validation_images':80,'GMM_iterations':int(dic.gmm.n_iter_),'independent_Fisher_max_error':delta,'model_reload_error':0.})
  new=.9*ref+.1*p;np.testing.assert_array_equal(new,d['primary'])
  for key,score in r['scores'].items():assert score=={'rmse':fv.rmse(y,d[key]),'fold_rmse':[fv.rmse(y[fid==k],d[key][fid==k]) for k in range(5)]}
  fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];boot_ix=np.random.default_rng(20270104).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref)[boot_ix]**2,1))-np.sqrt(np.mean((y-new)[boot_ix]**2,1));comp={'gain':fv.rmse(y,ref)-fv.rmse(y,new),'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist()};assert comp==r['comparison'];results[split]={'scores':r['scores'],'comparison':comp,'fold_audits':fold_audits,'residual_correlation':r['fisher008_residual_correlation'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']}
 sel=json.loads((e.OUT/'selection.json').read_text());assert sel['advance']==(results['original']['comparison']['gain']>0)
 if not sel['advance']:assert set(results)=={'original'}
 e.check();v={'passed':True,'results':results,'models_verified':models,'max_reload_error':reload_max,'independent_Fisher_formula_max_error':formula_max,'tokens_reextracted':400,'token_max_error':max_token,'dictionary_full_refits':replay,'fixtures':fixture,'eligible_for_candidate':len(results)==3 and all(a['comparison']['gain']>0 for a in results.values()),'best008_unchanged':True,'best008_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-008.csv')),'validation_source_sha256':fv.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_access':False,'cost':0,'new_packages':0,'limits':plan['bias']};name='fisher-067-validation.json' if not sel['advance'] or len(results)==3 else 'fisher-067-screen-validation.json';fv.write(_repo_path(ROOT, 'reports', name),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
