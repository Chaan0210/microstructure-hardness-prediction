"""Independent scope reconstruction, all train-local statistics and eigensolve."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pooled_residual_119 as p
import validate_nonlinear_residual_115 as v
import numpy as np,joblib
e=p.e;r=p.r;n=p.n

def validate(split):
 p.check();d=r.load(split);out=p.OUT/split;report=e.read(out/'report.json');assert e.sha(out/'predictions.npz')==report['predictions_sha256'];start=time.perf_counter();small=np.full((3,400),np.nan);pooled=np.full(400,np.nan);maxerr=0.;count=0
 for k in range(5):
  path=out/f'fold-{k}';rec=e.read(path/'report.json');assert e.sha(path/'report.json')==report['fold_reports_sha256'][f'fold-{k}/report.json'];assert e.sha(path/'predictions.npz')==rec['predictions_sha256'];T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);mapping={};records=[]
  for j in range(3):
   A,B,pa,trace=r.anchor(d,split,k,j);assert trace==rec['block_anchor_traces'][j];assert not set(A)&set(B) and set(A)|set(B)==set(T) and not (set(A)|set(B))&set(V);assert not set(d['g'][T])&set(d['g'][V]);records.append((B,d['y'][B]-pa))
   for i,q in zip(B,pa):assert int(i) not in mapping and int(i) not in A;mapping[int(i)]=float(q)
  assert set(mapping)==set(T);pa=np.array([mapping[int(i)] for i in T]);res=d['y'][T]-pa
  with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:save={x:z[x].copy() for x in z.files}
  np.testing.assert_array_equal(save['train_indices'],T);np.testing.assert_array_equal(save['validation_indices'],V);np.testing.assert_array_equal(save['anchor_predictions'],pa);np.testing.assert_array_equal(save['residual'],res)
  for name,ix,rr,expected in [(f'block-{j}',B,rr,save['block_corrections'][j]) for j,(B,rr) in enumerate(records)]+[('pooled',T,res,save['pooled_correction'])]:
   file=path/f'{name}.joblib';assert e.sha(file)==rec['models_sha256'][file.name];s=joblib.load(_repo_io(file));np.testing.assert_array_equal(s['train_indices'],ix);np.testing.assert_array_equal(s['validation_indices'],V);m=s['model'];assert m['kind']=='rbf1' and m['n_train']==len(ix) and m['alpha']==len(ix) and m['effective_df']<=1+1e-8
   manual=v.manual(m,d['X'][ix],rr,d['X'][V],d['blocks']);reload=n.predict(m,d['X'][V]);np.testing.assert_array_equal(reload,expected);err=float(abs(manual-expected).max());assert err<2e-9;maxerr=max(maxerr,err);count+=1
   if name=='pooled':pooled[V]=expected
   else:small[int(name[-1]),V]=expected
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:
  assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['block_corrections'],small);np.testing.assert_array_equal(z['pooled_correction'],pooled);np.testing.assert_array_equal(z['block_mean'],d['reference011']+small.mean(0));np.testing.assert_array_equal(z['pooled'],d['reference011']+pooled);np.testing.assert_array_equal(z['block_individual'],d['reference011'][None,:]+small)
  for key in report['scores']:assert float(np.mean((d['y']-z[key])**2))==report['scores'][key]['MSE']
 p.check();result={'passed':True,'split':split,'models_reloaded':count,'independent_rawstatistics_eigensolve_max_error':maxerr,'outer80_excluded_from_all_targets_and_fits':True,'every320target_crossfit_ownlabel_excluded':True,'no_inner_hyperparameter_selection':True,'report_sha256':e.sha(out/'report.json'),'seconds':time.perf_counter()-start};e.write(p.OUT/f'validation-{split}.json',result);e.emit(result)

def leakage():
 p.check();out=p.OUT/'leakage-replay';out.mkdir(exist_ok=False);count=0
 for split in e.SPLITS:
  d=r.load(split);alter=dict(d);alter['y']=d['y'].copy();V=np.flatnonzero(d['fid']==0);alter['y'][V]=10000+np.arange(len(V));small,pooled,rec=p.fit_fold(alter,split,0,out/split);old=p.OUT/split/'fold-0'
  with np.load(_repo_io(old/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(small,z['block_corrections']);np.testing.assert_array_equal(pooled,z['pooled_correction'])
  for name in ['block-0','block-1','block-2','pooled']:
   m=joblib.load(_repo_io(out/split/f'{name}.joblib'))['model'];om=joblib.load(_repo_io(old/f'{name}.joblib'))['model'];np.testing.assert_array_equal(m['coef'],om['coef']);np.testing.assert_array_equal(m['scale'].mean_,om['scale'].mean_);assert m['gamma']==om['gamma'] and m['residual_scale']==om['residual_scale'];count+=1
 result={'passed':True,'three_outerfolds_labels_perturbed':True,'models_and_predictions_exact':count,'upstream112_exclusion_retraining_certificate':e.sha(e.OUT/'validation-replay.json'),'new_upstream_retraining':False};e.write(p.OUT/'leakage-validation.json',result);e.emit(result)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=[*e.SPLITS,'leakage']);x=ap.parse_args();leakage() if x.mode=='leakage' else validate(x.mode)
