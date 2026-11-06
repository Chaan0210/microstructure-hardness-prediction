"""Independent union-find, level counts, Ridge algebra and leakage audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,argparse,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib
from scipy import ndimage as ndi
from sklearn.preprocessing import StandardScaler
import persistence_features_107 as f

def independent_diagram(x):
 h,w=x.shape;v=x.ravel().astype(int);parent=np.full(v.size,-1,dtype=int);birth=v.copy();pairs=[]
 def root(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 for i in np.argsort(v,kind='stable'):
  parent[i]=i;y,c=divmod(int(i),w)
  for j in ([i-w] if y else [])+([i+w] if y<h-1 else [])+([i-1] if c else [])+([i+1] if c<w-1 else []):
   if parent[j]<0:continue
   a,b=root(i),root(j)
   if a==b:continue
   if (birth[a],a)>(birth[b],b):a,b=b,a
   if v[i]>birth[b]:pairs.append((birth[b],v[i]))
   parent[b]=a
 return np.array(sorted(pairs),dtype=np.int16).reshape(-1,2)
def engineering():
 rng=np.random.default_rng(107);cases=[np.zeros((7,7),dtype=np.uint8),np.full((7,7),255,dtype=np.uint8),np.tile(np.arange(7,dtype=np.uint8),(7,1))]+[rng.integers(0,16,(7,9),dtype=np.uint8) for _ in range(40)]
 for x in cases:
  a=f.diagram(x);b=independent_diagram(x);np.testing.assert_array_equal(a,b)
  for t in np.unique(x):assert ndi.label(x<=t)[1]==int(((a[:,0]<=t)&(t<a[:,1])).sum())+int(t>=x.min())
 # Two diagrams with identical level counts but distinct birth/death pairing.
 a=np.array([[0,128],[64,192]]);b=np.array([[0,192],[64,128]]);ca,pa=f.encode(a,1);cb,pb=f.encode(b,1);np.testing.assert_array_equal(ca,cb);assert np.linalg.norm(pa-pb)>0
 return {'toy_images':len(cases),'union_find_diagrams_exact':True,'all_observed_threshold_counts_exact':True,'same_curve_different_pairing_PI_distance':float(np.linalg.norm(pa-pb))}
def validate(split):
 import persistence_study_107 as s
 start=time.perf_counter();s.check();d=s.inputs(split);r=s.read(s.OUT/split/'report.json');path=s.OUT/split/'development_predictions.npz';assert s.sha(path)==r['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 np.testing.assert_array_equal(arr['ids'],d['ids']);np.testing.assert_array_equal(arr['y'],d['y']);np.testing.assert_array_equal(arr['fold'],d['fid']);np.testing.assert_array_equal(arr['reference011'],d['ref']);reload=0.;algebra=0.;perturb=0.;replayed=0
 for arm in s.ARMS:
  X=d[arm]
  for k in range(5):
   p=s.OUT/split/f'{arm}-fold{k}.joblib';assert s.sha(p)==r['models_sha256'][p.name];a=joblib.load(_repo_io(p));tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va];m=a['model'];pred=m.predict(X[va]);np.testing.assert_array_equal(pred,arr[arm][va]);reload=max(reload,float(abs(pred-arr[arm][va]).max()))
   sc=StandardScaler().fit(X[tr]);np.testing.assert_array_equal(sc.mean_,m.scaler.mean_);np.testing.assert_array_equal(sc.scale_,m.scaler.scale_);assert m.ym==d['y'][tr].mean() and m.ys==d['y'][tr].std();A=sc.transform(X[tr]);B=sc.transform(X[va]);ac=A-A.mean(0);t=(d['y'][tr]-m.ym)/m.ys;tc=t-t.mean();dual=np.linalg.solve(ac@ac.T+m.alpha*np.eye(len(tr)),tc);manual=((B-A.mean(0))@ac.T@dual+t.mean())*m.ys+m.ym;err=float(abs(manual-pred).max());algebra=max(algebra,err);np.testing.assert_allclose(manual,pred,atol=1e-7,rtol=0)
   if split=='original' and k==0:
    dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]+=12345;p2,m2,rec=s.nested(X,dd,tr,va,k);np.testing.assert_array_equal(p2,pred);assert rec==a['detail'];np.testing.assert_array_equal(m2.ridge.coef_,m.ridge.coef_);replayed+=1
  np.testing.assert_array_equal(arr[arm+'_blend'],.9*d['ref']+.1*arr[arm])
 assert s.gate(d,arr)==r['gate'];s.check();report={'passed':True,'split':split,'models_verified':10,'reload_max_error':reload,'independent_dual_Ridge_max_error':algebra,'outer_train_scaler_target_exact':True,'original_fold0_botharms_complete_innerCV_and_outerfit_replay':replayed,'perturbed_outer_validation_labels_prediction_max_error':perturb,'report_sha256':s.sha(s.OUT/split/'report.json'),'source_sha256':s.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_used':False};s.write(_repo_path(ROOT, f'reports/persistence-107-{split}-validation.json'),report);s.emit(report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode');a=ap.parse_args();print(engineering()) if a.mode=='engineering' else validate(a.mode)
