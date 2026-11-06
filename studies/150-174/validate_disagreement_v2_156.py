"""Validation-only local-name repair; frozen156 fits, recipe and source preserved."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import disagreement_study_156 as study
from disagreement_study_156 import *
def verify_system(d,B,anchor,V,path):
 r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_array_equal(z['outer_indices'],V);np.testing.assert_array_equal(z['anchor'],anchor);res=d['y'][B]-anchor;np.testing.assert_array_equal(z['residual'],res);ip=np.empty_like(z['inner_predictions']);maxerr=0.;count=0;folds=list(GroupKFold(3,shuffle=True,random_state=r['foldseed']).split(d['X'][B],res,d['g'][B]))
 for t,(it,iv) in enumerate(folds):
  assert r['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(s.KINDS):
   file=path/f'inner-{t}-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train_indices'],B[it]);np.testing.assert_array_equal(obj['validation_indices'],B[iv]);independent=manual(obj['model'],d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);actual=predict(obj['model'],d['X'][B[iv]]);err=float(abs(actual-independent).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['inner_predictions'][ci,iv]);ip[ci,iv]=actual;count+=1
 np.testing.assert_allclose(np.mean((ip-res)**2,1),list(r['inner_MSE'].values()),atol=1e-12,rtol=0);w,wins=s.independent_weights((ip-res)**2,d['g'][B],z['draws']);np.testing.assert_array_equal(w,z['weights']);np.testing.assert_array_equal(wins,z['winners']);np.testing.assert_array_equal(w,list(r['weights'].values()));_,draw,_=s.weights((ip-res)**2,d['g'][B],r['bootseed']);np.testing.assert_array_equal(draw,z['draws']);pp=[]
 for ci,kind in enumerate(s.KINDS):
  file=path/f'full-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train_indices'],B);independent=manual(obj['model'],d['X'][B],res,d['X'][V],d['blocks']);actual=predict(obj['model'],d['X'][V]);err=float(abs(actual-independent).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['full_predictions'][ci]);pp.append(actual);count+=1
 np.testing.assert_allclose(sum(wi*pi for wi,pi in zip(w,pp)),z['primary'],atol=1e-12,rtol=0);return z['primary'],count,maxerr
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('split',choices=e.SPLITS);split=p.parse_args().split;study.verify_system=verify_system;study.validate(split);e.write(OUT/f'validation-v2-{split}.json',{'passed':True,'repair':'Rename local numeric result, preserving independent manual() function dispatch. No fit/prediction/selection change.','source_sha256':e.sha(Path(__file__)),'frozen_source_sha256':e.sha(_repo_path(ROOT, 'disagreement_study_156.py')),'validation_sha256':e.sha(OUT/f'validation-{split}.json')})
