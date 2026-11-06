"""Resume completed020 block without refitting; verify linear equality at matching query size."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate020_pipeline as production
from candidate020_pipeline import *
ROUND=[];REUSED=[]
def resume_system(d,B,anchor,V,foldseed,bootseed,path):
 if path.exists():
  r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_array_equal(z['outer_indices'],V);np.testing.assert_array_equal(z['anchor'],anchor);np.testing.assert_array_equal(z['residual'],d['y'][B]-anchor);assert r['foldseed']==foldseed and r['bootseed']==bootseed
  for name,h in r['models_sha256'].items():assert e.sha(path/name)==h
  REUSED.append(str(path.relative_to(ROOT)));return z['primary'],r
 return q.system(d,B,anchor,V,foldseed,bootseed,path)
def assert_linear_equivalence(d,B,z,oldz,j):
 error=float(abs(z['full_predictions'][:3]-oldz['candidate_predictions'][:3,B]).max());assert error<1e-10;ROUND.append(error)
 for ci,kind in enumerate(s.KINDS[:3]):
  src=old.OUT/f'block-{j}/{kind}.joblib';ref=joblib.load(_repo_io(src))['model'];np.testing.assert_array_equal(q.predict(ref,d['X'][B]),z['full_predictions'][ci])
def fit():
 check();start=time.perf_counter();d,P,cells=inputs(False);records=[]
 for j,(dd,A,B,M,D,short,z,rec) in enumerate(cells):
  ap,ac=a.affine(a.mixture(D,x.ARM),d['y'][A],a.mixture(M,x.ARM));anchor=.9*ap+.1*short;src=p.OUT/f'block-{j}/residual';r=e.read(src/'report.json');oz=a.readnp(src/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(anchor,oz['anchor_predictions']);path=OUT/f'block-{j}';_,br=resume_system(dd,B,anchor,B,a.r.CVSEED+j,s.SEED+j,path);nr=a.readnp(path/'predictions.npz',br['predictions_sha256']);np.testing.assert_array_equal(nr['inner_predictions'][:3],oz['inner_predictions'][:3]);r0=e.read(old.OUT/f'block-{j}/report.json');o0=a.readnp(old.OUT/f'block-{j}/predictions.npz',r0['predictions_sha256']);assert_linear_equivalence(dd,B,nr,o0,j);records.append({'block':j,'anchor_calibrator':ac,'feature_report_sha256':e.sha(OUT/f'feature-{j}/report.json'),'residual_report_sha256':e.sha(path/'report.json'),'weights':br['weights']});e.emit({'full500_block':j,'weights':br['weights']})
 r={'rows':500,'train_ids':d['ids'],'blocks':records,'new_residual_objects_including_none':60,'new_neural_fits':0,'linear_same_query_batch_exact018':True,'legacy_different_query_size_roundoff_max':max(ROUND),'unchanged018_affines_and_base':True,'decision_sha256':e.sha(OUT/'decision.json'),'seconds':time.perf_counter()-start,'holdout_subgroup_scored':False,'test_read':False};e.write(OUT/'fit.json',r);check();e.emit({'full500_complete':True,'seconds':r['seconds']})
if __name__=='__main__':
 torch.set_num_threads(2);fit();e.write(OUT/'fit-v2-addendum.json',{'passed':True,'reason':'Original exact-cache assertion compared predictions queried on500 rows with predictions queried onB166/167; BLAS roundoff observed. Same-query-size old018 model replay is byteexact. No model/recipe/selection change.','source_sha256':e.sha(Path(__file__)),'frozen_source_sha256':e.sha(_repo_path(ROOT, 'candidate020_pipeline.py')),'legacy_different_query_size_max_error':max(ROUND),'reused_blocks_without_refitting':REUSED,'fit_sha256':e.sha(OUT/'fit.json')})
