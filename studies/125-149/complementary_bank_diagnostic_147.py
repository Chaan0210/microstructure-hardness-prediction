"""Bounded cached-bank complementarity diagnostic for current018; no fitted weights."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap_selector_144 as s
import numpy as np
s.torch.set_num_threads(2);s.check();e=s.e;a=s.a;out={};sources={}
for split in e.SPLITS:
 sr=e.read(s.OUT/split/'report.json');z=a.readnp(s.OUT/split/'predictions.npz',sr['predictions_sha256']);qr=e.read(s.q.OUT/split/'report.json');old=a.readnp(s.q.OUT/split/'predictions.npz',qr['predictions_sha256']);assert z['ids'].tolist()==old['ids'].tolist();np.testing.assert_array_equal(z['y'],old['y']);res=z['y']-z['primary'];out[split]={}
 for name,key in [('physics57_hardselector015','reference015'),('unordered585_hardselector140','unordered')]:
  p=old[key];delta=p-z['primary'];A=float(np.mean(res*delta));B=float(np.mean(delta*delta));out[split][name]={'remaining_error_times_alternative_difference':A,'alternative_difference_MSE':B,'residual_difference_correlation':float(np.corrcoef(res,delta)[0,1]),'alternative_residual_correlation':float(np.corrcoef(res,z['y']-p)[0,1]),'difference_RMS':float(np.sqrt(B)),'alternative_RMSE':e.rmse(z['y'],p),'current018_RMSE':e.rmse(z['y'],z['primary'])}
 for f in [s.OUT/split/'report.json',s.OUT/split/'predictions.npz',s.q.OUT/split/'report.json',s.q.OUT/split/'predictions.npz']:sources[str(f.relative_to(ROOT))]=e.sha(f)
r={'diagnostic':'147 limited two retained representation banks versus018','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'banks':['physics57','physics57+unorderedBcov528'],'why_these':'Matched140banks and015 alreadyvalidated;no137curve ornewhyperparameter. Current018 usesphysics57+adjacency528.','results':out,'no_fitted_mixture_weights':True,'no_blended_candidate_score_reported':True,'no_new_fit_or_inference':True,'scope':'Only existing400developmentOOFcaches, noholdout/test. Descriptive historicalreuse; notindependentcandidatevalidation. Hardselectoralternatives versusbootstrap018 are notmatchedselectionalgorithm,so thisonlymotivatespropercontrolledcomparison.','sources':sources,'source_sha256':e.sha(Path(__file__)),'cost':0};e.write(_repo_path(ROOT, 'reports/complementary-bank-147-diagnostic.json'),r);e.emit(r)
