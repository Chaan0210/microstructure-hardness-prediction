"""Apply parent-requested pooledMSE rule to all three fixed completed partitions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,math,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_seed_mean_133 as x
import numpy as np
x.check();policy=x.e.read(x.OUT/'pooled-decision-policy.json');scores={};losses=[]
for split in x.e.SPLITS:
 r=x.e.read(x.OUT/split/'report.json');v=x.e.read(x.OUT/f'validation-{split}.json');assert v['passed'];p=x.OUT/split/'predictions.npz';assert x.e.sha(p)==r['predictions_sha256']
 with np.load(_repo_io(p)) as z:
  assert len(z['ids'])==len(set(z['ids'].tolist()))==400 and np.isfinite(z['primary']).all();A=(z['reference014']-z['y'])**2;B=(z['primary']-z['y'])**2;losses.append(np.stack([A,B]));scores[split]={'014_MSE':float(A.mean()),'mean3_MSE':float(B.mean()),'014_RMSE':float(np.sqrt(A.mean())),'mean3_RMSE':float(np.sqrt(B.mean()))}
L=np.stack(losses);mse=L.mean((0,2));manual=np.array([math.fsum(float(v) for v in L[:,i,:].ravel())/1200 for i in range(2)]);np.testing.assert_allclose(mse,manual,rtol=0,atol=1e-12);decision={'completed_splits':list(x.e.SPLITS),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_independent_validation_passed':True,'primary_criterion':'pooleduntrimmedMSEover3partitionsx400IDs; noallfold/allpartition/CIgate','pooled_MSE':{'014':float(mse[0]),'fixed_mean3':float(mse[1])},'pooled_RMSE':{'014':float(np.sqrt(mse[0])),'fixed_mean3':float(np.sqrt(mse[1]))},'lower_pooled_MSE':bool(mse[1]<mse[0]),'scores':scores,'per_partition_wins_diagnostic_only':sum(s['mean3_MSE']<s['014_MSE'] for s in scores.values()),'criterion_timing':'Parentrequestedpooledruleafterfirstresplitpositivealreadyobserved; priorsequentialrulepreservedandsuperseded. Noactualearlystoporcandidatechange.','policy_sha256':x.e.sha(x.OUT/'pooled-decision-policy.json'),'independent_scalar_aggregation_max_error':float(abs(mse-manual).max()),'014_retained':True,'full500_test_submission_authorized_in133':False,'source_sha256':x.e.sha(Path(__file__))};x.s.write(x.OUT/'confirmation-decision.json',decision);x.e.emit(decision)
