from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
out=_repo_path(HERE, 'candidate-flow46-v1');out.mkdir(exist_ok=True);path=_repo_path(out, 'frozen-recipe.json');assert not path.exists()
d=np.load(_repo_path(HERE, 'data.npz'));h=json.loads((_repo_path(HERE, 'domain-semantics/H300-recipe.json')).read_text());flow=np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz'));assert np.array_equal(flow['ids'],d['ids']);p=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(p))
r=dict(status='frozen_exploratory_candidate_not_confirmed_or_selected',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),name='flow46-v1',
 formula='p083 + eligible*(.384*(mean3Tab(P104+F46)-mean3Tab(P104)) +.128*(Ridge100(P104+F46)-Ridge100(P104)) +.128*(exp(logRidge100(P104+F46))-exp(logRidge100(P104))))',
 baseline_csv='artifacts/submission-candidate-083.csv',baseline_sha256='02c4c6d58e418b14b86c786ee532c1d0373b6ab930075a7df93029f6fed9187c',
 eligibility='Keep083predictionunchangedwherefrozen076failuremaskR.failed|S.failedistrue; readthepreviouslyverifiedincumbent-components.npz mask, alignedbyID.',
 failure_cache='experiments/082-fusion-20261018-0155/reproduction/incumbent-components.npz',
 physical_train=str(p.relative_to(ROOT)),physical_test='experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-test.npz',physical_keys=z['keys'].tolist(),physical_train_sha256=sha(p),
 flow_training_cache='experiments/supervised-renewal-20261018-0625/domain-semantics/H300-features.npz',flow_cache_sha256=sha(_repo_path(HERE, 'domain-semantics/H300-features.npz')),flow_keys=flow['flow_keys'].tolist(),
 flow_model=h['source_weights'],flow_model_sha256=h['source_weights_sha256'],flow_descriptor='domain_semantics.descriptors:46continuousfields;C4meanoriginal/90/180/270;image-local1/99percentiles;CPUfloat32nativeCPSAMforward. Semantic2560columnsareunused.',
 flow_source_sha256=sha(_repo_path(HERE, 'domain_semantics.py')),
 tabpfn=dict(version='3.5',weights='existinglocaltabpfn-v3.5-20260909.safetensors',n_estimators=8,seeds=[0,100,200],device='cpu',native_preprocessing=True,fingerprint='unchangedcheckpointdefault'),
 linear=dict(alpha=100.,imputer='trainingmedian,keep_empty_features=True',scaler='StandardScalerfitonlytrain',targets=['y','natural_log_y'],log_inverse='exp,noadditionalcalibration'),
 training='Readall500providedtrainlabels;fitbothpairedTabcontextsandlinearheadsusingall500originalrows. ExistingH300auxiliaryweightsarevalidcachedtrainingofitsdeclaredclean250subset;preservetheweightsandancestry.',
 inference='Onlyafterthisfreeze,readtestimages/features/IDsforfixedinferenceandschemaverification. Nofittingpreprocessing/calibration/featureselection/hyperparameterson test,notestpseudo-labels.',
 validation_status='Original400conditionalcombinedRMSE9.0233908132vs0839.2017068026. Full-systemconfirmationnotpassed;componentall500stresscheckinconclusive. ThisfiledoesnotchangefinalselectionorclaimsaboutPublic/Private.',
 submission_actions=False,cost=0,
 source_files={name:sha(_repo_path(HERE, name)) for name in ['domain_semantics.py','domain_semantics_readout.py','flow_linear_check.py']})
assert sha(_repo_path(ROOT, r['baseline_csv']))==r['baseline_sha256'];r['failure_cache_sha256']=sha(_repo_path(ROOT, r['failure_cache']));path.write_text(json.dumps(r,indent=2));print('FROZEN',path)
