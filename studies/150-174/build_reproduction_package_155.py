"""Create an additive best018-only saved-model inference package from audited136 runtime."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,shutil,hashlib,ast,datetime
from pathlib import Path
R=_RepoPath(__file__).resolve().parents[1];OLD=_repo_path(R, 'deliverables/reproduction-011-014-015-v136');P=_repo_path(R, 'deliverables/reproduction-018-v155')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(_repo_io(src),_repo_io(dst))
assert not P.exists();assert sha(_repo_path(R, 'deliverables/reproduction-011-014-015-v136.zip'))=='4427867a93a8e9ffb0526c467eaa570c5b6952a0f18a45adcb40df30cffad2ef'
protected={str(f.relative_to(R)):sha(f) for f in [*sorted((_repo_path(R, 'artifacts')).glob('submission-candidate-*.csv')),*sorted((_repo_path(R, 'deliverables')).glob('*.zip'))]};write(_repo_path(R, 'reports/reproduction-package-155-protected.json'),protected);P.mkdir()
for f in OLD.rglob('*'):
 if not f.is_file():continue
 rel=f.relative_to(OLD)
 if rel.parts[0] in ['vendor','weights','licenses','microstructure_v2'] or str(rel) in ['environment.json','requirements-current.lock','candidate_features.py','denoised_grain_features.py','grain_features.py','latent_features.py','compact_regression.py','cnn_model.py','mlp_head.py','residual_models.py','residual_linear.py','config/schema.json']:cp(f,_repo_path(P, rel))
# Runtime remains self-contained, with no original-study imports.
t=(_repo_path(OLD, 'runtime.py')).read_text();t=t.replace("base=baseline_predict(d['Xbase'],models/'baseline.joblib');", "base=np.zeros(len(d['ids']));")
t=t.replace("features=[]\n  for j,i", "features=[];spatial=[]\n  for j,i")
needle="features.append(torch.cat(parts,1).cpu().numpy()[0])"
t=t.replace(needle,needle+"\n    if arm=='B':spatial.append(spatial_descriptor(layers[-1][0].cpu().numpy()[0]))")
t=t.replace("result[arm]=np.stack(features);assert", "\n  if arm=='B':result['X']=np.column_stack([result['X'],np.stack(spatial)])\n  result[arm]=np.stack(features);assert")
start=t.index("W={'011':")
t=t[:start]+'''SPATIAL_PROJECTION=None
def spatial_descriptor(tokens):
 global SPATIAL_PROJECTION
 if SPATIAL_PROJECTION is None:SPATIAL_PROJECTION=np.linalg.qr(np.random.default_rng(20271201).normal(size=(768,32)),mode='reduced')[0]
 z=(tokens.astype(float)@SPATIAL_PROJECTION).reshape(36,36,32);h=np.diff(z,axis=0).reshape(-1,32);v=np.diff(z,axis=1).reshape(-1,32);mat=(h.T@h+v.T@v)/(2*(len(h)+len(v)));return mat[np.triu_indices(32)]
KINDS=('none','ridge1','ridge01','rbf1','rbf01')
def predict_all(d,models,device):
 assert d['X'].shape==(len(d['ids']),585)
 P,avg,short=components(d,models,device);P[:,2]=avg;w=np.array([0.,.1875,.1875,.125,.125]);raw=P@(w/w.sum());cal=read(models/'calibrator-018.json');base=.9*(cal['target_center']+cal['slope']*(raw-cal['prediction_center']))+.1*short;cor=[]
 for j in range(3):
  weights=np.array([read(models/f'weights-018-{j}.json')[kind] for kind in KINDS]);assert (weights>=0).all() and weights.sum()==1;pred=np.stack([residual_models.predict(joblib.load(models/f'residual-018-{j}-{kind}.joblib')['model'],d['X']) for kind in KINDS]);cor.append(weights@pred)
 pred=base+np.mean(cor,0);assert pred.shape==(len(d['ids']),) and np.isfinite(pred).all();return {'018':pred}
'''
ast.parse(t);(_repo_path(P, 'runtime.py')).write_text(t);cp(_repo_path(OLD, 'predict.py'),_repo_path(P, 'predict.py'))
assets={}
def asset(src,name):cp(_repo_path(R, src),_repo_path(P, 'models', name));assets[name]={'source':src,'sha256':sha(_repo_path(R, src))}
for name in ['kernel.joblib','shortB.pt',*[f'cnn-{seed}.pt' for seed in [20261003,20261023,20261043]],*[f'{arm}-mlp-seed-{seed}.pt' for arm in ['S','B'] for seed in [20261128,20261205,20261219]]]:
 rec=json.loads((_repo_path(OLD, 'config/assets.json')).read_text())[name];asset(rec['source'],name)
asset('artifacts/final-model-134/calibrator.json','calibrator-018.json')
for j in range(3):
 r=json.loads((_repo_path(R, f'artifacts/final-model-145/block-{j}/report.json')).read_text());write(_repo_path(P, f'models/weights-018-{j}.json'),r['weights']);assets[f'weights-018-{j}.json']={'source':f'artifacts/final-model-145/block-{j}/report.json:weights','sha256':sha(_repo_path(P, f'models/weights-018-{j}.json'))}
 for kind in ['none','ridge1','ridge01','rbf1','rbf01']:asset(f'artifacts/final-model-145/block-{j}/{kind}.joblib',f'residual-018-{j}-{kind}.joblib')
write(_repo_path(P, 'config/assets.json'),assets);write(_repo_path(P, 'config/candidates.json'),{'018':{'candidate':'018','expected_CSV_SHA256':sha(_repo_path(R, 'artifacts/submission-candidate-018.csv')),'parent_confirmed_Public_RMSE':11.5230652779,'formula':'.9*affine(.3KRR+.3CNNmean3identity+.2Smean3+.2Bmean3)+.1shortB+mean3bootstrap_residual585','D4_inference':False,'feature_support':'Train-fitted MinMax clipping remains enabled for all585 residual features; failed154 not adopted.'}})
write(_repo_path(P, 'config/training-recipe.json'),{'scope':'Saved-model inference reproduction only. No standalone018 training driver is shipped or validated.','base':'Full500 compact26 KRR;3 TinyCNN80epoch identity seeds20261003/23/43 meanlast5;S/B frozen official DINO504 descriptors layers5/8/11 CLS/patchmean/populationSD with3 MLPheads each;shortB12epoch head.','calibration':'One affine from full500 threefold OOF raw mixture. Each honest B166/167 uses anchor A333/334; anchor affine fitted from deep3fold OOF restricted to A. A-only base/shortB predicts B.','residual':'physics57 plusfixedQR32 Btoken adjacent energy528. Five candidates none,Ridge1,.1,RBF1,.1. Each B innerGroupKFold3 CVseed20271013+j; train-only MinMaxclip+StandardScaler/blockweights, residualRMS;lambda*n. RBF trainingmedian positive distance gamma, traincenter/trace normalization.512groupbootstrap winner-frequency weights seed20271204+j;fit allcandidates B;mean3 corrections.','lineage':'Original project candidate015_pipeline.py, candidate017_pipeline.py and candidate018_pipeline.py generated these frozen assets. Consolidated fulltraining reproduction remains unverified; do not claim exact retraining.','no_data':'No raw images, target CSVs or feature caches are distributed. Fitted kernel support vectors and scaler statistics are model parameters.'})
write(_repo_path(P, 'config/build-provenance.json'),{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_package':'reproduction-011-014-015-v136','parent_archive_SHA256':sha(_repo_path(R, 'deliverables/reproduction-011-014-015-v136.zip')),'builder_SHA256':sha(Path(__file__)),'runtime_source_SHA256':sha(_repo_path(OLD, 'runtime.py')),'upstream_018_verification_SHA256':sha(_repo_path(R, 'reports/candidate018-verification.json')),'source_code_SHA256':{f:sha(_repo_path(R, f)) for f in ['spatial_joint_140.py','bootstrap_selector_144.py','candidate018_pipeline.py']},'old_packages_preserved':True,'network_disabled':True,'cost':0})
(_repo_path(P, 'README.md')).write_text('''# Best018 saved-model inference package\n\nThis additive version covers Publicbest018 (parent-confirmed RMSE11.5230652779). The older136 archive covers011/014/015 and is preserved.019 and failed154 are not adopted.\n\nUse the existing uv Python environment with versions in environment.json. No installation, download or network is needed or permitted by the runtime. Official DINO weights and source/license provenance are included. Put the competition sample_submission.csv and test/ directory under a local data/ directory, or pass a relative --data-dir.\n\n```sh\npython -I -B predict.py --check-environment\npython -I -B predict.py --data-dir data --output-dir output --device mps\npython -I -B predict.py --data-dir data --features output/features.npz --output-dir reload --device mps\n```\n\nOutput directories must be new to preserve existing results. All paths are local; inference forbids fitting and backward, and network sockets are blocked. The saved feature cache is checked against every image hash and the sample CSV. IDs preserve sample order. CPU is supported with small numerical differences; original M1Pro MPS is the reference.\n\nThe model uses3-seed identity CNN means, not019 D4. Spatial residual features are the fixed32-channel projected Btoken nearest-neighbor joint energy. The512-bootstrap procedure's final weights and all15 full-block residual objects are shipped. Feature clipping is unchanged from018.\n\nSee VALIDATION.json for actual isolated-directory raw1000, reload and CPU evidence. This is saved-model inference reproduction, not an end-to-end fulltraining reproduction. config/training-recipe.json describes the original training procedure; no consolidated018 training driver or exact retraining guarantee is included. Kernel support vectors/scalers are learned model parameters. Raw competition data, target CSVs and feature caches are excluded from the archive.\n''')
for f in P.rglob('*.py'):ast.parse(f.read_text())
write(_repo_path(R, 'reports/reproduction-package-155-build.json'),{'package':str(P.relative_to(R)),'builder_SHA256':sha(Path(__file__)),'source_hashes':{str(f.relative_to(P)):sha(f) for f in P.rglob('*.py')},'training_driver_included':False,'raw_data_or_caches_included':False,'cost':0});print('best018 package built')
