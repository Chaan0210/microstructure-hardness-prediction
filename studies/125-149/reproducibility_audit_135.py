"""Targeted audit on development rows only; no fits or test prediction reads."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,hashlib,ast,zipfile,datetime,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,torch,joblib
import candidate015_pipeline as p
import candidate011_pipeline as p11
from sklearn.preprocessing import StandardScaler,MinMaxScaler
OUT=_repo_path(ROOT, 'artifacts/reproducibility-audit-135')
def read(path):return json.loads((_repo_path(ROOT, path)).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def forbidden(*args,**kwargs):raise AssertionError('Unexpected fit/backward during inference audit')
def main():
 start=time.perf_counter();torch.set_num_threads(2);OUT.mkdir(exist_ok=True);assert not any(OUT.iterdir())
 protected={f'artifacts/submission-candidate-{c}.csv':sha(_repo_path(ROOT, f'artifacts/submission-candidate-{c}.csv')) for c in ['011','014','015']}
 for c in ['011','014','015']:
  m=read(f'artifacts/submission-candidate-{c}.manifest.json');assert protected[f'artifacts/submission-candidate-{c}.csv']==m['submission_sha256']
 evidence=['reports/candidate011-verification.json','artifacts/pipeline-audit-103/report.json','reports/candidate014-verification.json','artifacts/final-model-123/training-validation.json','reports/candidate015-verification.json','artifacts/final-model-134/training-validation.json','artifacts/cnn-seed-mean-133/validation-original.json']
 evidence={n:sha(_repo_path(ROOT, n)) for n in evidence}
 split=read('artifacts/run-001/split_manifest.json');ids=sorted(split['development_folds']);assert len(ids)==400 and not set(ids)&set(split['holdout'])
 # Read only feature arrays, never y from full500 archives.
 full=read('artifacts/final-model-065/report.json');features={};cachechecks={}
 for arm,devpath in [('S','artifacts/dino-multilayer-044/development_features.npz'),('B','artifacts/dinob-062/development_features.npz')]:
  with np.load(_repo_path(ROOT, full[f'train_{arm}_cache']),allow_pickle=False) as z:fullids=z['ids'].tolist();F=z['X'];ix=[fullids.index(i) for i in ids];F=F[ix]
  with np.load(_repo_path(ROOT, devpath),allow_pickle=False) as z:di=z['ids'].tolist();D=z['X'][[di.index(i) for i in ids]].reshape(400,-1)
  np.testing.assert_array_equal(F,D);features[arm]=F;cachechecks[arm]={'rows':400,'exact':True,'full_cache_sha256':sha(_repo_path(ROOT, full[f'train_{arm}_cache']))}
 with np.load(_repo_path(ROOT, 'artifacts/final-model-075/training-inputs.npz'),allow_pickle=False) as z:
  fi=z['ids'].tolist();ix=[fi.index(i) for i in ids];images=z['images'][ix];K=z['K'][ix]
 with np.load(_repo_path(ROOT, 'artifacts/nested-calibration-074/development_features.npz'),allow_pickle=False) as z:
  di=z['ids'].tolist();np.testing.assert_array_equal(images,z['images'][[di.index(i) for i in ids]])
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:
  di=z['ids'].tolist();np.testing.assert_array_equal(K,z['X'][[di.index(i) for i in ids]])
 with np.load(_repo_path(ROOT, 'artifacts/final-model-114/train57.npz'),allow_pickle=False) as z:
  fi=z['ids'].tolist();X=z['X'][[fi.index(i) for i in ids]]
 with np.load(_repo_path(ROOT, 'artifacts/run-022/development_features.npz'),allow_pickle=False) as z:
  di=z['ids'].tolist();np.testing.assert_array_equal(X,z['X'][[di.index(i) for i in ids]])
 cachechecks.update(CNN128={'rows':400,'exact':True},compact26={'rows':400,'exact':True},physics57={'rows':400,'exact':True})
 # Runtime guards cover the actual helper calls exercised below, not arbitrary future code.
 StandardScaler.fit=forbidden;MinMaxScaler.fit=forbidden;torch.Tensor.backward=forbidden;p.n.fit=forbidden
 saved=torch.load(_repo_path(ROOT, 'artifacts/final-model-086/head/model.pt'),map_location='cpu',weights_only=True);short=p.e.Small(saved['scaler_mean'].numpy(),saved['scaler_scale'].numpy(),20270215);short.load_state_dict({k:saved['state'][k] for k in short.state_dict()});B=features['B'][:6]
 q=p.e.small_predict(short,B,np.arange(6),'cpu')*saved['target_std']+saved['target_mean'];old=p11.infer(B,saved,'cpu');np.testing.assert_array_equal(q,old)
 cnnchecks=[]
 for seed in p.x.SEEDS:
  file=p.cnnpath(seed);obj=torch.load(_repo_io(file),map_location='cpu',weights_only=True);a=p.c.cp.predict_cnn(images[:6],file,'cpu');b=p.x.predict(obj,{'images':images[:6]},np.arange(6),'cpu');np.testing.assert_array_equal(a,b);cnnchecks.append({'seed':seed,'development_vs_production_predictor_exact':True})
 batchchecks={}
 for candidate,folder in [('014','final-model-123'),('015','final-model-134')]:
  for j in range(3):
   m=joblib.load(_repo_path(ROOT, f'artifacts/{folder}/block-{j}/residual/selected.joblib'))['model'];t=p.n.predict(m,X[:6]);u=np.concatenate([p.n.predict(m,X[i:i+1]) for i in range(6)]);err=float(abs(t-u).max());assert err<1e-10;batchchecks[f'{candidate}/{j}']={'kind':m['kind'],'query_batch_composition_max_error':err}
 kernel=joblib.load(_repo_path(ROOT, 'artifacts/final-model-017/kernel.joblib'));k=kernel['model'];assert len(kernel['train_ids'])==500
 # No refit: inspect the existing production selection evidence and training code.
 selection=read('artifacts/final-model-017/decision.json')['kernel_selection'];assert selection['n_development']==400 and selection['selected']['bandwidth']==k.bandwidth and selection['selected']['alpha']==k.alpha
 archives={}
 for file in sorted((_repo_path(ROOT, 'deliverables')).glob('*.zip')):
  with zipfile.ZipFile(file) as z:names=z.namelist();archives[file.name]={'file_count':len(names),'candidate_entrypoints_present':{c:any(Path(n).name==f'candidate{c}_pipeline.py' for n in names) for c in ['011','014','015']},'DINO_LICENSE_present':any('dinov2' in n and Path(n).name=='LICENSE' for n in names)}
 src=_repo_path(ROOT, read('vendor/dinov2-official/provenance.json')['source_dir']);sprov=read('vendor/dinov2-official/provenance.json');bprov=read('vendor/dinov2-base-062/provenance.json');smoke=read('artifacts/dino-frozen-037/smoke.json')
 for name in ['LICENSE','README.md','MODEL_CARD.md']:assert sha(src/name)==smoke['metadata_sha256'][name]
 assert sha(src/'LICENSE')==bprov['license_sha256'] and sha(src/'MODEL_CARD.md')==bprov['model_card_sha256']
 weights=[]
 for name,rel,meta in [('original DINOv2 ViT-S/14','vendor/dinov2-official/dinov2_vits14_pretrain.pth',sprov['dinov2_vits14_pretrain.pth']),('original DINOv2 ViT-B/14','vendor/dinov2-base-062/dinov2_vitb14_pretrain.pth',bprov)]:
  assert sha(_repo_path(ROOT, rel))==meta['sha256'];weights.append({'model':name,'path':rel,'official_url':meta['url'],'bytes':meta['bytes'],'sha256':meta['sha256'],'license':'Apache-2.0','hash_is_local_integrity_not_publisher_signature':True})
 licenses={'scope':'Current candidates011/014/015; supplements historical software-licenses.json without altering it','code_commit':sprov['commit'],'weights':weights,'license_file':str((src/'LICENSE').relative_to(ROOT)),'license_sha256':sha(src/'LICENSE'),'model_card_sha256':sha(src/'MODEL_CARD.md'),'model_card_original_models':['ViT-S/14','ViT-B/14'],'nonused_licenses':'Cell-DINO/XRay-DINO licenses do not describe these original non-register S/B weights. No such weights used.','attribution':'Meta Platforms, Inc. and affiliates; preserve upstream LICENSE and model-card/provenance in any future code package.','historical_document_issue':'reports/software-licenses.json says no external pretrained weights/model code; valid for its early snapshot, stale for current candidates.','new_downloads':0}
 write(_repo_path(ROOT, 'reports/software-licenses-current-135.json'),licenses)
 static=[]
 targets={'candidate014_pipeline.py':['inference'],'candidate015_pipeline.py':['inference'],'candidate011_pipeline.py':['infer','test_inputs'],'candidate008_pipeline.py':['predict_checkpoint'],'candidate_pipeline.py':['predict_cnn'],'nonlinear_residual_115.py':['predict'],'compact_regression.py':['predict']}
 for name,funcs in targets.items():
  tree=ast.parse((_repo_path(ROOT, name)).read_text())
  for f in ast.walk(tree):
   if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)) and f.name in funcs:
    bad=[getattr(n.func,'attr','') for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['fit','fit_transform','partial_fit','backward','step']];assert not bad;static.append({'file':name,'function':f.name,'line':f.lineno,'direct_fit_calls':bad})
 result={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed_targeted_checks':True,'scope':'Development400 feature comparisons and six development-row helper predictions only; full500 feature caches inspected without y arrays. No test features/predictions decoded; no holdout scoring; no fit. Three final CSVs hashed only.','prior_evidence_reused':evidence,'feature_routes':cachechecks,'shortB_011_vs014_015_loader_exact':True,'CNN_identity_predictors':cnnchecks,'residual_query_batch_independence':batchchecks,'runtime_fit_backward_guard_passed':True,'direct_inference_AST_scan':static,'static_limit':'Direct call scan plus exercised leaf helpers and prior full-inference evidence; not a universal proof for arbitrary future code.','actual_differences':[{'item':'compact26 KRR final hyperparameter selection scope','severity':'procedure difference, not test leakage','finding':'Final017 selects its nine-grid bandwidth/alpha on development400 then refits statistics/kernel on500; nested models select within each current training subset. Thus same estimator/preprocessing/grid, not literally all500 inner-CV hyperparameter selection. Inherited unchanged by011/014/015.','source':'candidate_pipeline.py:134; frozen_validation.py:231','actual_parameters':{'bandwidth':k.bandwidth,'alpha':k.alpha},'existing_selection_evidence':selection,'action':'Preserve selected candidates. Any change to all500 selection is a new procedure, not silent repair; validate only a predeclared development analogue before promotion.'},{'item':'epoch-based update counts','finding':'Fixed80 CNN and12 shortB epochs cause more updates on500 than320; nested cells smaller again. Intended current rule, not identical optimizer-step budgets. Previously103/128 examined this; no repeat proposed.'},{'item':'feature clipping versus output clipping','finding':'KRR and physics57 use training-fit MinMaxScaler(clip=True). Manifest clipping=false describes final predictions, not feature transforms. Query clipping uses saved train bounds; no test-fitted bounds.'},{'item':'reproduction packaging','severity':'code-submission readiness gap','finding':'Existing source zip contents below omit current011/014/015 entrypoints; saved-project replay passed previously, clean standalone judge reproduction not yet demonstrated. Production check() depends on historical study reports/source hashes.','action':'Before code submission, build one minimal offline relative-path inference package for the finally selected candidate and smoke-test in an isolated project-local folder with network disabled; preserve LICENSE/provenance and use existing packages.'},{'item':'historical license inventory','finding':'Early software-licenses.json no-pretrained statement is stale. Added current scoped registry without modifying historical evidence.'}],'archives':archives,'licenses_report':'reports/software-licenses-current-135.json','no_numerical_prediction_bug_demonstrated':True,'rules':{'url':'https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai','checked_utc_date':'2026-10-04','method':'Read-only official public page via web tool; no account or submission action','observed_requirements':['Official publicly released pretrained weights with permissive allowed license','All prediction models executable locally; no remote-model-only API','No external training data or test training','Relative input/output paths; UTF-8; working imports; OS/library versions; Private Score reproduction'],'assessment':'Inspected data/model lineage is consistent with these requirements; final organizer ruling and clean code-package reproduction are not certified.','AI_coding_assistance':'Public rules do not explicitly resolve AI coding assistance; local prediction model uses no remote API. No claim of organizer-specific endorsement.'},'recommendation':'No additional model training prompted by this audit. Close code-package/dependency and current-license documentation gaps first. Preserve011/014/015 and defer upload.','protected':protected,'source_sha256':sha(Path(__file__)),'new_training':False,'new_test_inference':False,'cost':0,'new_install':False,'uploaded':False,'seconds':time.perf_counter()-start}
 for file,h in protected.items():assert sha(_repo_path(ROOT, file))==h
 write(_repo_path(OUT, 'report.json'),result);write(_repo_path(ROOT, 'reports/reproducibility-audit-135-summary.json'),result);print(json.dumps({'passed':True,'feature_routes_exact':list(cachechecks),'new_numerical_bug_found':False,'production_KRR_selection_scope_difference':True,'code_archive_readiness_gap':True,'seconds':result['seconds']}))
if __name__=='__main__':main()
