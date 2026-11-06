"""Project-only builder: packages fresh176 code/models/evidence without raw data."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import hashlib,json,shutil,zipfile,ast
ROOT=Path(__file__).resolve().parent;PROJECT=_RepoPath(__file__).resolve().parents[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,r):
 with p.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
def main():
 test=read(_repo_path(ROOT, 'test-run/inference-validation.json'));assert test['passed'];assert read(_repo_path(ROOT, 'run/training-validation.json'))['passed'];assert read(_repo_path(ROOT, 'run/training-comparison.json'))['passed'];assert read(_repo_path(ROOT, 'run/test-comparison.json'))['CSV_bytes_exact'];out=_repo_path(PROJECT, 'deliverables/reproduction-018-v176');out.mkdir(exist_ok=False)
 plan=read(_repo_path(ROOT, 'run/plan.json'));sources=list(plan['sources'])+['validate_full500.py','infer.py','export_models.py','selftest.py','requirements-current.lock']
 for name in sources:
  dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(_repo_path(ROOT, name),dest)
 for folder in ['config','licenses','weights','models']:
  shutil.copytree(_repo_path(ROOT, folder),out/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copyfile(_repo_path(ROOT, 'source-history/change-record.json'),out/'source-history/change-record.json')
 evidence=out/'evidence';evidence.mkdir()
 for name in ['plan.json','post-training-plan.json','environment.json','train-feature-plan.json','train-feature-validation.json','train-feature-comparison.json','training-validation.json','validation-addendum.json','comparison-addendum.json','comparison-provenance-correction.json','training-comparison.json','test-comparison.json']:
  shutil.copyfile(_repo_path(ROOT, 'run', name),evidence/name)
 for name in ['test-feature-plan.json','test-feature-validation.json','inference-validation.json']:shutil.copyfile(_repo_path(ROOT, 'test-run', name),evidence/name)
 shutil.copyfile(_repo_path(ROOT, 'run/full/report.json'),evidence/'full500-fit.json')
 execution=read(next((_repo_path(ROOT, 'run')).glob('execution-*.json')));write(evidence/'training-resources.json',{'seconds':execution['seconds'],'resource_trace':execution['resource_trace'],'historical_project_files_blocked_by_audit_hook':execution['historical_project_files_blocked_by_audit_hook']})
 shutil.copyfile(_repo_path(ROOT, 'README-package.md'),out/'README.md')
 write(out/'PACKAGE.json',{'version':176,'candidate':'018','purpose':'Fresh full500 exact reproduction, not a new performance candidate','fitted_artifact_reuse':{'fresh176':True,'previous_regressors_CNN_heads_calibrators_residuals_loaded_during_fit':0,'official_frozen_DINO_weights_reused':['S','B'],'configuration_and_code_reused':True},'dependency_artifacts_bundled':False,'raw_data_labels_and_feature_caches_bundled':False,'models_contain_training_statistics_and_kernel_support_features':True,'license_scope':'Original DINOv2 S/B Apache-2.0; upstream notices preserved. Project-specific code carries no new license grant here. Raw competition data not included.','fresh_training_plan_sha256':sha(_repo_path(ROOT, 'run/plan.json')),'verification_scope':'Fresh raw500 full hierarchy trained; fresh raw1000 inference. Archive isolated verification is a separate external report; raw2-image smoke plus fresh176-cache1000 inference. No duplicate full500 training from extracted ZIP.'})
 files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};write(out/'MANIFEST.json',{'version':176,'expected_CSV_sha256':test['CSV_sha256'],'files':files})
 for p in out.rglob('*.py'):ast.parse(p.read_text())
 for p in out.rglob('*'):
  if p.is_file() and p.suffix in ['.py','.json','.md','.lock']:
   s=p.read_text();assert '/Users/' not in s and 'sk-proj-' not in s,str(p)
 zip_path=out.with_suffix('.zip')
 with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(out.parent))
 with zipfile.ZipFile(zip_path) as z:
  assert z.testzip() is None;names=z.namelist();assert len(names)==len(set(names));assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
  for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==sha(out.parent/n)
 record={'passed':True,'directory':str(out.relative_to(PROJECT)),'ZIP':str(zip_path.relative_to(PROJECT)),'ZIP_sha256':sha(zip_path),'ZIP_bytes':zip_path.stat().st_size,'files':len(files)+1,'manifest_sha256':sha(out/'MANIFEST.json'),'all_archive_member_hashes_exact':True,'raw_data_labels_caches_excluded':True,'private_absolute_paths_excluded':True};write(_repo_path(ROOT, 'run/package-build.json'),record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
