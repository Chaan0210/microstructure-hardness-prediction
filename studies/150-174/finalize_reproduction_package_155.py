"""Seal independently replayed018 saved-model package, preserving all prior submissions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,hashlib,zipfile,re,datetime,ast
from pathlib import Path
import numpy as np
R=_RepoPath(__file__).resolve().parents[1];P=_repo_path(R, 'deliverables/reproduction-018-v155');I=_repo_path(R, 'deliverables/verification-155/package')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvread(p):
 with p.open(newline='') as f:r=csv.DictReader(f);assert r.fieldnames==['ID','hardness'];rows=list(r)
 return [r['ID'] for r in rows],np.array([float(r['hardness']) for r in rows])
protected=read(_repo_path(R, 'reports/reproduction-package-155-protected.json'))
for f,h in protected.items():assert sha(_repo_path(R, f))==h,f
contract=read(_repo_path(I, 'contract.json'));assert contract['passed'] and not contract['source_imports_outside_package']
ids,expected=csvread(_repo_path(R, 'artifacts/submission-candidate-018.csv'));assert len(ids)==len(set(ids))==1000;checks={}
for stage in ['validation-raw','validation-reload','validation-cpu']:
 f=_repo_path(I, stage, 'candidate-018.csv');ii,p=csvread(f);assert ii==ids and np.isfinite(p).all();error=float(abs(p-expected).max());assert error<(1e-3 if stage=='validation-cpu' else 1e-10);checks[stage]={'max_error_vs018':error,'CSV_SHA256':sha(f),'byte_exact_vs018':f.read_bytes()==(_repo_path(R, 'artifacts/submission-candidate-018.csv')).read_bytes(),'rows':1000,'ID_order_exact':True,'finite':True,'duplicate_ids':0}
assert (_repo_path(I, 'validation-raw/candidate-018.csv')).read_bytes()==(_repo_path(I, 'validation-reload/candidate-018.csv')).read_bytes()
with np.load(_repo_path(I, 'validation-raw/features.npz')) as z:
 assert z['ids'].tolist()==ids and z['X'].shape==(1000,585);X=z['X'].copy();features={k:z[k].copy() for k in ['S','B']}
with np.load(_repo_path(R, 'artifacts/final-model-114/test57.npz')) as z:np.testing.assert_array_equal(z['X'],X[:,:57])
with np.load(_repo_path(R, 'artifacts/final-model-141/test-adjacent.npz')) as z:np.testing.assert_array_equal(z['X'],X[:,57:])
m=read(_repo_path(R, 'artifacts/submission-candidate-008.manifest.json'))
for arm in ['S','B']:
 file=_repo_path(R, m[f'test_{arm}_cache']);assert sha(file)==m[f'test_{arm}_sha256']
 with np.load(_repo_io(file)) as z:np.testing.assert_array_equal(z['X'],features[arm])
for file in P.rglob('*.py'):
 rel=file.relative_to(P);assert sha(file)==sha(_repo_path(I, rel)),str(rel);ast.parse(file.read_text())
validation={'passed':True,'candidate':'018','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'isolated_directory':True,'original_project_study_imports':False,'network_fit_backward_guards':True,'raw1000_test_images_reextracted':True,'raw_physics57_spatial528_and_S_B_features_exact_vs_original':True,'checks':checks,'source_identical_to_isolated_validation':True,'full500_retraining_performed':False,'standalone018_training_driver_included':False,'remaining_limit':'Saved-model inference verified; consolidated fulltraining and exact retraining weights not reproduced.','test_fit_or_selection':False,'cost':0,'uploaded':False};write(_repo_path(P, 'VALIDATION.json'),validation)
files=[]
for f in sorted(P.rglob('*')):
 if not f.is_file():continue
 rel=str(f.relative_to(P));assert not f.is_symlink() and '__pycache__' not in rel;assert f.suffix in ['.py','.json','.md','.lock','.pt','.pth','.joblib'] or rel=='licenses/DINOv2-LICENSE',rel
 if f.suffix in ['.py','.json','.md','.lock'] or rel=='licenses/DINOv2-LICENSE':assert not re.search(r'/Users/[^\s\"\']+|sk-proj-[A-Za-z0-9_-]+|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',f.read_text()),rel
 files.append(f)
manifest={str(f.relative_to(P)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in files};write(_repo_path(P, 'MANIFEST.sha256.json'),manifest);archive=_repo_path(R, 'deliverables/reproduction-018-v155.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(P.rglob('*')):
  if f.is_file():z.write(f,str(_repo_path(Path(P.name), f.relative_to(P))))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(manifest)+1
 for name,m in manifest.items():assert hashlib.sha256(z.read(P.name+'/'+name)).hexdigest()==m['sha256']
for f,h in protected.items():assert sha(_repo_path(R, f))==h,f
report={'passed':True,'archive':str(archive.relative_to(R)),'archive_SHA256':sha(archive),'archive_bytes':archive.stat().st_size,'file_count':len(manifest)+1,'validation':validation,'old_artifacts_preserved':protected,'raw_competition_data_and_feature_caches_excluded':True,'all_archive_members_hash_verified':True,'privacy_scan_passed':True,'cost':0,'uploaded':False,'remaining_limit':validation['remaining_limit']};write(_repo_path(R, 'reports/reproduction-package-155-summary.json'),report)
s=read(_repo_path(R, 'reports/spatial-support-154-summary.json'));assert not s['advance'] and s['validation_passed'];research={'status':'154 failed;018 preserved; no new candidate CSV','public019':read(_repo_path(R, 'reports/candidate019-public-submission.json')),'diagnostic153':read(_repo_path(R, 'reports/support-diagnostic-153.json'))['feature_support_summary'],'154':s,'interpretation':'Spatial clipping exposes many rows but amplitude loss did not correlate with larger squared errors; fixed unclipping did not improve pooled development. No causal claim based on tiny Public019/016 degradation. Stop this transform without variants. Model complementarity remains, but no new global weights or size-based extrapolation chosen.','next_candidate':None,'unexecuted_next_hypothesis':{'path':'reports/next-hypothesis-after154.json','sha256':sha(_repo_path(R, 'reports/next-hypothesis-after154.json')),'status':'Proposal only, no fitted candidate'},'new_test_candidate_fit':False,'reproduction_package':{'archive':report['archive'],'SHA256':report['archive_SHA256'],'scope':'Saved-model018 inference only; prior136 excludes018 and is preserved.'},'remaining_limit':validation['remaining_limit'],'technical_blockers':[],'no_further_submission_before_parent_local_midnight':True,'uploaded':False,'cost':0};write(_repo_path(R, 'reports/research-153-155-summary.json'),research)
f=_repo_path(R, 'README.md');backup=_repo_path(R, 'reports/README-before-155-final.md');backup.write_bytes(f.read_bytes());addition='\n## 153–155: 공간 범위 제한 검증 및 best018 재현 패키지\n\n153은 현재018의 모델 상보성과 학습 규모별 공간 특징 절단을 진단했습니다. 내부70–72장 모델의 공간 좌표2.76%/표본99.1%,외부106–107장 모델의 좌표1.85%/표본96.6%에서 범위를 벗어난 값이 있었습니다. 절단 크기와 큰 오차의 양의 관계는 확인되지 않았습니다. 공간528개 특징만 제한을 해제하는 단일154 통제는 통합RMSE12.83654465→12.83712366으로 악화하여 기각했습니다.900개 모델 독립검증·30개 새 적합 재현·라벨배제 검증을 통과했으며 변형 탐색·전체500 후보학습·새CSV는 없습니다.\n\n기존136은011/014/015만 포함합니다. 새 best018 저장모델 추론 패키지:`'+report['archive']+'`\nSHA256:`'+report['archive_SHA256']+'`\n독립 디렉터리에서 원본test1000 특징 재추출,018 예측/MPS재로드/CPU추론/ID/finite/순서/파일해시를 검증했습니다. 데이터·정답CSV·특징캐시·개인경로는 배포물에 포함하지 않습니다. 전체018 재학습 드라이버와 정확한 재학습 재현은 포함·검증되지 않았습니다. 기존ZIP·018·019는 보존했고 업로드는 없습니다.\n\n기록:`reports/support-diagnostic-153.json`, `reports/spatial-support-154-summary.json`, `reports/reproduction-package-155-summary.json`, `reports/research-153-155-summary.json`.\n';f.write_text(f.read_text()+addition);write(_repo_path(R, 'reports/readme-update-155.json'),{'before_sha256':sha(backup),'after_sha256':sha(f),'summary_sha256':sha(_repo_path(R, 'reports/research-153-155-summary.json'))});print(json.dumps({k:report[k] for k in ['passed','archive','archive_SHA256','archive_bytes','file_count']}))
