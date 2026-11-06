"""Verify isolated outputs, seal allowlisted package, preserve old artifacts."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,csv,hashlib,zipfile,datetime,ast,shutil,re
import numpy as np
R=_RepoPath(__file__).resolve().parents[1];P=_repo_path(R, 'deliverables/reproduction-011-014-015-v136');I=_repo_path(R, 'deliverables/verification-136/package')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows(p):
 with p.open(newline='') as f:r=csv.DictReader(f);assert r.fieldnames==['ID','hardness'];v=list(r)
 return [a['ID'] for a in v],np.array([float(a['hardness']) for a in v])

def main():
 protect=json.loads((_repo_path(R, 'reports/reproduction-package-136-protected.json')).read_text())
 for name,h in protect.items():assert sha(_repo_path(R, name))==h,name
 result={}
 for c in ['011','014','015']:
  ref=_repo_path(R, f'artifacts/submission-candidate-{c}.csv');ids,expected=rows(ref);assert len(ids)==len(set(ids))==1000;checks={}
  for stage in ['validation-raw','validation-reload','validation-cpu']:
   file=_repo_path(I, stage, f'candidate-{c}.csv');got,pred=rows(file);assert got==ids and np.isfinite(pred).all();err=float(abs(pred-expected).max());assert err<(1e-3 if stage=='validation-cpu' else 1e-10);checks[stage]={'rows':1000,'ID_order_exact':True,'duplicates':0,'finite':True,'max_abs_error_vs_original':err,'CSV_SHA256':sha(file),'byte_exact_vs_original':file.read_bytes()==ref.read_bytes()}
  assert c=='011' or (_repo_path(I, f'validation-raw/candidate-{c}.csv')).read_bytes()==(_repo_path(I, f'validation-reload/candidate-{c}.csv')).read_bytes();result[c]={'expected_SHA256':sha(ref),'checks':checks}
 # All shipped runtime source must match the final isolated reload code.
 code=[]
 for file in P.rglob('*.py'):
  rel=file.relative_to(P);assert sha(file)==sha(_repo_path(I, rel)),str(rel);code.append(str(rel));ast.parse(file.read_text())
 contract=json.loads((_repo_path(R, 'reports/reproduction-package-136-isolated-contract.json')).read_text());assert contract['passed'] and contract['source_imports_outside_isolated_package']==[]
 validation={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidates':result,'011_roundoff':json.loads((_repo_path(R, 'reports/reproduction-package-136-011-roundoff.json')).read_text()),'independent_directory':True,'original_project_study_imports':False,'verified_source_files':len(code),'raw1000_feature_extraction':True,'final_code_MPS_reload_and_CPU_inference':True,'fit_backward_guard_in_final_inference':True,'network_disabled':True,'environment_versions_exact':True,'training_entrypoint_import_and_plan_checked':True,'full_training_execution':False,'full_training_exact_state_reproduction':'NOT VERIFIED; new consolidated training driver was not run end-to-end. Prior project training evidence is not substituted for this check.','data_in_distribution':False,'new_model_training':False,'new_install':False,'cost':0,'uploaded':False}
 write(_repo_path(P, 'VALIDATION.json'),validation)
 # Allowlist path classes and inspect all human-readable payloads for local personal paths/secrets.
 forbidden=[];files=[]
 for file in sorted(P.rglob('*')):
  if not file.is_file():continue
  rel=str(file.relative_to(P));assert '__pycache__' not in rel and file.suffix not in ['.csv','.npz','.png','.jpg','.zip','.pyc'],rel;assert not file.is_symlink();assert len(file.parts)>len(P.parts)
  assert file.suffix in ['.py','.json','.md','.lock','.pt','.pth','.joblib'] or rel=='licenses/DINOv2-LICENSE',rel
  if file.suffix in ['.py','.json','.md','.lock'] or rel=='licenses/DINOv2-LICENSE':
   txt=file.read_text(encoding='utf-8');hits=re.findall(r'/Users/[^\s\"\']+|sk-proj-[A-Za-z0-9_-]+|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',txt)
   if hits:forbidden.append(rel)
  files.append(file)
 assert not forbidden,forbidden
 manifest={str(f.relative_to(P)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in files};write(_repo_path(P, 'MANIFEST.sha256.json'),manifest)
 archive=_repo_path(R, 'deliverables/reproduction-011-014-015-v136.zip');assert not archive.exists()
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for file in sorted(P.rglob('*')):
   if file.is_file():z.write(file,str(_repo_path(Path(P.name), file.relative_to(P))))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;assert len(z.namelist())==len(manifest)+1
  for name,m in manifest.items():assert hashlib.sha256(z.read(P.name+'/'+name)).hexdigest()==m['sha256']
 for name,h in protect.items():assert sha(_repo_path(R, name))==h,name
 out={'passed':True,'archive':str(archive.relative_to(R)),'archive_SHA256':sha(archive),'archive_bytes':archive.stat().st_size,'file_count':len(manifest)+1,'validation':validation,'all_archive_members_hash_verified':True,'privacy_scan_passed':True,'allowed_contents':'source, model weights/support vectors, derived split metadata, environment, provenance/license, documentation and aggregate verification only','raw_competition_data_and_feature_caches_excluded':True,'previous_ZIPs_and_candidates_preserved':protect,'standalone_directory':str(P.relative_to(R)),'cost':0,'uploaded':False,'remaining_limit':'Full consolidated training driver not executed; no claim of exact retraining weights. Saved-model1000 inference checked.'};write(_repo_path(R, 'reports/reproduction-package-136-summary.json'),out);print(json.dumps({k:out[k] for k in ['passed','archive','archive_SHA256','archive_bytes','file_count']}))
 # README is repository documentation; preserve prior bytes.
 rd=_repo_path(R, 'README.md');before=sha(rd);assert before==json.loads((_repo_path(R, 'reports/readme-update-135.json')).read_text())['after_sha256'];backup=_repo_path(R, 'reports/README-before-package136.md')
 with backup.open('xb') as f:f.write(rd.read_bytes())
 text=f'''\n\n## 오프라인 재현 패키지136 (011/014/015, 비공개·미업로드)\n\n새 패키지:`{out['archive']}`\nSHA256:`{out['archive_SHA256']}`\n\n세 후보를 명시한 모델·원본이미지 추론·full500 학습 진입점, 현재26개 의존성 고정정보, DINO 출처/라이선스/해시, 데이터배치와 seed·중첩선택 명령을 포함합니다.\n기존 ZIP과011/014/015를 보존했고 원본 이미지·정답CSV·test 특징캐시·사용자경로·credential은 넣지 않았습니다.\n학습된 커널의 support vector와 정규화 통계는 모델의 필수 구성으로 포함됩니다.\n독립 디렉터리에서 기존uv환경만 사용해 원본 test1000 특징을 새로 추출하고 세 후보 예측을 비교했습니다.\n최종 코드로 MPS 모델 재로딩 및 CPU 추론도 확인했습니다. 구체적인 오차·CSV bytes·해시는 패키지 `VALIDATION.json`과 보고서에 기록했습니다.\n전체 학습 명령은 실행하지 않았으며, 새 통합 학습 드라이버의 전체 재학습 가중치 일치는 검증하지 않았습니다.\n학습 import·구문·plan·저장 모델 설정 확인을 전체 재학습 재현으로 과장하지 않습니다.\n설치·추가모델학습·test튜닝·업로드·비용 없음. 상세:`reports/reproduction-package-136-summary.json`.\n'''
 with rd.open('a') as f:f.write(text)
 write(_repo_path(R, 'reports/readme-update-136.json'),{'before_sha256':before,'after_sha256':sha(rd),'backup':str(backup.relative_to(R)),'archive_SHA256':out['archive_SHA256']})
if __name__=='__main__':main()
