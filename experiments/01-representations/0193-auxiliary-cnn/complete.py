from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json,csv,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
s=read(_repo_path(R, 'summary.json'));assert read(_repo_path(R, 'validation.json'))['passed']
for name in ['preflight','study','validate','summarize']:assert read(_repo_path(R, name+'-exit.json'))['exit_code']==0
for p,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, p))==h,p
for p,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, p))==h,p
assert sha(_repo_path(R, 'complete.py'))==read(_repo_path(R, 'completion-code-lock.json'))['complete.py']
p=_repo_path(P, 'reports/strategy-experiment-register.csv')
with (_repo_path(R, 'register-before-final.csv')).open('xb') as f:f.write(p.read_bytes())
with p.open(newline='') as f:rows=list(csv.DictReader(f));cols=list(rows[0])
for row in rows:
 if row['실험']=='193':
  row['상태']='완료: 사전기준 통과, 후속 실행 전 보고' if s['gate_passed'] else '완료: 고정 CE 가설 기각, best021 유지'
  row['결과와 당시 기준·split']='원본 개발400 5fold;best021 RMSE %.10f; '%s['best021_RMSE']+'; '.join(a+' 단독 %.10f / 고정10%%결합 %.10f'%(z['standalone'],z['blend10']) for a,z in s['scores'].items())
  row['해석·검증 한계']='동일400 반복선택,조건부CI는선택편향미보정.회귀전용-detach CPU/MPS 8update exact;15모델재로드·5fold train통계·3군 fold0전체80epoch labelpoison exact 검증통과. true vs detach 단독 gain %.8f, vs permuted %.8f. 셔플CE는난도/gradient차이도있음.'%(s['paired_comparisons']['true_vs_detach']['standalone']['gain'],s['paired_comparisons']['true_vs_permuted']['standalone']['gain'])
  row['다음행동']='결과보고 후 대기. 계수/seed/epoch 탐색·자동재분할·전체500·test·submission 없음. pooling은공식수식확인대기 유지.'
  row['근거파일']='reports/auxiliary-cnn-193-summary.json;experiments/auxiliary-cnn-193/plan.json;summary.json;validation.json;completion.json';row['갱신UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
with p.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
manifest={str(t.relative_to(R)):sha(t) for t in R.rglob('*') if t.is_file() and not any(v in ['cache','config','tmp','__pycache__'] for v in t.relative_to(R).parts)}
write(_repo_path(R, 'final-artifact-manifest.json'),manifest)
completion={'study':193,'status':'completed','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision':s['decision'],'gate_passed':s['gate_passed'],'main_fits':15,'full_poison_replays':3,'all_required_exit_codes':0,'protected_files_verified':len(read(_repo_path(R, 'preservation.json'))),'README_unchanged':True,'best021_SHA256':sha(_repo_path(P, 'artifacts/submission-candidate-021.csv')),'summary_SHA256':sha(_repo_path(R, 'summary.json')),'report_SHA256':sha(_repo_path(P, 'reports/auxiliary-cnn-193-summary.json')),'manifest_SHA256':sha(_repo_path(R, 'final-artifact-manifest.json')),'register_SHA256':sha(p),'core_main_fit_seconds':s['core_main_fit_seconds'],'monitored_compute_validation_seconds':s['monitored_compute_validation_seconds'],'peak_RSS_plus_MPS_conservative':s['peak_RSS_plus_MPS_conservative'],'swap_growth':0,'cost':0,'new_install_download_holdout_test_resplit_full500_CSV_upload':False,'CSV_note':'Only experiment-management CSV; no prediction/submission CSV.','no_expansion':True,'remaining_blockers':[]}
write(_repo_path(R, 'completion.json'),completion);print(json.dumps(completion,ensure_ascii=False));print('completion_SHA256',sha(_repo_path(R, 'completion.json')))
