from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
P=_RepoPath(__file__).resolve().parents[3];R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,z):
 with p.open('x') as fp:json.dump(z,fp,indent=2,allow_nan=False)
plan=read(_repo_path(R, 'plan.json'));orig=read(_repo_path(R, 'original/report.json'));splits=['original']+(['20261031','20261101'] if orig['advance'] else []);results={};valid={};resource={}
for split in splits:
 results[split]=read(_repo_path(R, split, 'report.json'));valid[split]=read(_repo_path(R, split, 'validation.json'));assert valid[split]['passed']
passed=all(r['advance'] for r in results.values())
for path in R.glob('*-resources.json'):
 rr=read(path);trace=rr['trace'];resource[path.stem]={'seconds':rr['seconds'],'peak_RSS':max(z['peak_RSS'] for z in trace),'peak_conservative_RSS_plus_MPS':max(z['conservative_RSS_plus_MPS'] for z in trace),'peak_MPS_driver':max(z['MPS_driver'] for z in trace),'max_swap_growth':max(z['swap_bytes']-rr['initial_swap_bytes'] for z in trace),'minimum_system_free_percent':min(z['system_memory_free_percent'] for z in trace)}
assert all(sha(_repo_path(P, n))==h for n,h in plan['protected'].items())
summary={'study':182,'UTC':datetime.now(timezone.utc).isoformat(),'decision':'eligible_for_parent_review' if passed else 'close_WPH_family_keep021','current_best':'021','source_duplicate_audit':read(_repo_path(R, 'duplicate-audit.json')),'synthetic':read(_repo_path(R, 'synthetic-validation.json')),'feature_validation':{k:v for k,v in read(_repo_path(R, 'feature-validation.json')).items() if k!='image_sha256'},'results':results,'second_process_validation':valid,'resources':resource,'runtime_failure_recovery':read(_repo_path(R, 'backend-amendment.json')),'failure_preserved':'experiments/wph-182/attempt1-scipy-threadpool','protected_unchanged':plan['protected'],'plan_sha256':sha(_repo_path(R, 'plan.json')),'code_lock_sha256':sha(_repo_path(R, 'code-lock.json')),'resplits_run':orig['advance'],'new_holdout_test_access':False,'new_full500_fit':False,'new_CSV':False,'upload':False,'new_install_download_payment':False,'performance_limit':'Development400 repeatedly reused with adaptive history. Resplits reuse same400; conditionalbootstrap not independent confirmation.','method_limit':'Custom fixed15D WPH subset, not full published WPH/PS reproduction; no hardness improvement claim from texture papers.'}
out=_repo_path(P, 'reports/wph-182-summary.json');write(out,summary)
if not passed:write(_repo_path(P, 'reports/WPH-family-closure-182.json'),{'closed':True,'reason':'Prespecified gate failed','scores':{s:r['RMSE'] for s,r in results.items()},'gate':plan['gate'],'do_not_retry':'No alternative scales/orientations/pooling/head/weight rescue in this family.','keep':'021','summary_sha256':sha(out)})
shutil.copyfile(_repo_path(P, 'README.md'),_repo_path(P, 'reports/README-before-182.md'))
scores=orig['RMSE'];note=f'''# 182 WPH 검증 완료 — {'부모 검토 후보' if passed else 'best021 유지, WPH 계열 종료'}

원본 개발400 OOF RMSE:021 **{scores['021']:.10f}**, amplitude-only **{scores['AMP']:.10f}**, WPH **{scores['WPH']:.10f}**.
고정4scale×8방향의 인접 스케일 phase-doubling 중심공분산15D와 동일wavelet amplitude15D를 같은 honest nested 잔차 학습으로 비교했습니다.
원본 진입기준 통과:`{orig['advance']}`. 재분할 실행:`{orig['advance']}`. 합성 위상변이·회전/반사 검증, 기존018/021 exact 재현, 모든900개 모델의 별도 프로세스 재로딩 및 train-local 통계 검증을 통과했습니다.
초기 SciPy worker 오류는 로그·원본을 보존했고 수식 변경 없이 NumPy FFT로 복구했습니다. holdout/test/full500·신규CSV·설치·다운로드·결제·업로드 없음. 기존021/018 등23개 보호파일 유지.
기록:`reports/wph-182-summary.json`, 소스·계획:`experiments/wph-182/`.

'''
(_repo_path(P, 'README.md')).write_text(note+(_repo_path(P, 'README.md')).read_text())
(_repo_path(R, 'README.md')).write_text('''# Study182: fixed phase-harmonic covariance versus amplitude control

Read plan.json and duplicate-audit.json first. Locked custom 4-scale, 8-axial-direction filterbank produces15 phase and15 amplitude covariances. Same residual models, alpha rules, groupfolds and bootstrap seeds. Existing baseline replay is exact. Only development400 image IDs are allowed; model-fitting processes prohibit rawdata opens.

Source order: prepare.py; features.py/synthetic.py; extract.py; readout.py original; validate.py original. Code hashes were recorded before feature extraction and training. Run each stage in a fresh directory when reproducing; outputs refuse overwrite. build_readout.py is generation provenance, not an execution step to repeat over the final readout source.

The initial SciPy multi-worker FFT run failed before producing a cache or fitting models. attempt1-scipy-threadpool retains original sources/locks/logs. backend-amendment.json records the switch to existing NumPy FFT and SciPy worker1 verification, without feature/head/gate changes. Synthetic checks were rerun before development extraction.

See ../../reports/wph-182-summary.json for scores, decisions, validation and resource evidence. No model or CSV is promoted by this folder. The papers concern texture/randomprocess synthesis, not hardness prediction; this small custom subset is not full WPH/Portilla-Simoncelli replication.
''')
print(json.dumps({'decision':summary['decision'],'RMSE':scores,'summary':str(out),'summary_sha256':sha(out),'protected_unchanged':len(plan['protected']),'resources':resource},indent=2))
