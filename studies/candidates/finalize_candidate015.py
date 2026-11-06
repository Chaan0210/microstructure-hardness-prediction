"""Report the locked133 production candidate without test-based selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate015_pipeline as p
e=p.e
p.check();decision=e.read(p.OUT/'decision.json');fit=e.read(p.OUT/'fit.json');tv=e.read(p.OUT/'training-validation.json');v=e.read(_repo_path(ROOT, 'reports/candidate015-verification.json'));assert v['valid'] and tv['passed'] and v['CSV_SHA256']==e.sha(p.CSV)
old=e.read(_repo_path(ROOT, 'artifacts/problem-definition-111/plan.json'))['existing_report_sha256']
for name,h in old.items():assert e.sha(_repo_path(ROOT, name))==h,name
assert e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-014.csv'))=='2c5ce4ca43d75a7f47fbf7dd6c155db9c5c090d86f23b5999a0f6920def4f6a0'
summary={'candidate':'015','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified_ready_parent_submission','formula':decision['formula'],'fixed_seeds':decision['fixed_seeds'],'development':decision['development'],'pooled_RMSE_gain_vs014':decision['pooled_RMSE_gain'],'conditional95_gain_CI':decision['conditional95_gain_CI'],'limits':decision['limits'],'full500_training':{'n_train':500,'new_nested_CNN_fits':24,'full500_CNN_checkpoints_reused':3,'new_affine_fits':4,'new_residual_models_including_none':48,'fit_seconds':fit['seconds'],'selected_residuals':[r['chosen'] for r in fit['blocks']],'all500_OOF_used_only_to_fit_calibrator':True,'holdout_subgroup_scored':False},'verification':v,'training_validation':{'new_CNN_models_verified':tv['new_nested_CNN_models_checked'],'full500_CNN_checks':tv['full500_CNN_checks'],'affine_fits_independently_verified':tv['affine_fits_independently_verified'],'residual_models_reloaded':tv['residual_models_reloaded'],'manual_kernel_max_error':tv['manual_kernel_max_error'],'nested_label_exclusion_certificate':'artifacts/cnn-seed-mean-133/validation-original.json'},'deliverables':{'CSV':str(p.CSV.relative_to(ROOT)),'model_directory':str(p.OUT.relative_to(ROOT)),'manifest':str(p.CSV.with_suffix('.manifest.json').relative_to(ROOT)),'components':str(p.CSV.with_suffix('.components.csv').relative_to(ROOT)),'verification_report':'reports/candidate015-verification.json'},'CSV_SHA256':e.sha(p.CSV),'014_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-014.csv')),'test_scope':decision['test_scope'],'prior_test_aggregate_inspection':decision['prior_inspection'],'execution_issue_repaired':e.read(_repo_path(ROOT, 'reports/candidate015-initial-prepare-failure.json')),'cost':0,'new_install':False,'uploaded':False,'execution_blockers':[],'submission_coordination':'Upload remains with parent; previous Mac lock/file-chooser issue not rechecked here.','source_sha256':e.sha(Path(__file__))}
e.write(_repo_path(ROOT, 'reports/candidate015-summary.json'),summary)
readme=_repo_path(ROOT, 'README.md');before=e.sha(readme);assert before==e.read(_repo_path(ROOT, 'reports/readme-update-133.json'))['after_sha256'];backup=_repo_path(ROOT, 'reports/README-before-candidate015.md')
with backup.open('xb') as f:f.write(readme.read_bytes())
text=f'''

## 후보015: 고정 CNN 3-seed 평균의 전체500 학습 및 검증 (미업로드)

133의 고정 seed20261003/20261023/20261043을 동일 비중으로 평균하고 identity 추론을 사용했습니다.
seed·가중치·CNN 구조·80epoch/마지막5epoch 가중치 평균 규칙은 변경하지 않았습니다.
전체500 CNN 세 체크포인트는 동일 학습 절차의 기존024에서 검증 후 재사용했습니다.024의 D4 추론은 사용하지 않았습니다.
누락된 nested CNN24개, affine4개, physics57 잔차 선택/최종 모델48개(none 포함)를 새로 학습했습니다.
KRR/S/B/shortB는 검증된 전체500 모델을 재사용했습니다. 전체500 OOF는 보정 학습용이며 검증 점수가 아닙니다.
기존 holdout100을 별도로 평가하지 않았습니다.

개발 pooled RMSE:014의12.895301459 →015 절차12.859781154, 개선0.035520305.
조건부 ID 묶음95% 개선 구간은[-0.026804586,0.099789302]로0을 포함합니다.
원본/첫 재분할은 개선됐고 마지막 재분할은13.045857558→13.107923156으로 악화됐습니다.
같은400장과 겹치는 재분할을 반복 사용한 선택 편향, 고정 모델 조건부 구간의 한계가 남습니다.
pooled 기준은 첫 재분할 결과를 본 후 부모 지시로 명확해졌으며 실제 조기 중단이나 seed/가중치 변경은 없었습니다.

훈련 검증 후 test1000개를 추론했습니다. ID·순서·중복·누락·finite·범위, 모델 재로딩 및 CSV bytes 재현을 통과했습니다.
새 nested CNN24개와 전체500 CNN3개를 CPU/MPS 재로딩으로 검증했고, affine4개와 잔차48개를 독립 계산했습니다.
최종 CPU/MPS 최대 차이:{v['CPU_MPS_final_max_error']:.10g}; 독립 OLS/커널 오차:{v['independent_OLS_kernel_final_max_error']:.10g}.
예측 범위:{v['prediction_range']}. clipping·test 튜닝 없음.

CSV:`artifacts/submission-candidate-015.csv`
SHA256:`{e.sha(p.CSV)}`

014를 그대로 보존했습니다. 비용0, 설치 없음, 업로드 없음. 업로드는 부모 작업에서 조율합니다.
캐시 조사 중 과거004 보고서의 test 요약 통계가 표시됐으나 이미 고정된015 설정 선택에 사용하지 않았습니다.
상세:`reports/candidate015-summary.json`, `reports/candidate015-verification.json`.
'''
with readme.open('a') as f:f.write(text)
e.write(_repo_path(ROOT, 'reports/readme-update-candidate015.json'),{'before_sha256':before,'after_sha256':e.sha(readme),'backup':str(backup.relative_to(ROOT)),'reason':'Fixed133 full500 production candidate015 and independent verification'})
p.check();pres={'passed':True,'015_CSV_SHA256':e.sha(p.CSV),'014_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-014.csv')),'old_reports_verified':len(old),'packages_unchanged':len(p.a.check()['environment']),'summary_sha256':e.sha(_repo_path(ROOT, 'reports/candidate015-summary.json')),'README_sha256':e.sha(readme),'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/candidate015-preservation.json'),pres);e.emit(pres)
