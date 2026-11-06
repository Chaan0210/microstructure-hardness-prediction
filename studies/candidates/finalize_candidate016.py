"""Handoff the fixed137 candidate016 while preserving submittedbest015."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate016_pipeline as p
e=p.e
p.check();dev=e.read(_repo_path(ROOT, 'reports/quadratic-calibration-137-summary.json'));decision=e.read(p.OUT/'decision.json');fit=e.read(p.OUT/'fit.json');tv=e.read(p.OUT/'training-validation.json');v=e.read(_repo_path(ROOT, 'reports/candidate016-verification.json'));assert v['valid'] and tv['passed'] and v['CSV_SHA256']==e.sha(p.CSV);assert dev['decision']['eligible'];pub=e.read(_repo_path(ROOT, 'reports/candidate015-public-submission.json'))
assert e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-015.csv'))==pub['CSV_SHA256'];package=_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip');assert e.sha(package)==e.read(_repo_path(ROOT, 'reports/reproduction-package-136-summary.json'))['archive_SHA256']
summary={'candidate':'016','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified_ready_parent_submission_coordination','current_public_best':'015','public_result015_parent_reported':{'RMSE':pub['public_rmse'],'submission_id':pub['submission_id'],'independently_queried_here':False},'formula':decision['formula'],'fixed_lambda':1.,'extra_curvature_effective_df':.5,'development':dev,'full500_fit':{'new_quadratic_calibrators':4,'new_residual_models_including_none':48,'new_neural_fits':0,'fit_seconds':fit['seconds'],'selected_residuals':[r['chosen'] for r in fit['blocks']],'calibrator':e.read(p.OUT/'calibrator.json'),'full500_OOF_used_only_for_calibration_training':True,'holdout_subgroup_scored':False},'training_validation':tv,'verification':v,'CSV_SHA256':e.sha(p.CSV),'deliverables':{'CSV':str(p.CSV.relative_to(ROOT)),'model_directory':str(p.OUT.relative_to(ROOT)),'manifest':str(p.CSV.with_suffix('.manifest.json').relative_to(ROOT)),'components':str(p.CSV.with_suffix('.components.csv').relative_to(ROOT)),'development_report':'reports/quadratic-calibration-137-summary.json','verification_report':'reports/candidate016-verification.json'},'limits':dev['limits'],'selection_reason':'Single pre-fixed137 procedure lowered all3partition RMSE andpooled MSE, independentlyvalidated. Allpartitionwins descriptive; pooledcriterionfixedbeforeoutcomes. Conditionalgainintervalcontainszero.','preservation':{'015_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-015.csv')),'014_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-014.csv')),'011_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'reproduction136_zip_sha256':e.sha(package)},'cost':0,'new_install':False,'uploaded':False,'test_fit_or_tuning':False,'execution_blockers':[],'submission_coordination':'Parent controls submission.015remainscurrentpublicbest;016hasnopublicresult.014remainsunsubmitted.','source_sha256':e.sha(Path(__file__))};e.write(_repo_path(ROOT, 'reports/candidate016-summary.json'),summary)
rd=_repo_path(ROOT, 'README.md');before=e.sha(rd);assert before==e.read(_repo_path(ROOT, 'reports/readme-update-best015-public.json'))['after_sha256'];backup=_repo_path(ROOT, 'reports/README-before-137-candidate016.md')
with backup.open('xb') as f:f.write(rd.read_bytes())
text='''

## 규제된2차 보정137 및 후보016 (검증 완료, 미제출)

015의 고정 KRR/CNN3seed/S/B 혼합 비중은 그대로 두고 affine 보정에 곡률 항 하나를 추가했습니다.
074의 단일 affine,077·125의 성분별 선형 비중 학습,115의 물리특징 커널 잔차와 구별됩니다.
073의 선형 보정 필요성이 출발점이며 곡률의 유용성이 사전에 입증된 것은 아닙니다. Public015 점수로 곡률·비중·표본 정보를 추정하지 않았습니다.

학습 OOF 예측을 train mean/std로 표준화한z에서z²의 상수·선형 성분을 제거하고 남은q를train RMS로 정규화했습니다.
기존 affine 계수는 비규제이며 곡률 항의 penalty는1로 고정해 해당1차원OLS의 절반(추가df0.5)을 사용합니다.
Outer320 보정은 middleOOF, 잔차anchor213/214 보정은 deepOOF에서 fit하고 physics57 잔차 선택·모델을 새 타깃으로 다시 학습했습니다.
차수·규제·비중 탐색이나 clipping은 없습니다. 세 기존 분할 pooled untrimmed MSE 판단을 결과 전에 고정했습니다.

| 분할 |015|137/016 절차|RMSE 개선|
|---|---:|---:|---:|
'''
for row in dev['per_partition']:text+=f"|{row['split']}|{row['reference015_RMSE']:.9f}|{row['primary_RMSE']:.9f}|{row['gain']:.9f}|\n"
d=dev['decision'];text+=f"\nPooled RMSE:{d['reference015_pooled_RMSE']:.9f}→{d['primary_pooled_RMSE']:.9f}; 개선{d['pooled_RMSE_gain']:.9f}.\n조건부 ID 묶음95% 개선 구간:{d['conditional_ID_cluster95_gain_CI']}.0을 포함하므로 개선의 불확실성이 남습니다.\n"
text+='''
같은400개를 반복 연구한 선택 편향, 겹치는 재분할, 고정 예측에 조건부인 불확실성, quadratic 외삽과 학습량 차이를 유지해서 해석해야 합니다.
60개 곡률 보정의 독립 augmented least-squares,720개 물리 모델 재로딩/독립 계산 및 outer/innerB 타깃 변경 불변성을 검증했습니다.
신경망과 기저 표현은133의 학습범위·타깃 제외 검증을 인용해 재사용했습니다. 새 신경망 학습은 없습니다.

양성 결정을 고정한 후 전체500에 곡률 보정4개와 물리 잔차 모델48개(none 포함)를 다시 fit했습니다.
이전holdout100은 최종 학습에만 포함했으며 별도 점수/선택에 사용하지 않았습니다.
훈련 검증 후 test1000을 추론했고 ID/순서/중복/finite/범위, 모델재로딩과 CSV바이트 재현 및 CPU/MPS 대조를 통과했습니다.
'''
text+=f"\n016 CSV:`{p.CSV.relative_to(ROOT)}`\nSHA256:`{e.sha(p.CSV)}`\n예측 범위:{v['prediction_range']};CPU/MPS최대 차이:{v['CPU_MPS_final_max_error']:.10g}.\n"
text+='''
현재Publicbest015를 보존합니다.016은 미제출이며 Public/private 개선은 미확인입니다.014 역시 미제출입니다.
기존 재현ZIP136은011/014/015용으로 보존했으며016을 포함한다고 주장하지 않습니다.
설치·비용·추가외부자료·test튜닝·업로드 없음. 제출은 부모 작업에서 조율합니다.
상세:`reports/quadratic-calibration-137-summary.json`, `reports/candidate016-summary.json`, `reports/candidate016-verification.json`.
'''
with rd.open('a') as f:f.write(text)
e.write(_repo_path(ROOT, 'reports/readme-update-137-candidate016.json'),{'before_sha256':before,'after_sha256':e.sha(rd),'backup':str(backup.relative_to(ROOT)),'reason':'Document single fixed137 nestedstudy and verified016; retainpublicbest015'})
p.check();out={'passed':True,'016_ready':True,'016_CSV_SHA256':e.sha(p.CSV),'current_public_best':'015','preserved':summary['preservation'],'summary_sha256':e.sha(_repo_path(ROOT, 'reports/candidate016-summary.json')),'development_summary_sha256':e.sha(_repo_path(ROOT, 'reports/quadratic-calibration-137-summary.json')),'README_sha256':e.sha(rd),'old200_reports_not_needlessly_rechecked':True,'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/candidate016-preservation.json'),out);e.emit(out)
