"""Final candidate014 handoff, explicit selection limits, and preserved artifacts."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate014_pipeline as p
e=p.e;a=p.a
p.check();dev=e.read(_repo_path(ROOT, 'reports/component-single-ablation-122-summary.json'));decision=e.read(p.OUT/'decision.json');fit=e.read(p.OUT/'fit.json');v=e.read(_repo_path(ROOT, 'reports/candidate014-verification.json'));assert v['valid'] and v['CSV_SHA256']==e.sha(p.CSV);assert dev['decision']['selected_configuration']==p.ARM
chosen=next(x for x in dev['all_configurations'] if x['configuration']==p.ARM);gain=-chosen['RMSE_change_vs013'];ci=[-x for x in chosen['ID_cluster95_RMSE_change_CI'][::-1]]
summary={'candidate':'014','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified_ready_parent_submission','chosen_configuration':p.ARM,'formula':decision['formula'],'selection_reason':'Amongexact4authorizedstructures,baseline795-onlyremovalhadlowestpooleduntrimmedMSE;compact26,CNN,S/BandshortBretained. No allpartition-improvement orCIgate.','development':dev,'pooled_RMSE_gain_vs013':gain,'conditional95_gain_CI':ci,'full500_training':{'changed_pipeline_retrained':True,'new_affine_fits':4,'new_residual_blocks':3,'new_residual_models_including_none':48,'new_neural_fits':0,'remaining_independent_models_reused_trained_on500':True,'selected':[x['chosen'] for x in fit['blocks']],'fit_seconds':fit['seconds'],'calibrator':e.read(p.OUT/'calibrator.json'),'holdout_subgroup_scored':False},'verification':v,'deliverables':{'CSV':str(p.CSV.relative_to(ROOT)),'model_directory':str(p.OUT.relative_to(ROOT)),'manifest':str(p.CSV.with_suffix('.manifest.json').relative_to(ROOT)),'components':str(p.CSV.with_suffix('.components.csv').relative_to(ROOT)),'development_report':'reports/component-single-ablation-122-summary.json','verification_report':'reports/candidate014-verification.json'},'CSV_SHA256':e.sha(p.CSV),'zeroing_explanation_ko':'기존 baseline795 출력을 fold 학습평균으로 대체하고 후속 모델을 고정하면 pooled RMSE13.593111로 악화됐다. 해당 경로를 제거하고 남은 비중을 정규화한 뒤 중첩 affine 보정과 잔차 타깃/모델을 다시 학습하면12.895301이었다. 전자는 현재 모델 출력 개입의 민감도, 후자는 바뀐 학습 절차의 성능이다. raw신경망 가중치 zeroing이나 특징 중요도로 혼동하지 않는다.','limits':['121결과를 본 뒤 같은400장으로122를 선택했다. 재분할과nested fit은 역사적 선택 편향을 제거하지 않는다.','조건부 ID bootstrap은 고정된 모델의 불확실성만 반영하며 연구 선택·재학습 변동을 포함하지 않는다. pooled 개선 구간은0을 포함한다.','남은 모델의 상대 비중, affine 보정과 잔차 함수도 함께 바뀐다. baseline795 특징의 순수 인과 효과나 물리정보가 불필요하다는 증거가 아니다.','Production residualblock166/167,anchor333/334 versusdevelopment106/107,213/214 학습량 차이가 남는다. Public/private 성능은 아직 모른다.'],'test_scope':{'new_test_inference_after_fit_and_training_validation':True,'test_fit_or_tuning':False,'earlier_prior_artifact_inspection':dev['scope']['prior_test_artifact_inspection']},'preservation':{'013_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-013.csv')),'011_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'prior_candidates_unchanged':True},'cost':0,'new_install':False,'uploaded':False,'submission_coordination':'Parent controls upload. Candidate014ready;013retained.','execution_blockers':[],'evidence_sha256':{name:e.sha(_repo_path(ROOT, name)) for name in ['reports/component-single-ablation-122-summary.json','reports/candidate014-verification.json','artifacts/component-single-ablation-122/decision.json','artifacts/final-model-123/decision.json','artifacts/final-model-123/fit.json','artifacts/final-model-123/training-validation.json','artifacts/submission-candidate-014.csv','artifacts/submission-candidate-014.manifest.json']},'source_sha256':e.sha(Path(__file__))};e.write(_repo_path(ROOT, 'reports/candidate014-summary.json'),summary)
old=e.read(_repo_path(ROOT, 'artifacts/problem-definition-111/plan.json'))['existing_report_sha256']
for name,h in old.items():assert e.sha(_repo_path(ROOT, name))==h,name
readme=_repo_path(ROOT, 'README.md');before=e.sha(readme);assert before==e.read(_repo_path(ROOT, 'reports/readme-update-121.json'))['after_sha256'];backup=_repo_path(ROOT, 'reports/README-before-122-candidate014.md')
with backup.open('xb') as f:f.write(readme.read_bytes())
labels={'full013':'기존013','no_legacy_physics':'baseline795+compact26 제거','no_baseline795':'baseline795만 제거 (014)','no_compact26':'compact26만 제거'}
text='''

## 하위 회귀 경로 분리122 및 후보014 (완료, 미업로드)

121에 이어 baseline795만 제거 / compact26 KRR만 제거의 두 고정 비교를 실행했습니다.
기존013과121의 회귀군 전체 제거까지 정확히 네 구성을 비교했고, 같은 nested affine·잔차 재학습을 적용했습니다.
주지표는 세 개발 분할 pooled untrimmed MSE이며 모든 분할/모든 fold 개선이나 CI 양수를 필수 조건으로 두지 않았습니다.

| 구성 | 원본 RMSE | 재분할1 | 재분할2 | pooled RMSE |
|---|---:|---:|---:|---:|
'''
for row in dev['all_configurations']:
 vals=[x['RMSE'] for x in row['per_partition']];text+=f"| {labels[row['configuration']]} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {row['pooled_RMSE']:.6f} |\n"
text+=f'''
선택은 baseline795만 제거입니다. 새로운 보정 전 혼합은0.3×compact26+0.3×CNN+0.2×Smean3+0.2×Bmean3입니다.
이 혼합의 새 affine 보정에0.9, 기존 full500 shortB에0.1을 적용하고 새 honest physics57 잔차3개 평균을 더합니다.
물리 도메인 경로 compact26과57차원 잔차는 유지합니다. 많은 특징의 baseline이 본질적으로 나쁘거나 인과적으로 무의미하다는 주장은 아닙니다.

013 대비 pooled RMSE 개선은{gain:.6f}, 조건부 ID 묶음95% 개선 구간은[{ci[0]:.6f}, {ci[1]:.6f}]입니다.

| 분할 | 013 대비 RMSE 개선 | 조건부95% 개선 구간 |
|---|---:|---:|
'''
for row in chosen['per_partition']:
 interval=[-x for x in row['paired_ID_bootstrap95_RMSE_change_CI'][::-1]];text+=f"| {row['split']} | {-row['RMSE_change_vs013']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] |\n"
text+='''
121의 결과를 본 뒤122를 같은400개로 수행했으므로 선택 편향이 남습니다. 위 구간은 모델을 고정한 조건부 불확실성이며 연구 선택이나 재학습 변동을 포함하지 않습니다.

단순 출력 대체와 재학습의 차이: baseline795 출력을 fold 학습평균으로 대체하고 기존 보정/잔차를 유지하면13.593111로 악화됩니다.
경로를 제거하고 비중·보정·잔차를 올바르게 재구성하면12.895301입니다. 서로 다른 추정 대상이며 raw 가중치 zeroing을 특징 중요도로 사용하지 않았습니다.

전체500 파이프라인을 고정해 affine4개와 잔차 선택/최종 모델48개(none 포함)를 새로 학습·검증했습니다.
나머지 기저 모델은 각자 독립적으로 이미 전체500을 학습했으며 설정이 동일해 재사용했습니다. 새 신경망 학습을 수행했다고 주장하지 않습니다.
전체500 OOF는 보정 학습에만 사용했고 이전 holdout100의 별도 점수를 계산하지 않았습니다.

훈련 검증 후1000개 test를 남은 KRR/CNN/S/B/shortB 모델에서 새로 추론했습니다.
모든 모델 재로딩 예측과 CSV bytes가 정확히 재현됐고, ID순서/중복/누락/finite 검증 및1000개 이미지 해시 확인을 통과했습니다.
CPU/MPS 최종 차이는9.66e-6, 독립 OLS/커널 최종 오차는1.14e-13 미만입니다.
'''
text+=f"\n제출 파일: `artifacts/submission-candidate-014.csv`\nSHA256: `{e.sha(p.CSV)}`\n예측 범위: [{v['prediction_range'][0]:.8f}, {v['prediction_range'][1]:.8f}], clipping 없음.\n"
text+='''
구현 조사 중 기존008/011 구성 CSV의 헤더와 첫 행이014 고정 전에 표시됐다는 사실을122 계획에 기록했습니다.
그 값으로 학습·순위·설정 선택을 하지 않았고 후보 선택은 개발 OOF만 사용했습니다. 새 test 추론은014 학습·검증 후 실행했습니다.

기존013·011·012와 이전 보고서를 보존했습니다. 비용·설치·test 튜닝·외부 업로드는 없습니다. 업로드는 부모 작업에서 조율합니다.
상세: `reports/candidate014-summary.json`, `reports/candidate014-verification.json`, `reports/component-single-ablation-122-summary.json`.
'''
with readme.open('a') as f:f.write(text)
e.write(_repo_path(ROOT, 'reports/readme-update-122-candidate014.json'),{'before_sha256':before,'after_sha256':e.sha(readme),'backup':str(backup.relative_to(ROOT)),'reason':'Document122fixedsingleremovals,selected014full500fit,verificationandlimits'})
p.check();out={'passed':True,'014_ready':True,'014_CSV_SHA256':e.sha(p.CSV),'013_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-013.csv')),'011_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'previous200_reports_preserved':len(old),'packages_unchanged':len(a.check()['environment']),'summary_sha256':e.sha(_repo_path(ROOT, 'reports/candidate014-summary.json')),'development_summary_sha256':e.sha(_repo_path(ROOT, 'reports/component-single-ablation-122-summary.json')),'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/candidate014-preservation.json'),out);e.emit(out)
