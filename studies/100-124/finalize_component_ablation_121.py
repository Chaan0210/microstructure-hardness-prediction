"""Preserve incumbents and append auditable ablation findings to repository README."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import component_ablation_121 as a
e=a.e;p=a.check();report=_repo_path(ROOT, 'reports/component-ablation-121-summary.json');r=e.read(report);assert r['decision']['keep013'] and r['status']=='completed_diagnostic_keep013'
old=e.read(_repo_path(ROOT, 'artifacts/problem-definition-111/plan.json'))['existing_report_sha256']
for name,h in old.items():assert e.sha(_repo_path(ROOT, name))==h,name
for name,h in e.read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256'].items():assert e.sha(_repo_path(ROOT, name))==h,name
readme=_repo_path(ROOT, 'README.md');before=e.sha(readme);assert before==e.read(_repo_path(ROOT, 'reports/readme-update-120.json'))['after_sha256'];backup=_repo_path(ROOT, 'reports/README-before-121.md')
with backup.open('xb') as f:f.write(readme.read_bytes())
labels={'no_DINO':'DINO 전체','no_CNN':'기존 CNN','no_legacy_physics':'기존 수작업 특징 직접 회귀','no_new_residual':'새 물리 특징 잔차 보정'}
text='''

## 후보013 구성요소 ablation 121 (완료, 진단)

후보013의 실제 계보는 다음과 같습니다. 계수는 affine 보정 이전의 비중입니다.

- 008 = 0.375×baseline795 + 0.1875×compact26 KRR + 0.1875×TinyCNN128 + 0.125×DINO-S mean3 + 0.125×DINO-B mean3
- 009 = 학습 fold 내부 OOF로 맞춘 affine(008)
- 011 = 0.9×009 + 0.1×DINO-B short head
- 013 = 011 + 학습 범위를 분리한 physics57 잔차 모델3개의 평균

baseline795는 형태·텍스처·통계 특징의 ET/Ridge 앙상블이며 정확한 물리 방정식이 아닙니다.
compact26에는 Hall–Petch 대리량(결정립 직경의 역제곱근), 상/공극 분율, 형상/정렬, 선절편 특징이 들어갑니다.
새 잔차 보정은 compact26+TV31의57개 특징을 이용합니다. 따라서 도메인 지식은 실제 입력과 잔차 모델에 적용되어 있습니다.
이 영상 대리량이 참 결정립 마스크나 절대 물리 크기임을 입증한 것은 아닙니다.

네 가지 구성군을 하나씩 제거했습니다. 앞의 세 경우에는 남은008 비중을 합1로 정규화하고,
중첩 OOF에서 affine 보정 및115 잔차 선택/학습까지 다시 수행했습니다.
DINO 제거에는 기존 S/B head뿐 아니라011의 short B head도 포함됩니다.
독립적으로 학습된 나머지 기저 모델은 같은 fold의 저장 모델을 재사용할 수 있어 새 신경망 학습이 필요하지 않았습니다.
반대로 입력 특징·공유 가중치 자체를 삭제하는 실험은 그 기저 모델을 다시 학습해야 하며 이번 분석에 포함하지 않았습니다.
새 잔차 제거만은 예측합에서 잔차를 빼는 것과 정확히 같으며011로 돌아갑니다.

| 제거 구성 | 후속 보정·잔차 재학습 RMSE | 013 대비 변화 | 고정 모델에서 학습평균 대체 RMSE |
|---|---:|---:|---:|
'''
for row in r['ablation_table']:text+=f"| {labels[row['removed_group']]} | {row['refit_pipeline_RMSE']:.6f} | {row['refit_minus013_RMSE']:+.6f} | {row['frozen_trainmean_replacement_RMSE']:.6f} |\n"
text+=f'''
기준013 pooled RMSE는 {r['reference013_pooled_RMSE']:.6f}입니다. 같은400개를 세 번 재분할한1200쌍의 오류를 합산했습니다.
고정 모델 민감도는 해당 분기의 출력만 outer 학습 타깃 평균으로 바꾸고 기존 보정·잔차를 유지한 값입니다.
이는 재학습 ablation과 다른 질문이며, 신경망 가중치를 임의로0으로 만들거나 물성 예측값을0으로 만드는 분석이 아닙니다.

기존 수작업 특징 직접 회귀군 제거는 pooled RMSE를0.130882 낮췄지만 원본 분할은0.005715 악화했고 두 재분할에서 개선됐습니다.
조건부 ID 묶음95% RMSE 변화 구간은[-0.409853,+0.149781]이며 반복 연구의 선택 편향을 제거하지 않습니다.
이 구성군의 학습평균 대체는 오히려 RMSE14.261699로 크게 악화되어, 단순 zeroing의 손실을 곧바로 중요도로 해석하면 다른 결론에 이릅니다.
그 제거 모델에도57개 물리 특징의 잔차 보정은 남아 있고, 재학습한 잔차 보정은 RMSE13.041851→12.951140의 이득을 보였습니다.
따라서 물리 도메인 특징 전체가 불필요하다는 결론이 아닙니다.

CNN의 단독 RMSE는14.621919로 여섯 원시 구성 중 가장 나쁘지만, CNN을 제거하면 합산 RMSE가0.178306 악화됩니다.
단독 모델 점수만으로 구성요소를 버리면 상보성을 놓칠 수 있습니다.
DINO-B 장기/단기 head의 오차 상관은0.962915, baseline795/compact26은0.916419입니다.
높은 상관만으로 중복·제거 가능성을 확정하지 않습니다.

476개 계보 입력 해시,15개 외부 CNN의 CPU 재로딩,2160개 잔차 모델 재로딩과135개 내부 선택을 검증했습니다.
독립 OLS/커널 계산 최대 오차는6.49e-14 미만입니다. 세 제거 경로에서 외부 라벨과 제거된 출력 전체를 교란해 결과 불변을 확인했습니다.
최초 비트 단위 재실행 검사는5.68e-14 차이로 실패해 보존했고, 기존 OLS 허용오차1e-10을 사용한 후속 검증을 통과했습니다.

후속 제안(아직 미실행): 기존 회귀군을 baseline795와compact26으로 나눠 두 가지 단일 제거만 고정해 같은 중첩 재학습을 수행합니다.
이는 어느 직접 회귀 경로가 상대적으로 중복되거나 비중이 맞지 않는지 구분하는 실험입니다. 새 특징/커널/가중치 탐색으로 확대하지 않습니다.
필요한 모든 캐시는 이미 있으며 새 신경망 학습 없이 실행할 수 있습니다.

이번 작업은 진단이므로 후보013을 보존합니다. 새 제출 CSV·전체500 재학습·test 추론·holdout 평가·설치·비용·업로드는 없습니다.
`reports/component-ablation-121-metrics.csv`는 분석 지표 표이며 제출 파일이 아닙니다.
상세 결과: `reports/component-ablation-121-summary.json`; 계획: `artifacts/component-ablation-121/plan.json`.
'''
with readme.open('a') as f:f.write(text)
e.write(_repo_path(ROOT, 'reports/readme-update-121.json'),{'before_sha256':before,'after_sha256':e.sha(readme),'backup':str(backup.relative_to(ROOT)),'reason':'Documentfourfixedcomponentablations,refitversusfrozenestimands,andkeep013'})
a.check();v={'passed':True,'candidate013_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-013.csv')),'candidate011_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'prior_reports_preserved':len(old),'packages_unchanged':len(p['environment']),'summary_sha256':e.sha(report),'metrics_table_sha256':e.sha(_repo_path(ROOT, 'reports/component-ablation-121-metrics.csv')),'sources_sha256':{name:e.sha(_repo_path(ROOT, name)) for name in ['component_ablation_121.py','validate_component_ablation_121.py','validate_component_ablation_v2_121.py','summarize_component_ablation_121.py','finalize_component_ablation_121.py']},'new_submission_CSV':False,'metrics_CSV_is_analysis_only':True,'full500_test_executed':False,'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/component-ablation-121-preservation.json'),v);e.emit(v)
