from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def read(n):return json.loads((_repo_path(R, n)).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
s=read('summary.json');v=read('validation.json');t=read('training-complete.json');p=read('plan.json');pres=read('preservation.json')
assert v['passed'] and read('score-validation.json')['passed'] and t['fits']==20 and v['verification_retrain_fits']==4 and s['original211_point_estimates_CIs_top5_exact']
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
for n,h in read('code-lock.json').items():assert sha(_repo_path(R, n))==h,n
resources={};peak=0;swap=0;seconds=0
for name in ['preflight','training','validation','score']:
 z=read(name+'-resources.json');mx=max(a['conservative_RSS_plus_MPS'] for a in z['trace']);sw=max(a['swap']-z['initial_swap'] for a in z['trace']);elapsed=z['trace'][-1]['seconds'];assert z['access']=={'raw_data_attempts':0,'outside_writes':0};peak=max(peak,mx);swap=max(swap,sw);seconds+=elapsed;resources[name]={'peak_RSS_plus_MPS_bytes':mx,'swap_growth':sw,'seconds':elapsed}
assert peak<6e9 and swap==0 and not list(R.glob('*STOP*')) and not list(R.rglob('*.csv'))
write(_repo_path(R, 'resources-summary.json'),{'stages':resources,'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':swap,'monitored_seconds':seconds,'cost':0})
rows=[]
for label,scope in [('원본 211',s['individual_partitions']['original']),('재분할 20261031',s['individual_partitions']['20261031']),('재분할 20261101',s['individual_partitions']['20261101']),('두 재분할 pooled',s['ID_cluster_aggregates']['two_repartitions']),('세 분할 pooled',s['ID_cluster_aggregates']['all_three_partitions'])]:
 for name,tag in [('standalone','CNN 단독'),('fixed_ensemble','고정 첫 seed 치환')]:
  z=scope[name];ci=z['conditional_CI95'];rows.append(f"| {label} | {tag} | {z['baseline_RMSE']:.8f} | {z['candidate_RMSE']:.8f} | {z['gain']:+.8f} | [{ci[0]:+.8f}, {ci[1]:+.8f}] | {z['top5_removed_gain']:+.8f} |")
trainrows=[];foldrows=[]
for split,d in s['individual_partitions'].items():
 for name in ['standalone','fixed_ensemble']:
  z=d[name];foldrows.append(f"- {split} {name}: 개선 fold {z['positive_folds']}/5, fold별 개선 {[round(a['gain'],8) for a in z['fold_results']]}.")
 for arm in ['control','huber']:
  z=s['train_validation'][split][arm];a=s['optimization_diagnostics'][split][arm];trainrows.append(f"| {split} | {arm} | {z['pooled_train_RMSE_over1600_repeated_train_observations']:.8f} | {z['OOF_RMSE']:.8f} | {z['OOF_minus_train_gap']:.8f} | {100*a['clip5_fraction']:.2f}% |")
report=f'''# 212: 고정 MSE–Huber 정책의 재분할 안정성

**{s['decision']}**

원본 211의 단독 평균 개선 +0.43933650, 95% CI [-0.02846487, +0.91677690] 및 사전 기준 미달 판단은 그대로다. 212는 이후 명시 의뢰된 두 기존 재분할의 안정성 확인이며 원본 판단을 소급 변경하지 않는다.

## 동일 정책과 실제 작업

MSE 대 2 × Huber(delta 1) 한 정책이다. 전이값은 fold train 타깃 표준편차 1개이며, 배율 2는 중앙 구간의 r²·기울기 2r·곡률 2를 MSE와 맞춘다. 전체 gradient norm이나 AdamW update 크기를 같게 맞추는 방식은 아니다. LR·clip·decay를 변경하지 않았고, 작은 최적화 손실값 자체를 개선으로 해석하지 않는다.

기존 20261031/20261101 각각 5-fold(320/80), seed 20261003 + fold, BOX 128, train 전용 입력·타깃 표준화, D4, AdamW lr 0.002/weight decay 0.01, cosine 80 epoch/eta_min 0.0001, batch 32, gradient clip 5, 마지막 5 epoch 동일 평균을 유지했다. 대비 감소·EMA·손실 탐색은 없다. Plan과 실행·평가 소스는 학습 전에 잠갔다.

새 주 학습 20회 및 각 재분할 fold 0 양 군 validation label poison 검증 4회, 총 24회/19,200 update다. 원본 211은 재학습하지 않았다.

## 대조 재현과 데이터 분리

- 10개 MSE 대조의 기존 checkpoint 및 209 마지막 평균 state·epoch 76–80 state·legacy FP32 OOF exact.
- 209의 모든 800 update 샘플 순서·D4·LR·loss·gradient norm·CPU/MPS RNG와 최종 optimizer/scheduler exact. 209에서 저장된 update별 RNG를 사용했으며, 201에 없는 과거 RNG 기록을 비교했다고 주장하지 않는다.
- 두 군의 초기 state, 모든 800 update 입력 hash·order·D4·LR·CPU/MPS RNG exact.
- 20개 모델의 마지막 5 state 독립 재평균 및 재로드 검증. MPS 최대 오차 {v['max_MPS_prediction_error']:.8g}, CPU 최대 오차 {v['max_CPU_prediction_error']:.8g}.
- 각 재분할 fold 0 양 군에서 validation 타깃을 NaN/1e30으로 바꾼 80 epoch replay의 state·OOF·stream·최종 optimizer exact. 전 fold train 정규화도 validation 타깃 변경에 불변.
- 기존 고정 앙상블 계보와 coefficient를 다시 구성해 저장된 201 결과와 exact임을 확인했다. 기술 검증 뒤 점수를 계산했다.

## 세 분할 및 400 ID 묶음 결과

개선량은 MSE RMSE − Huber RMSE다. 모든 분할은 같은 400개 ID다. Pooled는 ID별 분할 squared error 평균의 전체 평균에서 제곱근을 계산한다. 예측 평균 앙상블이 아니다.

Bootstrap은 400 ID cluster를 10,000회 재표집하며 같은 ID의 모든 분할 관측을 함께 포함한다. 1,200개 독립 표본으로 다루지 않았다. 211의 seed 202610211과 draw 배열을 그대로 사용해 원본 점추정·CI·상위 5개 제외 결과가 exact임을 검증했다.

| 분할 | 비교 | MSE RMSE | Huber RMSE | 개선량 | 조건부 95% CI | 상위 5개 ID 제외 개선 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

상위 5개 제외는 각 비교의 ID별 평균 squared error 이득이 가장 큰 5개 ID를 전체 반복 관측과 함께 제거한다. 원본 211의 상위 5개 ID를 고정 제외한 각 분할 및 pooled 감도도 summary.json에 별도로 보존했다.

{chr(10).join(foldrows)}

고정 앙상블은 Q = P021 + 0.081 × 기존 split/fold 보정 기울기 × (새 첫 seed CNN − 기존 첫 seed CNN)이다. CNN 3개 seed 중 첫 seed의 출력 치환이며, 새 성분을 추가한 blend가 아니다. 나머지 두 seed·보정·잔차는 고정했다. Control Q는 각 분할의 기존 021과 exact다. 전체 후단을 재학습한 효과로 해석하지 않는다.

## 학습·검증 오차와 판정

Train RMSE는 각 분할의 1,600개 중복 train 관측으로 계산했다. Clipping 비율은 증강 batch의 학습 경로 통계다. 모든 fold 및 공통 MSE·scaled Huber·gradient norm·꼬리 비율은 summary.json/streams.npz에 있다.

| 분할 | 손실 | Train RMSE | OOF RMSE | OOF−train | clip 5 비율 |
|---|---|---:|---:|---:|---:|
{chr(10).join(trainrows)}

추가 두 재분할 각각의 사전 단독 기준은 개선 ≥ 0.02, 개선 fold ≥ 3, CI 하한 > 0, 상위 5개 제외 개선 > 0이다. 두 재분할 및 세 분할 pooled에서는 개선·CI 하한·상위 5개 제외 개선이 모두 양수인지 확인한다. 실제 판정은 다음과 같다.

{json.dumps(s['stability_criteria'],ensure_ascii=False,indent=2)}

이 안정성 판정과 원본 211의 역사적 기준 미달은 함께 보존한다. 결과와 관계없이 안정성 판단 후 정지하며 전체 500장·후단 nested 재학습·제출을 자동 실행하지 않는다. Best021 유지.

## 한계와 보존

같은 개발 400개 ID를 반복 사용했다. CI는 저장된 학습 결과에 조건부이며 재학습·seed 변동·누적 선택 편향을 보정하지 않는다. 211 결과 이후 의뢰된 고정 정책 확인이며 새로운 독립 검증이 아니다.

기존 파일 {len(pres)}개의 SHA256 및 best021 해시가 모두 유지됐다. Best021 SHA256: {pres['artifacts/submission-candidate-021.csv']}.
CPU 2 threads/MPS, 최대 RSS+MPS {peak/1e9:.3f} GB, swap 증가 0, 감시 단계 합계 {seconds:.1f}초, 비용 0.

설치·다운로드·외부 AI·원시 train/holdout/test 읽기·전체 500장 학습·CSV·업로드·제출은 없다. 프로젝트 내부 새 212 코드·결과·관리표만 작성했다. 211·209·204의 기존 기록은 보존했다.

근거: plan.json, code-lock.json, preflight.json, training-complete.json, split/fold별 state와 streams, validation.json, score-validation.json, bootstrap.npz, summary.json, resources-summary.json.
'''
with (_repo_path(R, 'REPORT.md')).open('x') as f:f.write(report)
top={**s,'report':'experiments/huber-repartition-212/REPORT.md','protected_file_count':len(pres),'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':swap,'monitored_seconds':seconds}
write(_repo_path(P, 'reports/huber-repartition-212-summary.json'),top)
q=s['ID_cluster_aggregates']['all_three_partitions']
completion=f'''# 212 완료 관리표

{s['decision']}

| 항목 | 상태 | 범위 |
|---|---|---|
| 고정 손실 재분할 대조 | 완료 | 20261031/20261101 각각 5-fold, MSE 대 2 × Huber(delta 1), 원본 211 재학습 없음. |
| 원본 211 | 기존 판정 유지 | 평균 개선은 있지만 CI가 0 포함. 사전 기준 미달 사실과 모든 원본 보존. |
| 세 분할 단독 pooled | 진단 완료 | 개선 {q['standalone']['gain']:+.8f}, 400 ID cluster CI {q['standalone']['conditional_CI95']}. |
| 세 분할 고정 첫 seed 치환 | 진단 완료 | 개선 {q['fixed_ensemble']['gain']:+.8f}, CI {q['fixed_ensemble']['conditional_CI95']}. 추가 결합·후단 재학습 아님. |
| 재현 검증 | 통과 | 10개 MSE state/OOF/800 update RNG/optimizer exact; 20개 모델 재로드 및 poison replay 4회 통과. |
| 후속 | 정지 | 손실 탐색·전체 500장·자동 제출 없음. Best021 유지. |

[보고서](../experiments/huber-repartition-212/REPORT.md) · [구조화 결과](../reports/huber-repartition-212-summary.json) · [사전 계획](../experiments/huber-repartition-212/plan.json)

로드맵 범위 집계 완료 19개·부분 8개·미실시 3개는 유지한다. 211의 17번 완료는 수행 범위이며 성능 개선 확정을 의미하지 않는다. 기존 211 관리표를 덮어쓰지 않는다.
'''
with (_repo_path(P, 'docs/records/roadmap-audit/212-completion.md')).open('x') as f:f.write(completion)
write(_repo_path(P, 'docs/records/roadmap-audit/212-management.json'),{'study':212,'status':'완료','decision':s['decision'],'stability_criteria_passed':s['stability_criteria_passed'],'original211_status':'사전 기준 미달 유지','original211_retrained':False,'best021_maintained':True,'scope_counts':{'완료':19,'부분':8,'미실시':3},'EMA209_status':'보류','nested204_retry':False,'report':'experiments/huber-repartition-212/REPORT.md','new_training_running':False})
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
manifest={str(p.relative_to(P)):sha(p) for p in R.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ['artifact-manifest.json','completion.json']}
for p in [_repo_path(P, 'reports/huber-repartition-212-summary.json'),_repo_path(P, 'docs/records/roadmap-audit/212-completion.md'),_repo_path(P, 'docs/records/roadmap-audit/212-management.json')]:manifest[str(p.relative_to(P))]=sha(p)
write(_repo_path(R, 'artifact-manifest.json'),manifest);write(_repo_path(R, 'completion.json'),{'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'완료','protected_files':len(pres),'protected_unchanged':True,'report_SHA256':sha(_repo_path(R, 'REPORT.md')),'manifest_SHA256':sha(_repo_path(R, 'artifact-manifest.json')),'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'original211_unchanged':True,'cost':0,'remaining_blockers':[]})
print(json.dumps({'report':str(_repo_path(R, 'REPORT.md')),'decision':s['decision'],'peak_GB':peak/1e9,'protected_files':len(pres),'artifact_count':len(manifest)},ensure_ascii=False,indent=2))
