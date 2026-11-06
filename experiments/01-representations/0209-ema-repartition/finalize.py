from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
s=read(_repo_path(R, 'summary.json'));v=read(_repo_path(R, 'validation.json'));t=read(_repo_path(R, 'training-complete.json'));plan=read(_repo_path(R, 'plan.json'));pres=read(_repo_path(R, 'preservation.json'));protected={**pres,**plan['inputs_SHA256']};assert v['passed'] and t['fits']==10 and read(_repo_path(R, 'preflight.json'))['passed'] and read(_repo_path(R, 'score-validation.json'))['passed']
for n,h in protected.items():assert sha(_repo_path(P, n))==h,n
for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
assert plan['EMA']==read(_repo_path(P, 'experiments/update-ema-208/plan.json'))['EMA']
resources={};peak=0;sw=0;seconds=0
for name in ['preflight','training','validation','score']:
 r=read(_repo_path(R, f'{name}-resources.json'));pk=max(z['conservative_RSS_plus_MPS'] for z in r['trace']);growth=max(z['swap']-r['initial_swap'] for z in r['trace']);peak=max(peak,pk);sw=max(sw,growth);seconds+=r['trace'][-1]['seconds'];assert r['access']=={'raw_data_attempts':0,'outside_writes':0};resources[name]={'peak_RSS_plus_MPS_bytes':pk,'swap_growth':growth,'seconds':r['trace'][-1]['seconds'],'access':r['access']}
assert peak<6e9 and sw==0 and not list(R.glob('*STOP*')) and not list(R.rglob('*.csv'))
write(_repo_path(R, 'resources-summary.json'),{'stages':resources,'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':sw,'monitored_seconds':seconds,'cost':0})
def row(label,name,z):
 ci=z['conditional_CI95'];fold=f"{z['positive_folds']}/5" if 'positive_folds' in z else '해당 없음'
 return f"| {label} | {name} | {z['baseline_RMSE']:.10f} | {z['EMA_RMSE']:.10f} | {z['gain']:+.10f} | [{ci[0]:+.10f}, {ci[1]:+.10f}] | {fold} | {z['top5_removed_gain']:+.10f} |"
rows=[];aggrows=[];foldrows=[];sens=[];criteria=[]
for label,z in s['individual_partitions'].items():
 for name,k in [('CNN 단독','standalone'),('고정 첫 seed 치환','fixed_ensemble')]:
  q=z[k];rows.append(row(label,name,q));sens.append(f"| {label} | {name} | {q['original208_top5_excluded_gain']:+.10f} |")
 for a,b in zip(z['standalone']['fold_results'],z['fixed_ensemble']['fold_results']):foldrows.append(f"| {label} | {a['fold']} | {a['baseline_RMSE']:.8f} | {a['EMA_RMSE']:.8f} | {a['gain']:+.8f} | {b['baseline_RMSE']:.8f} | {b['EMA_RMSE']:.8f} | {b['gain']:+.8f} |")
for label,z in s['ID_cluster_aggregates'].items():
 for name,k in [('CNN 단독','standalone'),('고정 첫 seed 치환','fixed_ensemble')]:
  aggrows.append(row(label,name,z[k]));sens.append(f"| {label} | {name} | {z[k]['original208_top5_excluded_gain']:+.10f} |")
for label,q in s['stability_criteria'].items():criteria.append(f"- {label}: "+', '.join(f"{k}={'통과' if z else '미달'}" for k,z in q.items()))
maxstate=max(x['EMA_direct_state_max_error'] for x in t['folds']);maxpred=max(x['checks']['update_ema']['independent_float64_state_prediction_error'] for x in v['folds']);maxcpu=max(a['CPU_MPS_error'] for x in v['folds'] for a in x['checks'].values())
ens=s['ID_cluster_aggregates']['all_three_partitions']['fixed_ensemble'];ensemble_conclusion='고정 첫 seed 치환의 전체 세 분할 조건부 CI도 양수다.' if ens['conditional_CI95'][0]>0 else '고정 첫 seed 치환의 전체 세 분할 조건부 CI는 0을 포함하거나 음수이므로, 단독 CNN의 결과를 앙상블 이득의 확정으로 확대 해석하지 않는다.'
report=f'''# 209: 고정 전체 update EMA의 기존 재분할 안정성

**{s['decision']}** {ensemble_conclusion} Best021은 유지한다. 이 작업에서 새 제출이나 최종 평가 선택은 하지 않았다.

CNN 단독은 원본과 두 재분할 모두 점 추정치가 개선됐다. 그러나 20261031은 개선 +0.0557580의 CI가 [−0.0262294, +0.1382911]로 0을 포함해, 분할별 기준 중 이 한 항목이 미달했다. 20261101과 두 재분할·세 분할 pooled의 단독 기준은 통과했다. 이를 EMA 전반의 무효로 해석하지 않는다. 반면 고정 앙상블의 첫 seed 치환은 두 재분할에서 각각 −0.0038733, −0.0060012 악화했고, 세 분할 pooled도 −0.0020414였다. 단독 개선만으로 현재 앙상블 확대나 전체 nested 재학습을 정당화할 근거는 충분하지 않다.

## 고정 정책과 범위

208의 EMA 설정을 JSON 값까지 동일하게 유지했다. Decay **0.99**, warmup **0**, 초기값 **E0 = θ0**, CPU float32 독립 복사본에서 각 `optimizer.step()` 직후 `Et = 0.99 E(t−1) + 0.01 θt`를 누산하고 **800번째 update 직후 E800** 하나를 평가했다. Bias correction, 추가 checkpoint 평균, epoch·seed·decay 탐색은 없다. 기존 평균 대조는 같은 학습 경로의 epoch 76–80 state를 기존 순서대로 1/5씩 더했다.

기존 재분할 20261031·20261101에서 각각 5fold, 총 **10회 주 학습·8,000 gradient update**를 수행했다. 208 원본 결과는 재학습 없이 보존하여 포함했다. 같은 400개 ID, BOX128 입력, seed 20261003+fold, batch32, 80 epoch, train 전용 입력·타깃 표준화, D4, MSE, AdamW·cosine LR·clip 5를 유지했다. BatchNorm 0개, GroupNorm 4개, buffer 0개다.

## 사전 확인과 학습 경로 검증

- 학습 전에 두 재분할의 원래 021 checkpoint, 201 detach의 main state 및 legacy FP32 OOF가 **10개 모두 exact**임을 재확인했다. Fold별 ID·정규화·seed도 동일했다. 121/133/140/144/177의 예측·보정·잔차 계보를 재구성하여 best021과 정확히 일치시켰다.
- 8,000 update의 순서·D4·LR·MSE·main gradient norm이 과거 201과 정확히 같았다. 각 초기 state, 마지막 5 epoch main state, 평균 및 OOF도 exact였다. 저장된 **201 최종 main AdamW 파라미터별 moment·step·하이퍼파라미터와 scheduler 상태**도 직접 비교해 exact를 확인했다.
- 같은 clipped gradient를 받는 별도 AdamW 파라미터 복제본과 매 update live 파라미터·moment·step을 비교했고, EMA 갱신 전후 모델·optimizer·CPU/MPS RNG 불변도 각각 8,000회 확인했다. 복제본에는 별도 forward/backward가 없으므로 추가 gradient 학습은 **0회**다.
- 201은 update별 RNG 해시를 저장하지 않았다. 과거 RNG 직접 비교는 시작·종료 상태에 한정한다. 209는 모든 update의 RNG를 기록했으며 EMA가 RNG를 소비하지 않음을 확인했다. 과거 update별 RNG를 직접 비교했다는 주장은 하지 않는다.
- Validation 타깃을 NaN으로 바꿔도 train 정규화가 같았다. 학습 중 validation 점수에 따른 선택은 없었으며, 두 재분할의 모든 학습·검증을 마친 후 점수를 계산했다.

독립 float64 닫힌 가중 합 검산의 state 최대 차이는 **{maxstate:.3g}**, CPU 예측 차이는 **{maxpred:.3g}**였다. 저장 E0와 초기 모델, 최종 모델과 E800, 마지막 5 state의 재구성 평균은 exact였다. MPS 재로드 예측 오차는 0, CPU/MPS 최대 차이는 **{maxcpu:.3g}**로 사전 허용오차 이내다.

## 분할별 결과

개선량은 기존 평균 RMSE − EMA RMSE이며 양수가 개선이다. 아래 CI는 209에 사전 고정한 공통 paired ID bootstrap 10,000회, seed 202610209로 계산했다. 원본 208의 점 추정치와 제외 ID·제외 개선량은 exact로 보존됐다. 208의 당시 CI(seed 202610208)도 `summary.json`에 별도 저장했다.

| 분할 | 비교 | 기존 RMSE | EMA RMSE | 개선량 | 조건부 95% CI | 개선 fold | 상위 5개 이득 ID 제외 개선 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 같은 ID를 묶은 통계

세 분할은 **동일한 400개 ID의 반복측정**이다. Bootstrap에서 한 ID의 분할별 결과를 항상 함께 재표집했다. Pooled RMSE는 ID별 분할 squared error를 평균한 뒤 전체 ID 평균의 제곱근이다. 예측을 평균해 새 앙상블을 만든 수치가 아니며, 1,200개의 독립 관측 또는 독립 holdout으로 취급하지 않는다. Top 5는 해당 비교의 ID별 평균 squared error 개선이 큰 다섯 ID를 제외했으며, 해당 ID의 모든 분할 관측을 함께 뺐다.

| 집계 | 비교 | 기존 RMSE | EMA RMSE | 개선량 | ID cluster 조건부 95% CI | 개선 fold | 상위 5개 ID 제외 개선 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(aggrows)}

조건부 CI는 저장된 학습 결과의 ID 표집 불확실성만 다룬다. 누적된 같은 400장의 선택 편향, seed 변동, 재학습 불확실성을 보정하지 않는다. 사전 분할 안정성 기준은 각 신규 분할에서 208과 같은 단독 개선 ≥ 0.02·3/5 fold 이상 개선·CI 하한 > 0·상위 5개 제외 개선 > 0, 두 재분할 및 세 분할 pooled에서 개선·CI 하한·상위 5개 제외 개선 모두 > 0이었다. 이는 자원 배분 기준이며 선택 편향을 보정한 확증 검정이 아니다.

{chr(10).join(criteria)}

## 전체 fold

| 분할 | Fold | CNN 기존 | CNN EMA | CNN 개선 | 고정 치환 기존 | 고정 치환 EMA | 고정 치환 개선 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(foldrows)}

208 원본에서 정한 상위 5개 ID를 모든 분할에 동일하게 제외한 별도 감도는 다음과 같다. 각 분할에서 유리한 제외 ID를 새로 고른 감도와 구분한다. 정확한 ID 목록은 `summary.json`에 저장했다.

| 분할·집계 | 비교 | 원본 208의 다섯 ID 제외 개선 |
|---|---|---:|
{chr(10).join(sens)}

## 고정 앙상블 해석과 종료

식은 `Q = P021 + 0.081 × 기존 split/fold 보정 기울기 × (EMA 첫 seed CNN − 기존 첫 seed CNN)`이다. 대조 출력의 Q는 best021과 exact다. **기존 CNN 3seed 중 첫 seed 출력 하나를 치환**했으며, 새 CNN을 추가한 blend가 아니다. 나머지 두 seed·다른 모델·calibrator·잔차 선택과 가중치는 그대로 뒀다. 학습 내부의 CNN 예측을 바꾸어 보정·잔차를 전부 재적합한 결과가 아니므로, 전체 nested 재학습의 효과나 최종 제출 개선으로 단정하지 않는다.

이번 범위는 분할 안정성 확인까지다. 전체 nested 재학습 여부는 이 근거를 보고 별도로 판단한다. 새 decay·seed·epoch 탐색과 후단 재학습을 시작하지 않았다. 204의 CNN 3→6 seed 기각 분기를 반복하지 않았다.

기존 파일·입력 **{len(protected)}개**의 SHA256을 실행 전후 확인했다. Best021 CSV SHA256은 `{protected['artifacts/submission-candidate-021.csv']}`다. 원본 208 산출물과 기존 관리표도 보존했다. CPU 2개, 기존 uv·로컬 MPS만 사용했으며 관측 최대 RSS+MPS는 **{peak/1e9:.3f} GB**, swap 증가 **0**, 감시 단계 합계 **{seconds:.1f}초**였다. 작성 시간을 포함한 전체 벽시계 시간은 아니다.

비용은 0이다. 설치·다운로드·외부 AI·raw train/holdout/test 접근·전체 500장 학습·신규 CSV·제출은 없었다. 기존 개발 400장 캐시만 사용했다.

근거: `plan.json`, `preflight.json`, `training-complete.json`, 각 fold의 `fit.json`·`streams.npz`·`optimizer-verification.pt`·EMA state, `validation.json`, `score-validation.json`, 분할별 `OOF.npz`·`ensemble-OOF.npz`, `summary.json`, `resources-summary.json`.
'''
with (_repo_path(R, 'REPORT.md')).open('x') as f:f.write(report)
write(_repo_path(R, 'interpretation.json'),{'decision':s['decision'],'ensemble_interpretation':ensemble_conclusion,'best021_maintained':True,'whole_nested_retraining_started':False,'new_policy_search_started':False,'new_submission':False,'remaining_blockers':[],'scope_complete':True})
top={**s,'report':'experiments/ema-repartition-209/REPORT.md','protected_file_count':len(protected),'best021_SHA256':protected['artifacts/submission-candidate-021.csv'],'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':sw,'all10_historical_weights_OOF_final_optimizer_exact':True}
write(_repo_path(P, 'reports/ema-repartition-209-summary.json'),top)
q=s['ID_cluster_aggregates']['all_three_partitions']['standalone'];f=s['ID_cluster_aggregates']['all_three_partitions']['fixed_ensemble']
management=f'''# 209 완료 관리표

{s['decision']}

| 항목 | 상태 | 결과·남은 질문 |
|---|---|---|
| 208 고정 전체 update EMA | 재분할 검증 완료 | Decay 0.99, warmup 0, E0 = θ0, 매 update 직후 누산, 최종 E800. 설정 변경 없음. |
| 기존 두 재분할 | 완료 | 각 5fold, 신규 주 학습 총 10회·8,000 update. 원본 208은 재학습 없이 포함. |
| 기존 CNN 및 optimizer 재현 | 통과 | 10개 가중치·OOF·최종 main optimizer·scheduler exact. 과거 RNG는 시작·종료만 직접 비교 가능. |
| CNN 단독 세 분할 pooled | 완료 | 개선 {q['gain']:+.10f}, ID cluster CI {q['conditional_CI95']}, 상위 5개 제외 {q['top5_removed_gain']:+.10f}. |
| 고정 첫 seed 치환 | 진단 완료 | 개선 {f['gain']:+.10f}, ID cluster CI {f['conditional_CI95']}, 상위 5개 제외 {f['top5_removed_gain']:+.10f}. 추가 blend가 아님. |
| 전체 nested 재학습 | 미실시 | 기존 보정·잔차는 고정했다. 전체 재적합의 효용은 아직 알 수 없으며 별도 판단 대상이다. |
| 최종 평가·제출 | 미실시 | Best021과 기존 CSV·모든 보호 파일 보존. 비용 0. |

같은 400개 ID를 세 번 사용했다. 독립 holdout 또는 1,200개 독립 표본이 아니며, 조건부 CI는 누적 선택 편향과 재학습·seed 변동을 보정하지 않는다. 새로운 decay·seed·epoch 탐색이나 후단 학습을 시작하지 않았다.

[전체 보고서](../experiments/ema-repartition-209/REPORT.md) · [구조화 결과](../reports/ema-repartition-209-summary.json) · [사전 계획](../experiments/ema-repartition-209/plan.json)
'''
with (_repo_path(P, 'docs/records/roadmap-audit/209-completion.md')).open('x') as out:out.write(management)
for n,h in protected.items():assert sha(_repo_path(P, n))==h,n
manifest={str(p.relative_to(P)):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','completion.json','finalize.log']}
for p in [_repo_path(P, 'reports/ema-repartition-209-summary.json'),_repo_path(P, 'docs/records/roadmap-audit/209-completion.md')]:manifest[str(p.relative_to(P))]=sha(p)
write(_repo_path(R, 'artifact-manifest.json'),manifest);write(_repo_path(R, 'completion.json'),{'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'완료','protected_files':len(protected),'protected_unchanged':True,'report_SHA256':sha(_repo_path(R, 'REPORT.md')),'manifest_SHA256':sha(_repo_path(R, 'artifact-manifest.json')),'best021_SHA256':protected['artifacts/submission-candidate-021.csv'],'cost':0,'remaining_blockers':[],'scientific_limits':s['limitations']})
print(json.dumps({'report':str(_repo_path(R, 'REPORT.md')),'decision':s['decision'],'peak_GB':peak/1e9,'protected_files':len(protected),'artifact_count':len(manifest),'monitored_seconds':seconds},ensure_ascii=False,indent=2))
