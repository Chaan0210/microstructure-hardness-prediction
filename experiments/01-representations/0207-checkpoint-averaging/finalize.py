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
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
s=read(_repo_path(R, 'summary.json'));v=read(_repo_path(R, 'validation.json'));plan=read(_repo_path(R, 'plan.json'));pres=read(_repo_path(R, 'preservation.json'));assert v['passed']
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
r=[read(_repo_path(R, f'{name}-resources.json')) for name in ['inference','score']];peak=max(z['conservative_RSS_plus_MPS'] for x in r for z in x['trace']);swap=max(z['swap']-x['initial_swap'] for x in r for z in x['trace']);seconds=sum(x['trace'][-1]['seconds'] for x in r);assert peak<=6e9 and swap==0
assert all(x['access']=={'raw_data_attempts':0,'outside_writes':0} for x in r);assert not list(R.rglob('*.csv')) and not list(R.glob('*STOP*'))
q=s['comparisons']['standalone'];f=s['comparisons']['fixed_ensemble'];rows=[]
for name,z in [('CNN 단독',q),('고정 앙상블 진단',f)]:rows.append(f"| {name} | {z['control_RMSE']:.10f} | {z['candidate_RMSE']:.10f} | {z['gain']:+.10f} | [{z['conditional_CI95'][0]:+.10f}, {z['conditional_CI95'][1]:+.10f}] | {z['positive_folds']}/5 | {z['top5_removed_gain']:+.10f} |")
maxstate=max(x['EMA_float64_direct_state_error'] for x in v['folds']);maxpred=max(x['EMA_independent_state_prediction_error'] for x in v['folds']);maxcpu=max(y['CPU_MPS_max_error'] for x in v['folds'] for y in x['reload'].values());sourcecount=read(_repo_path(R, 'duplicate-audit.json'))['code_files_searched']
audit=f'''# EMA와 checkpoint 평균 중복 감사

**현재 마지막 5 epoch 평균은 EMA와 다르다. 동일 학습 경로에서 평균 정책만 바꾼 기존 대조는 찾지 못했으며, 207에서 저장 checkpoint만 사용한 한 가지 후반 지수 평균 대조를 완료했다.** 206 대비 감소는 종료 상태로 유지하고 그 후보 checkpoint는 사용하지 않았다.

| 항목 | 상태·근거 | 중복 판단과 남은 범위 |
|---|---|---|
| 마지막 5 epoch 동일 평균 | 완료. `frozen_validation.py`의 `epoch >= 75`에서 `state/5`를 누적. 76–80 epoch 각각 20%. 009·129·133·193·204·206에 적용. | 기존 기본 정책이다. 지수 감쇠가 없어 EMA가 아님. |
| 여러 seed의 예측 평균 | 완료. 133은 각 seed 안의 마지막 5 state 평균 후 3개 seed의 예측을 동일 평균. 204는 3→6 seed와 후단 재학습으로 pooled 개선 0.005785, CI [-0.026274, 0.037320] 및 두 재분할 악화로 기각. | 서로 다른 학습 경로의 예측 평균이며 한 경로의 checkpoint EMA와 다름. 반복하지 않음. |
| 평균 시점과 학습 예산 | 완료. 128은 5개 정규화된 후반 시점을 동일 평균하되 update 예산을 변경. 작은 표본 400→800 update RMSE 15.418684→15.597598. | 평균 정책 단독 대조가 아님. |
| 최종 state 대 평균 state | 부분. 131/132는 마지막 state와 기존 평균을 사용한 경로가 다르지만, 2-view batching·LR 계단·readout·seed 등도 달라짐. `mse_readout_132.py`에 단일 schedule 대조가 아님을 명시. | 평균 방식만의 효과로 해석할 수 없음. 동일 대조를 했다고 간주하지 않음. |
| 같은 마지막 5 state의 지수 평균 | **207 완료.** 206 control의 동일 경로 checkpoint 25개를 재사용. 학습 0. 단독 개선 {q['gain']:+.10f}, 고정 앙상블 개선 {f['gain']:+.10f}; 두 CI 모두 0 포함, 상위 5개 이득 제외 시 둘 다 악화. | 하나의 사전 정책만 검증했고 기준 미달로 종료. decay·구간 탐색 없음. |
| 매 update 또는 전체 학습 EMA | 미실시. 현재 원래 경로의 모든 800 update state와 epoch1–75 state가 저장되지 않음. | 207을 전체 EMA 검증으로 부르지 않음. 이 범위의 정확한 대조에는 별도 재실행이 필요. |

최상위 및 실험별 Python 코드 {sourcecount}개를 명칭·API로 검색하고 관련 학습 루프와 계획을 읽었다. 명칭 검색만으로 임의의 수제 구현 부재를 증명하지는 않으므로 조사 범위를 명시한다. 기본 코드의 동일 평균, 133·128·132의 실제 대조 내용과 206의 저장 자료를 함께 확인했다.

현재 원본 경로의 `artifacts/run-009/fold-*.pt`에는 평균 state만 있다. 206 control에는 fold마다 `epoch-76.pt`부터 `epoch-80.pt`까지와 최종 평균 모델이 있다. 207에서 25개 state hash를 검증하고 동일 평균을 재구성하여 206 및 run009와 bitwise exact임을 확인했다. 따라서 마지막 5 state 범위의 비교에는 새 학습이 필요하지 않았다.

전체 update EMA를 별도로 요청하면 최소한 첫 seed의 원본 5fold를 동일 800 update씩 재실행해야 한다. 필요량은 **5회 학습·4,000 update·128,000 image presentation**이다. 재실행 중 EMA를 별도 누산하고 마지막 5 state 동일 평균도 함께 보관하면 학습 경로를 같게 유지할 수 있다. 전처리·seed·D4·loss·optimizer·80 epoch를 고정하고 decay 하나를 결과 전에 정하며 early stopping·epoch 선택을 하지 않는 설계가 필요하다. 이번에는 실행하거나 예약하지 않았다. 기존 3개 seed 전체를 바꾸는 질문은 최소 15개 outer 학습이 필요하며, 후단 전체 재학습은 또 다른 범위다.

[207 전체 보고서](../experiments/checkpoint-averaging-207/REPORT.md), [중복 조사 상세](../experiments/checkpoint-averaging-207/duplicate-audit.json), [기존 평균 구현](../frozen_validation.py), [133](../cnn_seed_mean_133.py), [128](../cnn_update_budget_128.py), [132](../mse_readout_132.py).
'''
with (_repo_path(P, 'docs/records/roadmap-audit/ema-averaging-audit.md')).open('x') as out:out.write(audit)
report=f'''# 207: 저장된 마지막 5 checkpoint의 평균 정책 대조

**{s['decision']}** 단독과 고정 앙상블의 점추정은 소폭 개선했지만 CI가 0을 포함하고 상위 5개 이득을 제외하면 모두 악화했다. decay·평균 구간·checkpoint 탐색 없이 종료한다.

## 고정한 비교

기존 코드의 동일 평균은 `U = (θ76 + θ77 + θ78 + θ79 + θ80)/5`이다. 지수 평균은 과거 state의 가중치가 기하급수적으로 감소하므로 다른 정책이다.

이번 후보 하나는 `m75 = 0`, `m_e = 0.5 × m_(e−1) + 0.5 × θe`를 e=76…80에만 적용한 뒤 `E = m80/(1−0.5^5)`로 정규화했다. 최종 가중치는 76…80 순서로 **[1, 2, 4, 8, 16]/31**이다. 한 checkpoint 지연마다 가중치를 절반으로 낮추는 간단한 정책을 점수 계산 전에 잠갔다. 최적 decay라는 근거는 없으며 후보를 더 만들지 않았다.

이는 **마지막 5 epoch에 한정한 정규화 지수 평균**이다. 매 optimizer update EMA 또는 80 epoch 전체를 누산한 EMA가 아니다. 개별 epoch의 예측·성능은 계산하지 않았고 최종 epoch 선택이나 early stopping도 없다.

206의 **원래 control 경로**에서 저장한 25개 checkpoint를 사용했다. 원본 5fold, seed 20261003+fold, 기존 BOX 128 입력, train 전용 정규화, D4·MSE·AdamW·80 epoch 학습 경로가 완전히 같다. 206 대비 감소 후보를 재사용하지 않는다. 새로운 gradient 학습과 optimizer update는 모두 0이다.

## 검증

- 25개 state hash와 fold의 320/80 ID·group 분리, train 전용 정규화 및 seed를 확인했다. 동일 평균의 float32 연산을 재현하여 206 control과 run009의 state가 모두 bitwise exact였다. 다섯 fold의 OOF 역시 기존 첫 seed의 legacy float32 예측과 정확히 같았다.
- 지수 평균 재귀를 float64 직접 가중 합으로 별도 확인했다. State 최대 오차 {maxstate:.3g}, 독립 구성 state의 CPU 예측 차이 최대 {maxpred:.3g}로 사전 기준 안이었다. 모든 값은 유한하며 구성 state의 성분별 최소·최대 안에 있었다.
- GroupNorm 모델에 running-stat buffer가 없음을 확인해 BN 통계 재적합을 하지 않았다. 모델은 eval·gradient 비활성 상태로 추론했다. MPS 재로드 오차 0, CPU/MPS 예측 최대 오차 {maxcpu:.3g}. 추론 전후 state는 불변이었다.
- Validation 타깃을 NaN으로 바꿔도 train 정규화가 같았다. 가중치 평균 함수는 타깃을 사용하지 않는다. 검증 완료 후 아래 점수를 계산했다.

## 결과

개선량은 기존 동일 평균 RMSE − 고정 지수 평균 RMSE다. CI는 같은 400개 ID의 paired bootstrap 10,000회(seed 202610207), 저장된 학습 결과에 조건부인 구간이다.

| 비교 | 동일 평균 RMSE | 지수 평균 RMSE | 개선량 | 조건부 95% CI | 개선 fold | 상위 5개 이득 제외 개선 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

단독 fold 개선량은 {q['fold_gains']}이며, 고정 앙상블은 {f['fold_gains']}이다.

사전 단독 판정 기준은 개선 ≥ 0.02, 3/5 fold 이상 개선, CI 하한 > 0, 상위 5개 이득 제외 개선 > 0 및 기술 검증 통과였다. 실제 결과는 `{s['criteria']}`이다. 자원 배분 기준이며 누적 선택 편향을 보정한 유의성 검정이 아니다.

앙상블 진단은 194·206과 같은 `Q = P021 + coef_fold × (새 첫 seed CNN − 기존 첫 seed CNN)`, `coef_fold = 0.081 × 기존 fold 보정 기울기`를 고정했다. 기존 평균을 대입한 Q는 best021과 정확히 같다. 다른 seed·보정·잔차를 고정한 출력 치환으로, **후단 전체 재학습의 성능이 아니다.** 후보의 작은 개선을 근거로 best021을 교체하지 않는다.

## 범위와 남은 질문

한 seed의 같은 개발 400장에 대한 조건부 대조다. Seed 분산·재학습 불확실성·반복 선택 편향을 반영하지 않으며 독립 holdout 검증이 아니다. 이번 결과는 이 후반 5개 state 정책만 다루며 전체 EMA의 성능을 결론내리지 않는다.

전체 update EMA는 이전 state가 없어 재구성할 수 없다. 최소한 첫 seed 원본5fold의 결정론적 5회·4,000 update 재실행과 online EMA 누산이 필요하다. 지금 실행하지 않았으며 상세 범위는 [중복 감사](../../roadmap-audit/ema-averaging-audit.md)에 적었다.

기존 파일 {len(pres)}개의 SHA256을 보존했다. Best021 CSV SHA256은 `{pres['artifacts/submission-candidate-021.csv']}`다. 무료 CPU 2 threads/MPS, 최대 RSS+MPS {peak/1e9:.3f} GB, swap 증가 0, 감시한 추론·점수 계산 단계 합계 {seconds:.1f}초였다. 신규 학습·설치·다운로드·holdout/test·새 CSV·제출·비용 발생은 없다. 개발 400장의 기존 캐시만 읽었으며 206은 종료 상태로 유지했다.

근거: `plan.json`, `duplicate-audit.json`, `validation.json`, `OOF.npz`, `summary.json`, `bootstrap.npz`, fold별 저장 평균 모델과 검증 기록.
'''
with (_repo_path(R, 'REPORT.md')).open('x') as out:out.write(report)
top={**s,'report':'experiments/checkpoint-averaging-207/REPORT.md','best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'preserved_file_count':len(pres),'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':swap,'monitored_stage_seconds':seconds,'saved_checkpoints_reused':25,'uniform_control_exact_all5':True}
write(_repo_path(P, 'reports/checkpoint-averaging-207-summary.json'),top)
completion=f'''# 207 완료 관리표

{s['decision']}

| 항목 | 상태 | 근거·결과 |
|---|---|---|
| 마지막 5 epoch 동일 평균과 EMA 구분 | 완료 | 기존 가중치 1/5씩. 207은 같은 state에 [1,2,4,8,16]/31. 전체 update EMA와 구분. |
| 중복 확인 | 완료 | 133 seed 예측 평균, 128 학습 예산, 131/132 학습 경로 대조와 다름. 독립 평균 정책 대조는 찾지 못함. |
| 추가 학습 없는 비교 | 완료 | 206 control의 checkpoint 25개 재사용. 새로운 학습 0, 개별 epoch 점수 0, 후보 정책 1개. |
| 원본 5fold 단독 | 기준 미달 | {q['control_RMSE']:.10f} → {q['candidate_RMSE']:.10f}. 개선 {q['gain']:+.10f}, CI {q['conditional_CI95']}, 상위 5개 제외 {q['top5_removed_gain']:+.10f}. |
| 고정 앙상블 진단 | 비선택 보고 | {f['control_RMSE']:.10f} → {f['candidate_RMSE']:.10f}. 개선 {f['gain']:+.10f}, CI {f['conditional_CI95']}, 상위 5개 제외 {f['top5_removed_gain']:+.10f}. |
| 후속 | 정지 | 기존 동일 평균·best021 유지. Decay·구간 탐색 없음. 전체 EMA는 필요한 학습량만 보고하고 미실행. |

[전체 보고서](../experiments/checkpoint-averaging-207/REPORT.md) · [중복 감사](ema-averaging-audit.md) · [구조화 결과](../reports/checkpoint-averaging-207-summary.json)

기존 {len(pres)}개 파일과 best021 해시 보존. 신규 학습·설치·다운로드·holdout/test·CSV·제출·비용 0. 206은 종료 상태다.
'''
with (_repo_path(P, 'docs/records/roadmap-audit/207-completion.md')).open('x') as out:out.write(completion)
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
manifest={str(p.relative_to(P)):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','completion.json','finalize.log']}
for n in ['reports/checkpoint-averaging-207-summary.json','docs/records/roadmap-audit/ema-averaging-audit.md','docs/records/roadmap-audit/207-completion.md']:manifest[n]=sha(_repo_path(P, n))
write(_repo_path(R, 'artifact-manifest.json'),manifest);write(_repo_path(R, 'completion.json'),{'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'완료','report_SHA256':sha(_repo_path(R, 'REPORT.md')),'manifest_SHA256':sha(_repo_path(R, 'artifact-manifest.json')),'protected_files':len(pres),'protected_unchanged':True,'new_training':0,'cost':0,'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'remaining_execution_blockers':[],'remaining_scientific_scope':'전체 update EMA는 저장 경로 부족으로 미평가. 필요한 최소 재실행 설계만 보고.'})
print(json.dumps({'report':str(_repo_path(R, 'REPORT.md')),'decision':s['decision'],'peak_GB':peak/1e9,'preserved_files':len(pres),'artifacts':len(manifest),'new_training':0},ensure_ascii=False,indent=2))
