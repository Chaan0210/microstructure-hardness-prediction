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
s=read(_repo_path(R, 'summary.json'));v=read(_repo_path(R, 'validation.json'));t=read(_repo_path(R, 'training-complete.json'));plan=read(_repo_path(R, 'plan.json'));pres=read(_repo_path(R, 'preservation.json'));assert v['passed'] and t['fits']==5
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
resources={};peak=0;sw=0;seconds=0
for name in ['training','validation','score']:
 r=read(_repo_path(R, f'{name}-resources.json'));pk=max(z['conservative_RSS_plus_MPS'] for z in r['trace']);growth=max(z['swap']-r['initial_swap'] for z in r['trace']);peak=max(peak,pk);sw=max(sw,growth);seconds+=r['trace'][-1]['seconds'];assert r['access']=={'raw_data_attempts':0,'outside_writes':0};resources[name]={'peak_RSS_plus_MPS_bytes':pk,'swap_growth':growth,'seconds':r['trace'][-1]['seconds'],'access':r['access']}
assert peak<6e9 and sw==0 and not list(R.glob('*STOP*')) and not list(R.rglob('*.csv'))
write(_repo_path(R, 'resources-summary.json'),{'stages':resources,'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':sw,'monitored_seconds':seconds,'cost':0})
q=s['comparisons']['standalone'];f=s['comparisons']['fixed_ensemble'];rows=[]
for name,z in [('CNN 단독',q),('고정 앙상블 진단',f)]:rows.append(f"| {name} | {z['baseline_RMSE']:.10f} | {z['EMA_RMSE']:.10f} | {z['gain']:+.10f} | [{z['conditional_CI95'][0]:+.10f}, {z['conditional_CI95'][1]:+.10f}] | {z['positive_folds']}/5 | {z['top5_removed_gain']:+.10f} |")
maxstate=max(x['EMA_direct_state_max_error'] for x in t['folds']);maxpred=max(x['checks']['update_ema']['independent_float64_state_prediction_error'] for x in v['folds']);maxcpu=max(a['CPU_MPS_error'] for x in v['folds'] for a in x['checks'].values())
report=f'''# 208: 전체 optimizer update EMA 단일 정책 대조

**{s['decision']}** Decay·warmup·초기화·평가 시점을 바꾸거나 구간을 재탐색하지 않는다. Best021을 교체하지 않는다.

단독 CNN은 사전 기준을 모두 통과했다. 다섯 fold 모두 개선됐고 조건부 CI 하한 및 상위 5개 이득 제외 개선도 양수다. 고정 앙상블은 소폭 개선했으나 CI가 0을 포함하므로 그 이득은 불확실하다. 단독 기준 통과를 앙상블 채택이나 독립 검증 성공으로 확대 해석하지 않는다.

## 207과의 구분 및 사전 고정 정책

207은 저장된 epoch76–80의 다섯 checkpoint에 지수 가중치를 적용한 대조로 종료했다. 208은 원본 경로를 재실행하면서 **초기화부터 800개 optimizer update 전체를 누산한 EMA**다. 207 결과에서 decay를 선택하거나 유리한 epoch를 고르지 않았다.

- Decay는 **0.99**, warmup은 **0 update**다. 초기값 `E0 = θ0`는 무작위 초기 모델 파라미터의 독립 CPU float32 복사본이다.
- 각 `optimizer.step()`이 끝난 직후의 `θt`로만 `Et = 0.99 E(t−1) + 0.01 θt`를 800회 계산했다. Gradient·학습 모델·optimizer에 EMA를 되돌려 쓰지 않았다.
- Bias correction은 없다. 초기값의 최종 가중치는 `0.99^800 = {plan['EMA']['initial_weight_at800']:.12f}`다. 반감기는 약 {plan['EMA']['half_life_updates']:.2f} update다. 이 고정 시간 척도가 최적이라는 주장은 없다.
- 최종 평가 시점은 **800번째 update 직후 E800 하나**다. EMA checkpoint 평균, best epoch, early stopping, decay 탐색을 하지 않았다.
- 실행 검사에서 **BatchNorm 0개, GroupNorm 4개, named buffer 0개**였다. EMA는 학습 가능 파라미터만 다뤘고 running statistics 재보정 차이가 없다.

사전 계획은 `plan.json`에, 학습 및 평가 코드는 `code-lock.json`에 잠갔다. 원본 dev 400의 기존 5fold(320/80), seed 20261003+fold, 원본 BOX128 입력, train 전용 입력·타깃 정규화, D4, MSE, AdamW, cosine LR, batch32, 80 epoch를 유지했다. 206 대비 감소를 사용하지 않았다.

## 같은 학습 경로와 optimizer 검증

다섯 gradient 학습 경로만 실행했다. 각 경로에서 기존 마지막 5 epoch 동일 평균과 EMA를 동시에 보관했으므로, 두 결과 사이에 학습 데이터·RNG·optimizer 경로가 달라질 여지가 없도록 구성했다.

1. 모든 4,000 update의 샘플 순서·D4·LR·CPU/MPS RNG·손실·gradient norm이 저장된 206 control과 정확히 일치했다. 초기 state 및 각 fold의 epoch76–80 state, 기존 평균 모델, legacy float32 OOF도 bitwise exact였다.
2. 별도 forward/backward 없이 같은 clipped gradient를 받는 **검증용 AdamW 파라미터 복제본**을 뒀다. 모든 update에서 live와 복제본의 파라미터·`exp_avg`·`exp_avg_sq`·step counter·하이퍼파라미터를 정확히 비교했다. EMA 갱신 전후 live 모델·optimizer·CPU/MPS RNG 불변 검사도 4,000회 통과했다.
3. 기존 206은 optimizer checkpoint를 저장하지 않았다. 따라서 과거 optimizer 파일과 직접 비교했다고 주장하지 않는다. 기존 학습 스트림·state 재현과 이번 동일 gradient의 optimizer 복제 대조가 근거다.
4. 작업량은 **주 학습 5회·gradient update 4,000회**, 검증 복제 optimizer step 4,000회다. 복제본에는 별도 모델 forward/backward가 없으며 추가 gradient 학습은 0회다.

## EMA 및 데이터 분리 검증

독립 검산은 float64 닫힌 가중 합 `0.99^800 θ0 + Σ(0.01 × 0.99^(800−t) × θt)`으로 수행했다. 각 optimizer update 직후의 같은 θt를 사용했으며 새로운 평균 후보가 아니다. 실제 float32 EMA와 state 최대 차이는 {maxstate:.3g}, CPU 예측 최대 차이는 {maxpred:.3g}로 사전 기준 안이었다.

저장한 E0는 동일 seed의 초기 모델과, 최종 모델은 저장한 E800과 정확히 같았다. 마지막 5 state의 동일 평균도 별도로 재구성했다. MPS 재로드 오차는 0, CPU/MPS 예측 최대 차이는 {maxcpu:.3g}였다. 추론 중 state는 변하지 않았다.

Train/validation ID·group 분리와 320장 train 전용 정규화를 확인했다. Validation 타깃을 NaN으로 바꿔도 정규화는 같았다. 학습 중 validation RMSE를 계산하거나 checkpoint를 고르지 않았으며, 모든 검증이 끝난 뒤 아래 점수를 계산했다.

## 결과

개선량은 기존 마지막 5 epoch 평균 RMSE − EMA RMSE다. CI는 같은 400개 ID의 paired bootstrap 10,000회(seed202610208), 저장된 학습 결과에 조건부인 95% 구간이다.

| 비교 | 기존 평균 RMSE | 전체 update EMA RMSE | 개선량 | 조건부 95% CI | 개선 fold | 상위 5개 이득 제외 개선 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

단독 fold 개선량은 {q['fold_gains']}다. 고정 앙상블의 fold 개선량은 {f['fold_gains']}다.

사전 기준은 단독 개선 ≥ 0.02, 3/5 fold 이상 개선, 조건부 CI 하한 > 0, 상위 5개 이득 제외 개선 > 0 및 기술 검증 통과였다. 결과는 `{s['criteria']}`다. 이는 후속 자원 배분 기준이며 누적 선택 편향을 보정한 유의성 검정이 아니다.

고정 앙상블 식은 기존 194·206·207과 같은 `Q = P021 + 0.081 × 기존 fold 보정 기울기 × (새 첫 seed CNN − 기존 첫 seed CNN)`이다. 대조 Q는 best021과 정확히 같다. 다른 seed·보정·잔차를 고정한 **첫 seed 출력 치환 진단**이며 후단 전체 재학습 결과가 아니다.

## 한계와 보존

같은 개발 400장의 누적 재사용이며 독립 holdout이 아니다. 한 seed의 조건부 구간은 seed 분산·재학습 불확실성·누적 선택 편향을 보정하지 않는다. 결과는 decay0.99·warmup0·초기값θ0·최종E800의 한 정책에 한정한다. 모든 EMA 또는 가중치 평균 방식의 성능을 일반화하지 않는다. 실패 시 재탐색 없이 이 분기를 종료한다.

기존 파일 {len(pres)}개의 SHA256을 실행 전후 확인했다. Best021 CSV SHA256은 `{pres['artifacts/submission-candidate-021.csv']}`로 유지됐다. CPU2/MPS, 관측 최대 RSS+MPS {peak/1e9:.3f} GB, swap 증가 0, 감시 단계 합계 {seconds:.1f}초였다. 작성 시간을 포함한 전체 벽시계 시간은 아니다.

신규 설치·다운로드·외부 AI·raw train/holdout/test 파일 읽기·전체500 학습·새 CSV·제출·비용 발생은 없다. 허용된 개발400 기존 이미지·라벨·예측 캐시만 사용했다. 206과207은 종료 상태를 유지한다.

근거: `plan.json`, `training-complete.json`, fold별 `fit.json`, `streams.npz`, `optimizer-verification.pt`, EMA 시점별 state, `validation.json`, `OOF.npz`, `ensemble-OOF.npz`, `summary.json`, `resources-summary.json`.
'''
for old,new in [('epoch76–80','epoch 76–80'),('seed202610208','seed 202610208'),('decay0.99','decay 0.99'),('warmup0','warmup 0'),('초기값θ0','초기값 θ0'),('최종E800','최종 E800'),('CPU2','CPU 2'),('전체500','전체 500장'),('개발400','개발 400장의'),('206과207','206과 207')]:report=report.replace(old,new)
with (_repo_path(R, 'REPORT.md')).open('x') as out:out.write(report)
top={**s,'report':'experiments/update-ema-208/REPORT.md','preserved_file_count':len(pres),'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':sw,'monitored_stage_seconds':seconds,'historical_training_streams_exact':True,'optimizer_reference_checks':4000,'EMA_side_effect_checks':4000}
write(_repo_path(P, 'reports/update-ema-208-summary.json'),top)
completion=f'''# 208 완료 관리표

{s['decision']}

| 항목 | 상태 | 결과·범위 |
|---|---|---|
| 전체 update EMA | 완료 | Decay0.99, warmup0, E0=θ0, 매 update 직후 누산, 최종 E800. 한 정책만 실행. |
| 기존 평균 대조 | 정확히 재현 | 같은5fold 학습 경로에서 마지막5epoch 동일 평균 재구성. 이전 state·OOF·학습 스트림 exact. |
| Optimizer·RNG 통제 | 통과 | 동일 gradient의 AdamW 복제본 및 EMA 갱신 전후 상태 검사 각각4000회. 별도 gradient 학습 없음. |
| BatchNorm·buffer | 없음 | BatchNorm0, GroupNorm4, buffer0. Running statistics 재적합 없음. |
| 단독 OOF | 완료 | {q['baseline_RMSE']:.10f} → {q['EMA_RMSE']:.10f}, 개선 {q['gain']:+.10f}, CI {q['conditional_CI95']}, 상위5개 제외 {q['top5_removed_gain']:+.10f}. |
| 고정 앙상블 | 비선택 진단 | {f['baseline_RMSE']:.10f} → {f['EMA_RMSE']:.10f}, 개선 {f['gain']:+.10f}, CI {f['conditional_CI95']}, 상위5개 제외 {f['top5_removed_gain']:+.10f}. |
| 후속 | 정지 | 기준 충족 여부 {s['further_review_eligible']}. 기존 평균·best021 유지, decay·구간 재탐색 없음. |

[전체 보고서](../experiments/update-ema-208/REPORT.md) · [구조화 결과](../reports/update-ema-208-summary.json) · [사전 계획](../experiments/update-ema-208/plan.json)

207은 후반 checkpoint 지수 평균 대조였다. 기존 EMA 감사의 전체 update EMA 미실시 상태는 당시 기록으로 보존하고, 이번208에서 위 한 정책을 검증 완료했다. 206·207은 재개하지 않았다.

주 gradient 학습5회·4000update, 검증용 optimizer replay4000step. 기존 {len(pres)}개 파일과 best021 해시 유지. 새 설치·다운로드·holdout/test·CSV·제출·비용0.
'''
# Normalize compact technical shorthand in the user-facing Markdown only.
for old,new in [('seed202610208','seed 202610208'),('decay0.99','decay 0.99'),('warmup0','warmup 0'),('초기값θ0','초기값 θ0'),('최종E800','최종 E800'),('CPU2','CPU 2'),('전체500','전체 500장'),('개발400','개발 400장의'),('Decay0.99','Decay 0.99'),('같은5fold','같은 5fold'),('마지막5epoch','마지막 5 epoch'),('각각4000회','각각 4,000회'),('BatchNorm0','BatchNorm 0'),('GroupNorm4','GroupNorm 4'),('buffer0','buffer 0'),('상위5개','상위 5개'),('이번208','이번 208'),('학습5회','학습 5회'),('4000update','4,000 update'),('replay4000step','replay 4,000 step'),('비용0','비용 0')]:completion=completion.replace(old,new)
with (_repo_path(P, 'docs/records/roadmap-audit/208-completion.md')).open('x') as out:out.write(completion)
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
manifest={str(p.relative_to(P)):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','completion.json','finalize.log']}
for p in [_repo_path(P, 'reports/update-ema-208-summary.json'),_repo_path(P, 'docs/records/roadmap-audit/208-completion.md')]:manifest[str(p.relative_to(P))]=sha(p)
write(_repo_path(R, 'artifact-manifest.json'),manifest);write(_repo_path(R, 'completion.json'),{'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'완료','protected_files':len(pres),'protected_unchanged':True,'report_SHA256':sha(_repo_path(R, 'REPORT.md')),'manifest_SHA256':sha(_repo_path(R, 'artifact-manifest.json')),'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'cost':0,'remaining_blockers':[],'scientific_limits':s['limitations']})
print(json.dumps({'report':str(_repo_path(R, 'REPORT.md')),'decision':s['decision'],'peak_GB':peak/1e9,'protected_files':len(pres),'artifact_count':len(manifest)},ensure_ascii=False,indent=2))
