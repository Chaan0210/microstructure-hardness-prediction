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
s=read(_repo_path(R, 'summary.json'));v=read(_repo_path(R, 'validation.json'));pres=read(_repo_path(R, 'preservation.json'));plan=read(_repo_path(R, 'plan.json'));t=read(_repo_path(R, 'training-complete.json'));assert v['passed'] and read(_repo_path(R, 'score-validation.json'))['passed'] and read(_repo_path(R, 'control-validation.json'))['passed'] and t['candidate_fits']==5
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
for fname in ['code-lock.json','analysis-code-lock.json']:
 for n,h in read(_repo_path(R, fname)).items():assert sha(_repo_path(R, n))==h,n
resources={};peak=0;seconds=0
for stage in ['preflight','control','training','validation','score']:
 r=read(_repo_path(R, f'{stage}-resources.json'));pk=max(x['conservative_RSS_plus_MPS'] for x in r['trace']);sw=max(x['swap']-r['initial_swap'] for x in r['trace']);assert sw==0 and pk<6e9 and r['access']=={'raw_data_attempts':0,'outside_writes':0};sec=r['trace'][-1]['seconds'];resources[stage]={'peak_RSS_plus_MPS_bytes':pk,'seconds':sec,'swap_growth_bytes':sw,'access':r['access']};peak=max(peak,pk);seconds+=sec
assert resources['control']['seconds']+resources['training']['seconds']<900 and not list(R.rglob('*STOP*')) and not list(R.rglob('*.csv'))
write(_repo_path(R, 'resources-summary.json'),{'stages':resources,'max_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':0,'monitored_seconds':seconds,'cost':0})
co=s['comparisons'];tr=s['training_generalization_gap'];q=co['primary_uniform_vs_LLRD'];en=co['best021_vs_LLRD_blend'];labels={'primary_uniform_vs_LLRD':'Uniform → LLRD 단독','frozen100_vs_LLRD':'Frozen100 → LLRD 단독','head_only112_vs_LLRD':'Head-only112 → LLRD 단독','fixed_blend_uniform_vs_LLRD':'Uniform 10% 결합 → LLRD 10% 결합','best021_vs_LLRD_blend':'Best021 → LLRD 10% 결합'}
rows=[];concentration=[]
for name,z in co.items():
 ci=z['conditional_CI95'];rows.append(f"| {labels[name]} | {z['baseline_RMSE']:.10f} | {z['candidate_RMSE']:.10f} | {z['gain']:+.10f} | [{ci[0]:+.10f}, {ci[1]:+.10f}] | {z['positive_folds']}/5 | {z['top5_removed_gain']:+.10f} |")
 concentration.append(f"| {labels[name]} | {z['positive_ID_count']}/400 | {100*z['top5_share_of_positive_squared_error_improvement']:.2f}% | {', '.join(z['top5_removed_IDs'])} |")
foldrows=[]
for a,b in zip(tr['folds'],en['folds']):foldrows.append(f"| {a['fold']} | {a['baseline_train_RMSE']:.8f} | {a['candidate_train_RMSE']:.8f} | {a['baseline_OOF_RMSE']:.8f} | {a['candidate_OOF_RMSE']:.8f} | {a['OOF_gain']:+.8f} | {a['baseline_gap']:.8f} | {a['candidate_gap']:.8f} | {b['gain']:+.8f} |")
curve='\n'.join(f"| {ep} | {tr['baseline_train_curve'][str(ep)]:.8f} | {tr['candidate_train_curve'][str(ep)]:.8f} |" for ep in [0,3,6,9,12])
criteria='\n'.join(f"| {label} | {'통과' if all(z.values()) else '미달'} | "+', '.join(f"{k}: {'통과' if val else '미달'}" for k,val in z.items())+' |' for label,z in [('LLRD 대조',s['primary_criteria']),('최종 앙상블',s['ensemble_criteria'])])
if q['gain']<=0:
 gaptext='Train 오차는 더 낮아졌지만 OOF는 악화했고, train/OOF 격차도 더 커졌다. 이번 고정 LLRD가 과적합 격차를 줄였다는 근거는 없다.'
elif not s['primary_passed']:
 gaptext='OOF 점 추정치는 개선됐지만 사전 primary 기준을 충족하지 못했다. 격차 감소나 train 오차 증가만으로 일반화 개선이 입증됐다고 주장하지 않는다.'
else:gaptext='OOF 개선의 사전 기준은 충족했다. 다만 train 오차 변화와 OOF 개선을 분리하며, 같은 400장의 선택 편향을 보정한 확증 결과로 해석하지 않는다.'
maxcpu=max(x['CPU_MPS_max_error'] for x in v['folds']);recovery=s['recovery_diagnostics']
report=f'''# 210: ConvNeXt 마지막 세 블록의 고정 LLRD 대조

**{s['decision']}** 209 EMA는 보류 상태를 유지한다. 이번에는 EMA, 새 층, 입력 변경, 증강, loss 변경을 추가하지 않았다.

단독 Uniform 대비 개선은 **{q['gain']:+.10f}**, 조건부 95% CI는 **{q['conditional_CI95']}**다. Best021 대비 고정 10% 추가 결합 개선은 **{en['gain']:+.10f}**, CI는 **{en['conditional_CI95']}**다. 같은 모델의 OOF 개선과 현재 앙상블에 추가할 효용을 별도로 판정했다.

## 정책과 정확한 통제

학습 전 [기존 계획](../../roadmap-audit/next-experiment-210.md), `plan.json`, 실행 코드 해시를 고정했다. Ratio **0.8 하나**를 사용했다. 대조의 세 블록 LR은 모두 1e-5, 후보의 초기 LR은 순서대로 **7.868852459e-6 / 9.836065574e-6 / 1.229508197e-5**다. 마지막 stage 내부 blockwise LLRD이며 전체 backbone의 LLRD 대조가 아니다.

각 블록은 4,763,136개 파라미터다. `(0.8², 0.8, 1)`을 그 평균으로 정규화해 세 블록의 평균 LR을 맞췄다. Terminal LayerNorm 1,536개의 LR은 1e-5, head 49,281개의 LR은 1e-4로 유지했다. Tail 합계는 14,290,944개다. 각 파라미터는 한 그룹에만 포함됐으며 기존 파라미터 순서를 유지했다. 모든 1,440 update에서 그룹별 LR·평균 LR·cosine schedule을 확인했다.

**실제 weight decay factor도 동일하게 유지했다.** Optimizer의 내장 weight decay는 0이고, step 직전에 head는 `1 − 1e-4 × f(epoch) × 0.01`, 모든 tail 파라미터는 `1 − 1e-5 × f(epoch) × 0.01`을 곱했다. 후보 그룹의 변경된 LR을 이 축소식에 사용하지 않았다. 실제 factor를 update별로 저장하고 독립적으로 재계산했다. Nominal 평균 LR을 맞춘 것이 실제 AdamW 파라미터 이동량까지 같게 만든다는 뜻은 아니다.

기존 ConvNeXt-Tiny 로컬 가중치·native 256의 동결 prefix 캐시·MLP head·fold별 warm head·train RMS scaler·target 표준화를 재사용했다. AdamW β=(0.9,0.999), eps=1e-8, SmoothL1 beta 1, global clip 1, dropout 0.25, microbatch 8/effective batch 16, 12 epoch·240 update, 기존 seed·배치 순서를 유지했다. Backbone은 eval, head만 train이며 BN과 running buffer는 없다. Terminal LayerNorm affine과 마지막 세 블록만 기존과 같이 학습했다. Fixed final epoch 한 번만 평가했다.

## 대조 재현과 RNG 검증

1. 후보 학습 전에 기존 190의 **5개 fold OOF 및 최종 train RMSE**를 MPS에서 정확히 재로드했다. Train-only 정규화·ID·fold·로컬 가중치·캐시 해시와 모든 그룹의 범위를 확인했다.
2. Fold 0에서 ratio 1의 그룹 분할 대조를 한 번 재학습했다. 기존 190의 head·tail 최종 state와 OOF, 모든 **240 update LR·gradient trace**, 전체 feature trace, train 곡선·배치 순서가 **exact**였다. 따라서 이번 그룹 분할 코드 자체가 uniform 결과를 바꾸지 않는 것을 확인했다.
3. 190에는 update별 RNG 해시가 없다. 과거에 저장되지 않은 RNG와 비교했다고 주장하지 않는다. 신규 대조 fold 0의 RNG는 모두 저장했고 후보 fold 0과 직접 일치했다. 각 fold에서는 같은 seed의 `8×64` head dropout을 update당 두 번 소비하는 독립 RNG 재현과 **모든 update의 CPU/MPS RNG**를 비교했다. Optimizer 및 decay 연산 전후 RNG도 같았다.
4. 후보 5개 모델의 저장·MPS 재로드 예측은 exact였다. CPU/MPS 최대 차이는 **{maxcpu:.9f}**로 기존 허용오차 0.002 이내였다. 모든 학습 그룹은 실제로 변경됐고 동결 prefix의 원본 해시·trainable 범위를 확인했다. 최종 optimizer step counter는 모든 파라미터에서 240이었다.
5. Validation y를 NaN으로 바꿔도 train 통계는 exact였다. Train 함수에는 train 320개의 y만 전달했다. 이번 추가 full-gradient poison replay는 없으며, 기존 190의 poison 검증과 이번 정규화·입력 분리 검사를 구별한다.

작업량은 **기술 대조 1회 + 후보 5회 = 6회·1,440 gradient update**다. Head 재학습·inner CV·새 seed·재분할은 0회다. 결과를 본 뒤 비율을 수정하지 않았다. OOF 점수는 저장 모델 검증을 모두 통과한 뒤 계산했다.

## OOF와 고정 앙상블 결과

개선량은 대조 RMSE − 후보 RMSE다. CI는 400개 ID paired bootstrap 10,000회(seed 202610210)이며, 저장된 학습 결과에 조건부다. 상위 5개 제외는 각 비교에서 squared error 이득이 가장 큰 다섯 ID를 제외했다.

| 비교 | 대조 RMSE | 후보 RMSE | 개선량 | 조건부 95% CI | 개선 fold | 상위 5개 이득 제외 개선 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Frozen100 및 head-only112는 기존 저장 결과를 그대로 비교한 회복 진단이다. 각각을 점 추정치로 이겼는지는 `{recovery}`다. Uniform보다 덜 나쁘다는 것만으로 미세조정의 효용을 인정하지 않는다.

고정 앙상블은 **`0.9 × P021 + 0.1 × 후보`**다. 190과 같은 10% 추가 blend이며 209의 기존 CNN 첫 seed 출력 치환과 다르다. 기존 021의 calibration·residual이나 blend 비중을 재적합하지 않았다.

## Train/OOF 격차

| 조건 | 최종 train RMSE | OOF RMSE | OOF − train |
|---|---:|---:|---:|
| 기존 uniform | {tr['baseline_final_train_RMSE']:.10f} | {q['baseline_RMSE']:.10f} | {tr['baseline_OOF_minus_train_gap']:.10f} |
| 고정 LLRD | {tr['candidate_final_train_RMSE']:.10f} | {q['candidate_RMSE']:.10f} | {tr['candidate_OOF_minus_train_gap']:.10f} |

격차 감소는 **{tr['gap_reduction']:+.10f}**이며 `OOF 개선 {tr['OOF_gain']:+.10f} + train RMSE 증가 {tr['train_RMSE_increase']:+.10f}`로 분해된다. {gaptext}

Train 곡선은 각 fold train RMSE의 제곱을 평균한 뒤 제곱근을 취했다. 같은 ID가 여러 train fold에 등장하며, 이를 새 독립 표본으로 세지 않는다. Epoch별 OOF는 계산·선택하지 않았다.

| Adaptation epoch | Uniform train RMSE | LLRD train RMSE |
|---|---:|---:|
{curve}

Gradient clipping 비율은 기존 **{100*tr['baseline_clip_fraction']:.2f}%**, 후보 **{100*tr['candidate_clip_fraction']:.2f}%**다. 이 진단으로 완전 수렴이나 구조 한계를 증명하지 않는다.

## 전체 fold와 표본 집중

| Fold | 기존 train | LLRD train | 기존 OOF | LLRD OOF | OOF 개선 | 기존 격차 | LLRD 격차 | Best021 대비 결합 개선 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(foldrows)}

“상위 5개 비중”은 양수인 squared error 개선량 합계 중 이득 상위 다섯 ID가 차지하는 비중이다. 순개선이 작거나 음수일 때 불안정해지는 순개선 분모를 쓰지 않았다.

| 비교 | 개선 ID | 상위 5개 비중 | 제외 ID |
|---|---:|---:|---|
{chr(10).join(concentration)}

## 기준과 종료

사전 기준은 개선 ≥ 0.02, 3/5 fold 이상 개선, 조건부 CI 하한 > 0, 상위 5개 제외 개선 > 0 및 기술 검증 통과다. 동일 기준을 단독 matched 대조와 best021 대비 고정 결합에 각각 적용했다.

| 판단 | 결과 | 기준별 결과 |
|---|---|---|
{criteria}

{s['decision']} 209 EMA는 보류로 유지하며 이를 새 백본 개선의 보장으로 해석하지 않는다. 204의 대규모 nested 재학습을 반복하지 않았다. 다른 ratio·seed·epoch·optimizer·규제 강도로 확대하지 않는다.

## 한계와 보존

같은 개발 400장을 적응적으로 반복 사용했다. 독립 holdout이 아니며, CI는 누적 선택 편향·재학습·seed 변동을 보정하지 않는다. 이 결과는 한 백본의 마지막 세 블록·고정 비율 0.8에 한정한다. 실패를 모든 LLRD의 무효로 일반화하지 않는다.

기존 파일 **{len(pres)}개**의 SHA256을 전후 확인했다. Best021 CSV SHA256은 `{pres['artifacts/submission-candidate-021.csv']}`로 유지됐다. CPU 2개·기존 185 uv·로컬 MPS만 사용했다. 관측 최대 RSS+MPS **{peak/1e9:.3f} GB**, swap 증가 **0**, 감시 단계 합계 **{seconds:.1f}초**, 비용 **0**이다. 작성 시간을 포함한 전체 벽시계 시간은 아니다.

설치·다운로드·외부 API·raw train/holdout/test 접근·전체 500장 학습·신규 CSV·제출은 없다. 모든 신규 실험 파일은 `experiments/convnext-llrd-210`과 신규 요약·관리 보고서에 저장했다.

근거: `plan.json`, `preflight.json`, `control-validation.json`, fold별 `parameter-groups.json`·`rng-and-groups.npz`·`updates.npz`·`stability.json`·`report.json`, `validation.json`, `validation-train-predictions.npz`, `OOF.npz`, `ensemble-OOF.npz`, `score-validation.json`, `summary.json`, `resources-summary.json`.
'''
with (_repo_path(R, 'REPORT.md')).open('x') as f:f.write(report)
write(_repo_path(R, 'interpretation.json'),{'decision':s['decision'],'gap_interpretation':gaptext,'primary_passed':s['primary_passed'],'ensemble_passed':s['ensemble_passed'],'209_EMA_status':'보류','best021_maintained':True,'new_ratio_search':False,'nested_retraining':False,'blockers':[]})
write(_repo_path(P, 'reports/convnext-llrd-210-summary.json'),{**s,'report':'experiments/convnext-llrd-210/REPORT.md','protected_file_count':len(pres),'protected_files_unchanged':True,'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'max_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':0})
management=f'''# 210 완료 관리표

**{s['decision']}**

| 항목 | 상태 | 결과·범위 |
|---|---|---|
| 깊이별 LR 대조 | 완료 | 기존 ConvNeXt 마지막 세 블록의 정규화 ratio 0.8 한 정책. 평균 LR·실제 decay 고정. |
| 기존 대조 재현 | 통과 | 5개 OOF·train 재로드 exact. Fold 0 그룹 분할 재학습 state·OOF·전체 trace exact. |
| 단독 OOF | 완료 | {q['baseline_RMSE']:.10f} → {q['candidate_RMSE']:.10f}, 개선 {q['gain']:+.10f}, CI {q['conditional_CI95']}, 상위 5개 제외 {q['top5_removed_gain']:+.10f}. |
| Train/OOF 격차 | 진단 완료 | Train {tr['baseline_final_train_RMSE']:.8f} → {tr['candidate_final_train_RMSE']:.8f}. 격차 감소 {tr['gap_reduction']:+.8f}. {gaptext} |
| 고정 앙상블 | 완료 | Best021 대비 개선 {en['gain']:+.10f}, CI {en['conditional_CI95']}. 후보 10% 추가 blend이며 첫 seed 치환이 아님. |
| 로드맵 LLRD | 부분 완료 | 마지막 stage 내부의 한 비율 대조를 완료했다. 전체 backbone LLRD를 수행한 것은 아님. 이번 정책은 결과대로 종료·보관하며 비율 탐색 없음. |
| 후속 확대 | 미실시 | 209 EMA 보류 유지. 새 층·입력·loss·증강·nested 재학습·최종 평가 선택·제출 없음. |

신규 학습은 기술 대조 1회와 후보 5회, 총 1,440 update다. 기존 {len(pres)}개 파일·best021을 보존했고 비용은 0이다. 같은 400장의 조건부 결과이며 독립 holdout이 아니다.

[전체 보고서](../experiments/convnext-llrd-210/REPORT.md) · [구조화 결과](../reports/convnext-llrd-210-summary.json) · [기존 사전 계획](next-experiment-210.md)
'''
with (_repo_path(P, 'docs/records/roadmap-audit/210-completion.md')).open('x') as f:f.write(management)
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
manifest={str(p.relative_to(P)):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','completion.json','finalize.log']}
for p in [_repo_path(P, 'reports/convnext-llrd-210-summary.json'),_repo_path(P, 'docs/records/roadmap-audit/210-completion.md')]:manifest[str(p.relative_to(P))]=sha(p)
write(_repo_path(R, 'artifact-manifest.json'),manifest);write(_repo_path(R, 'completion.json'),{'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'완료','decision':s['decision'],'protected_files':len(pres),'protected_unchanged':True,'report_SHA256':sha(_repo_path(R, 'REPORT.md')),'manifest_SHA256':sha(_repo_path(R, 'artifact-manifest.json')),'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'cost':0,'remaining_blockers':[]})
print(json.dumps({'report':str(_repo_path(R, 'REPORT.md')),'decision':s['decision'],'primary_gain':q['gain'],'fixed_blend_gain_vs021':en['gain'],'peak_GB':peak/1e9,'protected_files':len(pres),'artifacts':len(manifest)},ensure_ascii=False,indent=2))
