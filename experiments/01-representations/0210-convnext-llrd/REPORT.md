# 210: ConvNeXt 마지막 세 블록의 고정 LLRD 대조

**사전 primary 기준 미달. 이 고정 LLRD 정책을 종료하고 비율 탐색 없이 best021을 유지한다.** 209 EMA는 보류 상태를 유지한다. 이번에는 EMA, 새 층, 입력 변경, 증강, loss 변경을 추가하지 않았다.

단독 Uniform 대비 개선은 **-0.0039440058**, 조건부 95% CI는 **[-0.01321240480664647, 0.005391779873496193]**다. Best021 대비 고정 10% 추가 결합 개선은 **-0.1723121648**, CI는 **[-0.2820342062686717, -0.060728247790295016]**다. 같은 모델의 OOF 개선과 현재 앙상블에 추가할 효용을 별도로 판정했다.

## 정책과 정확한 통제

학습 전 [기존 계획](../../../docs/records/roadmap-audit/next-experiment-210.md), `plan.json`, 실행 코드 해시를 고정했다. Ratio **0.8 하나**를 사용했다. 대조의 세 블록 LR은 모두 1e-5, 후보의 초기 LR은 순서대로 **7.868852459e-6 / 9.836065574e-6 / 1.229508197e-5**다. 마지막 stage 내부 blockwise LLRD이며 전체 backbone의 LLRD 대조가 아니다.

각 블록은 4,763,136개 파라미터다. `(0.8², 0.8, 1)`을 그 평균으로 정규화해 세 블록의 평균 LR을 맞췄다. Terminal LayerNorm 1,536개의 LR은 1e-5, head 49,281개의 LR은 1e-4로 유지했다. Tail 합계는 14,290,944개다. 각 파라미터는 한 그룹에만 포함됐으며 기존 파라미터 순서를 유지했다. 모든 1,440 update에서 그룹별 LR·평균 LR·cosine schedule을 확인했다.

**실제 weight decay factor도 동일하게 유지했다.** Optimizer의 내장 weight decay는 0이고, step 직전에 head는 `1 − 1e-4 × f(epoch) × 0.01`, 모든 tail 파라미터는 `1 − 1e-5 × f(epoch) × 0.01`을 곱했다. 후보 그룹의 변경된 LR을 이 축소식에 사용하지 않았다. 실제 factor를 update별로 저장하고 독립적으로 재계산했다. Nominal 평균 LR을 맞춘 것이 실제 AdamW 파라미터 이동량까지 같게 만든다는 뜻은 아니다.

기존 ConvNeXt-Tiny 로컬 가중치·native 256의 동결 prefix 캐시·MLP head·fold별 warm head·train RMS scaler·target 표준화를 재사용했다. AdamW β=(0.9,0.999), eps=1e-8, SmoothL1 beta 1, global clip 1, dropout 0.25, microbatch 8/effective batch 16, 12 epoch·240 update, 기존 seed·배치 순서를 유지했다. Backbone은 eval, head만 train이며 BN과 running buffer는 없다. Terminal LayerNorm affine과 마지막 세 블록만 기존과 같이 학습했다. Fixed final epoch 한 번만 평가했다.

## 대조 재현과 RNG 검증

1. 후보 학습 전에 기존 190의 **5개 fold OOF 및 최종 train RMSE**를 MPS에서 정확히 재로드했다. Train-only 정규화·ID·fold·로컬 가중치·캐시 해시와 모든 그룹의 범위를 확인했다.
2. Fold 0에서 ratio 1의 그룹 분할 대조를 한 번 재학습했다. 기존 190의 head·tail 최종 state와 OOF, 모든 **240 update LR·gradient trace**, 전체 feature trace, train 곡선·배치 순서가 **exact**였다. 따라서 이번 그룹 분할 코드 자체가 uniform 결과를 바꾸지 않는 것을 확인했다.
3. 190에는 update별 RNG 해시가 없다. 과거에 저장되지 않은 RNG와 비교했다고 주장하지 않는다. 신규 대조 fold 0의 RNG는 모두 저장했고 후보 fold 0과 직접 일치했다. 각 fold에서는 같은 seed의 `8×64` head dropout을 update당 두 번 소비하는 독립 RNG 재현과 **모든 update의 CPU/MPS RNG**를 비교했다. Optimizer 및 decay 연산 전후 RNG도 같았다.
4. 후보 5개 모델의 저장·MPS 재로드 예측은 exact였다. CPU/MPS 최대 차이는 **0.000104913**로 기존 허용오차 0.002 이내였다. 모든 학습 그룹은 실제로 변경됐고 동결 prefix의 원본 해시·trainable 범위를 확인했다. 최종 optimizer step counter는 모든 파라미터에서 240이었다.
5. Validation y를 NaN으로 바꿔도 train 통계는 exact였다. Train 함수에는 train 320개의 y만 전달했다. 이번 추가 full-gradient poison replay는 없으며, 기존 190의 poison 검증과 이번 정규화·입력 분리 검사를 구별한다.

작업량은 **기술 대조 1회 + 후보 5회 = 6회·1,440 gradient update**다. Head 재학습·inner CV·새 seed·재분할은 0회다. 결과를 본 뒤 비율을 수정하지 않았다. OOF 점수는 저장 모델 검증을 모두 통과한 뒤 계산했다.

## OOF와 고정 앙상블 결과

개선량은 대조 RMSE − 후보 RMSE다. CI는 400개 ID paired bootstrap 10,000회(seed 202610210)이며, 저장된 학습 결과에 조건부다. 상위 5개 제외는 각 비교에서 squared error 이득이 가장 큰 다섯 ID를 제외했다.

| 비교 | 대조 RMSE | 후보 RMSE | 개선량 | 조건부 95% CI | 개선 fold | 상위 5개 이득 제외 개선 |
|---|---:|---:|---:|---:|---:|---:|
| Uniform → LLRD 단독 | 16.7641850464 | 16.7681290522 | -0.0039440058 | [-0.0132124048, +0.0053917799] | 3/5 | -0.0083899946 |
| Frozen100 → LLRD 단독 | 16.5046246216 | 16.7681290522 | -0.2635044306 | [-0.4037477985, -0.1279834499] | 0/5 | -0.3263964968 |
| Head-only112 → LLRD 단독 | 16.6151232037 | 16.7681290522 | -0.1530058485 | [-0.2622036286, -0.0461736755] | 0/5 | -0.1969058322 |
| Uniform 10% 결합 → LLRD 10% 결합 | 12.9586385806 | 12.9585129405 | +0.0001256401 | [-0.0007947179, +0.0010218393] | 2/5 | -0.0003770331 |
| Best021 → LLRD 10% 결합 | 12.7862007758 | 12.9585129405 | -0.1723121648 | [-0.2820342063, -0.0607282478] | 0/5 | -0.2259211224 |

Frozen100 및 head-only112는 기존 저장 결과를 그대로 비교한 회복 진단이다. 각각을 점 추정치로 이겼는지는 `{'beats_frozen100_point': False, 'beats_head_only112_point': False}`다. Uniform보다 덜 나쁘다는 것만으로 미세조정의 효용을 인정하지 않는다.

고정 앙상블은 **`0.9 × P021 + 0.1 × 후보`**다. 190과 같은 10% 추가 blend이며 209의 기존 CNN 첫 seed 출력 치환과 다르다. 기존 021의 calibration·residual이나 blend 비중을 재적합하지 않았다.

## Train/OOF 격차

| 조건 | 최종 train RMSE | OOF RMSE | OOF − train |
|---|---:|---:|---:|
| 기존 uniform | 3.8087147847 | 16.7641850464 | 12.9554702617 |
| 고정 LLRD | 3.7666448293 | 16.7681290522 | 13.0014842229 |

격차 감소는 **-0.0460139613**이며 `OOF 개선 -0.0039440058 + train RMSE 증가 -0.0420699555`로 분해된다. Train 오차는 더 낮아졌지만 OOF는 악화했고, train/OOF 격차도 더 커졌다. 이번 고정 LLRD가 과적합 격차를 줄였다는 근거는 없다.

Train 곡선은 각 fold train RMSE의 제곱을 평균한 뒤 제곱근을 취했다. 같은 ID가 여러 train fold에 등장하며, 이를 새 독립 표본으로 세지 않는다. Epoch별 OOF는 계산·선택하지 않았다.

| Adaptation epoch | Uniform train RMSE | LLRD train RMSE |
|---|---:|---:|
| 0 | 5.84985451 | 5.84985451 |
| 3 | 5.26504412 | 5.25320405 |
| 6 | 4.57871114 | 4.55904372 |
| 9 | 4.07537244 | 4.04425207 |
| 12 | 3.80871478 | 3.76664483 |

Gradient clipping 비율은 기존 **99.75%**, 후보 **99.75%**다. 이 진단으로 완전 수렴이나 구조 한계를 증명하지 않는다.

## 전체 fold와 표본 집중

| Fold | 기존 train | LLRD train | 기존 OOF | LLRD OOF | OOF 개선 | 기존 격차 | LLRD 격차 | Best021 대비 결합 개선 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.13397417 | 4.09069439 | 17.52687573 | 17.52398770 | +0.00288803 | 13.39290156 | 13.43329331 | -0.28772218 |
| 1 | 3.64113234 | 3.60051744 | 15.22549909 | 15.22166678 | +0.00383231 | 11.58436676 | 11.62114934 | -0.07532927 |
| 2 | 3.70322937 | 3.66111029 | 18.56245263 | 18.57952511 | -0.01707248 | 14.85922326 | 14.91841482 | -0.26135041 |
| 3 | 3.90428818 | 3.85799893 | 15.19720864 | 15.18993330 | +0.00727534 | 11.29292046 | 11.33193437 | -0.00939489 |
| 4 | 3.63683660 | 3.59898258 | 17.04882716 | 17.06249243 | -0.01366527 | 13.41199056 | 13.46350986 | -0.20603751 |

“상위 5개 비중”은 양수인 squared error 개선량 합계 중 이득 상위 다섯 ID가 차지하는 비중이다. 순개선이 작거나 음수일 때 불안정해지는 순개선 분모를 쓰지 않았다.

## 기준과 종료

사전 기준은 개선 ≥ 0.02, 3/5 fold 이상 개선, 조건부 CI 하한 > 0, 상위 5개 제외 개선 > 0 및 기술 검증 통과다. 동일 기준을 단독 matched 대조와 best021 대비 고정 결합에 각각 적용했다.

| 판단 | 결과 | 기준별 결과 |
|---|---|---|
| LLRD 대조 | 미달 | technical: 통과, gain_at_least_0_02: 미달, at_least3_folds: 통과, CI_lower_positive: 미달, top5_removed_positive: 미달 |
| 최종 앙상블 | 미달 | technical: 통과, gain_at_least_0_02: 미달, at_least3_folds: 미달, CI_lower_positive: 미달, top5_removed_positive: 미달 |

사전 primary 기준 미달. 이 고정 LLRD 정책을 종료하고 비율 탐색 없이 best021을 유지한다. 209 EMA는 보류로 유지하며 이를 새 백본 개선의 보장으로 해석하지 않는다. 204의 대규모 nested 재학습을 반복하지 않았다. 다른 ratio·seed·epoch·optimizer·규제 강도로 확대하지 않는다.

## 한계와 보존

같은 개발 400장을 적응적으로 반복 사용했다. 독립 holdout이 아니며, CI는 누적 선택 편향·재학습·seed 변동을 보정하지 않는다. 이 결과는 한 백본의 마지막 세 블록·고정 비율 0.8에 한정한다. 실패를 모든 LLRD의 무효로 일반화하지 않는다.

기존 파일 **410개**의 SHA256을 전후 확인했다. Best021 CSV SHA256은 `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b`로 유지됐다. CPU 2개·기존 185 uv·로컬 MPS만 사용했다. 관측 최대 RSS+MPS **1.226 GB**, swap 증가 **0**, 감시 단계 합계 **308.5초**, 비용 **0**이다. 작성 시간을 포함한 전체 벽시계 시간은 아니다.

설치·다운로드·외부 API·raw train/holdout/test 접근·전체 500장 학습·신규 CSV·제출은 없다. 모든 신규 실험 파일은 `experiments/01-representations/0210-convnext-llrd`과 신규 요약·관리 보고서에 저장했다.

근거: `plan.json`, `preflight.json`, `control-validation.json`, fold별 `parameter-groups.json`·`rng-and-groups.npz`·`updates.npz`·`stability.json`·`report.json`, `validation.json`, `validation-train-predictions.npz`, `OOF.npz`, `ensemble-OOF.npz`, `score-validation.json`, `summary.json`, `resources-summary.json`.
