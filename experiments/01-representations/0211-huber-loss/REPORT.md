# 211: 현재 TinyCNN의 MSE–Huber 고정 대조

**사전 기준 미달. delta 1·배율 2의 Huber 손실 분기를 종료하고 best021 유지.** 전이값·배율·다른 손실·seed·분할을 결과에 따라 탐색하지 않았다. Best021 유지.

## 중복 감사와 사전 고정

058은 동결 DINO 3456차원 특징의 100 epoch MLP head에서 SmoothL1 대비 half-MSE를 실행했다. 당시 결합 RMSE는 13.2415969550 → 13.2423921046이었다. 현재 TinyCNN 전체 학습 경로와 다르다. TinyCNN/cnn_controls 관련 소스 107개에 robust loss API와의 같은 파일 교집합은 없었다. 현재 frozen_validation.py와 206/208/209는 MSE이며 169는 타깃 밀도 가중 MSE였다. 확인한 기록에서 동일 대조가 없어 211을 실행했다.

개발 400장의 기존 5-fold(320/80), seed 20261003 + fold, BOX 128, train 전용 입력·타깃 표준화, D4, AdamW lr 0.002/weight decay 0.01, cosine 80 epoch/eta_min 0.0001, batch 32, gradient clip 5, 800 update, 마지막 5 epoch state 동일 평균을 유지했다. 손실만 바꿨다. 대비 감소와 EMA는 적용하지 않았다.

Plan과 실행·평가 소스는 학습 전에 잠갔다. 기본 쓰기 경로 불일치로 첫 단계에서는 감사만 했고, 이후 사용자 프로젝트 작업 승인 증거에 따라 명령별 승인으로 진행했다. 영구 권한·보안 설정 변경은 없다. 첫 코드 생성 명령의 문자열 구문 오류는 Python 실행 전에 발생해 파일 작성·학습이 없었고, 기록을 보존한 뒤 수정했다.

## 전이값과 gradient scale

표준화 잔차 r에 대해 대조는 mean(r²), 후보는 2 × Huber(delta=1)이다. 후보는 |r| ≤ 1에서 r², 밖에서 2|r|−1이다. 중앙 구간의 출력 기울기 2r 및 곡률 2와 mean reduction이 MSE와 같다. 꼬리 기울기는 ±2로 제한한다.

전이값 1은 fold train 타깃 표준편차 1개라는 고정 단위다. 잔차 표준편차 추정값이나 정규오차 효율 최적값을 주장하지 않는다. 큰 잔차를 라벨 오류로 전제하지 않았다. 전이값·배율은 결과 전에 각각 한 값으로 고정했다.

전체 parameter gradient norm, clipping 빈도, AdamW update가 같다는 뜻은 아니다. 이 차이는 꼬리 제한의 결과이므로 LR·clip·decay를 보정하지 않았다. 최적화 손실값의 크기 대신 원단위 RMSE와 공통 unweighted MSE를 비교한다.

## 실행과 재현

주 학습 10회(두 군 × 5-fold), validation label poison 검증 재학습 2회, 총 12회/9,600 gradient update를 실행했다.

- 5개 MSE 대조의 기존 평균 state·epoch 76–80 state·legacy FP32 OOF exact. 206의 모든 800 update 샘플 순서·D4·LR·loss·gradient norm·CPU/MPS RNG exact. 최종 AdamW optimizer와 scheduler도 저장된 208과 exact.
- 양 군 초기 state와 각 fold 전체 800 update 입력 hash·샘플 순서·D4·LR·CPU/MPS RNG exact.
- 10개 모델의 마지막 5 checkpoint 독립 재평균, ID/group 분리, train 전용 통계 검증. MPS 재로드 최대 오차 0, CPU 최대 오차 3.0517578e-05.
- Fold 0 양 군 validation 타깃을 NaN/1e30으로 바꾼 전체 80 epoch replay에서 state·OOF·stream·최종 optimizer exact. 전 fold 정규화도 validation 타깃 변경에 불변.
- CPU/MPS Huber 수식·출력 gradient·경계 및 중앙 MSE 일치 fixture 통과. 기술 검증 뒤 validation RMSE 계산. Checkpoint 선택 없음.

## 원본 400장 OOF

개선량은 MSE 대조 RMSE − Huber RMSE다. 동일 ID paired bootstrap 10,000회(seed 202610211)의 조건부 95% CI다. 상위 5개 제외는 각 비교에서 표본별 제곱 오차 이득이 가장 큰 5개 ID를 제거한 결과다.

| 비교 | MSE RMSE | Huber RMSE | 개선량 | 조건부 95% CI | 개선 fold | 상위 5개 제외 개선 |
|---|---:|---:|---:|---:|---:|---:|
| CNN 단독 | 15.0323178431 | 14.5929813432 | +0.4393364999 | [-0.0284648725, +0.9167769032] | 3/5 | +0.1303984986 |
| 고정 앙상블 진단 | 12.7862007758 | 12.7587574662 | +0.0274433095 | [-0.0131157755, +0.0695382562] | 3/5 | +0.0004764075 |

단독 fold 개선량: [0.7753976931071414, -0.3879154498850852, -0.0529809758354034, 0.6162429758444539, 1.2830803758667244].
고정 앙상블 fold 개선량: [0.05951612703511344, -0.025308482279390532, -0.014282377628592968, 0.012382740231442924, 0.10422090177590349].

사전 기준은 단독 개선 ≥ 0.02, 개선 fold ≥ 3, CI 하한 > 0, 상위 5개 제외 개선 > 0 및 기술 검증 통과다. 실제 판정: {'technical_validation': True, 'gain_at_least_0_02': True, 'at_least3_folds': True, 'conditional_CI_lower_positive': False, 'top5_removed_positive': True}.

고정 앙상블은 Q = P021 + 0.081 × 기존 fold 보정 기울기 × (새 첫 seed CNN − 기존 첫 seed CNN)이다. Control Q는 best021과 exact다. 나머지 seed·보정·잔차는 고정했으며 전체 후단 재학습이나 새 후보 채택을 뜻하지 않는다.

## 학습·검증 및 최적화

Train RMSE는 각 fold 320개, 총 1,600개 중복 train 관측으로 계산했다. Gradient 및 꼬리 비율은 증강 batch 4,000 update 평균으로, 최종 averaged model 오차와 평가 대상이 다르다.

| 손실 | Train RMSE | OOF RMSE | OOF−train | 평균 unclipped gradient norm | clip 5 비율 | 꼬리 비율 |
|---|---:|---:|---:|---:|---:|---:|
| MSE | 9.88568501 | 15.03231784 | 5.14663283 | 2.696545 | 8.82% | 15.05% |
| 2 × Huber | 10.01675741 | 14.59298134 | 4.57622393 | 2.142382 | 3.25% | 14.92% |

공통 unweighted MSE, optimized loss, scaled Huber 및 fold별 train/validation RMSE 전체 수치는 summary.json과 streams.npz에 보존했다. 서로 다른 목적함수의 작은 숫자 자체를 개선으로 해석하지 않는다.

## 범위와 보존

같은 개발 400장의 누적 재사용이며 독립 holdout이 아니다. 조건부 CI는 누적 선택 편향·재학습·seed 불확실성을 보정하지 않는다. 결과는 delta 1·배율 2 한 정책에 한정한다. 204의 6 seed nested 분기를 반복하지 않았고 209는 보류 상태다.

기존 파일 1012개의 SHA256 모두 일치. Best021 SHA256은 7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b로 유지됐다. CPU 2 threads/MPS, 관측 최대 RSS+MPS 0.697 GB, swap 증가 0, 감시 단계 합계 194.5초다.

신규 설치·다운로드·외부 AI·원시 train/holdout/test 데이터 읽기·전체 500장 학습·새 CSV·업로드·비용은 없다. 개발 400장 기존 캐시만 사용했다.

근거: plan.json, code-lock.json, training-complete.json, fold별 state/stream/control-reproduction, validation.json, OOF.npz, ensemble-OOF.npz, bootstrap.npz, summary.json, resources-summary.json.
