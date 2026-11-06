# 212: 고정 MSE–Huber 정책의 재분할 안정성

**사전 분할 안정성 기준을 충족하지 못했다. 원본 211의 CI 기준 미달 판단과 best021을 유지하고 보고 후 정지한다.**

원본 211의 단독 평균 개선 +0.43933650, 95% CI [-0.02846487, +0.91677690] 및 사전 기준 미달 판단은 그대로다. 212는 이후 명시 의뢰된 두 기존 재분할의 안정성 확인이며 원본 판단을 소급 변경하지 않는다.

## 동일 정책과 실제 작업

MSE 대 2 × Huber(delta 1) 한 정책이다. 전이값은 fold train 타깃 표준편차 1개이며, 배율 2는 중앙 구간의 r²·기울기 2r·곡률 2를 MSE와 맞춘다. 전체 gradient norm이나 AdamW update 크기를 같게 맞추는 방식은 아니다. LR·clip·decay를 변경하지 않았고, 작은 최적화 손실값 자체를 개선으로 해석하지 않는다.

기존 20261031/20261101 각각 5-fold(320/80), seed 20261003 + fold, BOX 128, train 전용 입력·타깃 표준화, D4, AdamW lr 0.002/weight decay 0.01, cosine 80 epoch/eta_min 0.0001, batch 32, gradient clip 5, 마지막 5 epoch 동일 평균을 유지했다. 대비 감소·EMA·손실 탐색은 없다. Plan과 실행·평가 소스는 학습 전에 잠갔다.

새 주 학습 20회 및 각 재분할 fold 0 양 군 validation label poison 검증 4회, 총 24회/19,200 update다. 원본 211은 재학습하지 않았다.

## 대조 재현과 데이터 분리

- 10개 MSE 대조의 기존 checkpoint 및 209 마지막 평균 state·epoch 76–80 state·legacy FP32 OOF exact.
- 209의 모든 800 update 샘플 순서·D4·LR·loss·gradient norm·CPU/MPS RNG와 최종 optimizer/scheduler exact. 209에서 저장된 update별 RNG를 사용했으며, 201에 없는 과거 RNG 기록을 비교했다고 주장하지 않는다.
- 두 군의 초기 state, 모든 800 update 입력 hash·order·D4·LR·CPU/MPS RNG exact.
- 20개 모델의 마지막 5 state 독립 재평균 및 재로드 검증. MPS 최대 오차 0, CPU 최대 오차 1.5258789e-05.
- 각 재분할 fold 0 양 군에서 validation 타깃을 NaN/1e30으로 바꾼 80 epoch replay의 state·OOF·stream·최종 optimizer exact. 전 fold train 정규화도 validation 타깃 변경에 불변.
- 기존 고정 앙상블 계보와 coefficient를 다시 구성해 저장된 201 결과와 exact임을 확인했다. 기술 검증 뒤 점수를 계산했다.

## 세 분할 및 400 ID 묶음 결과

개선량은 MSE RMSE − Huber RMSE다. 모든 분할은 같은 400개 ID다. Pooled는 ID별 분할 squared error 평균의 전체 평균에서 제곱근을 계산한다. 예측 평균 앙상블이 아니다.

Bootstrap은 400 ID cluster를 10,000회 재표집하며 같은 ID의 모든 분할 관측을 함께 포함한다. 1,200개 독립 표본으로 다루지 않았다. 211의 seed 202610211과 draw 배열을 그대로 사용해 원본 점추정·CI·상위 5개 제외 결과가 exact임을 검증했다.

| 분할 | 비교 | MSE RMSE | Huber RMSE | 개선량 | 조건부 95% CI | 상위 5개 ID 제외 개선 |
|---|---|---:|---:|---:|---:|---:|
| 원본 211 | CNN 단독 | 15.03231784 | 14.59298134 | +0.43933650 | [-0.02846487, +0.91677690] | +0.13039850 |
| 원본 211 | 고정 첫 seed 치환 | 12.78620078 | 12.75875747 | +0.02744331 | [-0.01311578, +0.06953826] | +0.00047641 |
| 재분할 20261031 | CNN 단독 | 14.27018716 | 14.58656477 | -0.31637761 | [-0.74693569, +0.12188366] | -0.49327180 |
| 재분할 20261031 | 고정 첫 seed 치환 | 12.62190242 | 12.66315513 | -0.04125271 | [-0.08174011, -0.00079560] | -0.06024224 |
| 재분할 20261101 | CNN 단독 | 14.55309989 | 14.59169709 | -0.03859720 | [-0.48837061, +0.41487570] | -0.22813907 |
| 재분할 20261101 | 고정 첫 seed 치환 | 13.03244921 | 13.04056687 | -0.00811766 | [-0.05216632, +0.03511698] | -0.03030043 |
| 두 재분할 pooled | CNN 단독 | 14.41233773 | 14.58913116 | -0.17679342 | [-0.51163164, +0.15909357] | -0.31530552 |
| 두 재분할 pooled | 고정 첫 seed 치환 | 12.82881821 | 12.85324633 | -0.02442812 | [-0.05614018, +0.00712985] | -0.03877719 |
| 세 분할 pooled | CNN 단독 | 14.62191891 | 14.59041467 | +0.03150424 | [-0.22300134, +0.29417748] | -0.08834886 |
| 세 분할 pooled | 고정 첫 seed 치환 | 12.81462814 | 12.82182741 | -0.00719927 | [-0.03163859, +0.01708439] | -0.01886900 |

상위 5개 제외는 각 비교의 ID별 평균 squared error 이득이 가장 큰 5개 ID를 전체 반복 관측과 함께 제거한다. 원본 211의 상위 5개 ID를 고정 제외한 각 분할 및 pooled 감도도 summary.json에 별도로 보존했다.

- original standalone: 개선 fold 3/5, fold별 개선 [0.77539769, -0.38791545, -0.05298098, 0.61624298, 1.28308038].
- original fixed_ensemble: 개선 fold 3/5, fold별 개선 [0.05951613, -0.02530848, -0.01428238, 0.01238274, 0.1042209].
- 20261031 standalone: 개선 fold 1/5, fold별 개선 [-0.32006461, -0.70224146, 0.13580093, -0.50186701, -0.19952591].
- 20261031 fixed_ensemble: 개선 fold 0/5, fold별 개선 [-0.04978717, -0.06195662, -0.04193472, -0.03430661, -0.01727812].
- 20261101 standalone: 개선 fold 1/5, fold별 개선 [-0.27937945, -0.1673776, -0.22901592, 0.55507465, -0.05086229].
- 20261101 fixed_ensemble: 개선 fold 2/5, fold별 개선 [-0.09906914, 0.02249047, -0.03472685, 0.0790918, -0.01198454].

고정 앙상블은 Q = P021 + 0.081 × 기존 split/fold 보정 기울기 × (새 첫 seed CNN − 기존 첫 seed CNN)이다. CNN 3개 seed 중 첫 seed의 출력 치환이며, 새 성분을 추가한 blend가 아니다. 나머지 두 seed·보정·잔차는 고정했다. Control Q는 각 분할의 기존 021과 exact다. 전체 후단을 재학습한 효과로 해석하지 않는다.

## 학습·검증 오차와 판정

Train RMSE는 각 분할의 1,600개 중복 train 관측으로 계산했다. Clipping 비율은 증강 batch의 학습 경로 통계다. 모든 fold 및 공통 MSE·scaled Huber·gradient norm·꼬리 비율은 summary.json/streams.npz에 있다.

| 분할 | 손실 | Train RMSE | OOF RMSE | OOF−train | clip 5 비율 |
|---|---|---:|---:|---:|---:|
| original | control | 9.88568501 | 15.03231784 | 5.14663283 | 8.82% |
| original | huber | 10.01675741 | 14.59298134 | 4.57622393 | 3.25% |
| 20261031 | control | 9.75966922 | 14.27018716 | 4.51051794 | 9.62% |
| 20261031 | huber | 10.69028244 | 14.58656477 | 3.89628233 | 2.62% |
| 20261101 | control | 9.83988344 | 14.55309989 | 4.71321645 | 9.50% |
| 20261101 | huber | 10.00537973 | 14.59169709 | 4.58631736 | 2.00% |

추가 두 재분할 각각의 사전 단독 기준은 개선 ≥ 0.02, 개선 fold ≥ 3, CI 하한 > 0, 상위 5개 제외 개선 > 0이다. 두 재분할 및 세 분할 pooled에서는 개선·CI 하한·상위 5개 제외 개선이 모두 양수인지 확인한다. 실제 판정은 다음과 같다.

{
  "20261031": {
    "gain_at_least_0_02": false,
    "at_least3_positive_folds": false,
    "CI_lower_positive": false,
    "top5_removed_positive": false
  },
  "20261101": {
    "gain_at_least_0_02": false,
    "at_least3_positive_folds": false,
    "CI_lower_positive": false,
    "top5_removed_positive": false
  },
  "two_repartitions": {
    "gain_positive": false,
    "CI_lower_positive": false,
    "top5_removed_positive": false
  },
  "all_three_partitions": {
    "gain_positive": true,
    "CI_lower_positive": false,
    "top5_removed_positive": false
  }
}

이 안정성 판정과 원본 211의 역사적 기준 미달은 함께 보존한다. 결과와 관계없이 안정성 판단 후 정지하며 전체 500장·후단 nested 재학습·제출을 자동 실행하지 않는다. Best021 유지.

## 한계와 보존

같은 개발 400개 ID를 반복 사용했다. CI는 저장된 학습 결과에 조건부이며 재학습·seed 변동·누적 선택 편향을 보정하지 않는다. 211 결과 이후 의뢰된 고정 정책 확인이며 새로운 독립 검증이 아니다.

기존 파일 1181개의 SHA256 및 best021 해시가 모두 유지됐다. Best021 SHA256: 7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b.
CPU 2 threads/MPS, 최대 RSS+MPS 0.757 GB, swap 증가 0, 감시 단계 합계 384.1초, 비용 0.

설치·다운로드·외부 AI·원시 train/holdout/test 읽기·전체 500장 학습·CSV·업로드·제출은 없다. 프로젝트 내부 새 212 코드·결과·관리표만 작성했다. 211·209·204의 기존 기록은 보존했다.

근거: plan.json, code-lock.json, preflight.json, training-complete.json, split/fold별 state와 streams, validation.json, score-validation.json, bootstrap.npz, summary.json, resources-summary.json.
