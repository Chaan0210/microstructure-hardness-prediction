# 210 제안: ConvNeXt 마지막 stage의 제한된 LLRD 대조

**계획만 작성했다. 학습·추론·설치·다운로드를 시작하지 않았으며 실행 코드를 만들지 않았다.** 209 EMA는 보류 상태를 유지한다. 이 제안은 EMA를 추가하지 않는 별개 가설이다.

## 질문과 대조

190의 ConvNeXt-Tiny에서 이미 학습했던 마지막 세 CNBlock 안에서, **평균 LR과 실제 weight decay를 유지하면서 깊이별 LR 배분만 바꾸면** 기존 uniform LR보다 OOF가 개선되는가?

190에서 head-only 대비 last stage의 train RMSE는 5.379704→3.808715로 낮아졌으나 OOF는 16.615123→16.764185로 악화했다. 이는 제한된 과적합 완화 대조의 동기다. 어느 블록이 과도하게 변했는지, LLRD가 도움이 되는지는 미확인이다. ResNet 189의 종료된 확대 분기는 재개하지 않는다.

- **대조 U:** 190 `last_stage112`의 세 블록과 terminal LayerNorm에 모두 초기 LR 1e-5. 기존 결과를 해시 확인 후 사용.
- **후보 D:** 아래의 단일 정규화 depth ratio 0.8. Trainable 범위·파라미터 수는 U와 동일.
- `frozen100` 및 `head_only112`의 기존 OOF는 학습을 추가하지 않는 회복 여부 진단으로만 함께 보고한다. 새 후보 arm이나 선택 grid가 아니다.

## 하나의 변화만 분리하는 LR·weight decay 규칙

각 CNBlock은 저장 checkpoint에서 **4,763,136개**의 파라미터를 가진다. 세 블록의 원래 순서를 유지하고 `r = (0.8², 0.8, 1)`을 평균 `r̄ = (0.64+0.8+1)/3`으로 나눈다. 일반적인 최고 LR 고정 방식 대신 평균을 맞춰, 전체 tail LR을 낮춘 효과와 깊이별 배분 효과를 분리한다. 이는 **마지막 stage 내부의 blockwise LLRD**이며, 전체 backbone LLRD라고 부르지 않는다.

| 파라미터 | 기존 초기 LR | 후보 초기 LR |
|---|---:|---:|
| `features[7][0]` | 1.000000000e-5 | 7.868852459e-6 |
| `features[7][1]` | 1.000000000e-5 | 9.836065574e-6 |
| `features[7][2]` | 1.000000000e-5 | 1.229508197e-5 |
| Terminal LayerNorm affine | 1.000000000e-5 | 1.000000000e-5 |
| 기존 MLP head | 1.000000000e-4 | 1.000000000e-4 |

후보는 앞 블록 LR을 낮추고 마지막 블록 LR을 높인다. 모든 층의 LR을 낮추는 정책이 아니다. 세 블록의 파라미터 수가 같으므로 파라미터 수 가중 평균 LR도 동일하다. Ratio 0.8은 결과를 보고 선택한 값이 아닌, 인접 블록에 20% 차이를 주는 **한 번의 탐색용 고정값**이다. 최적값이라는 근거는 없고 추가 ratio를 시험하지 않는다.

모든 LR에 190의 동일 epoch별 cosine factor `f(e)=0.1+0.9(1+cos(πe/12))/2`, `e=0…11`을 곱한다. Terminal norm과 head는 기존 LR·schedule을 유지한다. 평균 명목 LR을 맞춘 것이 실제 AdamW 업데이트 norm이나 모델 이동량까지 같게 만든다는 주장은 하지 않는다.

**Weight decay 혼입을 막는다.** 190은 `optimizer.weight_decay=0`으로 두고 optimizer step 전에 각 파라미터를 `1 − lr × 0.01`로 직접 축소했다. 후보에서 변경된 블록 LR을 그대로 축소식에 쓰면 depth별 규제 강도까지 달라진다. 따라서 두 군 모두:

- 세 블록과 terminal norm의 실제 축소 factor: `1 − (1e-5 × f(e)) × 0.01`.
- Head의 실제 축소 factor: `1 − (1e-4 × f(e)) × 0.01`.
- AdamW의 gradient update에만 표의 각 LR을 사용한다. 내장 weight decay는 두 군 모두 0.

이로써 실제 decay factor를 기존과 같게 둔다. Global clip 1과 head/tail의 합산 순서, 파라미터 순서도 그대로 유지한다. 후보의 학습 경로가 달라진 뒤 gradient 값이나 clip 빈도까지 대조와 같아야 한다고 요구하지 않는다.

## 고정할 나머지 조건

| 항목 | 고정 조건 |
|---|---|
| 데이터 | 기존 원본 development 400의 5fold, train 320/validation 80. 같은 ID/group·fold를 재사용. |
| 입력·캐시 | 190 native 256 grayscale→RGB, ImageNet 정규화, `features[:7]`의 고정 prefix 캐시 `(400,768,8,8)` float32. Resize·crop·증강 추가 없음. |
| 가중치 | 로컬 `experiments/01-representations/0190-modern-backbones/weights/convnext_tiny-983f1562.pth`. SHA256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`. 파일·캐시 존재와 해시 확인 완료. 다운로드 경로 호출 금지. |
| 학습 범위 | 마지막 CNBlock 3개와 terminal LayerNorm affine 1,536개, 합계 tail 14,290,944개 및 기존 head. Earlier prefix는 계속 동결. |
| Head·전처리 | 기존 `Linear768→64, GELU, Dropout0.25, Linear64→1`의 fold별 100 epoch warm head·train RMS scaler·target 표준화 값을 재사용. Warm head 재학습이나 새 inner selection 없음. |
| 예산·순서 | 추가 adaptation 12 epoch, fold당 240 optimizer update, microbatch 8/effective batch 16. 기존 adaptation seed `202710186+100×fold+99`, NumPy 순서와 head dropout RNG를 동일하게 시작. |
| Optimizer·손실 | AdamW β=(0.9,0.999), eps=1e-8, SmoothL1 beta 1, global clip 1. 실제 decay는 위 고정식. |
| Mode·평가 | Backbone eval로 stochastic depth 비활성화, head만 train. BN 없음, LayerNorm 통계 재보정 없음. 고정 마지막 epoch 하나, TTA·EMA·마지막 checkpoint 평균·early stopping 없음. |

## 향후 실행이 명시적으로 지시될 경우의 최소 순서

1. 원본 파일과 기존 190의 5개 warm head·tail·OOF·캐시 해시를 확인한다. 저장 모델 재로드로 기존 OOF를 재현한다. 기존 uv 환경 `experiments/01-representations/0185-pretrained-cnn/.venv`만 사용한다.
2. **Fold 0에서 ratio=1의 그룹 분할 대조를 1회 재학습**한다. 세 블록을 별도 optimizer group으로 나눈 코드가 190의 기존 uniform 경로를 바꾸지 않는지 확인한다. 기존 마지막 가중치·OOF·배치 순서·LR·gradient trace가 exact여야 한다. 불일치하면 중단하고 원인을 보고하며, 결과를 보고 허용오차를 풀거나 전체 대조를 자동 재학습하지 않는다.
3. 통과하면 고정 ratio 0.8 후보를 **5fold에서 각 1회** 학습한다. 총 신규 gradient 학습은 후보 5회+기술 대조 1회=**6회, 1,440 update**다. 새 head 학습·inner CV·새 seed·재분할·nested 후단 학습은 없다.
4. Train 전용 통계·label slicing을 확인하고 validation y poison으로 전처리 불변을 검사한다. 모든 block/head LR, 실제 decay factor, 순서·RNG 해시, train 곡선·gradient/clip·feature trace를 저장한다. Prefix 고정·학습 대상 누락/중복 없음·terminal norm 범위를 검증한다.
5. 5개 후보를 MPS·CPU로 재로드한다. MPS 저장·재로드는 exact, CPU/MPS 허용오차는 190의 0.002를 유지한다. 기술 검증을 마친 뒤 모든 fold와 OOF 점수를 함께 계산한다. 중간 validation 성능으로 epoch나 정책을 선택하지 않는다.

이는 최소 대조의 **실행 제안**이지 현재 실행 권한이나 완료 사실이 아니다. 기술 대조가 실패하면 후보 학습을 시작하지 않는다. 미세조정의 관측 이동량·clip 비율은 해석용이며 추가 튜닝 축으로 쓰지 않는다.

## 판단과 종료 규칙

- **가설 확인의 primary:** 후보 D 대 uniform U의 단독 RMSE 개선. 개선 ≥ 0.02, 3/5 fold 이상 개선, 400개 ID paired bootstrap 10,000회(seed 202610210)의 조건부 95% CI 하한 > 0, 이득이 큰 상위 5개 ID 제외 개선 > 0, 기술 검증 통과를 모두 요구한다.
- **회복 여부:** 기존 `frozen100`(16.504625) 및 `head_only112`(16.615123)와의 paired 차이도 전부 보고한다. Uniform(16.764185)보다 덜 나쁘다는 결과만으로 미세조정의 효용을 주장하지 않는다. Train/OOF 격차 감소만으로 성공 판정하지 않는다.
- **현재 앙상블 효용:** 190과 같은 `0.9×P021 + 0.1×D`를 고정해서 보고한다. 이는 새 후보의 10% 추가 blend이며, 209처럼 기존 CNN 한 seed를 치환하는 식이 아니다. Best021 대비 개선 ≥ 0.02·3/5 fold·CI 하한 > 0·상위 5개 제외 개선 > 0을 별도로 요구한다. 가중치 탐색은 없다.
- 단독 대조만 통과하면 해당 가설의 신호로만 기록한다. 앙상블 채택·최종 평가 선택·대규모 nested 재학습·제출로 자동 연결하지 않는다. 실패하면 이번 ratio 정책을 종료하며 다른 ratio·epoch·seed를 이어서 탐색하지 않는다.
- 같은 400장의 적응적 재사용이다. 조건부 CI는 누적 선택 편향과 재학습·seed 분산을 보정하지 않으며 독립 holdout이라고 부르지 않는다.

자원 상한은 CPU 2개, RSS+MPS 6 GB, swap 증가 0, 비용 0이다. 190에서 ConvNeXt 마지막 stage 5회는 약 227초였으므로 6회는 약 273초의 **과거 기록 기반 추정**이며 보장 시간이 아니다. 향후 최초 대조에서 실측하고 단일 fit 120초·학습 합계 900초의 보수적 상한을 넘으면 중단한다. 설치·다운로드·외부 API·holdout/test·전체 500장 학습·새 CSV·제출은 제안 범위에도 없다.

[중복 및 근거 감사](llrd-overfit-audit.md) · [확인한 파일·수치·해시](llrd-overfit-verification.json) · [209 보류 상태](209-status-after-review.md)
