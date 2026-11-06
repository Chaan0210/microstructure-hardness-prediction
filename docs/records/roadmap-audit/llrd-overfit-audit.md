# 185–190 대조: LLRD와 사전학습 CNN의 과적합 완화

**낮은 LR·head/backbone 차등 LR·제한 미세조정의 도입은 이미 완료됐다. 깊이별 LR 배분 대조는 미실시다.** 기존 ResNet 분기는 종료 상태를 유지한다. 새 구조나 다운로드 없이 ConvNeXt의 이미 학습한 마지막 세 블록 안에서 LR 배분 하나를 비교할 수 있다. 다만 과적합의 원인을 특정한 것은 아니므로 **제한된 가설 확인 계획으로만 정리하며 실행하지 않는다.** 최종 앙상블 개선을 기대할 강한 근거는 아직 없다.

209의 현재 관리 상태는 [보류](209-status-after-review.md)다. TinyCNN 단독 EMA 개선을 다른 백본에도 보장되는 효과로 가정하지 않는다. 204의 대규모 nested 재학습을 자동 반복하지 않는다.

## 실제 수행한 비교

아래는 반복 사용한 개발 400장의 원본 5fold 결과다. “결합”은 각 후보를 기존 best021에 **10% 추가한 고정 blend**이며, 209의 기존 CNN 첫 seed 출력 치환과 다르다. Best021 OOF RMSE는 12.7862007758이다.

| 실험 | 실제 변경·학습 설정 | 결과 | 중복 판단·남은 질문 |
|---|---|---|---|
| **185 완료** | ResNet50·EfficientNet-B0. 전체 256 영상을 고정 224로 resize. 동결 특징의 head 100 epoch, 추가 head만 12 epoch, 마지막 stage+head 12 epoch의 세 arm. 미세조정은 AdamW, tail 1e-5/head 3e-4. | ResNet 단독 18.691826 / 18.909963 / 16.710664, 결합 12.916410 / 12.924531 / 12.901771. B0 단독 17.522142 / 17.625291 / 17.461952, 결합 12.992189 / 12.995323 / 12.988033. 모두 021 미달. | 낮은 LR, head/backbone 차등 LR, 마지막 stage 미세조정, 추가 head 학습을 통제한 비교는 완료. 깊이별 decay는 없음. |
| **186 완료·수치 한계 기록** | 같은 두 백본에서 outer train 안의 3fold로 네 recipe 선택. AdamW tail/head 1e-5/1e-4 또는 1e-4/1e-3, SGD 1e-3/1e-2 또는 1e-2/1e-1. 모두 12 epoch. | 선택 arm의 ResNet 단독/결합 16.084005/12.907364, B0 17.094628/13.004753. ResNet의 AdamW·SGD family 결합도 12.907992·12.885459로 미달. | optimizer·전체 tail LR 재탐색은 중복. 작은 feature train 분산으로 한 innerfold의 Z가 약 22,984까지 증폭한 이슈가 있어 이를 순수 과적합이나 백본 한계로 해석하지 않음. |
| **187 완료** | 186과 같은 네 recipe·선택 절차를 유지하고 feature 표준편차에 train RMS floor만 추가. 각 train에서 scaler와 100 epoch head를 새로 적합. | 선택 ResNet 단독/결합 15.536522/12.909169, B0 17.253049/12.992554. strict CPU 검증 통과. ResNet은 다섯 fold 모두 AdamW high 선택. | 수치 안정성 수리는 완료됐지만 앙상블 개선으로 이어지지 않음. 스케일 수리와 정규화 강도 대조는 다른 질문. |
| **188 완료** | ResNet layer4·187에서 고정한 recipe와 예산으로 고정 224→native 256만 변경. 무작위 scale augmentation 아님. | 단독 15.536522→15.423283, 결합 12.909169→12.897651. 결합 개선 CI [−0.037185, +0.063599]. | 원해상도 최초 도입은 중복. 일관된 근거 부족으로 해상도 탐색 종료. |
| **189 완료·분기 종료** | ResNet layer4→layer3+4. 두 stage에 **같은** tail LR 1e-4, head 1e-3. 추가 layer 전용 LR 없음. | 단독 15.536522→15.489643, 결합 12.909169→12.872903. 단독 2/5 fold 개선, 결합 개선 CI [−0.045277, +0.115355]. 여전히 021 미달. | 더 많은 층을 푸는 비교는 완료. LLRD 대조가 아님. ResNet 확대 분기 종료를 유지. |
| **190 완료** | ConvNeXt-Tiny·Swin-V2-T, native 256. frozen100 / head_only112 / last_stage112. AdamW tail 1e-5/head 1e-4, 마지막 stage 전체는 같은 LR. ConvNeXt는 기존 세 CNBlock 및 terminal LayerNorm만 학습. | ConvNeXt 단독 16.504625 / 16.615123 / 16.764185, 결합 12.952081 / 12.953058 / 12.958639. Last stage는 matched head 대비 단독 **−0.149062**, CI [−0.250839, −0.048067], 개선 0/5. Swin 단독 16.593247 / 16.712738 / 16.947878, 결합 12.904978 / 12.907535 / 12.918068. | 새 백본 도입·동결 대비 마지막 stage 학습은 반복하지 않음. ConvNeXt 검증 통과. Swin의 strict 재현 검증 실패 기록은 유지하며 이번 후보에서 제외. ConvNeXt 내부 block별 LR 대조는 없음. |

근거: 각 [185](../../../experiments/01-representations/0185-pretrained-cnn/plan.json)·[186](../../../experiments/01-representations/0186-cnn-optimizer/plan.json)·[187](../../../experiments/01-representations/0187-cnn-scale-floor/plan.json)·[188](../../../experiments/01-representations/0188-cnn-resolution/summary.json)·[189](../../../experiments/01-representations/0189-cnn-unfreeze/summary.json)·[190](../../../experiments/01-representations/0190-modern-backbones/summary.json)의 계획·결과 및 실제 optimizer 그룹 코드. 확인 파일 해시는 [검증 기록](llrd-overfit-verification.json)에 있다.

## train/validation 격차가 말해 주는 범위

Train 수치는 각 fold의 train RMSE 제곱을 평균한 뒤 제곱근을 취한 진단이다. 같은 ID가 여러 fold의 train에 반복되므로 새로운 표본 수로 세지 않는다. OOF는 각 ID의 validation 예측 하나로 계산한 기존 결과다.

| 조건 | 최종 train RMSE | 기존 OOF RMSE | 해석 |
|---|---:|---:|---|
| 185 ResNet 미세조정 | 1.437116 | 16.710664 | train 적합 자체는 가능. 불안정 feature scale의 영향도 분리해야 함. |
| 186 ResNet 선택 recipe | 2.100922 | 16.084005 | 낮은 train 오차만으로 optimizer 선택 성공이나 일반화 개선을 주장할 수 없음. |
| 187 ResNet layer4, RMS floor | 1.718471 | 15.536522 | 수치 안정성 수리 뒤에도 큰 격차가 남음. |
| 188 ResNet native 256 | 1.639855 | 15.423283 | 단순 해상도 확대가 격차를 해결하지 못함. |
| 189 ResNet layer3+4 | 3.931462 | 15.489643 | train 오차도 증가했고 clip 비율은 99.5%. 깊이 확대가 더 잘 최적화됐다는 근거가 없음. |
| 190 ConvNeXt frozen100 | 5.849855 | 16.504625 | 미세조정 이전부터 큰 격차가 존재. tail만이 유일한 원인이라고 할 수 없음. |
| 190 ConvNeXt head_only112 | 5.379704 | 16.615123 | 추가 head 학습은 train을 개선하면서 OOF를 5/5 fold 악화시킴. |
| 190 ConvNeXt last_stage112 | 3.808715 | 16.764185 | matched head보다 train은 더 낮지만 OOF는 5/5 fold 악화. 제한 미세조정의 과적합 완화를 검토할 직접 동기. |

ConvNeXt last stage의 train 곡선은 epoch 0/3/6/9/12에서 **5.849855 / 5.265044 / 4.578711 / 4.075372 / 3.808715**다. Head-only는 **5.849855 / 5.745508 / 5.577513 / 5.438275 / 5.379704**다. Train 개선과 최종 OOF 악화는 과적합과 양립하는 관찰이다. 다만 중간 epoch별 OOF 곡선이 저장된 것은 아니므로 최적 early-stopping epoch를 사후 선택할 수 없다. Clip 비율도 last stage 99.75%, head-only 71.67%여서 “학습이 완전히 수렴했다”거나 “깊은 블록의 LR이 반드시 더 커야 한다”는 결론은 나오지 않는다.

## 로드맵 항목의 현재 상태

| 항목 | 상태 | 중복 판단·남은 질문 |
|---|---|---|
| 낮은 LR·head/backbone 차등 LR | **완료** | 185–190에서 실행. 이를 LLRD 최초 적용이라고 바꾸어 반복하지 않음. |
| 깊이별 LLRD | **미실시** | 189 두 stage와 190 세 블록은 각각 하나의 tail LR 그룹이었다. 실행 코드·계획에서 깊이별 decay를 확인하지 못함. 전체 backbone 확대 없이 마지막 세 블록 안의 상대 배분만 새 질문으로 남김. |
| 동결 prefix·제한 stage 학습 | **완료** | 185–190. ResNet 동결 범위 확대의 동일 비교는 종료. |
| Head dropout 0.25·weight decay 0.01·global clip 1 | **부분** | 모두 실제 사용했으나 강도만 바꾸는 대조는 미실시. “규제가 없었다”는 전제는 틀림. 동시에 바꾸지 않음. |
| Feature RMS floor·타깃 표준화 | **완료** | 187 이후 train 전용 RMS floor, 전 실험 train 전용 target 표준화. 재도입은 중복. |
| SmoothL1/Huber·BN 고정·backbone eval | **부분** | SmoothL1 beta 1 사용, ResNet BN affine/stat 고정, ConvNeXt LayerNorm 학습. Backbone eval로 stochastic depth 비활성화. 각각의 단독 제거·강도 비교는 미실시. |
| 학습 길이·early stopping | **부분** | frozen100→head-only112는 완료됐고 ConvNeXt에서 악화. 다른 epoch 탐색이나 inner 전용 early stopping은 미실시. 이번 비교에서는 고정 12 epoch 유지. |
| 사전학습 백본 EMA | **미실시·이번 제외** | 208–209는 TinyCNN이다. ConvNeXt에 같은 이득을 가정하지 않으며 LLRD와 동시에 넣지 않음. |
| 새 구조·추가 unfreeze·해상도/optimizer grid | **기존 비교 종료** | 185–190과 실질적으로 같은 제안은 실행하지 않음. 127의 학습곡선·오차 상관도 새로 반복하지 않음. |

## 가장 작은 새 대조의 정당화와 한계

[210 계획](next-experiment-210.md)은 **ConvNeXt 마지막 stage의 세 블록에 대한 평균 LR을 맞춘 blockwise LLRD** 한 정책이다. 기존의 uniform tail LR과 비교하며 trainable parameter 수, head, feature scale, 입력, 학습 길이를 유지한다. 먼저 190 대조 재현이 가능해야 한다.

이는 낮은 train 오차와 실제 OOF 악화가 함께 관찰된 조건에서 **상대적인 깊이별 업데이트 배분을 바꾸는 것이 도움이 되는가**라는 제한된 질문이다. “LLRD가 과적합을 해결한다”는 확인된 사실이나 최종 앙상블 성공 예측은 아니다. 앞 블록 손상이 원인이라는 직접 증거는 없고, head-only도 악화했기 때문에 tail LR만으로 전체 문제를 해결할 가능성을 과장하지 않는다.

정책을 이론상 분리할 수 있으므로 기존 실험과 동일한 반복으로 닫지는 않는다. 다만 **계획만 보관하고 실행은 보류**한다. 한 정책에서 실패하면 ratio·seed·epoch를 바꾸며 탐색하지 않는다. 성공해도 기존 frozen/head-only 비교와 best021 결합 조건을 따로 통과해야 하며, 자동 nested 재학습이나 제출로 이어지지 않는다.
