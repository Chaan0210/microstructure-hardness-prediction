# 다음 단일 가설: TinyCNN 128/256의 동일 업데이트 예산 비교

**상태: 제안만 작성. 실험 번호 미부여, 학습·자원 probe·재분할 모두 미실행.** 이 문서는 215의 결과를 바꾸지 않으며 후속 실행을 자동 예약하지 않는다.

기존 run009의 Tiny128/80 epoch RMSE는 15.032318, run010의 Tiny256/40 epoch는 14.621452였다. 학습 길이와 cosine 스케줄도 달라 해상도의 순수 효과로 해석할 수 없다. 남은 항목 중 프로젝트 내부 근거와 실행 가능성을 고려해 이 혼재를 해소하는 한 비교를 우선 제안한다. 개선 규모나 성공 확률을 보장하지 않는다.

## 고정 비교

| 항목 | 정확한 조건 |
|---|---|
| 유일한 비교 정책 | A: 기존 BOX128 입력. B: 같은 시야의 native256 입력. 두 군 모두 80 epoch. |
| 자료 | 215 plan.json의 동일 400 ID·동일 original 5-fold, fold마다 train320/validation80. 타깃·그룹은 기존 development 전용 캐시에서만 로드한다. 원시 train.csv 전체나 holdout/test는 열지 않는다. |
| A 입력 | 원본 grayscale256 → 기존 PIL BOX resize128의 uint8 출력 → float32/255. 기존 nested-calibration-074 캐시와 exact 재구성 확인. |
| B 입력 | 같은 원본 grayscale256 → resize 없이 float32/255. crop·zoom·종횡비 변환 없음. 허용 ID와 원본 SHA256을 215 기록에 대조. |
| 모델 | 동일 cnn_controls.TinyCNN, 91,905 parameters, 동일 GroupNorm·SiLU·head·dropout0.15. 해상도 외 구조 변경 없음. |
| 정규화 | 입력은 각 군의 해당 fold 원본 train 픽셀 전체 mean/population std, std floor1e-6. 타깃은 동일 train320의 mean/population std. per-image 정규화·validation 통계 사용 없음. |
| seed/노출 | 모델 torch.manual_seed(20261003+fold), NumPy default_rng(20261003+fold). 매 epoch train320 permutation, batch별 rot90 및 reflection은 기존 순서 그대로. 각 영상 80회. |
| 증강 | 두 군 같은 D4만 사용. 206·213–215의 대비·밝기·감마·블러, noise, crop, scale 증강 없음. |
| 손실/optimizer | train-standardized MSE, AdamW lr0.002/weight_decay0.01, gradient norm clip5. 기존 코드 경로의 optimizer defaults 유지. |
| 스케줄/예산 | batch32, 10 updates/epoch ×80 = 800 updates/fit. CosineAnnealingLR T_max80, eta_min0.0001. early stopping·best epoch 선택 없음. |
| 최종 state | epoch76–80 state를 기존 순서의 동일 가중 평균(각 state/5를 누적)으로 저장. EMA 없음. |
| 평가 | 원본 validation 입력, batch32, TTA·출력 clipping 없음. 기존 float32 역표준화 경로 유지. |
| 환경 | 같은 프로젝트 uv .venv, 두 군 같은 CPU/MPS 장치, CPU/BLAS2 threads. 설치·다운로드·비용 발생 없음. |

이는 **같은 업데이트 수**의 비교이며 동일 FLOPs나 시간 예산은 아니다. 동일 CNN의 픽셀 receptive field가 차지하는 원본 시야 비율도 달라지므로, 손실 없는 정보량만 분리하는 실험이 아니라 입력 해상도 정책의 비교다.

## 실행 가능성과 대조 재사용

1. 점수 확인 전에 plan, 입력 ID/fold, source hash, 검증 기준을 새 출력 폴더에 잠근다. 완료된 213–215 폴더를 재실행하지 않는다.
2. native256 train32개로 **8 update 자원 probe 한 번**만 수행한다. seed와 optimizer를 처음부터 초기화한 별도 모델이며 가중치를 버리고 성능 선택에 쓰지 않는다. probe도 학습 update로 따로 기록한다. CPU/MPS 합산 보수적 RSS+MPS가 6,000,000,000 bytes를 넘거나 swap이 증가하면 중단한다. batch·epoch를 줄이는 구제 실험은 하지 않는다. probe가 가능하다는 사실만으로 전체 단계의 메모리 제한 검사를 생략하지 않는다.
3. A fold0을 80 epoch 재현한다. 기존206 state·epoch76–80·OOF와 800회 order/D4/LR/loss/gradient norm/RNG가 exact이고, 나머지4개 대조도 source/hash·정규화·재추론이 검증되면 기존 A 5개를 재사용한다.
4. 기존 대조와 exact 재현되지 않으면 저장된 A OOF를 대응 대조로 쓰지 않는다. 원인을 기록하고 현 실행의 A fold0을 포함해 **같은80 epoch의 새 A 5개**를 확보한다. B도 같은 환경·경로로5개만 학습한다. source/입력 오류가 원인이면 비교 자체를 중단하고 오류부터 수정한다.
5. B 5개를 학습하고 fold0의 validation 타깃을 NaN으로 바꾼 80 epoch 재현1회로 누출을 검사한다. 새 A 대조 경로를 쓴 경우 A fold0의 라벨 교란 재현도1회 한다.

예상 실행 수는 exact 대조 재사용 시 **완전 fit7회/5,600 updates + probe8 updates**(B5, A 재현1, B 교란1)다. 대조 재사용 불가 시 **완전 fit12회/9,600 updates + probe8 updates**(새 A5, B5, 양 군 교란2)다. 중단·실패한 시도도 별도로 세며 성공 fit 수에 숨기지 않는다. 자원 probe와 본 학습 모두 새로운 지시 이후에만 실행한다.

## 기술 검증과 통계

- ID·fold·그룹 분리, train-only mean/std, validation 타깃 교란에 대한 정규화 불변성을 전 fold에서 확인한다.
- 두 군의 초기 state, 모든800 order/D4/LR/CPU·MPS RNG를 대조한다. 입력과 이후 gradient/state는 해상도 효과로 달라지는 것이 정상이다.
- 모든 새 모델의 저장5state를 독립 재평균한다. MPS 재로드 예측은 exact, CPU와의 최대 절대오차 허용치는 기존 기준0.001로 고정한다.
- 라벨 교란 replay의 state·OOF·stream·최종 optimizer/scheduler가 해당 원래 fit과 exact여야 한다.
- 400 ID별 예측·잔차·squared-error 이득을 JSON/NPZ로 저장한다. CSV 생성 없음.
- paired bootstrap은 동일400 ID를 10,000회 재표집, seed **202610256**으로 고정한다. 동일 ID의 양 군 관측을 함께 뽑고 RMSE 차이를 계산한다. 직접 인덱싱으로 재검산한다.
- 상위5개 제외는 각 비교에서 `SE_control−SE_candidate`가 가장 큰5개 ID를 따로 제거한다. 동률은 원래 ID 배열 순서를 유지하는 stable sort로 처리한다. 전체·fold별 이득, CI, 제외 ID와 제외 후 이득을 모두 보고한다.

고정 첫 seed 치환은 `Q_arm=P021+coef_fold*(C_arm−C_old_first)`이며 `coef=0.081*기존 slope`를 그대로 사용한다. exact A 재사용 시 Q_A=P021이어야 한다. 새 A를 학습한 fallback에서는 **Q_A가 P021과 달라질 수 있음**을 명시하고, `Q_A−Q_B`의 대응 정책 효과와 `P021−Q_B`의 현재 기준 대비 차이를 따로 보고한다. 그 둘을 혼동하거나 baseline 차이를 해상도 이득으로 포함하지 않는다.

## 사전 판단과 중단

단독 screening은 기술 검증 통과, RMSE 이득≥0.02, 개선fold≥3/5, paired95%CI 하한>0, 상위5개 제외 이득>0을 모두 요구한다. 고정 첫 seed 치환은 별도 결과로 보고한다. 전체 앙상블 후보로 후속 검토를 제안하려면 best021 대비 고정 치환에서도 같은≥0.02/3fold/CI하한/상위5개 기준을 충족해야 한다. fallback이면 새 A 대비 paired 치환 이득도 양수이고 CI 하한·상위5개 제외 이득이 양수여야 한다.

어느 기준이든 통과했다고 즉시 best021을 교체하지 않는다. 원본 첫 비교에서 항상 보고 후 정지한다. 같은 개발400의 누적 선택 편향과 seed·재학습 변동은 이 CI에 반영되지 않는다. 추가 재분할·다른 해상도/epoch·seed 확대·전체 nested 재학습은 자동 실행하지 않는다. 독립 확인 및 전체 파이프라인 개선은 후속 판단으로 남긴다. 기존204 대규모 nested 분기를 다시 시작하는 계획이 아니다.

원본·best021을 보존한다. no install/download/AI/test/holdout/full500/CSV/submission 조건을 유지하며 다른 작업자의 스킬·diagram 파일은 수정하지 않는다.
