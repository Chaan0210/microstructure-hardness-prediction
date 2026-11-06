# 213: 밝기 단일 증강의 첫 고정 비교

**CNN 단독 평균은 개선됐으나 사전 CI 기준 미달이다. 고정 첫 seed 치환도 확실한 개선이 아니므로 best021을 유지한다. 이번 첫 비교를 완료하고 추가 학습 없이 정지한다.**

| 비교 | 대조 RMSE | 밝기 RMSE | 이득(대조−밝기) | 조건부 paired 95% CI | 상위 5 ID 제외 이득 | 개선 fold |
|---|---:|---:|---:|---:|---:|---:|
| CNN 단독 | 15.03231784 | 14.83530291 | +0.19701493 | [-0.03537271, +0.43466043] | +0.05275347 | 4/5 |
| 고정 첫 seed 치환 | 12.78620078 | 12.77899632 | +0.00720445 | [-0.01303299, +0.02756500] | -0.00343315 | 4/5 |

단독의 사전 기준 중 이득 ≥0.02, 개선 ≥3/5 fold, 상위 5개 제외 양수는 통과했으나 CI 하한 >0은 실패했다. 앙상블 이득은 작고 상위 5개 제외 시 음수다. 상위 5개는 각 비교의 squared-error 이득이 가장 큰 ID를 따로 제거했다. 목록·fold 이득은 summary.json, 모든 400 ID의 타깃·예측·잔차·제곱오차 이득은 ID-residuals.json과 OOF.npz에 저장했다. CSV는 생성하거나 수정하지 않았다.

## 적용 전 검토와 유효성

HANDOFF, roadmap-30, 등록부와 212 REPORT를 확인했다. 등록부 CSV에는 204까지의 19개 행만 있어 최신 상태는 개별 206–212 기록을 함께 사용했다. 211 Huber 원본 개선과 212 두 재분할 악화는 보존했으며 213은 MSE를 유지했다.

206은 `mu + c*(x-mu)`로 영상별 평균을 보존하는 대비 감소다. 213은 `x + b`로 평균만 이동시키는 별도 비교다. `cnn_controls.py`의 전처리는 원본256 → PIL uint8 BOX128 → float32 /255이며, fold train320 전체 평균·표준편차로 고정 정규화한다. 영상별 평균 제거가 없고 TinyCNN readout에 입력 평균도 직접 들어간다. 따라서 밝기 이동은 정확히 상쇄되지 않는다.

원본 data/train 중 기존 개발400 ID만 EDA했다. 최초 읽기 전용 탐색과 기록 가능한 preflight를 수행했으며 이전 holdout100과 test는 열지 않았다. 원본은 모두 256×256 grayscale이다. 각 영상 밝기 평균의 범위는 0.497697–0.655347, 중앙값 0.570484, 영상별 표준편차 중앙값 0.074272다. 원본 일부에는 0 또는 255의 극소수 픽셀이 있어 원본에 직접 밝기를 더하면 포화 가능성이 있다. **실제 변환은 기존 BOX128 이후의 float 입력에 적용**했다. 원본 400장을 기존 경로로 다시 읽어 캐시와 exact 일치를 확인했으며 캐시 전체 범위는 0.066667–0.901961이다. 원본 min/max/mean/std/대비/포화/gradient energy/Laplacian std의 ID별 배열은 raw-quality-EDA.npz에 보존했다. 이 값들은 열화나 물성의 정답이 아니다.

강도는 각 fold train 영상 평균 IQR의 10%와 2/255 중 작은 값으로 사전 고정했다. 다섯 delta는 [0.004517949, 0.004596627, 0.0047158, 0.004767555, 0.004431969]이며 8비트 약 1.13–1.22 단계다. 강도 선택에는 validation과 타깃·성능을 쓰지 않았다. ID hash와 epoch의 고정4주기로 원본40회/밝게20회/어둡게20회를 배정해 추가 RNG 소비를 피했다.

정규화 뒤 이동은 0.06314–0.06751 표준편차다. 양 방향 모두 클리핑0, 재양자화0, 동일 좌표·크기·종횡비 유지, 픽셀 순위/동률 exact 보존, 대비와 공간 gradient 오차 1e-6 미만을 확인했다. 가상의 uint8 재양자화 오차도 preflight.json에 수치화했으나 실제 학습에는 적용하지 않았다. 결정립의 공간 형태에 영향을 주는 resampling은 추가하지 않았다. **이러한 연산 검사는 hardness 라벨의 물리적 불변성을 증명하지 않는다.** 특히 blur는 얇은 경계·결정립/상 측정도 바꿀 수 있으므로 밝기·감마·blur를 모두 라벨 보존 변환으로 간주하지 않는다.

## 대조·학습·검증 근거

원본400의 기존5fold(320/80), seed20261003+fold, TinyCNN128, D4, MSE, batch32, AdamW lr0.002/weight decay0.01, cosine80epoch/eta_min0.0001, clip5, 마지막5 checkpoint 동일 평균을 유지했다. 바꾼 변수는 훈련 입력의 밝기 이동뿐이다. 검증 입력은 원본이다.

- 새 주 학습 **5회**, 대조80epoch 재현 **1회**, 밝기 validation 라벨 NaN 교란80epoch 재현 **1회**: 총 **7회, 5,600 update**. 기존 대조5개는 재사용했다. 강도 탐색·감마·blur·재분할 학습은0회다.
- 206 당시 manifest의 소스·대조 파일 **54개 hash**를 확인했다. 대조 fold0의 평균 state와 epoch76–80 state, OOF, 모든800 order/D4/LR/loss/gradient norm/CPU·MPS RNG가 exact 재현됐다. 5개 대조 state는 기존 run009와 exact이고 전부 재추론했으며 ID/타깃/fold/대조 OOF 배열도 206과 exact다.
- 후보와 대조는 각 fold 초기 state 및 모든800 order/D4/LR/CPU·MPS RNG가 동일하다. 각 train영상의 노출 횟수는 20/20/40 exact다.
- 대조·후보10개 모두 저장된 마지막5 state를 독립 재평균하고 CPU/MPS 재로드했다. MPS 예측 최대오차0, CPU 최대오차 0.0000305176다.
- 전fold train 통계와 강도는 validation 이미지/타깃 교란에 불변이다. fold0 밝기 모델은 validation 타깃을 NaN으로 바꿔도 80epoch state·OOF·stream·최종 optimizer/scheduler가 exact다. 대조 검증은 기존206 라벨 교란과 이번 exact 경로 재현에 기반한다.
- 학습 전 plan/코드 hash를 잠갔다. 학습 후 validation/score의 복사된 설명 문구2곳만 수정했고, 원문·이전 lock·변경기록을 보존했다. 학습 코드·정책·점수·기준은 바뀌지 않았다.
- 고정 앙상블은 `Q=P021+0.081*slope_fold*(new_first_CNN-old_first_CNN)`이다. 194 계보 및 177 best021과 exact임을 재확인했다. 나머지2seed·보정·잔차가 고정된 출력 치환이며 전체 후단 재학습 효과가 아니다.
- paired bootstrap은 동일400 ID를 10,000회 재표집(seed202610213)했다. 직접 인덱싱 재계산으로 모든 bootstrap/CI/상위5개 제외/ID별 이득을 검증했다.

훈련 RMSE(중복 train1,600개)는 대조 9.885685, 밝기 10.374771다. 같은400 개발 표본을 누적·적응적으로 반복 선택했다. CI는 고정된 fit에 조건부이며 선택 편향·seed 변화·재학습 불확실성을 보정하지 않는다. 새 독립 평가로 해석하지 않는다.

## 다음 판단과 보존

단독 +0.1970, 4/5 fold, 상위5개 제외 +0.05275는 **약한 탐색적 신호**다. 채택 기준을 통과한 신호로 바꾸어 표현하지 않는다. 우선순위는 낮지만 별도로 이 가설의 안정성을 확인한다면 기존20261031/20261101 두 재분할에서 똑같은 강도 산식·노출·seed·MSE·예산·고정 치환만 유지하고 각각5fold를 비교한다. 강도는 각 새 fold train에서 같은 산식으로 계산하며 결과를 보고 조정하지 않는다. 각각 및 pooled에서 paired ID CI·상위5개 제외·양 분할 악화를 보고, 같은400이므로 독립 검증이라는 주장은 하지 않는다. 정확한 기존 대조를 hash/경로로 검증해 재사용하고 재현 불가 시만 동예산 대조를 실행한다. **이번 세션에서는 제안만 하고 실행하지 않았다.** 감마와 blur는 별개의 미수행 항목으로 남겼다.

기존파일 643개와 원본400 이미지 hash 보존, best021 SHA256 `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b` 유지. 기존 uv .venv, CPU2 threads/MPS를 사용했고 측정 최대 RSS+MPS는 0.731GB, swap 증가0, 비용0이다. 설치·다운로드·AI·회사자료·Slack·인증정보·test·이전holdout·전체500학습·새CSV·업로드·제출은 없었다. 모든 새 결과는 이213 폴더에만 저장했다. 차단 사항은 없다.

근거: plan.json, preflight.json, raw-quality-EDA.npz, training-complete.json, validation.json, final-audit.json, summary.json, ID-residuals.json, bootstrap.npz, preservation.json 및 fold별 state/streams.
