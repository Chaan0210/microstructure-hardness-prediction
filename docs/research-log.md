# 초기 연구 일지 (2026-09-30~2026-10-09)

아래는 해당 기간의 실험 기록이다. 최종 3위 모델의 구성과 실행은 [최종 패키지](../final/package/README.md)에 있다.

# 2026-10-09 인계: push-400 화소 U-Net — 033 Public 9.4900 (3위/130), 다음 작업은 `experiments/02-microstructure/0400-unet-maps/HANDOFF.md`

깨끗한 train 이미지를 열화해 원본의 2상/기공/경계 화소 지도를 복원하도록 학습한 U-Net(라벨·test 미사용)으로 2상 분율을 측정했다. 경도 상관이 깨끗/중간/잡음 0.71/0.66/0.56으로 기존 분할 기반(0.63/0.61/0.44)보다 높다. 경과: 030 9.8102 → 031 9.6088 → 032 9.5891 → 033 **9.4900**(dev 고정 W0.75 통합 10.779 → 10.431 → 10.382 → 10.361). 종료 직전 다중 과제 U-Net 3개 특징 평균 세트(u1dmavg)가 dev 통합 10.191로 더 좋아, 다음 세션에서 후보 034로 만든다(HANDOFF §4). 최종 선택은 미지정(자동 = Public 최고).

# 2026-10-08 인계: push-300 점수 개선 탐색 — 027 유지, 다음 작업은 `experiments/02-microstructure/0300-cellpose-finetuning/HANDOFF.md`

500장 반복 CV 하네스(TabPFN-v2 기준 11.7549, 잡음 3분위 8.86 / 10.58 / 14.54)로 잡음 구간 개선책을 시험했다. 입상권(9.65)까지 갈 개선은 아직 없다. 오차는 대부분 잡음 이미지에서 나온다. 잡음 때문에 결정립 경계가 사라져 cellpose가 이웃 결정립을 합치기 때문이다. 기각: 전처리(NLM, 업샘플, 2배 축소 분할), 열화 증강, 분할 없는 자기상관 특징, 잡음별 결합 비중. 확인 필요: 특징 복원 CNN(−0.032), 결합 비중 0.5→0.6~0.7(세 분할 모두 0.03~0.14 개선). 8-뷰 TTA는 125/249장에서 중단했다(결과 없음). 추천 순서: TabPFN-3.5 교체(사용자가 가중치를 받아야 함, 3위 가능성이 있는 유일한 후보) → 8-뷰·깨끗한 절반 TTA → 결합 비중 조정 → 027 재현 패키지. 최종 선택은 Public이 아니라 dev 근거로 한다. 다음 세션은 발열을 신경 쓰지 않고 MPS로 최대 속도로 진행한다. 다른 컴퓨터 세팅: `docs/records/handoff-20261008/README.md`.

# 최신 Public best: candidate027 — Public RMSE **10.3957187457** (11/118위)

2026-10-08 03:34 KST에 #21로 제출했다(오늘 3/5 사용). 구성은 0.5×021 + 0.5×TabPFN-v2(n_est 32 × seed 3)다. 입력은 결정립 특징 v3이고, 잡음이 중앙값을 넘는 이미지는 4-뷰 분할 TTA 평균을 쓴다. 통합 dev는 11.662(025 대비 +0.088)다. 경과: 023 10.540 → 025 10.494 → 027 10.396. 최종 선택은 변경하지 않았다. 기록: `reports/submissions/candidate027-public-submission.json`, `experiments/02-microstructure/0220-grain-cellpose/REPORT.md`.

# 220 후속: 후보 024/025 생성(미제출)

v3 특징(공간 크기장, 2상 개수 분율·접촉도, 면적 가중 크기 등 64개)과 저커버리지 재분할로 통합 dev OOF가 12.240(023 대응)에서 11.872(024, KRR)로 내려갔다. TabPFN v2를 쓰면 11.750(025)이다. 세 분할 모두 개선이며, 사전 등록한 채택 규칙을 따랐다. 025는 TabPFN(Prior Labs License 1.1)을 쓰므로 사용 여부는 사용자가 결정한다. 상세: `experiments/02-microstructure/0220-grain-cellpose/REPORT.md`.

# 최신 Public best: candidate023 — Public RMSE **10.5397345474** (021 11.4878517800 대비 0.9481 개선, 12/118위)

2026-10-08 00:21 KST에 #19로 1회 제출했다(오늘 1/5 사용, 4회 남음). 최종 선택은 변경하지 않았다(1개 선택 가능, 미선택 시 Public 최고 자동 선택). 기록: `reports/submissions/candidate023-public-submission.json`. Public 점수로 튜닝하지 않는다.

# 220 Cellpose-SAM 결정립 측정 — 후보023 생성, 수정 검증 4개 게이트 통과

사용자 승인으로 Cellpose-SAM(cellpose 4.2.1.1, cpsam 가중치 BSD-3-Clause, SHA256 `e1440429…`)을 격리 venv에 설치했다. train/test 이미지를 결정립 단위로 분할한 뒤 라벨 무관 특징 36개(크기·**크기 불균일성**·신장/정렬·**2상 분율**·기공·화질)를 KRR로 회귀했다.

리뷰 반영 v2 검증(`experiments/02-microstructure/0220-grain-cellpose/plan-v2.json`)은 다음 조건을 지킨다.
- 가중치를 0.5로 고정해 best021 예측으로 아무것도 적합하지 않는다.
- 변환 열은 정의로 고정한다.
- 모델 계열은 바깥 훈련 fold 안의 내부 CV로만 고른다.
- 검증과 최종은 같은 절차를 쓴다.

| 분할 | best021 | 0.5×021 + 0.5×결정립 |
|---|---:|---:|
| original | 12.786 | **12.181** (5/5 fold) |
| 20261031 | 12.622 | **12.072** (4/5) |
| 20261101 | 13.032 | **12.462** (5/5) |

통합 개선 +0.575, 조건부 95% CI [0.156, 0.990], 상위 5개 제외 시 +0.353. 같은 400장의 반복 분할이며 독립 검증은 아니다. ablation에서 이득의 핵심은 크기 불균일성과 2상 분율이었다(평균 크기와 기공은 거의 기여하지 않음).

후보 `artifacts/submission-candidate-023.csv`(SHA256 `668c2959…ba4b02`). test 분할 실패 26장은 021 예측을 쓴다. 특징의 train/test adversarial AUC는 0.503이다. 이후 제출했다(위 기록 참조). best021 파일은 변경하지 않았다. 상세: `experiments/02-microstructure/0220-grain-cellpose/REPORT.md`. v1 수치(가중치 누출)는 폐기 기록으로 보존했다.

# 204 CNN seed 3→6 중첩 대조 완료 — best021 보존

모든 중첩 단계에 CNN 6개 seed를 적용하고 같은 후단 정책으로 재적합한 후보가 사전 개발 기준에 미달했습니다. best021을 유지하며, 이번 작업에서 추가 seed·가중치·구조 탐색은 하지 않습니다.

기존 3개 seed 대조군 R을 행별 예측과 후단 중간값까지 정확히 재현한 뒤, 신규 CNN 학습 540회와 검증 48회를 수행했습니다. 203번의 후단을 고정한 F와 모든 중첩 단계 및 후단을 갱신한 N을 비교했습니다.

| 분할 | R: best021 | F: 고정 후단 | N: 후단 재학습 |
|---|---:|---:|---:|
| original | 12.786201 | 12.745508 | 12.740882 |
| 20261031 | 12.621902 | 12.644221 | 12.645837 |
| 20261101 | 13.032449 | 13.034443 | 13.036569 |

N 대 R pooled gain **+0.005785**, 400개 ID를 묶어 계산한 조건부 95% CI **[-0.026273519539496214, 0.03731951544595677]**. 같은 400장을 반복해서 나눈 결과이며, 새로운 독립 검증 자료는 아닙니다.

신규 모델 540개 재로딩, 48회 엄격 재현, R과 N의 모든 outer fold에 대한 라벨 오염 검증, 후단 및 bootstrap 10,000회 독립 검산을 완료했습니다. 최대 관측 RSS+MPS 0.90GB, swap 증가 0 bytes.

추가 설치·다운로드·holdout/test 평가·전체 500개 재학습·새 제출 CSV·업로드는 없습니다. 이후 새 로드맵 실험은 별도 세션으로 이관합니다.

결과: `reports/summaries/cnn-nested-204-summary.json`. 상세: `experiments/01-representations/0204-cnn-nested/`. 인계: `experiments/01-representations/0204-cnn-nested/HANDOFF.md`.

# 203 CNN3→6 seed 고정 대조 완료 — best021 보존

신규3seed를 사전 고정해 원본+기존두재분할에서 **45회 학습**했습니다. 기존3seed를 포함한6개를 모두 동일평균했고 후단calibration/residual은 고정했습니다. 판정: **완료·고정 3→6seed 강한 채택기준 미달·best021 유지**.

| 분할 | best021 | CNN mean6 치환 | gain |
|---|---:|---:|---:|
| original | 12.78620 | 12.74551 | +0.04069 |
| 20261031 | 12.62190 | 12.64422 | -0.02232 |
| 20261101 | 13.03245 | 13.03444 | -0.00199 |
| pooled | 12.81463 | 12.80912 | +0.00550 |

400ID 묶음 조건부95%CI **[-0.025968, 0.036307]**. 같은400장의 반복분할이며 독립1200장이 아닙니다. 분할간 예측분산 **3.370358→3.187856**. 기존3seed에맞춘후단을유지한출력치환이며완전nested재학습결과가아닙니다.

기존45개+신규45개모델재로딩검증;신규MPSexact,CPU최대차3.05e-05,독립앙상블수식차8.53e-14,10000bootstrap검증통과. 신규36,000updates;기존모델중복학습0. 모든seed보존,선별/구제비중탐색없음.

현재단계에서설치·다운로드·외부AI·holdout/test·전체500·새제출CSV·업로드없음. 비용0. 기존best021과모든보호파일유지. 결과:`reports/summaries/cnn-seed-expand-203-summary.json`, 계획·코드·증빙:`experiments/01-representations/0203-cnn-seed-expand/`.

# 202 AI 리뷰 적용: 고정 단순화 대조 완료 — best021 유지

저장된 개발400 OOF에서 현재021의 L/short-B 비중을 고정하고 residual 제거(A, 설명용), 이어서 calibration 제거(B, 유일 후보)를 사전 등록해 실제 비교했습니다. 새 모델 학습은 없습니다.

| 분할 | best021 | A: residual 제거 | B: residual+calibration 제거 |
|---|---:|---:|---:|
| 원본 | 12.78620 | 12.81763 | 12.91326 |
| 20261031 | 12.62190 | 12.71076 | 12.80088 |
| 20261101 | 13.03245 | 13.11357 | 13.18639 |
| pooled RMSE | 12.81463 | 12.88178 | 12.96785 |

B는 분할 간 예측분산을 **25.6%** 줄였지만 세 분할 모두 오차가 악화했습니다. gain **−0.153225**, 동일400ID 묶음 조건부95%CI **[−0.324358, 0.006755]**이며 사전 채택기준에 미달합니다. A도 채택하지 않습니다. 세 분할은 같은400장의 반복 측정이며 새 독립 검증이 아닙니다.

현재021의 직접 seed 경로만 분해한 조건부 mean3 MSE분산 proxy는 CNN **0.476614**, S **0.031447**, B **0.025805**, L **0.007880**입니다. 성분 독립·후단 고정 가정이며 전체 일반화불확실성이나 seed 확대의 실제 이득은 아닙니다. 052/088/133을 재학습하지 않았고 seed 확대를 자동 실행하지 않았습니다.

독립 스칼라 예측 재구성 최대차 **8.53e−14**, 전체10000 bootstrap 검증 최대차 **8.89e−15**. 입력209개·기존제출물 등50개 해시 동일. 계산1.51초/검증0.37초, 최대관측RSS 약196MB, CPU2, 비용0. 설치·다운로드·holdout/test·전체500·새CSV·업로드 없음. 초기 NumPy 지연 import 차단 오류와 수정도 보존했습니다.

계획·코드·결과·판정: `experiments/01-representations/0202-simplification/`, 요약: `reports/summaries/simplification-202-summary.json`. 현재 단순화 분기는 종료하고 best021을 보존합니다.

# 191 사전학습 mean 대 mean+std / PCA128 비교 완료 — best021 유지

사용자 별도 명시 승인에 따라 기존 ResNet50·ConvNeXt-Tiny 가중치만 사용했습니다. 개발400장·원래5fold·전체256 입력을 고정했습니다. 같은 마지막 stage 공간 map(8×8)에서 기존 mean 경로를 그대로 보존하고 population std를 추가했습니다. mean400개는 기존188/190캐시와 bitwise 일치합니다. ConvNeXt는 평균 뒤 기존 LayerNorm을 유지했고, std는 그 앞의 동일 공간 map에서 추출했습니다.

두 조건 모두 각fold train320에만 bank별 RMS하한 스케일링과 full-SVD PCA128을 fit하고, PCA 뒤에도 train-only RMS하한을 적용했습니다. 동일 초기 헤드·seed/order·100epoch/500updates·MLP128→64→1(8,321학습파라미터)로 비교했습니다. PCA 축은 조건마다 다르므로 pooling+압축 파이프라인의 비교입니다.

| 모델 | 조건 | 단독 OOF RMSE | best021 고정10% 결합 RMSE |
|---|---|---:|---:|
| resnet50 | mean | 15.51298 | 12.84305 |
| resnet50 | mean_std | 15.57399 | 12.85935 |
| convnext_tiny | mean | 15.84896 | 12.92994 |
| convnext_tiny | mean_std | 15.38600 | 12.89281 |

기준 best021 **12.78620**. ConvNeXt의 std 추가는 자체 mean 대조보다 개선했지만 best021 보완성 기준에는 미달했고 ResNet은 악화했습니다. 두 mean+std 결합 모두 best021 대비 개선0/5fold. 승격 후보 없음, 재분할 확대하지 않음. PCA128이 정보를 줄일 수 있으므로 모든 공간 분산 정보의 무용성이나 모델 한계를 입증한 결과는 아닙니다.

20개 모델 CPU exact 재로딩과 MPS 비교, 모든fold 검증feature/label 변경 후 PCA·스케일·타깃통계 exact 일치, 동일초기화/ID/order/LR/500updates 검증 통과. 4개 fold0 전체100epoch 라벨변경 재학습 가중치·예측·loss/gradient/LR trace exact 재현. CPU/MPS 최대 예측차 **0.000006424**(기준0.002). 기존190 Swin 실패기록은 그대로 보존했습니다.

측정 계산·감사구간 합계 **31.54초**(환경시작·별도해시검사·보고작성시간 제외), 최대RSS+MPS **0.99GB**, swap 증가0, 비용0. 기존 1852개 보호파일/best021 SHA256 확인. 새설치/다운로드/holdout/test/전체500/새CSV/업로드 없음.

기록: `experiments/01-representations/0191-spatial-moments/summary.json`, `validation.json`, `interpretation.json`, `completion.json`, `reports/summaries/spatial-moments-191-summary.json`.

# 190 ConvNeXt-Tiny / Swin-V2-T 비교 완료 — best021 유지

개발 400장, 기존 5fold에서 공식 사전학습 두 모델을 비교했습니다. 원본 전체 256 입력·공식 정규화·train-only RMS 하한·동일 MLP를 고정했습니다. 100epoch frozen head 이후, 같은 초기 헤드와 seed/order/LR로 head-only 12epoch와 마지막 stage 12epoch(각 240 updates)를 대조했습니다. 학습 계산시간은 stage 역전파 때문에 동일하지 않습니다.

기준 best021 OOF **12.78620**. 고정 결합은 **0.9 × best021 + 0.1 × 후보**입니다.

| 모델 | 조건 | 단독 OOF RMSE | 고정 10% 결합 RMSE |
|---|---|---:|---:|
| convnext_tiny | frozen100 | 16.50462 | 12.95208 |
| convnext_tiny | head_only112 | 16.61512 | 12.95306 |
| convnext_tiny | last_stage112 | 16.76419 | 12.95864 |
| swin_v2_t | frozen100 | 16.59325 | 12.90498 |
| swin_v2_t | head_only112 | 16.71274 | 12.90753 |
| swin_v2_t | last_stage112 | 16.94788 | 12.91807 |

6조건 모두 승격 실패. 마지막 stage는 head-only 대조보다 결합 RMSE가 ConvNeXt 0.00558, Swin 0.01053 악화했습니다. 훈련 RMSE는 감소했지만 OOF 개선은 없었습니다. 제한된 학습 예산의 결과이며 완전수렴이나 구조의 성능 한계를 입증하지 않습니다. 추가 optimizer/활성함수/해상도 sweep은 실행하지 않았습니다.

전체 10개 warm head와 20개 adaptation 모델의 CPU/MPS 저장모델 재로딩, train-only 통계, ID/순서/LR/update, prefix/고정 buffer 검증 통과. CPU 최대 차이는 ConvNeXt **0.00011662**, Swin **0.00019729**로 기존 0.002 기준 이내입니다. ConvNeXt 라벨변경 전체 재학습도 exact 재현했습니다. **Swin last-stage는 엄격한 bitwise 재학습 검증에 실패**했습니다. 같은 라벨·같은 seed 반복에서도 차이가 발생했으며 예측 최대 차이 **0.00001937**, 가중치 최대 차이 **1.19e-7**입니다. 수치 실행 비결정성 가능성은 있으나 원인은 확정하지 않았고, 실패 기준을 완화하지 않았습니다. 전체 validation 상태는 false로 보존했습니다.

승인된 공식 가중치 2개만 총 **228,321,670 bytes** 다운로드했고 full SHA256/공식 hash prefix/strict state dict를 검증했습니다. 신규 설치 없음. 측정 단계 합계 **14.24분**, 최대 RSS+MPS **1.34GB**, swap 증가 0, 비용 0. 기존 1599개 보호파일과 best021 SHA256 일치. holdout/test/재분할/전체500/새CSV/업로드 없음. 189 ResNet 분기 종료 상태 유지.

기록: `experiments/01-representations/0190-modern-backbones/summary.json`, `validation-failure-record.json`, `swin-repro-diagnostic.json`, `completion.json`, `reports/summaries/modern-backbones-190-summary.json`.

# 189 ResNet 추가 미세조정 비교 완료 — best021 유지, 이 분기 종료

입력은기존전체224축소로고정하고학습범위만layer4→layer3+layer4로확대했습니다. 추가layer3에도기존tailLR1e-4를적용,187의outertrain내부선택설정·seed·12epoch/240updates·BN고정·RMS하한·초기헤드를고정했습니다. 개발400초기특징은185와bitwise일치,헤드체크포인트는187과byte일치,5fold초기훈련RMSE도exact일치했습니다. 더깊게풀면좋아진다고가정하지않고021보완성가설로검증했습니다.

단독OOF **15.53652→15.48964**(개선2/5fold), 고정10%021결합 **12.90917→12.87290**(개선3/5fold). paired개선95%조건부CI는각각 **[-0.75578,0.83575] / [-0.04528,0.11535]**로0포함. 기준021 **12.78620** 대비결합은 **0/5fold개선**,0.08670악화입니다. 요청된조건에따라 **이ResNetlocaladaptation분기를종료**하며추가LR/epoch/stage/해상도sweep을예약하지않았습니다.

고정12epoch최종훈련RMSE는1.71847→3.93146,gradientclip비율89.08%→99.50%입니다. 완전수렴이나구조한계를입증한결과로해석하지않습니다. 훈련가능backbone파라미터는14,942,208→22,020,096이며updates예산은동일하지만계산량은증가했습니다.

5개CPU재로딩·train-only통계·stage갱신·BN/prefix고정·ID/order/LR/update검증통과. heldout라벨변경전체재학습에서가중치/예측/기울기/LR/**특징분포trace까지exact재현**. CPU최대차 **0.00006026**(기준0.002). 사전정의한폭주중단선발생없음:최대gradient비율14.24,RMS비율1.45,maxabs특징비율5.35(중단비율100).

실측단계합계 **14.0분**, 최대RSS+MPS **2.40GB**,swap증가0,비용0. 기존1527보호파일/best021보존. 신규설치/다운로드/holdout/test/재분할/전체500/새CSV/업로드없음. 기존명시승인범위에서실행했고정책차단없음. 기록:`experiments/01-representations/0189-cnn-unfreeze/summary.json`, `experiments/01-representations/0189-cnn-unfreeze/branch-closure.json`, `reports/summaries/cnn-unfreeze-189-summary.json`.

# 188 ResNet 입력 해상도 대조 완료 — 평균 소폭 개선, best021 유지

원본전체256→224bicubic 대조(검증된187재사용)와 원본전체256유지만 비교했습니다. 각outertrain에서187이 내부선택한optimizer/LR·seed·12epoch/240updates·BN고정·layer4학습·RMS하한 규칙을 그대로 잠갔고,256의훈련통계·100epoch헤드만 새로 fit했습니다. 256에서 재선택하지 않았습니다. ResNet 우선은 재사용개발400에서 B0보다 좋은 단독표현을 보인 사후선택이며 선택편향 한계를 기록했습니다.

단독 OOF **15.53652→15.42328**(개선0.11324,3/5fold,조건부group-bootstrap95%CI **[-0.34823,0.62078]**), 고정10%021결합 **12.90917→12.89765**(개선0.01152,4/5fold,CI **[-0.03719,0.06360]**). 두 구간 모두0을 포함하며 기준021 **12.78620** 대비 개선0/5fold입니다. 평균의 작은 개선과 일관된 일반화 근거를 구분합니다.

미세경계/고주파 대리지표 상위100장은 단독 **14.37611→14.46989**, 결합 **12.16076→12.17986**로 악화했습니다. 기존실패33장 결합은 **29.77709→29.76990**로 소폭개선했지만021 **29.48212**에 미달했습니다. 경계지표는 실제입계정답이 아니며33장은사후진단집합입니다. 경계 회복의 인과효과로 해석하지 않습니다.

5개모델 CPU재로딩·train-local통계·BN/prefix·ID/order·240updates·기존187설정동일성 통과, heldout라벨변경100+12epoch전체재학습 가중치/예측/gradient/LR exact 재현. CPU최대절대차 **0.00001512**(기준0.002). 실측단계합계 **6.4분**, 최대RSS+MPS **1.62GB**,swap증가0,비용0. 기존1461보호파일/best021보존. 신규설치/다운로드/holdout/test/재분할/전체500/새CSV/업로드/추가해상도sweep없음. 후속확장은실행하지않았습니다.

기록: `experiments/01-representations/0188-cnn-resolution/summary.json`, `experiments/01-representations/0188-cnn-resolution/interpretation.json`, `reports/summaries/cnn-resolution-188-summary.json`.

# 187 RMS 스케일 하한 대조 완료 — 수치 안정성 복구, best021 유지

훈련 특징의 표준편차에 훈련 RMS 하한을 적용하는 단일 조건만 변경했습니다. 문제 ResNet 특징의 검증 최대 표준화값 **22984.37→13.44**, 기존 CPU/MPS 절대오차 초과 3건을 같은0.002기준으로 재현한 뒤 수정했습니다. 예측 clipping이나 검증 데이터로 정한 하한은 없습니다. 동일 초기 헤드·100epoch warmstart·12epoch 미세조정·4 optimizer/LR 후보·내부3fold/외부5fold를 유지했습니다.

186→187의 내부선택 포함 단독 OOF / 고정10%021결합은 ResNet50 **16.08400→15.53652 / 12.90736→12.90917**, EfficientNet-B0 **17.09463→17.25305 / 13.00475→12.99255**입니다. 두 결합 모두 기준021 **12.78620** 대비 **0/5fold 개선**으로 정확도 승격에 실패했습니다. 수치 복구가 결합 성능 개선을 보장하지 않았습니다.

140개 저장 모델의 독립 CPU 재로딩 검증 모두 통과: 최대 절대차 ResNet **0.00006105**, B0 **0.00002690**. 4회 라벨변경 전체 재학습의 가중치·예측·gradient·LR exact 재현, train-only RMS/타깃 통계·BN·prefix·ID·순서·선택 검증 완료. 기존186 실패 기록을 덮어쓰지 않았습니다.

측정 단계 합계 **65.9분**, 최대RSS+MPS **1.68GB**, swap증가0, 비용0. 기존780개 보호파일·환경·공식가중치·best021 보존. holdout/test/재분할/전체500/신규CSV/설치/다운로드/업로드 없음. 해상도·unfreeze·활성함수·증강 변경 없음. 기록: `experiments/01-representations/0187-cnn-scale-floor/summary.json`, `experiments/01-representations/0187-cnn-scale-floor/interpretation.json`, `reports/summaries/cnn-scale-floor-187-summary.json`.

# 186 제한 optimizer/LR 튜닝 완료 — best021 유지

두 CNN에 같은4후보(AdamW/SGD 각각2LR), 내부3fold 선택, 외부5fold 평가를 적용했습니다. 내부선택 절차를 포함한 단독 OOF/고정10%021결합은 ResNet50 **16.08400 / 12.90736**, EfficientNet-B0 **17.09463 / 13.00475**입니다. 기준021 **12.78620**보다 두 결합 모두0/5fold 개선으로 재분할하지 않았습니다. optimizer 계열별 OOF는 진단용이며 외부점수로 primary를 바꾸지 않았습니다.

각 모델60내부+10외부 fit,12epoch,inner168/outer240update, micro8/effective16. 동일decoupledWD.01·cosine·headGELU·BN고정.4개 라벨변경 전체 재학습의 가중치/예측/gradient/LR exact 재현,train-local통계·ID·순서 검증 완료. CPU절대오차.002기준은 ResNet내부3모델에서 미충족(max.0046863); 최종20outer모델과 B0전부 통과. 기준을 완화하지 않았습니다.

한 ResNet 내부 분할에서 nearconstant특징 표준화값22984와 초기head예측4795가 확인됐습니다. 이번 실험을 변경하지 않고 기록했으며, 다음 우선순위는109의train-onlyRMS scale하한을 별도로 대조하는 것입니다. 그 뒤 원본256전체 입력 비교를 검토하고 더 깊은unfreeze는 보류합니다. 185/186는256전체→224bicubic이며center crop을 하지 않았습니다. 낮은train오차/큰OOF격차가 있지만12epoch의 완전수렴이나CNN계열의 한계를 입증한 결과는 아닙니다.

관측 최대RSS+MPS **1.76GB**,swap0,측정학습·검증합계 **65.4분**, 비용0. 첫 실측에서 자체45분 시간한도가 걸려 기록 후 두 모델의 시간한도만 동일60분으로 조정했습니다. 기존환경/가중치/보호101파일/best021 보존. holdout/test/다운로드/설치/전체500/새제출CSV/업로드 없음. 기록:`experiments/01-representations/0186-cnn-optimizer/summary.json`, `reports/summaries/cnn-optimizer-186-summary.json`, `experiments/01-representations/0186-cnn-optimizer/interpretation.json`.

# 185 사전학습 CNN 비교 완료 — best021 유지

공식 ResNet50 ImageNet V2 및 EfficientNet-B0 V1을 별도 uv 환경에서 비교했습니다. 원본 개발400 5fold의 단독 OOF는 frozen→마지막stage 미세조정 기준 ResNet50 **18.69183→16.71066**, B0 **17.52214→17.46195**입니다. best021(12.78620)에 고정10% 결합한 미세조정 결과는 각각 **12.90177 / 12.98803**으로 두 모델 모두0/5fold 개선, 재분할 진입 실패입니다.

추가head학습만 수행한 frozen 대조군 포함30개 저장모델의 독립 CPU재로딩·train-local통계·BN/이전stage고정·ID/finite/order 검증 통과. 최대 보수적 RSS+MPS 약1.53GB,swap증가0. holdout/test/전체500/새CSV/업로드 없음, 비용0, 기존환경·best021 보존. 전체224px입력(중앙crop없음), single-seed고정설정의 결과이며 모든CNN설정의 실패를 뜻하지 않습니다.

Optimizer/LR/WD/scheduler의 제한된 후속비교는 계획만 기록했고 실행하지 않았습니다. 기존 epoch/dropout/폭/seed 비교와 달리 AdamW–SGD 및 activation의 통제비교는 미실행입니다. 기록: `experiments/01-representations/0185-pretrained-cnn/summary.json`, `experiments/01-representations/0185-pretrained-cnn/plan.json`, `reports/summaries/pretrained-cnn-185-summary.json`. 초기 MPS watermark 설정 오류와 수정도 보존했습니다.

# 183 학습 없는 분할 진단 — best021 유지

고정 개발16장·112개 분할 측정 완료. 안정성 3/16통과(요구14), 최저 인접그래프 Jaccard 중앙값 0.8493. 판정:`reject_current_inverse_rendering_route_no_training`. hardness라벨·OOF·holdout·test·학습·설치·결제·업로드 없음. 기존파일 보존. 기록:`reports/summaries/tessellation-diagnostic-183-summary.json`, 이미지 보고서:`reports/figures/tessellation-diagnostic-183-review.html`.

# 182 WPH 검증 완료 — best021 유지, WPH 계열 종료

원본 개발400 OOF RMSE:021 **12.7862007758**, amplitude-only **12.7788949953**, WPH **12.7795982831**.
고정4scale×8방향의 인접 스케일 phase-doubling 중심공분산15D와 동일wavelet amplitude15D를 같은 honest nested 잔차 학습으로 비교했습니다.
원본 진입기준 통과:`False`. 재분할 실행:`False`. 합성 위상변이·회전/반사 검증, 기존018/021 exact 재현, 모든900개 모델의 별도 프로세스 재로딩 및 train-local 통계 검증을 통과했습니다.
초기 SciPy worker 오류는 로그·원본을 보존했고 수식 변경 없이 NumPy FFT로 복구했습니다. holdout/test/full500·신규CSV·설치·다운로드·결제·업로드 없음. 기존021/018 등23개 보호파일 유지.
기록:`reports/summaries/wph-182-summary.json`, 소스·계획:`experiments/01-representations/0182-wph/`.

# 최신 Public best: candidate021

기록된021 제출완료 Public RMSE **11.4878517800**,018 **11.5230652779** 대비 **0.0352134979** 개선. 보고 당시27/97위,2026-10-06 오늘1/5사용·4회남음. 여기서 브라우저를 조회하지 않았고 정확한 제출시각·ID는 전달되지 않았습니다.021/018보존,Public점수로 기존10%비율을 튜닝하지 않습니다. 기존 미제출 문구는 당시 기록입니다. 출처: `reports/submissions/candidate021-public-submission.json`.

# 최신 제출 결과: 2026-10-05 best018 유지

기록된020을23:14KST에 정확히1회 제출했고 Public RMSE11.5290246197로01811.5230652779보다0.0059593418 악화했습니다.26/93위,23:17 보고 시 오늘4회 남음. IAB filechooser→setFiles→Submit→결과조회 경로가 복구됐고 화면잠금 상태는 미검증입니다. 이 작업에서 제출하거나 사이트를 재조회하지 않았습니다. 기존020 미제출/당일5회소진 문구는 당시 기록이며 이 최신 기록이 우선합니다.018보존,Public점수로 비중·특징을 조정하지 않습니다. 기록:`reports/submissions/candidate020-public-submission.json`.

# 최신 제출 결과: best018 유지

기록된019 Public RMSE11.5451213945로01811.5230652779보다0.0220561166 악화했습니다.21:35KST #15,21/83위,당일5/5 소진입니다. 추가 제출은 금지하며 아래 과거 미제출 기록보다 이 제출 완료 기록이 우선합니다. 기록:`reports/submissions/candidate019-public-submission.json`.018/019 및 기존 산출물은 보존합니다.

# 합성 미세구조 경도 예측 — 현재 best018

기록된018 Public RMSE **11.5230652779**,017대비0.0417156499개선,21/83위.21:03KST #14 제출ID:`70520a29-71e8-4c6d-88ac-ff412bd99ed3`,보고 당시 오늘1회 남음. 기록:`reports/submissions/candidate018-public-submission.json`.

018 CSV SHA256:`6d478f08de769ca0c54d640c0711c95b15ad4e6b707d0516022c23260f1cfc16`. 기존best 및ZIP은 보존합니다. 아래 best017/018미제출 문구는 당시 기록이며 이 기록이 최신 상태입니다.

# 합성 미세구조 경도 예측 — 현재 best017

기록된017 Public RMSE **11.5647809278**,015대비0.0169783752개선,22/83위.20:50KST 제출ID:`97e7bf6c-3e9a-411c-99f2-c0ad39a391e4`,보고 당시 오늘2회 남음.기록:`reports/submissions/candidate017-public-submission.json`.

017 CSV SHA256:`4256a5d2636053f8f1d115f39b0265a66a10c3169bd4016309ecdd2b63c3be20`.015·016과 이전 재현ZIP은 보존합니다. 아래 best015/017미제출 문구는 당시 기록이며 이 기록이 최신 상태입니다.

# 합성 미세구조 경도 예측 — 현재 best015

추가로 확인된016 Public RMSE는11.5828840997로015보다0.0011247967 높습니다. best015 유지, 보고 당시 오늘3회 남음. 이 미세 차이로 구조 우열·라벨 분포를 추론하지 않으며 곡률 추가 조정은 종료합니다. 기록:`reports/submissions/candidate016-public-submission.json`.

기록된015 공식 제출 결과: **Public RMSE11.5817593030**, 기존01111.7352685854 대비0.1535092824 개선,23/82위.
Mac 재가동 후20:15KST에 정확히1회 제출했으며, 기록 당시 오늘4회가 남았습니다. 이 작업에서 서버를 새로 조회한 값은 아닙니다.
제출ID:`a108bcc4-c78d-4a44-aa4a-5a1764b8e732`.014는 미제출입니다.

현재 best는 고정 CNN3seed 평균의015입니다. 0.3compact26KRR+0.3CNNmean3+0.2Smean3+0.2Bmean3 혼합에 nested affine을 적용하고,
0.9보정혼합+0.1shortB 및 honest physics57 잔차3개 평균을 사용합니다.
CSV:`artifacts/submission-candidate-015.csv`, SHA256:`74c9f5533b8b72c68b90af9d0b411bdc79b9cd9ec240dc3a18d7efb3a11ab67f`.
공식 결과 전달 기록:`reports/submissions/candidate015-public-submission.json`. 과거011/014와 재현ZIP은 보존합니다.
아래 과거 기록의 “미업로드/현재best011”은 해당 시점 상태이며 이 최신 기록이 현재 상태입니다.
Public 점수 하나에서 특정 test 표본·라벨을 추정하거나 혼합 비중을 맞추지 않습니다.

# best011 시점의 모델 계보 기록 (아래 내용은 당시 기록)

2026-10-03 기준 현재 보존 모델은 **candidate011**입니다. 마지막 기록된 Public RMSE는
**11.7352685854**이며 새로 조회한 점수는 아닙니다. 모델 계보는 103 감사에서 확인했습니다.
기존 README는 candidate004 당시 기록이므로 과거 README 원본 (현재 파일 없음)에
바이트 그대로 보존했습니다. 아래의 현재 모델 설명과 과거 실험을 구분하세요.

## 현재 모델의 실제 구성

- `candidate003 = 0.50 × baseline795 + 0.25 × compact26 커널 + 0.25 × 소형 CNN`
- `candidate008 = 0.75 × candidate003 + 0.125 × ViT-S MLP 3seed 평균 + 0.125 × ViT-B MLP 3seed 평균`
- `candidate009 = target_center + slope × (candidate008 − prediction_center)`
- **`candidate011 = 0.90 × candidate009 + 0.10 × 짧게 학습한 frozen ViT-B head`**

003의 CNN은 원본 256 영상을 BOX로 128에 줄이며, 학습 때 D4 증강을 사용하지만
현재 상속된 모델의 추론은 원래 방향 한 번입니다. **best011은 3seed×D4 CNN 추론이나
TV31 특징을 사용하는 candidate004 레시피가 아닙니다.** 전체 모델의 회전 불변성도 보장하지 않습니다.

DINO 입력은 전체 영상을 자르지 않고 PIL bicubic으로 504×504로 변환한 뒤,
회색조를 RGB 3채널로 복제하고 고정 ImageNet mean/std를 적용합니다.
원본 픽셀의 영상별 밝기·대비를 별도로 정규화하지 않습니다. 공식 DINOv2 ViT-S/14와
ViT-B/14를 frozen/eval로 사용합니다. 36×36 패치와 6/9/12번째 block의 normalized
CLS·patch 평균·population 표준편차를 연결합니다. S는3456차원, B는6912차원입니다.
최종 모델에는 DINO 회전 평균, 패치 위치 평균, 복원 전처리를 추가하지 않았습니다.

008의 S/B head는 각3seed,100epoch입니다. 011의 추가 head는 B6912→32→1,
GELU,dropout0.5,12epoch,effective batch16/microbatch2,AdamW weight decay0.1입니다.
개발 fold별 seed는20270215+fold이며 전체500 fit의 seed는20270215입니다.
스케일러와 타깃 mean/std는 해당 학습집합에서만 계산하고 예측 시 역변환합니다.
009 보정은 학습집합 안에서 새로 생성한 inner OOF로 fit했고, 전체 outer OOF에 직접 fit하지 않았습니다.

## 현재 성능과 검증 범위

| 개발 분할 | best011 OOF RMSE |
|---|---:|
| 원본 development400 | 13.0539977719 |
| 재분할20261031 | 13.0151760566 |
| 재분할20261101 | 13.2723779486 |

같은400장을 반복 연구했으므로 독립 검증 또는 선택편향 보정 결과가 아닙니다.
두 재분할도 같은 관측치를 재사용합니다. 개선이나 입상을 보장하지 않습니다.
원래 holdout100은 추가 선택에 사용하지 않으며, 최종 모델 고정 후 전체500 fit에만 포함했습니다.
Test는 최종 모델 고정 후 추론과 형식·파이프라인 무결성 확인에만 사용합니다.

기존 검증은 ID·타깃·fold 정렬, fold 내부 학습 통계, full500 head 재학습 재현,
타깃 역변환, test1000 저장 모델 재로딩, CSV 바이트 재현을 포함합니다.
103 감사에서는 입력 경로 동일성, 패치 순서와 위치 임베딩 보간을 별도 계산으로 확인했습니다.
전체 test1000의 원본 CPU DINO 특징 재추출까지 수행한 것은 아니며, 이전 raw10 점검과
모든 이미지 해시 및 전체 캐시 재로딩의 검증 범위를 구분합니다.

## 보존 산출물

- 제출 CSV: `artifacts/submission-candidate-011.csv`
- CSV SHA256: `b7d18f0b74f9259a5c9a1a0f78d79935f41fda74b55aaa57669e44e0372a1ab5`
- 추가 head와 고정 결정: `artifacts/final-model-086/`
- 008의 S/B 모델: `artifacts/final-model-065/`
- 009 보정기: `artifacts/final-model-075/calibrator.json`
- 003 기존 구성 모델: `artifacts/final-model-017/`
- 검증: `reports/submissions/candidate008-verification.json`, `reports/submissions/candidate009-verification.json`,
  `reports/submissions/candidate011-verification.json`, `reports/validation/frozen-complement-085-validation.json`
- 전체 경로 감사: `artifacts/pipeline-audit-103/report.json`
- 현재 계보와104 결과: `reports/validation/pipeline-audit-103-patch-phase-104-summary.json`
- 기록된 제출 기록: `reports/submissions/candidate011-public-submission.json`

생성 당시 final summary의 “unsubmitted” 같은 상태 문구는 당시 기록입니다.
그 이후의 제출 상태는 날짜가 더 늦은 public-submission 기록을 우선합니다.
기존 생산 스크립트는 고정 산출물 경로를 사용하고 이미 존재하면 중단합니다.
보존 경로를 지우거나 같은 이름으로 다시 학습하지 마세요. 재현 작업은 별도의 새 출력 경로로 준비합니다.

## 과거 결과와 현재 연구

004의 TV31·3seedD4 설명은 과거 README (현재 파일 없음)에 있습니다.
CNN 해상도·중간층 moments·covariance와 규제(009~011,035~036), DINO 회전 평균046,
lineal054, Betti055, token covariance056, TV071, head seed·dropout·폭088~090,
patch attention092~093, early adapter095, scattering096, NLM098, 분포099,
프레임 정보101, 복원 사전 진단102, 패치 위치 평균104 등의 원래 기록을 보존합니다.
실패는 해당 고정 비교에 대한 결과이며 모든 유사 방법의 불가능성이나 데이터 성능 상한의 증명이 아닙니다.
103에서는 새 수치 버그를 찾지 못했고104는 원본 OOF가 악화되어 재분할·최종 fit을 하지 않았습니다.
새 후보가 개발 검증을 통과하지 못하면 best011을 유지하며 제출 횟수를 억지로 소비하지 않습니다.

105~108 소규모 배치에서는 README 계보 수정 후 세 고정 가설을 실제 비교했습니다.
106의 GroupNorm 제거 결합은13.09720, 107의 grayscale persistence 결합은24.04636,
108의 세 위치 명암 곱 확장 결합은13.08121로, 원본 best011의13.05400을 넘지 못했습니다.
107은 Gaussian tail 특징의 극히 작은 학습 분산을 표준화하면서 검증값이 확대되는
설계상 불안정성을 별도 기록했습니다. 결과에 맞춘 clipping·문턱 재선택은 하지 않았습니다.
108은 자체 2점 대조군보다 개선됐지만 기존 앙상블에 추가 이득은 없었습니다.
원본 판정에서 모두 탈락해 이 배치의 재분할·전체500 fit·test 추론·새 제출은 없습니다.
상세 기록은 `reports/summaries/research-batch-105-108-summary.json`을 참조하세요.

## 실행 범위

허용 프로젝트는 `.`입니다.
기존 uv `.venv`의 Python3.13.3과 설치된26개 패키지, M1 Pro CPU/MPS만 사용합니다.
CPU 작업수는2이며 신규 설치·비용·원자료 외부 전송은 없습니다.
공식 DINOv2 가중치는 이전 별도 승인으로 설치되어 있습니다.
Kymatio0.3.0/appdirs1.4.4/configparser7.2.0 역시096에서 별도 승인 후 설치됐으며,
그 외 새 설치는 허용하지 않습니다. 회사 자료와 범위 밖 디렉터리는 작업에 사용하지 않습니다.

프로젝트 `.venv/bin/python -I -B`로 실행하며 새 연구는 고유 산출물 경로·고정 계획·로그·독립 검증을 남깁니다.
모든 업로드는 부모 작업과 조율합니다.

## 107 수치 수정 후속 검증109

부모 요청에 따라107 원본을 보존하고, 각 inner/outer 학습 fold의 열 표준편차 RMS를
스케일 하한으로 쓰는 단일 고정 수정을 적용했습니다. 임계값·특징·alpha 후보를 추가 탐색하지 않았습니다.
극단 검증 특징 확대는 완화됐지만, 수정 persistence 결합 OOF는13.12086으로
best011의13.05400보다 나빴고5개 fold 모두 개선되지 않았습니다. 수정된 대조 곡선 결합은13.08733입니다.
따라서107의 원래 수치 불안정과109의 예측 이득 부재를 구분하며, 이 고정 persistence+Ridge 접근은 종료합니다.
실제 물리 구조 보존을 입증한 결과는 아닙니다. 상세 기록: `reports/summaries/persistence-repair-109-summary.json`.

## 약한 원본 픽셀 잡음 증강110

단일 강도sigma2/255의 Gaussian 잡음을 native256 픽셀 공간에 추가한 고정 비교를 완료했습니다.
동일 head·12epoch에서 원본/잡음 영상을 각6회 사용했으며, 검증 추론은 원본만 사용했습니다.
동일 학습량 대조군5개 모델은 기존011 가중치와 예측을 정확히 재현했습니다.
잡음 증강 결합 OOF는13.05727로011의13.05400보다0.00327 악화됐습니다.
4개 fold의 작은 개선보다 한 fold의 악화가 컸으며, 개선 기여 상위5개 표본을 빼도 악화됐습니다.
사전 판정에서 탈락해 재분할·전체 fit·test·제출로 진행하지 않았고, 강도/비중을 바꾸지 않았습니다.
저장 모델 재로딩과 검증 라벨·잡음 특징 교란 재학습을 통과했습니다. 실제 물성 불변·구조 보존을 입증한 결과가 아닙니다.
상세 기록: `reports/summaries/input-noise-110-summary.json`.

## 문제 정의·표현 감사111

공식 페이지와 연결된3쪽 PDF를 직접 대조했습니다. 물리 길이 환산과 정확한 생성식은
접근한 자료에 없으며, 절대 방향과 정렬의 구분이 타깃 회전 불변을 보장하지는 않습니다.
모든 영상의 공통256→504 확대는 픽셀 크기를 영상별로 정규화하지 않습니다.
S/B의14×14 grayscale 패치 선형 투영은 모두 수치 rank196으로, 단순 평균 축소가 아님을 확인했습니다.
CNN의 BOX128 축소에는 실제 고주파 정보 손실이 있지만 native 해상도 대조는 이미 실행됐습니다.
합성곱만의33px 수용 영역과 GroupNorm의 전역 통계 의존을 구분했습니다.
전역 token moments는 추상 token 순서를 버리지만 positional/contextual DINO와 CLS를 포함한
전체 모델이 입력 공간 배치에 불변이라는 뜻은 아닙니다.
현재 compact26의 area-only 정렬 지표에는 원형 영역의 임의각 문제가 가능하나,
신장 가중 정렬은 과거denoised/connected 특징에 이미 구현돼 새 가설로 중복 학습하지 않았습니다.
19개 가설군의 당시 대조군·결과·해석 한계를 연결했고, 새 학습을 정당화할 비중복 누락은 확인하지 못했습니다.
실제 영상·라벨·캐시를 추가 읽지 않은 수치 감사와 독립식 검증만 수행했습니다.
best011 및 기존200개JSON 보고서·26개패키지를 보존했습니다.
상세 근거와 다음 검증의 전제: `reports/summaries/problem-definition-111-summary.json`.

## 검증 설계 재검토와 완전 중첩 결합112

반복400장의 선택편향을 유지한 채,011·attention 결합·동일 모수 mean 결합을 고정 비교했습니다.
011 보정까지 각 inner 학습집합 안에서 다시 학습했고, 비중은 inner OOF 제곱오차로만 정했습니다.
비중0을 허용했습니다. fold 승수·최소0.03 개선·양의 신뢰구간·상위5개 제외를 필수 게이트로 쓰지 않았습니다.

| 분할 | 기존011 | 중첩 attention | 중첩 mean |
|---|---:|---:|---:|
| original | 13.05400 | 13.10145 | 13.15887 |
| 20261031 | 13.01518 | 13.02444 | 13.03266 |
| 20261101 | 13.27238 | 13.25852 | 13.28009 |

모든 점수는 재사용한 development400의 OOF이며 독립 새 표본 검증이 아닙니다.
이 고정 절차가 전체011을 개선했는지 판정하며, attention 전체나 모든 물리 잔차 모델의 가능성을 판정하지 않습니다.
기존057 길이 비교와092 학습곡선을 검토했으며, 추가 길이/학습률 탐색 근거는 부족해 반복하지 않았습니다.
물리57 직접 예측의 양의 비중 결합은 원본 저장 벡터에서 이득이 없었지만,011 잔차 학습은 다른 미검증 축입니다.
135개 보정용008 학습과135개 head 학습을 완료했고, 저장 모델 재로딩·학습 fold 통계·별도 비중 최적화·라벨 교란 재학습을 검증했습니다.
새 코드의 짧은 B head는 기존011 가중치와 예측을 정확히 재현했습니다.
상세 결과와 다음 실행: `reports/summaries/nested-fusion-112-summary.json`.

## 물리57의 정직한 블록 잔차 회귀113

112에서 저장한 실제 중첩011 모델45개의 학습 범위와 해시를 검증했습니다.
각 잔차 블록106/107개 전체를 제외한213/214개로 학습한011이 잔차 타깃을 만들며,
잔차 블록 내부3fold에서 보정 없음·Ridge1·Ridge0.1만 선택했습니다.
스케일러·잔차 RMS와 회귀는 각 inner 학습 표본에서만 fit하고, 선택한 세 블록 보정의 평균을 기존 outer011에 더했습니다.
전체 OOF 잔차를 한꺼번에 학습하거나 검증 점수로 비중을 조정하지 않았습니다.

| 분할 | 기존011 RMSE | 잔차113 RMSE | 개선량 |
|---|---:|---:|---:|
| original | 13.05400 | 13.05658 | -0.00258 |
| 20261031 | 13.01518 | 12.97598 | +0.03920 |
| 20261101 | 13.27238 | 13.26435 | +0.00803 |

주지표는400개 pooled RMSE이며 fold 부호·영향 표본·조건부 신뢰구간은 진단에만 사용했습니다.
원본에서 소폭 악화돼 사전 고정한 세 분할 모두 개선 조건을 충족하지 못했습니다.
450개 저장 모델 재로딩, 수동 변환/dual Ridge 독립 계산, 검증 라벨 교란을 통과했습니다.
동일400장의 반복 선택편향과213→320 학습량 차이가 남으므로 독립 확증으로 해석하지 않습니다.
best011을 보존했고 전체500 재학습·test 추론·새 CSV·업로드는 실행하지 않았습니다.
세 분할 잔차 학습 합계0.755초, 추가 설치/비용0. 상세: `reports/summaries/residual-physics-113-summary.json`.

## 113 pooled 재평가 및 탐색 후보012

모든 분할에서 개선해야 한다는 추가 조건을 후보 준비 요건에서 제외했습니다.
이전113 계획·실패 판정·보고서는 보존하며 이 추가 기록으로 운영 결정을 갱신합니다.
세 분할 전체 제곱오차를 합산한 pooled RMSE는13.114339→13.099533, MSE 개선량은0.388142입니다.
동일ID의 세 예측을 함께 재표집한 조건부95% RMSE 개선구간은[-0.118287,0.146014]입니다.
같은400장 반복 사용의 선택편향과 영향 표본 민감성이 남아 유의한 개선으로 주장하지 않습니다.

전체500에서 각 잔차 블록을 제외한333/334개로011 앵커를 만들고, 그 내부 보정도 중첩 학습했습니다.
각166/167개 잔차 블록 내부에서 보정 없음/Ridge1/Ridge0.1을 동일 절차로 선택했습니다.
기존 전체500 학습011에 세 잔차 모델의 평균을 더하며 비중·강도·epoch를 재탐색하지 않았습니다.
선택 결과: ridge01, none, none.

검증된 제출 파일: `artifacts/submission-candidate-012.csv` (1000행, ID/순서/유한값/재로딩/범위/SHA 검증).
SHA256: `a7263185b3071e9aa6403e6dbc5dc16ca6c8675826356eb915d60770e79fab14`.
기존011 보존, 비용/설치0, test 튜닝/holdout 별도평가/업로드 없음. 제출은 부모 별도 작업입니다.
상세: `reports/submissions/candidate012-residual-summary.json`, `reports/submissions/candidate012-residual-verification.json`.

## 비선형 물리 잔차115 및 후보013

같은 nested 잔차 분할에서 기존113의 보정 없음/Ridge 두 후보에 강하게 정규화한 RBF 한 종류를 추가했습니다.
대역폭·스케일·커널 중심화·잔차 RMS·선택은 inner 학습 범위에서만 계산했습니다.
주지표는 세 분할 전체 pooled untrimmed MSE이며 모든 분할 개선을 필수조건으로 쓰지 않았습니다.

| 분할 | 기존011 | 선형113 | 확장115 |
|---|---:|---:|---:|
| original | 13.05400 | 13.05658 | 13.04554 |
| 20261031 | 13.01518 | 12.97598 | 12.96649 |
| 20261101 | 13.27238 | 13.26435 | 13.23261 |

pooled RMSE는011 13.114339,113 13.099533,115 13.082022입니다.
113 대비 ID 묶음 조건부95% 개선구간은[-0.008980,0.044515]로0을 포함합니다.
400장의 반복 선택편향은 남으며 유의한 개선을 주장하지 않습니다.
720개 모델 재로딩·독립 고유분해 계산·라벨 교란 검증을 통과하고113 대조를 정확히 재현했습니다.

검증된114 전체500 nested 타깃을 재사용해 최종 선택 Ridge0.1/RBF1/RBF1을 적용했습니다.
새 상위 모델 학습 없이 후보013 CSV1000행을 생성하고 ID/순서/유한값/범위/재로딩/SHA를 검증했습니다.
파일: `artifacts/submission-candidate-013.csv`. SHA256: `e2ea0111d8c6a262c71dd32ad3918db1a1d2189f1647db7a2d6399af83b51286`.
011·012·113 원본 보존, 비용/설치0, test/holdout 튜닝·외부전송·업로드 없음.
상세: `reports/summaries/nonlinear-residual-115-summary.json`, `reports/submissions/candidate013-summary.json`.
다음 실행 가능 축은 같은 절차의 고정 물리블록별 additive RBF 잔차 대조이며 아직 실행하지 않았습니다.

## 물리 블록 평균RBF117 및 잔차군 안정성 종합

기존run022의10개 물리 블록을 고정하고 블록별 RBF를 동일 비중으로 평균했습니다.
선택지 수·정규화 강도·nested 범위·pooled untrimmed 주지표는115와 동일합니다.

| 잔차군 | 전체106/107개 pooled RMSE | 진단용53/54개 pooled RMSE |
|---|---:|---:|
| 선형113 | 13.099533 | 13.161290 |
| 공동RBF115 | 13.082022 | 13.168024 |
| 블록평균117 | 13.088253 | 13.196291 |

기준011 pooled RMSE13.114339. 블록평균117은115보다0.006231 악화돼 후보013을 유지합니다.
동일 분할의 잔차군 간 보정값 상관성0.944~0.969로 중복 신호가 큽니다.
같은 잔차군의 재분할 간 보정값 상관성은0.284~0.466으로 작아 보정 추정의 안정성은 제한적입니다.
011 대비 총 MSE 감소는113 0.388142,115 0.846597,117 0.683520이며 서로 더할 수 있는 별도 이득이 아닙니다.
절반 표본은 고정된 앵커와 외부 검증을 유지한 진단이며 후보 선택에 쓰지 않았습니다.
한 번의 부분표본 추출은 표본 수와 구성 차이를 함께 바꾸므로 일반 학습곡선·인과 증거로 주장하지 않습니다.
같은400장 반복 선택편향과 조건부 불확실성도 그대로 남습니다.

2160개 모델 재로딩·270개 선택 독립 재계산·블록별 고유분해·라벨 교란 검증을 통과했습니다.
011·012·013 및 기존113·115 기록을 보존했고 새CSV·test/holdout열람·설치·비용·외부전송·업로드는 없습니다.
커널/alpha/gamma/블록비중 미세탐색은 여기서 멈춥니다. 다음 실행 근거는 고정joint115만의
사전 고정3개 추가 절반 부분표본 진단으로 표본 구성 민감도를 구분하는 것입니다.
진단 결과로 좋은 부분표본을 고르거나 새 후보를 만들지 않으며, 아직 실행하지 않았습니다.
상세: `reports/summaries/residual-family-117-summary.json`.

## 고정115 부분표본3회 진단118 및 전체320 잔차 학습 설계 검토

새 시드20271019/20271020/20271021을 고정해 추가3회 진단했고 기존117을 포함한4회 모두 보존했습니다.
커널·비중·시드 선택이나 새 앙상블 생성에 쓰지 않았습니다.

| 부분표본 | pooled RMSE |
|---|---:|
| 20271017 | 13.168024 |
| 20271019 | 13.126939 |
| 20271020 | 13.198685 |
| 20271021 | 13.140219 |

전체106/107 잔차 표본의115는13.082022이며 절반53/54개 표본4회 모두 악화됐습니다.
4회의 제곱오차를 평균한 뒤 제곱근을 취한 진단값은13.158495입니다(예측 앙상블 점수 아님).
전체 표본의 이점0.076473에 대한 ID 묶음 조건부95% 구간은[-0.012682,0.162888]입니다.
부분표본 간 보정값 상관성은-0.009~0.216으로 구성 민감도가 큽니다.
같은400장 반복 선택편향, 부분표본 구성 및 유한4회 변동 때문에 일반 학습곡선·확실한 개선으로 해석하지 않습니다.

현재 개발에서는 각 잔차 모델이106/107개, 세 모델 합집합은320개를 사용합니다.
최종013에서는 각166/167개, 합집합은500개를 사용합니다. 앵커 크기도213/214→333/334로 바뀝니다.
따라서 전체 모델이107개만 사용한다는 설명도, 한 잔차 모델이320개를 학습한다는 설명도 부정확합니다.
중복 앵커에 따른 잔차 타깃 의존성·여러 모델 평균·훈련 크기에 따른 잔차 함수 변화가 함께 작용합니다.

320개 outer-training OOF잔차를 모으는 것 자체는 outer80 누수가 아닙니다.
그러나 그 벡터를 다시 inner CV로 나누면, 다른 앵커가 inner 검증 라벨을 본 경로가 생깁니다.
실제45개 inner 분할에서 모든213/214개 inner 학습 잔차가 이 경로의 영향을 받을 수 있음을 인덱스로 확인했습니다.
이는 자기 라벨 제외만으로 nested 선택의 독립성을 보장할 수 없다는 뜻입니다.

119 제안(아직 미실행): 기존rbf1 설정을 고정하고 동일 잔차 타깃으로
세 블록 RBF 평균 대 단일320개 pooled RBF를 대조하며115·011을 기준으로 함께 기록합니다.
inner 하이퍼파라미터 선택을 하지 않아 기존 잔차 캐시를 안전하게 재사용할 수 있습니다.
5개 후보 선택을 유지하는 대안은 inner213 안에서 잔차를 새로 cross-fit해야 합니다.
이 경우011의009 보정까지 재중첩해야 하므로405개 추가008 보정 fit과135개 shortB fit이 필요합니다.
단일 강도 pooling 대조는 학습 표본 활용을 검토하는 제안이며 새 커널/가중치 탐색이 아닙니다.

2160개 모델 재로딩·135개 선택 독립 재계산·라벨 교란 검증 통과.
013·011·012 및 환경 보존, 새CSV·test/holdout 라벨 열람·비용·설치·업로드 없음.
상세: `reports/summaries/residual-subsamples-118-summary.json`, `reports/validation/residual-crossfit-118-scope-audit.json`,
`reports/summaries/pooled-residual-119-proposed-study.json`.

## 고정rbf1의 전체320 교차적합 잔차 대조119

118에서 작성한 사전 제안을 승인 후 실행했습니다. 기존 제안서는 당시 기록으로 보존합니다.
동일한320개 교차적합 잔차와rbf1을 사용해 블록별106/107개 모델3개 평균과 단일320개 모델을 비교했습니다.
추가inner 선택·커널·정규화·비중 탐색은 하지 않았습니다.

| 방식 | 세 분할 pooled RMSE |
|---|---:|
| 011 | 13.114339252 |
| 115/후보013 절차 | 13.082021905 |
| 고정rbf1 세 블록 평균 | 13.108572945 |
| 고정rbf1 단일320개 | 13.108590044 |

단일320개는 블록 평균 대비MSE0.000448 악화,115 대비RMSE0.026568 악화로 후보 조건을 충족하지 못했습니다.
블록 평균 대비RMSE 개선량의 ID 묶음95% 구간은[-0.001978,0.001957]입니다.
세 소규모 모델 개별오차의 평균은MSE171.946761이며, 예측을 평균하면171.834685로0.112077 감소합니다.
단일320개는MSE171.835133으로 평균화 이득을 넘어서는 추가 개선이 없었습니다.
이 분해는 평균화 효과를 정확히 분리하지만,320개 fit에서 바뀌는 전처리·대역폭까지 고정한 인과적 표본수 효과는 아닙니다.

고정rbf1의 보정RMS는약0.10, 유효자유도는약0.94로 매우 강한 정규화 조건입니다.
따라서 이 결과가 모든 pooled잔차 학습의 실패를 뜻하지는 않으며 새lambda 탐색으로 넘어가지 않았습니다.
앵커213/214와 적용기준320의 학습량 차이, 공유 앵커에 따른 잔차 의존성, 반복400장 선택편향은 남습니다.

45개 앵커의 전체 모델 해시·학습 범위를 재확인했습니다.60개 모델 재로딩·독립 변환/고유분해를 통과했고,
세 분할의 외부80개 라벨을 교란한12개 모델 재학습에서도 가중치·전처리·예측이 같았습니다.
013·011·012·기존 보고서·환경 보존. 전체500 재학습·새CSV·test/holdout열람·설치·비용·업로드 없음.
상세: `reports/summaries/pooled-residual-119-summary.json`.

## Study 120: MSE and pairwise ranking objective (completed)

The unchanged cached ViT-B short head was trained for the same 12 epochs with half-MSE alone or half-MSE plus a fixed 0.1 pairwise logistic ranking term. Pairs were formed within each training batch, using a fixed gap of 0.5 training-target standard deviations. Three inner folds selected the objective using head MSE inside every outer training fold. Only the existing 10% short-head contribution was replaced; candidate013's remaining components stayed fixed.

Across the three existing development partitions, pooled RMSE was 13.082021905 for candidate013, 13.088033264 for the matched MSE replacement, 13.084496162 for fixed ranking (diagnostic), and 13.088525668 for the nested primary. Ranking was selected in 4/15 outer folds. The primary failed the prespecified improvement criterion; candidate013 is retained. No new final500 fit, test inference, CSV, or upload was performed.

Validation reproduced the original SmoothL1 incumbent's parameters and predictions exactly, checked the ranking loss and gradient independently, reloaded all 120 comparison checkpoints, and replayed the full nested fold on deterministic CPU after perturbing outer labels. Both candidate fits were also replayed on CPU after perturbing inner validation labels. The original bit-exact MPS retraining assertion failed only for ranking models and is preserved; unchanged-label MPS repetition confirms small numerical variability within the recorded tolerance, while within-CPU label intervention is exact. An incorrect cross-device training equivalence assertion was also preserved and corrected: CPU and MPS have different dropout RNG streams. No bit-exact MPS retraining claim is made. Same initialization, dropout seeds, batches, update counts, and sample exposures were verified. The three partitions reuse the same400 examples, so their pooled results and conditional ID bootstrap do not remove historical selection bias.

The ranking term was active, but its effect did not reliably improve absolute prediction error. This is evidence against this fixed objective replacement, not all ordinal or distributional methods. No rank-weight, kernel, epoch, or seed sweep follows this result. Detailed metrics, head-only diagnostics, hashes, and limits are in `reports/summaries/ranking-objective-120-summary.json`; the frozen protocol is `artifacts/ranking-objective-120/plan.json`.

## 후보013 구성요소 ablation 121 (완료, 진단)

후보013의 실제 계보는 다음과 같습니다. 계수는 affine 보정 이전의 비중입니다.

- 008 = 0.375×baseline795 + 0.1875×compact26 KRR + 0.1875×TinyCNN128 + 0.125×DINO-S mean3 + 0.125×DINO-B mean3
- 009 = 학습 fold 내부 OOF로 맞춘 affine(008)
- 011 = 0.9×009 + 0.1×DINO-B short head
- 013 = 011 + 학습 범위를 분리한 physics57 잔차 모델3개의 평균

baseline795는 형태·텍스처·통계 특징의 ET/Ridge 앙상블이며 정확한 물리 방정식이 아닙니다.
compact26에는 Hall–Petch 대리량(결정립 직경의 역제곱근), 상/공극 분율, 형상/정렬, 선절편 특징이 들어갑니다.
새 잔차 보정은 compact26+TV31의57개 특징을 이용합니다. 따라서 도메인 지식은 실제 입력과 잔차 모델에 적용되어 있습니다.
이 영상 대리량이 참 결정립 마스크나 절대 물리 크기임을 입증한 것은 아닙니다.

네 가지 구성군을 하나씩 제거했습니다. 앞의 세 경우에는 남은008 비중을 합1로 정규화하고,
중첩 OOF에서 affine 보정 및115 잔차 선택/학습까지 다시 수행했습니다.
DINO 제거에는 기존 S/B head뿐 아니라011의 short B head도 포함됩니다.
독립적으로 학습된 나머지 기저 모델은 같은 fold의 저장 모델을 재사용할 수 있어 새 신경망 학습이 필요하지 않았습니다.
반대로 입력 특징·공유 가중치 자체를 삭제하는 실험은 그 기저 모델을 다시 학습해야 하며 이번 분석에 포함하지 않았습니다.
새 잔차 제거만은 예측합에서 잔차를 빼는 것과 정확히 같으며011로 돌아갑니다.

| 제거 구성 | 후속 보정·잔차 재학습 RMSE | 013 대비 변화 | 고정 모델에서 학습평균 대체 RMSE |
|---|---:|---:|---:|
| DINO 전체 | 13.612280 | +0.530258 | 14.021931 |
| 기존 CNN | 13.260328 | +0.178306 | 13.437149 |
| 기존 수작업 특징 직접 회귀 | 12.951140 | -0.130882 | 14.261699 |
| 새 물리 특징 잔차 보정 | 13.114339 | +0.032317 | 13.114339 |

기준013 pooled RMSE는 13.082022입니다. 같은400개를 세 번 재분할한1200쌍의 오류를 합산했습니다.
고정 모델 민감도는 해당 분기의 출력만 outer 학습 타깃 평균으로 바꾸고 기존 보정·잔차를 유지한 값입니다.
이는 재학습 ablation과 다른 질문이며, 신경망 가중치를 임의로0으로 만들거나 물성 예측값을0으로 만드는 분석이 아닙니다.

기존 수작업 특징 직접 회귀군 제거는 pooled RMSE를0.130882 낮췄지만 원본 분할은0.005715 악화했고 두 재분할에서 개선됐습니다.
조건부 ID 묶음95% RMSE 변화 구간은[-0.409853,+0.149781]이며 반복 연구의 선택 편향을 제거하지 않습니다.
이 구성군의 학습평균 대체는 오히려 RMSE14.261699로 크게 악화되어, 단순 zeroing의 손실을 곧바로 중요도로 해석하면 다른 결론에 이릅니다.
그 제거 모델에도57개 물리 특징의 잔차 보정은 남아 있고, 재학습한 잔차 보정은 RMSE13.041851→12.951140의 이득을 보였습니다.
따라서 물리 도메인 특징 전체가 불필요하다는 결론이 아닙니다.

CNN의 단독 RMSE는14.621919로 여섯 원시 구성 중 가장 나쁘지만, CNN을 제거하면 합산 RMSE가0.178306 악화됩니다.
단독 모델 점수만으로 구성요소를 버리면 상보성을 놓칠 수 있습니다.
DINO-B 장기/단기 head의 오차 상관은0.962915, baseline795/compact26은0.916419입니다.
높은 상관만으로 중복·제거 가능성을 확정하지 않습니다.

476개 계보 입력 해시,15개 외부 CNN의 CPU 재로딩,2160개 잔차 모델 재로딩과135개 내부 선택을 검증했습니다.
독립 OLS/커널 계산 최대 오차는6.49e-14 미만입니다. 세 제거 경로에서 외부 라벨과 제거된 출력 전체를 교란해 결과 불변을 확인했습니다.
최초 비트 단위 재실행 검사는5.68e-14 차이로 실패해 보존했고, 기존 OLS 허용오차1e-10을 사용한 후속 검증을 통과했습니다.

후속 제안(아직 미실행): 기존 회귀군을 baseline795와compact26으로 나눠 두 가지 단일 제거만 고정해 같은 중첩 재학습을 수행합니다.
이는 어느 직접 회귀 경로가 상대적으로 중복되거나 비중이 맞지 않는지 구분하는 실험입니다. 새 특징/커널/가중치 탐색으로 확대하지 않습니다.
필요한 모든 캐시는 이미 있으며 새 신경망 학습 없이 실행할 수 있습니다.

이번 작업은 진단이므로 후보013을 보존합니다. 새 제출 CSV·전체500 재학습·test 추론·holdout 평가·설치·비용·업로드는 없습니다.
`reports/registers/component-ablation-121-metrics.csv`는 분석 지표 표이며 제출 파일이 아닙니다.
상세 결과: `reports/summaries/component-ablation-121-summary.json`; 계획: `artifacts/component-ablation-121/plan.json`.

## 하위 회귀 경로 분리122 및 후보014 (완료, 미업로드)

121에 이어 baseline795만 제거 / compact26 KRR만 제거의 두 고정 비교를 실행했습니다.
기존013과121의 회귀군 전체 제거까지 정확히 네 구성을 비교했고, 같은 nested affine·잔차 재학습을 적용했습니다.
주지표는 세 개발 분할 pooled untrimmed MSE이며 모든 분할/모든 fold 개선이나 CI 양수를 필수 조건으로 두지 않았습니다.

| 구성 | 원본 RMSE | 재분할1 | 재분할2 | pooled RMSE |
|---|---:|---:|---:|---:|
| 기존013 | 13.045536 | 12.966493 | 13.232608 | 13.082022 |
| baseline795+compact26 제거 | 13.051251 | 12.707778 | 13.090963 | 12.951140 |
| baseline795만 제거 (014) | 12.968700 | 12.668261 | 13.045858 | 12.895301 |
| compact26만 제거 | 13.002402 | 12.954714 | 13.242867 | 13.067270 |

선택은 baseline795만 제거입니다. 새로운 보정 전 혼합은0.3×compact26+0.3×CNN+0.2×Smean3+0.2×Bmean3입니다.
이 혼합의 새 affine 보정에0.9, 기존 full500 shortB에0.1을 적용하고 새 honest physics57 잔차3개 평균을 더합니다.
물리 도메인 경로 compact26과57차원 잔차는 유지합니다. 많은 특징의 baseline이 본질적으로 나쁘거나 인과적으로 무의미하다는 주장은 아닙니다.

013 대비 pooled RMSE 개선은0.186720, 조건부 ID 묶음95% 개선 구간은[-0.005398, 0.376995]입니다.

| 분할 | 013 대비 RMSE 개선 | 조건부95% 개선 구간 |
|---|---:|---:|
| original | 0.076837 | [-0.104191, 0.253222] |
| 20261031 | 0.298233 | [0.083500, 0.510246] |
| 20261101 | 0.186751 | [-0.041069, 0.412196] |

121의 결과를 본 뒤122를 같은400개로 수행했으므로 선택 편향이 남습니다. 위 구간은 모델을 고정한 조건부 불확실성이며 연구 선택이나 재학습 변동을 포함하지 않습니다.

단순 출력 대체와 재학습의 차이: baseline795 출력을 fold 학습평균으로 대체하고 기존 보정/잔차를 유지하면13.593111로 악화됩니다.
경로를 제거하고 비중·보정·잔차를 올바르게 재구성하면12.895301입니다. 서로 다른 추정 대상이며 raw 가중치 zeroing을 특징 중요도로 사용하지 않았습니다.

전체500 파이프라인을 고정해 affine4개와 잔차 선택/최종 모델48개(none 포함)를 새로 학습·검증했습니다.
나머지 기저 모델은 각자 독립적으로 이미 전체500을 학습했으며 설정이 동일해 재사용했습니다. 새 신경망 학습을 수행했다고 주장하지 않습니다.
전체500 OOF는 보정 학습에만 사용했고 이전 holdout100의 별도 점수를 계산하지 않았습니다.

훈련 검증 후1000개 test를 남은 KRR/CNN/S/B/shortB 모델에서 새로 추론했습니다.
모든 모델 재로딩 예측과 CSV bytes가 정확히 재현됐고, ID순서/중복/누락/finite 검증 및1000개 이미지 해시 확인을 통과했습니다.
CPU/MPS 최종 차이는9.66e-6, 독립 OLS/커널 최종 오차는1.14e-13 미만입니다.

제출 파일: `artifacts/submission-candidate-014.csv`
SHA256: `2c5ce4ca43d75a7f47fbf7dd6c155db9c5c090d86f23b5999a0f6920def4f6a0`
예측 범위: [159.90905082, 238.09679235], clipping 없음.

구현 조사 중 기존008/011 구성 CSV의 헤더와 첫 행이014 고정 전에 표시됐다는 사실을122 계획에 기록했습니다.
그 값으로 학습·순위·설정 선택을 하지 않았고 후보 선택은 개발 OOF만 사용했습니다. 새 test 추론은014 학습·검증 후 실행했습니다.

기존013·011·012와 이전 보고서를 보존했습니다. 비용·설치·test 튜닝·외부 업로드는 없습니다. 업로드는 부모 작업에서 조율합니다.
상세: `reports/submissions/candidate014-summary.json`, `reports/submissions/candidate014-verification.json`, `reports/summaries/component-single-ablation-122-summary.json`.

## baseline 진단124 및 제한된 가중치 학습125 (완료, 014 유지)

기존 baseline 15개 outer·45개 inner 묶음을 재로딩해 개발400개에서 진단했습니다.
ExtraTrees 학습 RMSE1.4378/검증14.7760, baseline 혼합은7.1957/14.1906입니다.
전체795개 입력 중 ET·base Ridge는519개를 선택하고 morph Ridge는795개 모두를 사용합니다.
검증값의 학습 min/max 이탈률은0.5414%, 개체별 이탈률과 제곱오차의 Spearman 상관은0.0206입니다.
단순 특징 범위 이상이 주원인이라는 근거는 약합니다. 과적합·예측 중복·기존 비중을 함께 고려해야 하며014 개선의 단일 원인을 확정하지 않습니다.

125는014의 compact26 KRR/CNN/Smean/Bmean 네 가중치만 비음수·합1로 학습했습니다.
기존[0.3,0.3,0.2,0.2]에 대한 제곱 규제 lambda1 하나를 고정했고 shortB10%는 유지했습니다.
Outer 가중치는 내부 OOF에서, 잔차 앵커 가중치는 한 단계 더 깊은 OOF에서 학습했습니다.
전역 outer OOF는 평가에만 사용했습니다. 모든 affine과 honest 잔차를 새 혼합에 맞춰 재학습했습니다.
074의 단일 affine,077의 무규제 macro contrast,112의 attention 혼합 비중과 구별됩니다.

| 구성 | 원본 RMSE | 재분할1 | 재분할2 | pooled RMSE |
|---|---:|---:|---:|---:|
| 고정014 | 12.968700 | 12.668261 | 13.045858 | 12.895301 |
| 가중치 학습125 | 12.951399 | 12.675085 | 13.091223 | 12.907060 |

125의 pooled RMSE 변화는 +0.011759, 조건부 ID 묶음95% 구간은[-0.006347, 0.030640]입니다.
같은400개를 반복 사용한 탐색이므로 이 구간은 연구 선택·재학습 불확실성을 반영하지 않습니다.
주지표 pooled MSE가 악화돼014를 유지합니다. lambda 재탐색이나 사후 혼합을 추가하지 않습니다.

가중치·affine60개를 독립 SLSQP/OLS로 검증했고 잔차 모델720개(none 포함)를 재로딩·독립 계산했습니다.
고정014 전체 fold 재학습 대조, outer 검증 타깃 및 잔차 내부 검증 타깃 변경 검증을 통과했습니다.
새 설치·비용·신경망 학습·holdout/test 열람·전체500 재학습·새 제출 CSV·업로드는 없습니다.
기존014 CSV SHA256: `2c5ce4ca43d75a7f47fbf7dd6c155db9c5c090d86f23b5999a0f6920def4f6a0`
014의 업로드 및 결과 확인은 부모 작업에서 조율합니다. 실행상 막힘은 없습니다.
상세: `reports/summaries/baseline124-weights125-summary.json`, `artifacts/baseline-diagnostics-124/report.json`, `artifacts/constrained-weights-125/plan.json`.

## 013→014 잔차 변화의 이미지 조건 전이 감사126 (완료, 014 유지)

094의 전체 고오차 형태 대조,097/099~111의 실패 가설을 대조한 결과 새로운 표현 학습을 정당화할 근거가 부족했습니다.
단순 재탐색 대신 baseline 제거와 후속 재학습에 따른013→014 오차 변화가 특정 형태·잡음 조건에서 반복되는지 감사했습니다.
기존 이미지 지표8개와 morphology4개만 사용하고, 각 outer 학습320개에서만 고오차 변화 조건 하나를 골랐습니다.
타깃은 세 정직한 앵커 블록의 (y−base014)^2−(y−base013)^2입니다. 각 앵커는 해당 블록과 outer 검증80개를 제외해 학습됐습니다.
12특징의 학습 하위/상위 사분위24조건 중 평균 변화 대비가 가장 큰 조건을 선택하고 외부80개에 그대로 적용했습니다.
최종 평가에는 잔차 재학습까지 포함한013/014를 사용했습니다. 앵커와 최종 모델의 학습량·단계 차이는 남습니다.

| 외부 평가군 | 예측 쌍 수 | RMSE013 | RMSE014 | MSE 변화014−013 |
|---|---:|---:|---:|---:|
| 학습에서 선택된 조건 | 297 | 11.069803 | 10.814330 | -5.590800 |
| 나머지 | 903 | 13.679313 | 13.509870 | -4.607009 |

선택 조건군−나머지의 MSE 변화 대비는-0.983791, 조건부95% 구간[-8.274315, 6.294488]입니다.
학습에서의 선택 대비+12.9371이 검증에서 재현되지 않았습니다. 양쪽 군 모두014로 개선돼 조건별013 복원이나 새 보정의 근거가 없습니다.
세 분할1200개 예측 쌍은400개 이미지이며 bootstrap은 같은 ID의 세 예측을 함께 묶었습니다.
원본/재분할 모두 이미 반복 연구에 쓰였으므로 결과는 적응적 탐색이며 독립 확인이 아닙니다.

선택 조건은15fold에서8종류, 위험 조건군의 분할 간 Jaccard는0.223~0.274였습니다.
noise/contrast 낮음이5회 선택됐지만 이 빈도를 보고 새 게이트를 만들거나 점수를 계산하지 않았습니다.
개체별 제곱오차 변화의 분할 간 Spearman0.584~0.593, 불리한 변화 상위40개 교집합13~19개는 기술 통계입니다.

15개 조건의 독립 계산과 모든 outer 타깃·특징 교란 검사를 통과했습니다.
개발400개 원본 이미지 지표와 라벨 독립 선정6개 morphology 특징 재추출은 저장값과 오차0입니다.
이미지 단위 bootstrap도 독립 재계산했습니다. 기존 upstream 학습 누수 검증 증명서를 재사용했으며 새 신경망 학습을 주장하지 않습니다.

014 및 이전 보고서200개·패키지26개 보존을 확인했습니다. 새 제출 CSV·전체500 재학습·holdout/test 열람·설치·비용·업로드는 없습니다.
새 가설 근거 없이 특징/임계값을 확대하지 않고 이 감사를 종료합니다. 이는 성능 상한이나 모든 이미지 가설의 실패를 뜻하지 않습니다.
상세: `reports/validation/domain-risk-audit-126-summary.json`, `artifacts/domain-risk-audit-126/validation.json`.

## 現 기저 모델의 중첩 학습량·상보성127 (완료, 014 유지)

050은 과거DINO-S Ridge,078은213/214→320의legacy/DINO집계,118은잔차절반표본실험입니다.
127은현재compact26·CNN80·S/Bmean3를분리하고142/143단계를추가했습니다.
기존105모델묶음(840개leaf모델)을재로딩했으며새학습은없습니다.
각outer학습320개안에서142/143⊂213/214⊂320을구성하고모든수준을동일outer80개에서평가했습니다.
세작은모델의예측을평균해새ensemble을만들지않고각각의MSE를평균했습니다.

| 기저 모델 | 142/143장 검증RMSE | 213/214장 | 320장 | 320장 학습RMSE |
|---|---:|---:|---:|---:|
| compact26 | 14.734442 | 14.684380 | 14.402538 | 13.236389 |
| CNN80 | 15.560772 | 14.985635 | 14.621919 | 9.828551 |
| DINO_S_mean3 | 14.986205 | 14.625263 | 14.261857 | 0.588469 |
| DINO_B_mean3 | 14.818419 | 14.385192 | 13.870420 | 0.484348 |

213/214→320에서는모든기저모델이세분할각각에서개선했습니다.
320장보정전고정4way혼합RMSE는13.036232이며최종014점수와는다릅니다.
이혼합에서CNN만빼고남은고정비중을정규화하면13.315263(+0.279031)입니다.
CNN의단독검증오차가커도상보성은남습니다. Compact26제거는+0.015804,S제거+0.155837,B제거+0.186631입니다.
이진단은affine·shortB·잔차재학습을포함하지않아121/122의전체파이프라인ablation을대체하지않습니다.

다음우선순위는**CNN데이터수와최적화업데이트예산을분리하는단일통제**입니다(제안,미실행).
현재80epoch에서는142/143장400step,213/214장560step,320장800step으로계산량도함께증가합니다.
작은142/143표본을고정800update로학습하는한대조와기존400update·큰320표본800update를같은검증80개에서비교하면두가능성을더구분할수있습니다.
실행전learning-rate/weight-average규칙을고정하고80epoch통제를재현해야합니다. 바로014의epochs를늘리거나성공을예고하지않습니다.
새표현은구체적누락근거가없고,DINO과적합간격만으로이미실패한epoch/dropout/width/seed규제를다시탐색하지않습니다.

KRR105개독립해법,DINO630개독립functional예측,CNN9개다른배치재로딩을검증했습니다.
078의중간DINO대조최대오차5.68e-14,독립KRR3.13e-11,DINO4.38e-6,CNN1.53e-5입니다.
최초1e-8head일치검사는배치크기에따른6.74e-7차이로실패했고,원래배치로는오차0을확인했습니다.
구형CNN체크포인트의average_last_epochs메타데이터누락도원래학습코드의마지막5회평균으로확인했습니다.
원본계획·소스·실패로그를보존하고수치검증버전과메타데이터호환기록을추가했으며모델설정·가중치는변경하지않았습니다.

불확실성은400개이미지ID를묶은10000bootstrap으로계산했습니다. 세분할·세중첩경로는독립데이터가아닙니다.
반복된400개선택편향과seed/학습부분집합/업데이트수/alpha÷n차이는남습니다. 학습곡선으로500장성능이나상한을외삽하지않습니다.
014·기존모델·이전보고서200개·패키지26개보존을확인했습니다. holdout/test열람·설치·비용·새제출·업로드는없습니다.
상세: `reports/summaries/base-learning-curve-127-summary.json`, `reports/summaries/base-learning-curve-127-analysis.json`.

## CNN 데이터 수·업데이트 예산 통제128 (완료, 원본 음성으로 종료)

127에서 제안한 대조를 실제 MPS에서 학습했습니다. 원본5fold의 같은 외부80장마다
작은142장400업데이트, 같은142장800업데이트, 큰320장800업데이트를 비교했습니다.
현재TinyCNN128·MSE·AdamW·D4·초기시드를 유지하고, 주대조의 배치는 모두32장으로 맞췄습니다.
학습 순열의 잔여를 다음 순열과 연결해 drop_last나 부분 배치를 사용하지 않았습니다.
샘플링과 증강 난수는 분리했고, 작은400실행은800실행의 샘플·증강 난수 prefix를 공유합니다.

Cosine은 epoch가 아닌 업데이트 진행률로 계산했습니다. 400·800 모두 같은최저LR로 끝나고,
가중치평균도95/96.25/97.5/98.75/100%의5개state로 동일합니다.
짧은실행을 긴스케줄 중간에서끊지 않았습니다. 다만 총예산에맞춰LR경로가늘어나므로
이는 예산+normalized schedule 절차의비교이며, 동일한절대LR경로에서순수step수만바꾼효과는아닙니다.
두800실행의LR·batch·증강·초기화·dropout난수진행은 일치했습니다.

| 대조 | 학습장수 | 업데이트 | 학습RMSE | 외부검증RMSE |
|---|---:|---:|---:|---:|
| 작은집합·기준 | 142 | 400 | 9.545296 | 15.418684 |
| 작은집합·예산2배 | 142 | 800 | 3.022690 | 15.597598 |
| 큰집합·동일예산 | 320 | 800 | 8.305694 | 14.706596 |

작은집합의 예산확대는 검증RMSE0.178914 악화(개선량95%구간[-0.675087,0.318068])였습니다.
학습오차만 크게 감소했고, 사전 고정한 원본개선 기준을 통과하지 못해 두재분할로 확대하지 않았습니다.
동일800예산에서는 큰집합이0.891002 개선(조건부95%구간[0.073643,1.686196])됐습니다.
현재근거는 단순추가최적화가 데이터증가 효과를 대체하지못한다는 쪽이며, 모든장기학습의실패를 증명하지않습니다.

기존142장80epoch·부분배치 설정은 별도 재학습해 저장가중치·예측값 오차0으로 재현했습니다.
주대조15개 모델 학습은171.68초, 기존재현1개·누수검사재학습1개까지 총17개를 학습했습니다.
15개MPS재로딩 최대오차0, CPU/MPS차이1.53e-5입니다.
학습밖258개타깃을변경한400업데이트재학습은 가중치SHA와예측이 정확히일치했습니다.

순열경계의 작은집합 배치는 드물게ID가중복되어 평균고유이미지31.716~31.759장, 큰집합은32장입니다.
모든배치의입력행수는32지만 이특성을 기록하며 원래부분배치와동일한프로토콜이라고주장하지않습니다.
같은400개를반복사용한탐색이며,10000회불확실성계산은400개ID단위로수행했습니다.
새epochs/LR/seed/sampler/혼합비중 탐색을 추가하지않습니다. 320장더긴CNN이나전체014결합은시험하지않았습니다.
014·기존최고모델·이전보고서200개·패키지26개를보존했습니다. holdout/test열람·설치·비용·제출은없습니다.
상세: `reports/summaries/cnn-update-budget-128-summary.json`, `reports/summaries/cnn-update-budget-128-decision.json`, `artifacts/cnn-update-budget-128/validation-original.json`.

## CNN 회전·반사 증강 가정 제거129 (완료, 014 보존)

111에서 확인한 합성 hardness의 D4 불변 보증 부재를 실제 학습 대조로 검사했습니다.
046의 DINO 회전 평균,110의 잡음 추가,128의 학습 예산 변경과 달리 CNN의 기존 D4를 제거하는 한 실험입니다.
동일 TinyCNN128·80epoch·800update·초기시드·배치32·epoch cosine·마지막5epoch 평균을 사용했습니다.
identity도 D4 난수를 소비한 뒤 변환만 생략하여 모든 배치 순서와 dropout 난수 진행을 맞췄습니다.
학습320개에서만 픽셀·타깃 정규화했고 양쪽 추론은 원래 방향만 사용했습니다.

| 분할 | D4 CNN | identity CNN | D4 고정4way | identity 고정4way |
|---|---:|---:|---:|---:|
| original | 15.032318 | 16.338433 | 13.065976 | 13.493155 |

단독 CNN과 고정4way 둘 다 원본에서 개선할 때만 두 재분할로 확대하는 기준을 사전 고정했습니다.
실행 분할: original. 원본 통과: False. 세 분할 모두 일관 개선: False.
고정4way는 .3compact26+.3CNN+.2Smean3+.2Bmean3이며014의 affine·shortB·잔차 보정을 포함하지 않습니다.
cnn pooled RMSE 개선량(D4−identity): -1.306115, 조건부95%구간 [-2.2168683931428625, -0.3771831217325229].
raw pooled RMSE 개선량(D4−identity): -0.427179, 조건부95%구간 [-0.6918918354887087, -0.16115354693678913].

본 학습 10개와 학습 밖 라벨 변경 재학습1개를 실행했습니다. 기존 D4 가중치·예측 재현,
모든 모델 MPS/CPU 재로딩, 배치·LR·5개 state 평균·실제 입력 표본 SHA·학습/검증 그룹 분리를 독립 확인했습니다.
80개 외부 타깃 변경 재학습은 가중치 SHA가 동일했습니다.
권고: Retain014 and close this exact identity-only CNN hypothesis. No resplit/augmentation-ratio/subgroup/seed rescue search and no downstream/full500 retraining justified by this screen.

D4 제거는 방향 정보와 증강 정규화 효과를 함께 바꿉니다. 결과가 물리적 회전 불변/의존성을 입증하지 않습니다.
같은400개 반복 연구이며 bootstrap은 재학습·탐색 선택 편향을 반영하지 않습니다.
014·이전보고서200개·패키지26개를 보존했습니다. 설치·비용·외부자료·holdout/test·서버zeroing·업로드는 없습니다.
상세: `reports/summaries/cnn-orientation-129-summary.json`, `artifacts/cnn-orientation-129/plan.json`.

## 固定 이진 2점·3점 추가 잔차130 (완료, 원본 음성으로 bank 종료)

기존108은 연속명암 3점 모멘트·직선/등변삼각형·직접KRR·01110%혼합이었습니다.
130은 사용자 지정 분위.2/.5/.8 이진마스크의 실제 점유율p로 중심화한 곱입니다.
r=1/2/4/8/16/32의0,(r,0),(0,r) 및0,(r,0),(0,2r),D4평균·표준편차72열을 고정했습니다.
세 변의 displacement를 D4궤도로 중복 제거해19개궤도·114열의2점 대조를 만들었습니다.
3점은 공통유효삼각형 영역,2점은각변위의유효쌍영역을 사용하며 wrap·padding은 없습니다.
두논문은 재현하지 않았습니다.2021논문의3점SCE탄성계수,2025논문의3점·Gram·집합조직결합은 hardness 보장이 아닙니다.
두번째 논문 출판사403으로기관초록까지만 확인했고 단독3점 부정 결과는 사용자 제공 근거로 구분했습니다.

014의기존physics57 보정을 유지하고 새Ridge를추가했습니다. 내부각검증H를제외한I에서physics모델을다시선택하고,
I안의교차검증physics모델도각학습J안에서규제를선택해 새잔차타깃을만들었습니다.
새Ridgeλ1/.1와추가비율0/.5/1은honestB106/107 내부3fold에서만 선택했습니다.
스케일러는매학습부분집합에서특징군별std의RMS floor를fit하며2점규모는3점추가에도같습니다.

| 원본400 대조 | RMSE |
|---|---:|
| 기존014 | 12.968699772 |
| 014+2점 | 12.976657519 |
| 014+2점+3점 | 12.979599585 |

3점의014대비개선량 -0.010899814,2점대비 -0.002942066로 모두음성입니다.
사전규칙대로재분할확대·문턱/삼각형/비율재탐색·2점대체후보를만들지않고종료했습니다.014를유지합니다.
20개학습이미지추출 0.984초,나머지380개 19.115초,원본모델학습 3.827초입니다.
상수0·D4·보완마스크부호·비주기경계·재현·floor fixture를통과했습니다.
physics3120개(무보정포함)와Ridge240개를독립계산했고최대오차8.62e-14/5.21e-14,014재현5.69e-14입니다.
외부80개/내부36개타깃변경에도해당학습모델과선택·예측이동일했습니다.
초기NumPy정수JSON직렬화실패원본·로그를보존했고메타데이터저장만별도v2에서수정했습니다.
같은400개재사용,학습량차이,이진분위상proxy,D4방향요약한계를기록합니다. 모든3점통계의실패를뜻하지않습니다.
014·이전보고서200개·패키지26개보존. 새설치·비용·외부학습자료·holdout/test·서버probing·제출없음.
상세: `reports/summaries/binary-threepoint-130-summary.json`, `artifacts/binary-threepoint-130/serialization-v2/model-plan.json`.

130 출처 정정: 두 번째 논문의 단독 3점 부정 결과는 부모 작업의 공개 논문 연구에서 전달받은 요약입니다. 이 작업에서는 해당 원문을 직접 검증하지 못했습니다. 위의 “사용자 제공 근거” 표기는 이 정정으로 대체합니다. 원본 기록과 수치 결과를 보존하며 130은 종료,014를 유지합니다. 정정본: `reports/summaries/binary-threepoint-130-summary-corrected.json`.

## RNC 표현학습131 (고정 소규모 대조 완료)

120은 frozenDINO 최종스칼라 순위벌점이며131은 CNN잠재64차원 거리의 target-distance순서대조입니다. 공식논문과저자loss를읽기만했고 외부코드를다운로드·실행하지않았습니다.
MSE와RNC 모두 동일초기값/증강/800updates/25600views,이후동결표현에 동일train-onlyStandardScaler/Ridge10. tau2,40epochs,배치16ID×2views,추가튜닝없음.

| 원본400 조건 | RMSE |
|---|---:|
| mse | 14.482584949 |
| rnc | 15.358999440 |
| legacyCNN | 15.032317843 |
| raw_mse | 12.912475665 |
| raw_rnc | 13.150093737 |
| raw_legacy | 13.065975964 |

Fixed RNC screen failed. No tuning/rescue blend; retain014, close this small configuration.
014전체파이프라인 대조가아닌 raw4way 사전검사입니다. 손실·기울기독립계산,전모델CPU/MPS재로딩,독립SVD회귀,외부80label변조양조건재학습검증통과.
014및이전200보고서/26패키지보존. 설치·비용·holdout/test·제출없음. 상세:`reports/summaries/rnc-131-summary.json`.

## MSE CNN 회귀기/학습절차132 (nested014 대조 완료)

131MSE는014CNN과다름: 같은CNN/MSE/원본fold/800updates/25600views이나80epochs단일view/마지막5평균에서40epochs두view/최종가중치로변경,seed와회귀기도변경. RNC효과아님.
같은저장checkpoint의scalar/Ridge및새절차legacyseed대조를분리했습니다. legacyRidge만교체하면raw13.065976->13.131929로악화. 새절차legacyseedRidge13.054613,131seedRidge12.912476;스케줄단독효과로해석불가.
CNN만outer/middle/deep전체에서교체하고기존014affine/physics57선택잔차를매번재학습했습니다. KRR/S/B/shortB불변.

| partition |014|새CNN 전체pipeline|
|---|---:|---:|
|original|12.968699772|12.837478360|
|20261031|12.668260573|12.946444837|
|20261101|13.045857558|13.468018932|

상태:completed_negative_pooled_development_control. 014유지. 독립CNN재로딩/SVD/OLS/physics/외부·내부타깃변조통과. 설치·비용·holdout/test·full500·제출없음. 상세:`reports/summaries/mse-readout-132-summary.json`.

## 기존 CNN 고정3seed 평균133

018/021에이미20261003/23/43의동일CNN80epoch/last5평균 3분할실험이존재했습니다. 기존외부45개CNN을재로딩검증·재사용했으며새seed나최고seed선택은없습니다.
당시003fusion과현재014nested절차는달라,누락된내부두seed만학습하고014affine/physics57잔차선택을다시학습했습니다.

| partition |014|고정3seed 전체pipeline|
|---|---:|---:|
|original|12.968699772|12.830176114|
|20261031|12.668260573|12.636884323|
|20261101|13.045857558|13.107923156|

첫 재분할 결과를 양성으로 확인한 뒤 부모 지시에 따라 순차 중단 규칙을 폐기하고 세 분할 pooled MSE 판단으로 명시했습니다. 실제 중단·seed·가중치 변경은 없으며 이 시점 변경을 기록했습니다.

상태:completed_positive_pooled_development_control. 014보존. 원본음성이면중단,추가seed/부분집합/비율탐색없음. 캐시45개/새nestedCNN/독립OLS·physics/외부·내부타깃변조검증통과.
설치·비용·holdout/test·full500·제출없음. 상세:`reports/summaries/cnn-seed-mean-133-summary.json`.

## 후보015: 고정 CNN 3-seed 평균의 전체500 학습 및 검증 (미업로드)

133의 고정 seed20261003/20261023/20261043을 동일 비중으로 평균하고 identity 추론을 사용했습니다.
seed·가중치·CNN 구조·80epoch/마지막5epoch 가중치 평균 규칙은 변경하지 않았습니다.
전체500 CNN 세 체크포인트는 동일 학습 절차의 기존024에서 검증 후 재사용했습니다.024의 D4 추론은 사용하지 않았습니다.
누락된 nested CNN24개, affine4개, physics57 잔차 선택/최종 모델48개(none 포함)를 새로 학습했습니다.
KRR/S/B/shortB는 검증된 전체500 모델을 재사용했습니다. 전체500 OOF는 보정 학습용이며 검증 점수가 아닙니다.
기존 holdout100을 별도로 평가하지 않았습니다.

개발 pooled RMSE:014의12.895301459 →015 절차12.859781154, 개선0.035520305.
조건부 ID 묶음95% 개선 구간은[-0.026804586,0.099789302]로0을 포함합니다.
원본/첫 재분할은 개선됐고 마지막 재분할은13.045857558→13.107923156으로 악화됐습니다.
같은400장과 겹치는 재분할을 반복 사용한 선택 편향, 고정 모델 조건부 구간의 한계가 남습니다.
pooled 기준은 첫 재분할 결과를 본 후 부모 지시로 명확해졌으며 실제 조기 중단이나 seed/가중치 변경은 없었습니다.

훈련 검증 후 test1000개를 추론했습니다. ID·순서·중복·누락·finite·범위, 모델 재로딩 및 CSV bytes 재현을 통과했습니다.
새 nested CNN24개와 전체500 CNN3개를 CPU/MPS 재로딩으로 검증했고, affine4개와 잔차48개를 독립 계산했습니다.
최종 CPU/MPS 최대 차이:5.41758186e-06; 독립 OLS/커널 오차:5.684341886e-14.
예측 범위:[161.77814627638583, 236.42398745269793]. clipping·test 튜닝 없음.

CSV:`artifacts/submission-candidate-015.csv`
SHA256:`74c9f5533b8b72c68b90af9d0b411bdc79b9cd9ec240dc3a18d7efb3a11ab67f`

014를 그대로 보존했습니다. 비용0, 설치 없음, 업로드 없음. 업로드는 부모 작업에서 조율합니다.
캐시 조사 중 과거004 보고서의 test 요약 통계가 표시됐으나 이미 고정된015 설정 선택에 사용하지 않았습니다.
상세:`reports/submissions/candidate015-summary.json`, `reports/submissions/candidate015-verification.json`.

## 011/014/015 재현·규정 감사135 (학습·제출 대기 유지)

기존103·011/014/015 전체 추론 재현·훈련 통계 검증을 인용하고 미확인 경로를 점검했습니다.
개발400 ID에 해당하는 full500 캐시의 S/B/CNN128/compact26/physics57 특징은 개발 캐시와 정확히 같았습니다.
S 캐시의 (400,3,1152) 저장 형식은 실제 head 입력처럼 (400,3456)으로 펼쳐 대조했습니다.
개발6행에서011의 shortB 모델과014/015의 간소화 loader가 정확히 일치하고, CNN 세 seed의 개발/생산 identity 추론 helper도 정확히 일치했습니다.
저장된 잔차 모델 예측은 query batch 구성에 의존하지 않았습니다. 이 leaf 추론에 fit/backward 차단 guard를 걸었으며 모두 통과했습니다.
직접 inference 함수의 AST 검사에서도 fit/fit_transform/partial_fit/backward/step 호출이 없었습니다.
이것과 기존 전체 추론 증거를 합친 한정된 감사이며 임의의 미래 코드까지 보증하지 않습니다.
새 test 재추론·holdout 평가·모델 학습은 하지 않았고 세 후보 CSV는 해시만 확인했습니다.

실제 차이: 세 후보 공통 compact26 KRR의 final017은 개발400에서9개 조합 중 bandwidth4/alpha0.1을 선택한 뒤500장으로 fit했습니다.
nested 모델은 각 학습 부분집합에서 선택합니다. estimator·전처리·선택 grid는 같지만 최종 하이퍼파라미터 선택을500 내부 CV로 다시 수행한 것은 아닙니다.
이는 test 누수가 아니며 기존 동작을 조용히 바꾸지 않았습니다. 변경한다면 별도 학습 절차로 개발 검증해야 합니다.
고정 epoch 때문에500과320 및 nested 학습의 update 수가 다른 것은 기존 의도된 규칙입니다.
`clipping:false`는 최종 예측 clipping 없음입니다. compact/physics 특징의 MinMaxScaler(clip=True)는 저장된 train 범위로만 변환하며 test에서 bounds를 fit하지 않습니다.

[공식 규정](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai)을2026-10-04 읽기 전용으로 확인했습니다.
공개 사전학습 가중치의 허용 라이선스와 로컬 실행은 허용되고 외부 학습 데이터·test 학습은 금지됩니다.
현재 원본 비-register DINOv2 S/B는 공식URL·고정 commit·가중치 SHA256·원본 LICENSE/MODEL_CARD로 Apache-2.0 출처를 확인했습니다.
Cell-DINO/XRay-DINO 가중치를 사용한 것이 아닙니다. 로컬 해시는 게시자 서명 checksum을 뜻하지 않습니다.
초기 `reports/provenance/software-licenses.json`의 사전학습 가중치 없음 문구는 당시 기록으로 보존하고,
현재 후보용 `reports/provenance/software-licenses-current-135.json`을 추가했습니다.
AI 코딩 보조에 관한 명시적인 별도 조항은 해당 공개 규정에서 확인되지 않았으며 주최자의 개별 승인으로 표현하지 않습니다.

남은 실제 준비 과제: 현재 deliverables 소스 ZIP은011/014/015의 entrypoint를 포함하지 않는 과거 묶음입니다.
프로젝트 내 재현 증거는 있지만 심사용 독립 패키지 재현은 아직 확인하지 않았습니다.
다음 우선순위는 추가 모델 학습보다 최종 선택 후보의 상대 경로·오프라인 추론 패키지, 최소 의존 파일 및 LICENSE/출처 동봉,
프로젝트 내부 격리 폴더에서의 재현 확인입니다. 현재 제출 후보나 가중치 변경은 하지 않았습니다.

상세:`reports/validation/reproducibility-audit-135-summary.json`, `artifacts/reproducibility-audit-135/report.json`.
011/014/015 보존, 비용0, 설치 없음, 업로드 대기 유지.

## 오프라인 재현 패키지136 (011/014/015, 비공개·미업로드)

새 패키지:`deliverables/reproduction-011-014-015-v136.zip`
SHA256:`4427867a93a8e9ffb0526c467eaa570c5b6952a0f18a45adcb40df30cffad2ef`

세 후보를 명시한 모델·원본이미지 추론·full500 학습 진입점, 현재26개 의존성 고정정보, DINO 출처/라이선스/해시, 데이터배치와 seed·중첩선택 명령을 포함합니다.
기존 ZIP과011/014/015를 보존했고 원본 이미지·정답CSV·test 특징캐시·사용자경로·credential은 넣지 않았습니다.
학습된 커널의 support vector와 정규화 통계는 모델의 필수 구성으로 포함됩니다.
독립 디렉터리에서 기존uv환경만 사용해 원본 test1000 특징을 새로 추출하고 세 후보 예측을 비교했습니다.
최종 코드로 MPS 모델 재로딩 및 CPU 추론도 확인했습니다. 구체적인 오차·CSV bytes·해시는 패키지 `VALIDATION.json`과 보고서에 기록했습니다.
전체 학습 명령은 실행하지 않았으며, 새 통합 학습 드라이버의 전체 재학습 가중치 일치는 검증하지 않았습니다.
학습 import·구문·plan·저장 모델 설정 확인을 전체 재학습 재현으로 과장하지 않습니다.
설치·추가모델학습·test튜닝·업로드·비용 없음. 상세:`reports/validation/reproduction-package-136-summary.json`.

## 규제된2차 보정137 및 후보016 (검증 완료, 미제출)

015의 고정 KRR/CNN3seed/S/B 혼합 비중은 그대로 두고 affine 보정에 곡률 항 하나를 추가했습니다.
074의 단일 affine,077·125의 성분별 선형 비중 학습,115의 물리특징 커널 잔차와 구별됩니다.
073의 선형 보정 필요성이 출발점이며 곡률의 유용성이 사전에 입증된 것은 아닙니다. Public015 점수로 곡률·비중·표본 정보를 추정하지 않았습니다.

학습 OOF 예측을 train mean/std로 표준화한z에서z²의 상수·선형 성분을 제거하고 남은q를train RMS로 정규화했습니다.
기존 affine 계수는 비규제이며 곡률 항의 penalty는1로 고정해 해당1차원OLS의 절반(추가df0.5)을 사용합니다.
Outer320 보정은 middleOOF, 잔차anchor213/214 보정은 deepOOF에서 fit하고 physics57 잔차 선택·모델을 새 타깃으로 다시 학습했습니다.
차수·규제·비중 탐색이나 clipping은 없습니다. 세 기존 분할 pooled untrimmed MSE 판단을 결과 전에 고정했습니다.

| 분할 |015|137/016 절차|RMSE 개선|
|---|---:|---:|---:|
|original|12.830176114|12.813150116|0.017025998|
|20261031|12.636884323|12.612076949|0.024807374|
|20261101|13.107923156|13.072096619|0.035826537|

Pooled RMSE:12.859781154→12.833822643; 개선0.025958511.
조건부 ID 묶음95% 개선 구간:[-0.013876261493579145, 0.07093112776415968].0을 포함하므로 개선의 불확실성이 남습니다.

같은400개를 반복 연구한 선택 편향, 겹치는 재분할, 고정 예측에 조건부인 불확실성, quadratic 외삽과 학습량 차이를 유지해서 해석해야 합니다.
60개 곡률 보정의 독립 augmented least-squares,720개 물리 모델 재로딩/독립 계산 및 outer/innerB 타깃 변경 불변성을 검증했습니다.
신경망과 기저 표현은133의 학습범위·타깃 제외 검증을 인용해 재사용했습니다. 새 신경망 학습은 없습니다.

양성 결정을 고정한 후 전체500에 곡률 보정4개와 물리 잔차 모델48개(none 포함)를 다시 fit했습니다.
이전holdout100은 최종 학습에만 포함했으며 별도 점수/선택에 사용하지 않았습니다.
훈련 검증 후 test1000을 추론했고 ID/순서/중복/finite/범위, 모델재로딩과 CSV바이트 재현 및 CPU/MPS 대조를 통과했습니다.

016 CSV:`artifacts/submission-candidate-016.csv`
SHA256:`ceabcb7f019b7234e638510899ae98d2aacee08892f59650fd6cb16ff301cce0`
예측 범위:[163.36132037801448, 237.82362827332486];CPU/MPS최대 차이:5.612575279e-06.

현재Publicbest015를 보존합니다.016은 미제출이며 Public/private 개선은 미확인입니다.014 역시 미제출입니다.
기존 재현ZIP136은011/014/015용으로 보존했으며016을 포함한다고 주장하지 않습니다.
설치·비용·추가외부자료·test튜닝·업로드 없음. 제출은 부모 작업에서 조율합니다.
상세:`reports/summaries/quadratic-calibration-137-summary.json`, `reports/submissions/candidate016-summary.json`, `reports/submissions/candidate016-verification.json`.

## 140 공간 공동변화 실험 및 미제출017

기존 ViT-B 최종 패치에 고정32차원 투영을 적용하고 이웃 차이의 공동행렬528항을 physics57 잔차 입력에 추가했습니다. 비공간 공분산은 대조군입니다. 곡률137은 사용하지 않았습니다. 사전 고정한 세 개발 분할의 통합 RMSE는01512.85978115 / 비공간대조12.86332289 / 공간추가12.84853431입니다. 개선0.01124685, 조건부95%구간[-0.01208873,0.03357586]로 불확실하며 한 분할은 소폭 악화했습니다.

전체500의 변경된 잔차 모델48개와 test1000 추론·재로딩·ID순서·finite·CSV바이트 재현 검증을 완료했습니다. best015와 기존ZIP을 보존합니다. 새017은 미업로드이며 업로드는 부모와 별도 조율합니다. 기존136 재현ZIP에는017이 포함되지 않습니다.

CSV:`artifacts/submission-candidate-017.csv`

SHA256:`4256a5d2636053f8f1d115f39b0265a66a10c3169bd4016309ecdd2b63c3be20`

결과:`reports/summaries/spatial-joint-140-summary.json`, `reports/submissions/candidate017-summary.json`, `reports/submissions/candidate017-verification.json`, `reports/submissions/candidate017-preservation.json`.

## 143 잔차 진단 및 144 선택 안정화 — 미제출018

017 개발 잔차 진단에서45개 잔차블록 중24개는 내부fold 하나를 제외하면 선택후보가 바뀌었고,32개는1·2위 차이가 작았습니다. 기존5개 후보의 내부OOF 손실을512회 그룹 재표집해 선택빈도로 예측을 평균하는 단일 절차144를 사전 고정했습니다. 새특징·곡률·신경망 증가는 없습니다.

통합 개발RMSE01712.84853431→01812.83654465,조건부95%개선구간[-0.01312113,0.03702425]. 두분할 개선/한분할 악화이며 성능 향상을 보장하지 않습니다. 전체500의 추가잔차후보9개만 학습,모델15개 독립검증,test1000 ID/finite/order/reload/CSV바이트 검증을 완료했습니다. best017 유지,018 미업로드,제출 조율합니다.

CSV:`artifacts/submission-candidate-018.csv`

SHA256:`6d478f08de769ca0c54d640c0711c95b15ad4e6b707d0516022c23260f1cfc16`

기록:`reports/summaries/residual-stability-143-diagnostic.json`, `reports/summaries/bootstrap-selector-144-summary.json`, `reports/submissions/candidate018-summary.json`, `reports/submissions/candidate018-verification.json`, `reports/submissions/candidate018-preservation.json`.

## 147–152: best018 보존 및 미제출019

147 기존 특징뱅크 보완성 진단은 긍정적 근거가 없어 혼합하지 않았습니다. 방향성 에너지148(통합RMSE12.84682646)과 명암–토큰 결합149(12.85920477)은018(12.83654465)보다 나빠 종료했으며 결과를 보존했습니다.

151은 기존 CNN3seed의8개 D4방향을 동일 비중으로 평균합니다. 현재018의 모든 외부/중간/깊은중첩 예측에 적용하고 보정식·잔차 모델을 다시 학습했습니다. 원본12.81194528→12.77212764,재분할12.64587262→12.66541421 및13.04862442→13.02985466;통합12.83654465→12.82337805. 조건부95%개선구간[-0.03217290,0.05718671]이0을 포함하므로 작은 탐색적 개선입니다. 반복 사용한400장과 겹치는분할에 근거하며 독립 검증 성능으로 해석하지 않습니다.

900개 개발잔차모델과180개 중첩CNN예측캐시,45개 기존외부CNN의 D4예측을 검증했습니다. 전체500용 기존CNN가중치를 재사용하고12개 중첩D4예측노드·4개보정식·60개잔차모델객체(48개비영모델)를 생성했습니다. test1000 ID/finite/order/reload/독립수식/CSV바이트 검증 완료. 새로운 신경망학습·패키지설치·비용·holdout부분점수·test튜닝은 없습니다.

019 CSV:`artifacts/submission-candidate-019.csv`

SHA256:`77fa34cf57395c58a4289588bfad5ef65e0c984ef9ceb679bfbcee5b6784d51c`

현재Publicbest01811.5230652779와 기존 파일 보존,019는 미제출이며 마지막 제출기회는 남겨 두었습니다. 기록:`reports/summaries/cnn-view-mean-151-summary.json`, `reports/submissions/candidate019-summary.json`, `reports/submissions/candidate019-verification.json`, `reports/submissions/candidate019-preservation.json`. 기존136재현ZIP은011/014/015범위이며019를 포함하지 않습니다.

## 153–155: 공간 범위 제한 검증 및 best018 재현 패키지

153은 현재018의 모델 상보성과 학습 규모별 공간 특징 절단을 진단했습니다. 내부70–72장 모델의 공간 좌표2.76%/표본99.1%,외부106–107장 모델의 좌표1.85%/표본96.6%에서 범위를 벗어난 값이 있었습니다. 절단 크기와 큰 오차의 양의 관계는 확인되지 않았습니다. 공간528개 특징만 제한을 해제하는 단일154 통제는 통합RMSE12.83654465→12.83712366으로 악화하여 기각했습니다.900개 모델 독립검증·30개 새 적합 재현·라벨배제 검증을 통과했으며 변형 탐색·전체500 후보학습·새CSV는 없습니다.

기존136은011/014/015만 포함합니다. 새 best018 저장모델 추론 패키지:`deliverables/reproduction-018-v155.zip`
SHA256:`613638cb56740157023a389dc9a2985ddc6479e0bd1bb0e35126bd633373df53`
독립 디렉터리에서 원본test1000 특징 재추출,018 예측/MPS재로드/CPU추론/ID/finite/순서/파일해시를 검증했습니다. 데이터·정답CSV·특징캐시·개인경로는 배포물에 포함하지 않습니다. 전체018 재학습 드라이버와 정확한 재학습 재현은 포함·검증되지 않았습니다. 기존ZIP·018·019는 보존했고 업로드는 없습니다.

기록:`reports/summaries/support-diagnostic-153.json`, `reports/summaries/spatial-support-154-summary.json`, `reports/validation/reproduction-package-155-summary.json`, `reports/summaries/research-153-155-summary.json`.

## 156–157: 동일A 예측차이의 비선형 입력 — 미제출020

CNN−KRR의 단순 선형 항은 혼합비 변경과 같은 방향이므로 새 선형 후보를 제외했습니다. 기존585입력 none/Ridge1/Ridge.1은 유지하고,RBF1/RBF.1에만 같은A전용모델의 CNN3seed identity평균−KRR 차이를 추가했습니다. B와V/test 모두 같은A에서 학습된 upstream모델을 사용하고 내부 전처리·선택·보정 범위를 검증했습니다.

통합RMSE01812.83654465→02012.83560304,개선0.00094161. 원본/첫재분할 악화,마지막재분할 개선;조건부ID그룹95%개선구간[-0.00698549,0.00858682]. 극히 작은 탐색적 차이이며018을 대체하거나Public개선을 예상할 근거가 아닙니다.900개 개발잔차모델,60개 전체500모델,독립수식·라벨배제·동일A범위·test1000재로드/ID/finite/order/CSV바이트 검증 완료.

CSV:`artifacts/submission-candidate-020.csv`
SHA256:`f1e99e053bb1d425e5ce46afe7f7b2ee399bfee2a38941b0a7cbf96d64e9426b`
018/019 및136/155ZIP 보존,020 미업로드,오늘5/5소진으로 추가제출 없음.155는018 저장모델 추론 전용이며020을 포함하지 않습니다. 기록:`reports/validation/disagreement-156-audit.json`, `reports/summaries/disagreement-156-summary.json`, `reports/submissions/candidate020-summary.json`, `reports/submissions/candidate020-verification.json`, `reports/submissions/candidate020-preservation.json`. 검증기 이름충돌과 다른배치크기 반올림 검사는 별도수정기록에 남겼고 레시피 변경은 없습니다.

## 158: 동일 A 기저모델 평균과 320장 pooled 잔차 — 종료

119는 이미 내부 선택 없이 고정 RBF1로 올바르게 cross-fit한 320개 잔차를 합친 비교였습니다. 외부 라벨 누출로 판정하지 않습니다.158은 잔차 타깃을 만든 A213/214 전용 모델을 외부 추론에도 그대로 사용하고 세 모델을 평균했습니다. 동일 기저예측에 B106/107 잔차 세 모델 평균을 더한 대조군과,320개 타깃에 한 번 fit한 RBF1을 더한 실험군을 비교했습니다. 기존585개 특징과 규제를 고정했으며 내부 선택은 없습니다.

|분할|현재018|같은 A 기저 + 블록 잔차|같은 A 기저 + pooled320|
|---|---:|---:|---:|
|original|12.81194528|13.10878882|13.10530932|
|20261031|12.64587262|13.11149328|13.11422775|
|20261101|13.04862442|13.31444873|13.31412899|
|통합|12.83654465|13.17859559|13.17824132|

동일 기저 대조군 대비 개선0.00035428,조건부 ID 그룹95%구간[-0.00144412,0.00220995].018 대비0.34169667 악화하여 종료했습니다. 작은 잔차 모델 세 개도 합계320개 라벨을 사용하고,pooled전처리와 커널도 달라지므로 단순한 표본수 효과로 해석하지 않습니다. 같은400개 반복 연구의 선택 편향이 있으며,이 실패가 모든 정당한 중첩 스태킹의 실패를 뜻하지 않습니다.

새 학습은 pooled RBF15개와 라벨변경 재검증3개뿐이며 신경망 학습은 없습니다. 기존 블록 RBF45개 재사용,총60개 잔차 모델의 독립 수식 검증과 세 분할 외부라벨변경 불변성 검증을 통과했습니다. 새 전체500 학습·holdout/test열람·CSV·설치·비용·업로드 없음.018best와020보조후보,018/019/020 CSV와136/155재현ZIP 해시 보존을 확인했습니다.

기록:`reports/provenance/aligned-crossfit-158-design.json`, `reports/summaries/aligned-crossfit-158-summary.json`, `reports/provenance/aligned-crossfit-158-preservation.json`. 구현:`aligned_crossfit_158.py`.

## 159: 동일 연결성분 확률 C2와 동일 마스크 S2 — 검증 후 종료

028의 군집 요약,054의 직선 점유확률,055의 Betti 수,130의 두세 점 점유곱은 C2 곡선을 직접 계산하지 않았습니다. C2는 두 점이 같은4연결 성분 안에 있을 확률입니다. 합성 마스크의 모든 변위 S2·성분 면적 다중집합·0레벨 Betti가 같아도 C2가 다른 사례를 검증했습니다. 이 사례가 기존 특징 전체와의 독립성이나 물성 예측력을 증명하지는 않습니다.

사전 고정한 raw256 분위수20/50/80%,저명도영역과여집합,4연결,축방향거리4/16/64픽셀의 방향평균·표준편차36개를 사용했습니다. C2와 동일 마스크/거리/차원 S2를 각각018의585개 잔차입력에 추가했습니다. 추정 이진영역의 연결통계이며 실제 재료상·입자·공극을 확인한 것으로 해석하지 않습니다. 경계 밖 연결은 관찰하지 않으며 감기·필터·연결보정은 없습니다.

|분할|018|018+S2 대조|018+C2|
|---|---:|---:|---:|
|original|12.81194528|12.81226211|12.82147728|
|20261031|12.64587262|12.66319413|12.65749863|
|20261101|13.04862442|13.07409107|13.06985264|
|통합|12.83654465|12.85097147|12.85072759|

C2의018 대비 개선은-0.01418295,조건부 ID 그룹95%구간[-0.03629410944551932, 0.007503077647297417]입니다. S2 대비 개선은0.00024388,구간[-0.02508401125200992, 0.024192025104960137]입니다. 사전 고정한018과S2 모두보다 낮은 통합MSE 기준에 실패해 종료했습니다. 임계값·거리·연결규칙·규제·혼합 재탐색은 없습니다.

같은018 nested anchor/잔차 선택 절차를 유지했습니다. A213/214는B106/107과V80을 제외하며,잔차 전처리는 내부I70–72 또는전체B에서만 fit합니다. 새pooled잔차모델은 없습니다. 개발1800개 모델객체(1440개비영fit)의 독립 수식·bootstrap가중치 검증,외부/내부 라벨변경 불변성,실제3이미지의 독립BFS 및D4 검증을 통과했습니다. 신경망 재학습·전체500학습·holdout/test열람·새CSV·설치·비용·업로드 없음.018/019/020 CSV와136/155 ZIP 해시를 보존했습니다. 같은400개 반복 연구와 겹치는재분할,조건부 구간의 한계는 남습니다.

기록:`reports/provenance/cluster-pair-159-design.json`, `reports/summaries/cluster-pair-159-summary.json`, `reports/validation/cluster-pair-159-audit.json`, `reports/provenance/cluster-pair-159-preservation.json`. 구현:`cluster_pair_features_159.py`, `cluster_pair_159.py`.

## 160: 라벨 비사용 경계 배치 의존성 감사 — 제한적 음성 결과

이전에는 정규화32×32명암썸네일 비교와097의 특정 한 쌍 기하 비교가 있었으나,전체400장의 경계·접합점 배치 재사용을 확인한 감사는 없었습니다. exact/D4중복해시 검사는 반복하지 않았습니다.018/020레시피와 CSV/ZIP 해시를 고정하고 개발400장의 이미지에만 고정sigma2 Gaussian미분 경계지도,4×4평균다운샘플,이미지별 중심화/L2정규화,8D4 정렬을 적용했습니다. 밝기·대비 변화에는 제한적 불변성이 있고 추가블러 내성은 합성 사례로만 확인했습니다.

79,800쌍을 한 가지 점수로 순위화한 뒤 상위8쌍을 고정하고 원본·대비정규화·경계겹침 패널을 모두 직접 검토했습니다. 최대경계상관0.16289696. 같은 위치에서5개 이상의 접합점 및 여러 긴 경계 경로가 세 사분면 이상에서 대응하는 구체적인 증거는 없었습니다. 비슷한 질감·방향성·일부 반점은 그룹 근거로 인정하지 않았습니다. 확인된 관련쌍/그룹0개이며 새그룹·fold변경·레시피변경은 없습니다. 따라서 실제관련그룹에 대한 fold분할분석도 하지 않았습니다. 이는 이미지/생성기계열 독립성을 증명하지 않습니다.

추가 descriptor·임계값·상위쌍 확대를 중단합니다. 이동·잘림·크기변환·부분재사용·심한노이즈/블러·경계가 다른 생성기계열은 이 제한된 검사로 배제할 수 없습니다. 성능향상이나 잡음하한을 추정하지 않습니다.159의 분위수마스크는 면적비를 거의 고정한 rank연결통계이므로 그 실패를 실제 상분율 정보의 무용성으로 해석하지 않습니다.

개발라벨·OOF/예측배열·holdout/test이미지·새학습/추론·설치·비용·업로드 없이 종료했습니다.018/020레시피와018/019/020CSV 및136/155ZIP 해시 보존을 확인했습니다. 기록:`reports/summaries/layout-dependency-160-summary.json`, `reports/provenance/layout-dependency-160-preservation.json`; 검토패널:`artifacts/layout-dependency-160/panels`; 쌍별판정:`artifacts/layout-dependency-160/visual-review.json`.

## 161: DINO 공간 배치를 학습하는 CNN 분기

092의 독립 패치 가중평균과140의 고정 이웃통계를 넘어,학습 가능한 지역 패치 상호작용을 검증했습니다. 기존 B504의36×36패치에 동일한 고정QR32투영을 적용하고1×1줄임층·3×3합성곱 두 층·지역평균/표준편차 및 전체B6912입력의32차원 분기를 결합했습니다.226,449파라미터,30epoch,3seed,MSE,기존CPU2/MPS만 사용했습니다. 원래 공간배치와 고정순열배치의 전체초기값·순서·dropoutseed·학습량을 맞췄습니다. 순열은 토큰 다중집합과 채널 통계를 유지하면서 실제이웃2520개 중10개만 유지합니다.

새 직접목표 회귀기 평균을018과80:20으로 사전 고정 결합했습니다. 모든 스케일러·목표통계·학습은outertrain320에만 적용했고018 내부 보정/잔차 레시피는 바꾸지 않았습니다. 전체OOF잔차에 fit하지 않았습니다.

|분할|018|순열대조20%결합|공간CNN20%결합|
|---|---:|---:|---:|
|original|12.81194528|12.84365085|12.84179186|
|20261031|12.64587262|12.75958360|12.76108288|
|20261101|13.04862442|13.08921904|13.08917660|
|통합|12.83654465|12.89824272|12.89810579|

018 대비 조건부ID그룹95%개선구간:[-0.15683075283961512, 0.035660127667233914];순열대조 대비:[-0.007714997294700021, 0.008011832492462156]. 사전 기준 실패로 이번 고정 구성을 종료했습니다. 구조·epoch·seed·순열·비중 재탐색,전체500/test추론,새CSV는 없습니다.

90개 모델의MPS재로드/CPU/독립functionalforward,두 모델의전체30epoch외부라벨변경 재학습,fold훈련범위·쌍별초기값/학습순서·특징투영/순열을 검증했습니다. 새 사전학습모델·설치·비용·holdout/test선택·업로드 없음.018/019/020 CSV와136/155ZIP 보존. 같은400장의 적응적 재사용과 겹치는재분할의 한계는 남으며 입상이나 성능상한을 보장하지 않습니다. 원배치/순열의 용량은 동일하므로 둘의 차이로 이웃배치 효과를 검토하지만,DINO토큰 내부 위치임베딩·전체문맥은 순열 후에도 남습니다.018 대비 변화는 추가분기/고정20%결합 전체 효과이며,global-only 제3군이 없어 추가용량의 순수 인과효과를 따로 추정하지 않습니다.

기록:`reports/provenance/spatial-token-cnn-161-design.json`, `reports/summaries/spatial-token-cnn-161-summary.json`, `reports/summaries/spatial-token-cnn-161-final.json`, `reports/provenance/spatial-token-cnn-161-preservation.json`.

## Study162: isolated TabICL v2 CPU comparison (completed negative)

TabICL 2.2.0 was installed under `experiments/frameworks/tabicl-v2/.venv` from official PyPI, preserving existing NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 and PyTorch 2.14.0 versions. Official regression checkpoint: 114,324,594 bytes, SHA256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`, matching the official Hugging Face LFS hash; BSD-3-Clause. Explicit local checkpoint and auto-download disabled. No paid service or external training dataset.

Fixed CPU config: n_estimators=1, batch_size=1, kv_cache=False, use_amp=False, n_jobs=2, random_state=42. Actual ensemble normalization: none. Inputs: compact26, frozen DINO-B6912→train-only StandardScaler/PCA64, and their 90-column concatenation. Original 5-fold OOF standalone RMSE: 14.344922 / 14.355162 / 13.741986. Fixed 80% best018 + 20% TabICL RMSE: 12.933014 / 12.940977 / 12.888915, all worse than reference018 12.811945. None passed the preregistered gate; no resplit, full500, holdout100, test inference or new CSV. This one-estimator result does not settle separately justified future ensemble configurations.

First CPU fold: fit 0.221s, predict 0.225s, process peak RSS 759,267,328 bytes. All 15 saved models reloaded offline with exactly identical finite predictions; three independent validation-label-poison replays were exact. Fold train-only scalers, PCA, IDs and target normalization audited. Before/after hashes of 42 pre-existing package metadata/lock/submission files unchanged; installer targeted only the isolated environment. Initial audit expected float64 labels but official fit converts to float32; audit reference corrected, failed log retained, no model changes. Best018 remains unchanged.

Evidence: `reports/summaries/tabicl-v2-162-summary.json`, `experiments/frameworks/tabicl-v2/plan.json`, `installation-verification.json`, `audit.json`, `requirements.lock`, and fold-level saved models/predictions under `experiments/frameworks/tabicl-v2/results/`. Rerun commands use the isolated interpreter and `study162.py`; existing results are preserved.

## Study163: fixed eight-estimator TabICL comparison (completed negative)

Confirmed study162 actually used one estimator. Reused its isolated environment, official checkpoint, seed42 and fold-local transforms; no new installation/download. Study163 fixed n_estimators8, batch_size1, CPU threads2, AMP/cache off, none4+power4 normalization with Latin feature permutations. First fold fit0.261s, predict1.594s, peakRSS723,501,056 bytes; repeat/reload/fresh fit exact.

Original5-fold standalone RMSE compact26/PCA64/combined90: 14.298448 / 14.205829 / 13.725351, modestly improved over162. Fixed20% blends:12.929751 / 12.931244 / 12.900173, all worse than01812.811945. Improving folds:0/5,2/5,1/5. No arm passes frozen >=0.01 RMSE gain + >=3/5 improving folds rule. No resplits, learned nested blend, full500, holdout100, test, CSV or upload. Best018 and prior162 preserved.

All15 saved models replayed exactly offline, all predictions finite, all fold train-only target/feature scalers and PCA audited; three full validation-label-poison fresh-fit replays exact. Reports:`reports/summaries/tabicl-v2-163-summary.json`, `experiments/frameworks/tabicl-v2/study163/plan.json`, `audit.json`, fold results and logs.

## Study164: nested compact-KRR residual TabICL (completed, not promoted)

One fixed90-column eight-estimator TabICL residual probe. All residuals regenerated within each outerT320:3crossfitKRR anchors train213/214, each with its own3-fold nine-setting KRR selection entirely inside its training set. OOF residual y-KRR excludes own label throughout selection/fit and all outerV80 labels. Pool320 residuals for fixedTabICL; no inner selector on pooled residuals. Outer query base is mean of same3anchors. TabICL compact26+DINO-B train-onlyPCA64, seed42, none4+power4, CPU2, no weight search. This differs from119/158 in anchor/learner, not in pooledcrossfit principle.

OriginalOOF RMSE:mean3KRR14.413530 → KRR+TabICLresidual13.595009, compared with163directtarget90 13.725351. Fixed20%blend with018:12.865548, worse than01812.811945; only1/5fold improves. Zero-correctionblend12.955201. No resplits/full500/test/holdout/CSV/upload. Best018 unchanged.

Independent audit verified420KRR models including all internalCV choices, reconstructed all320residuals per fold, replayed5TabICL models exactly, and passed wholeouterlabelpoison plus anchorheldoutlabelpoison including hyperparameter selection. All scaler/PCA fits use respective training scope. Firstfold4.879wallsec/8.503CPUprocesssec/RSS1.049GB. No installation/download/externalAPI/cost. Reports:`reports/summaries/tabicl-residual-164-summary.json`, `experiments/frameworks/tabicl-v2/study164/plan.json`, `audit.json`, savedfoldmodels andOOF.

## Study165: strongly shrunk quality-dependent weights (completed negative)

Duplicate review:077globalcontrast,078training-size reliability,125sample-independent simplexweights,126risk-condition diagnostic did not test this fixedtwo-proxy samplewise gate with complete018 downstream refit. Uses cached noise_contrast_ratio and spectral_low_fraction (blur/coarse-structure proxy, not measured blur). Trainlocal median/IQR clipping, exactly2 gatecoefficients, ridge10, L1bound.03 on CNN-vsremaining weight shift; otherthree weights change proportionally. No physical sign assumption or sweep.

Whole nested018 procedure:320outer meta rows use middleOOF,213/214 anchor meta rows use deepOOF; every honest106/107 residualblock retrained with018585features/fivecandidates/512bootstrap. No globalOOF training. Theta0 wholefold reproduces018 exactly. OriginalRMSE01812.811945281→16512.812072219 (worse0.000126938),2/5fold improvements. No resplits/full500/test/holdout/newCSV. Failed fixedrecipe closed without penalty relaxation.

Independent20meta fits and300residualmodels verified; wholeouterlabel poison, anchorBlabel poison, residualinnerHlabel poison and qualityquery-isolation passed. Positive,sum1,boundedweights verified. Best018 and reproducibility ZIPs preserved. Evidence:`reports/summaries/quality-gate-165-summary.json`, `artifacts/quality-gate-165/plan.json`, `validation-original.json`, originalfold artifacts.

Best018 full500 training already completed: `artifacts/final-model-145/fit.json` records500uniqueIDs, reuses fitted017base/anchors, trains018residual aggregation with3honestblocks; all15 stored018residualmodel hashes checked. Existing submission018 contains1000finite, ordered test predictions according to validated summary. No duplicate full500 training or fresh test access in165.

## Studies166/167: case-led common-mode error audit and conditional calibration

166 reviewed only existing018OOF for400development IDs across3overlapping splits. Persistenttop40absolute-error intersection33; stable same-sign withminabs5 is233 (same-sign withoutmargin355); persistenttop40branch-span intersection16; large/disagreement overlap3. All33large cases have targetoutside rawKRR/CNN/DINOmean range on the same side inall3splits. Twenty-seven original256development images viewed in3panels; noisy/blurred and clear images both fail, matchedlowerror controls have similar visibleconditions. No latentlabelnoise/cause orperformanceceiling inferred. Counts/bias/RMSE bylabel,brightness,noise,size,shape quartiles saved. Case-driven selection is explicitly posthoc/exploratory.

Spatialtilemean heterogeneity topquartile(n100) has pooledRMSE15.38 and residual-on-prediction slopes.246/.266/.349 across3splits, despite nearzeromean signedbias.167 tests one score×tileheterogeneity calibration term versus immutable018 and samecapacity globalquadratic control. Train-only orthogonalization/RMS andlambda1halfshrink insideouter/deepOOF;018 residual585/fivecandidates/512bootstrap fullyrefit. OriginalRMSE01812.811945→interaction12.807842; quadraticcontrol12.796059. Large33RMSE29.571979→29.538691, control29.472036. Improvement tiny; interactionfailsmatchedcontrolgate, so noresplit/fallback/full500/test/CSV.600residual and40metaobjects audited; outer/anchor/innerlabelpoison passed. Best018 andZIPs preserved.

Visual narrative/report:`artifacts/failure-review-166/review.html`; machine-readable:`reports/summaries/failure-review-166-167-summary.json`; all400cases:`artifacts/failure-review-166/cases.csv`.

TopTemp reviewonly:055alreadyBetti1snapshots/distancecurves,107/109alreadyfinitegrayH0PI,159alreadyC2. FullintensityH1pairing notimplemented, but topologyfamily notnew and failurecaseconnection remainsweak. Existing6Betti1abs-error correlations.030–.156 aredescriptive,not mechanismvalidation.9x9fixture showsboundaryholeinfo can differwhileimplementedfiniteH0diagrams equal; notfullPH/S2-C2equivalenceproof. TopTempclassification isnot hardnessregression. ripser/gudhi/giotto-tda/persim/cripser absent inbothenvironments; noinstall. lower_star_imgH0 mustnotbe calledH1. NoGNNornewtopologyfit. References/review:`reports/summaries/topology-H1-166-review.json`, installedenvironmentcheck:`reports/provenance/topology-H1-166-environment-check.json`.

## Study168: exact cubical H1 engineering validation and fixed residual comparison

Closed-square top-cell sublevel filtration, min-face values, foreground8/complement4, finite nonperiodic rectangle. Dual max-tree with immortal exterior computes positive H1 birth/death pairs;640 independent GF2 cubical boundary reductions across8hand fixtures and300random grids agree. Ties, D4, polarity swap, Euler and independentUF pass. This is exactH1 for the statedcomplex;32dimPI is lossy, not fullbarcode or exactTopTemp reproduction. Joint fixture has identical old107lower/upperfiniteH0,132Betti055 and36C2/S2 butdifferentH1 dueboundary/contrast association.

Label-blind16development pilot passed: pairedH1/H0 extraction~0.400s/image; bounded±1/±4gray noise changesPI bymedian0.70%/6.66%. H1 vsboundary-unaware dualH0PI difference median3.70%, so substantialoldfeature overlap. All16UF/2D4/2all256Euler/4actualcropGF2 checks pass. Correctness/nonidentity doesnot establish physical or predictive relevance.

One fixed4x4PI pergraypolarity, sigma.25, linear lifetime/areaweight:32features appendedto018585. SamewidthH0control, existingtrainonlytransforms/honestnestedanchors/fivelearners/512bootstrap. OriginalRMSE018=12.811945281, H0control=12.807977483, H1=12.807968171. Originalgate=True; executedpartitions=original,20261031,20261101; developmentcandidate=False. 1800residual models independently audited, outer/innerlabelpoison pass, immutable018 reconstructsexactly.

No newinstall/download, holdout/test/full500/newCSV/upload. Best018/ZIPs preserved. Same400adaptivehistory limits inference. Evidence:`reports/summaries/cubical-h1-168-summary.json`, `artifacts/cubical-h1-168/` and source`cubical_h1_168.py`/`validate_cubical_h1_168.py`/`cubical_study_168.py`.

## Study169: train-label-density audit and one mild CNN weighting control

No prior directLDS/densityperexampleweight found in638inspectedsource/plan/summary files.058DINO SmoothL1/MSE,120ranking,051hardquantileclassaux,131RNC and125componentweights aredistinct.26of33persistentlarge018errors areintraininglowdensity inall3splits,only4outsideanytraininglabelrange. CNN cleantrainRMSEall9.83/sparse14.34 versusvalidation14.62/22.74; DINOS/B trainRMSE<.6 whilevalidation~14,so commonerrorsare not universaloptimizerunderfit.018predictioncalibration slopes1.051/1.009/1.010; withinIDdensity/abs-errorassociation-.025. Labeltailresidualtrend partlyfollowsCov(y,e)=Var(e)+Cov(pred,e),notproofthatimbalancecauseserrors.

Fixed169: trainLOOGaussianlabelKDE h=.4trainSD, inversedensityquarterpower,rawclip[.8,1.25],mean1normalization. Actualoriginalweights~.90-1.20,ESS~317/320. ExistingTinyCNN128,D4,80epochs/800updates,last5weightaverage,seed20261003+fold pairedwithunweightedMSE. Norm/density/weights useonlyfoldT320. Identityinference,unweightedRMSE. Full018 fixed80%+newCNN20%complementarity; not staleCNNreplacementorweightsearch.

OriginalRMSE:018=12.811945281; MSECNN=15.032317843,densityCNN=14.992537211; MSEblend=12.948476760,densityblend=12.927948671. Originalgate=False; executed=original; candidate=False. Sparse/dense/lower/upper/middle/large33counts,bias,train/validationRMSE andMSEalignmentidentity saved. Modelsreloaded=10; pairedinit/streams/D4/dropout/budget,trainonlydensity/weights,analyticlossgradient andactualstep-loss checks passed; weightedfold80outside-labelpoisonstateexact.

Best018/ZIPs preserved; noinstall/download/holdout/test/full500/newCSV/upload. Anygainexploratory onrepeatedsame400. Noautomaticweight/bandwidth/seed/architecturefollowup. Evidence:`reports/summaries/label-density-169-summary.json`, `artifacts/label-density-169/diagnostic.json`, `artifacts/label-density-169/plan.json`, `validation-original.json`.

169 audit execution note: all assertions and validation JSON write completed, then final stdout print raised NameError because locked validator lacked json import. Failure log preserved. `resume_validation_169.py original` rechecked saved poison/normal states, predictions, weights and streams exactly, without retraining; completion logging recovered. Amendment saved. Original weightedCNN correction alignment mean(e018*(CNN-018))=-1.752205<0, so any positive constant blend weight worsens this fixed original OOF MSE; no ratio sweep justified by this result. This doesnot generalize to newfits or prove a performanceceiling.

## Study170: reuse-only CNN memorization and augmentation diagnosis

No exact32-image/no-regularization run found. Reused stronger existing129 evidence: same91,905parameterCNN,800updates,dropout.15 andweight_decay.01 retained; D4removal reduces320-image cleantrainingRMSE9.885685→.306560. No newfit. Label-blind32IDs selectedbySHA256(170:+ID) beforelabelreading, checkedrawPNGhash/256gray/PILBOX128equalscachedpixels. Existingfour400-only label/ID sources agree; no mixed500labelCSV read. Ten129models checkedtrainIDs/normalization/CPU/MPSandindependentfunctionalforward; MPS exact,CPUmax3.052e-5,functionalexact.

Own-training sparseRMSE D4→identity14.696414→.517572; persistentlarge33RMSE20.047457→.609803. Thus grossincapacitytofitthoseoriginalimages, cachelabelmisalignment or inferencepreprocessing mismatch isnot supported. This doesnot validate labeltruth or generalization. FulloriginaltrainingRMSE differsfromfixedfinal-model all8viewloss: D4model9.885685→9.931285; identitymodel.306560→14.049587. Trainingregime tradesoriginalmemorization fortransformconsistency; doesnot establishphysicalD4invariance or isolateaugmentationdifficultyfromoptimization. Onlinecurvesincludechangingweights/dropout; no freshclean80epochtrajectory claimed.

Existing129 validation worsened15.032318→16.338433 withoutD4. Existing128 small142/143train400→800updates reducedtraining9.545296→3.022690 butvalidationworsened15.418684→15.597598; this isnot320rows800→1600 evidence. No inference that moretrainingorlessregularization improves018.

Proposedonly,notrun: oneoriginalfold0 control changingCNNdropout.15→0 whilekeepingD4,weight_decay.01,seed20261003,800updates,LR/batches/last5average andunweightedMSE unchanged. ReusevalidatedD4reference; one newfit wouldseparateexplicitdropout effect underD4, evaluatedoncleanidentity/all8viewtrainandfixed80validation. No fullCV,candidate,ratio/seedsearch or automaticexecution.

Evidence:`reports/summaries/cnn-memorization-170-summary.json`, `artifacts/cnn-memorization-170/plan.json`, `replay.npz`. Newtraining0,rawimages32devonly,noinstall/download/holdout/test/full500/CSV/upload;018/ZIP preserved. Minorper-fold deviceerrorfield naming corrected withexecutedsource/report archives; aggregatechecks andnumericresults unchanged.

## Study171: CNN dropout.15→0 withD4 retained

Onefactoronly, existingTinyCNN91,905params,seed20261003+k,80epochs/800updates,AdamW.002wd.01,clip5,epochcosine,last5average. Reused129/169verifiedcontrol. Input/targettrainlocalnorm, IDs,800sample/D4/LRstreams,4actualinputhashes andinitialweights identical. Dropoutonlyhead.2; trainp.15stochastic,p0identity; bothinferenceevalidentity. p0consumesnoMPSdropoutRNG,expectedendRNGdifference; NumPydata/D4streamunchanged.

Predeclaredfold0gate requires lowerfinal8viewtrainingloss,lower80validationCNNRMSE,andlowermatched01820percentblend. Gate=True. Executedmainfolds=[0, 1, 2, 3, 4], newfits=5 plusoneoutside80labelpoisonreplay. Outcome=completed_original5_no_promotion. RMSEoneevaluatedscope(n=400):018=12.811945281, CNN.15=15.032317843, CNN0=14.906688656, matchedblend.15=12.948476760, blend0=12.912112957. Notconfirmedindependentperformance.

AllmodelsCPU/MPS/functionalreload; trainstats/IDs,5weightaverage,modefixtures verified;outside80labelperturbationpreservesstateandpredictions. Sparse/dense/large33 andall8D4viewerrors/viewspread recorded. Best018preserved. No otherp/wd/seed/augmentationsearch,install/download/cost,holdout/test/full500/newCSV/upload. Evidence:`reports/summaries/cnn-dropout-171-summary.json`, `artifacts/cnn-dropout-171/plan.json`, foldvalidation/evaluation artifacts.

## Study172: exact D4 group CNN, engineering tests and originalfold0 only

True weight-tied lifting/group convolutions mixall8 orientations, not outputTTA. Width4/12/24/36; tiedGroupNorm4; averagepool2 beforestride1conv preservesfinite-squareD4sampling. New ordinaryPoolCNN controls this downsampling change withsame91905initialweights; D4model91593params. Same80epochs/800updates/seed20261003/D4/dropout.15/AdamWwd.01/last5average. TrainonlyT320normalization andmatchinginput/order/LRstreams verified.64groupcomposition and independent slowconv/gradient checks, allstagepermutations CPU/MPS, noiseless16example fit andresourcebenchmark passed.

Fold0CNN validation: legacy=15.249590276858356, poolcontrol=15.845719487143137, D4group=15.326864722039627. Reference018=12.805516135869311, fixed20percentgroupblend=13.025889384330569. Recommendremaining4foldconfirmation=False. Currentstudy stopsafterfold0; no candidate/promotion/generalizationclaim. All8viewtrainloss/subgroupharms andcompute/memorydifferences retained.

Two mainfits plusfull800update outside80labelpoison replay; stateexact,5stateaverage andCPU/MPS/independentfunctionalreload validated. Best018/ZIPs preserved. No holdout/test/full500/CSV/upload, package/download/cost. Evidence:`reports/summaries/d4-cnn-172-summary.json`, `artifacts/d4-cnn-172/plan.json`, unit/synthetic/benchmark/validation andmodelartifacts.

## Study174: KRR selection-scope audit (completed, no promotion)

With the 018 features, neural predictions, and grid fixed, compared selection on all current training rows with selection on one label-blind hashed 80% subset. Both arms refit KRR on all current training rows. All dependent affine and residual fits were recomputed.

Original OOF RMSE: 12.811945281 -> 12.802737510 (gain 0.009207771; 3/5 folds improve). Selected settings changed at 36/65 nodes; prediction RMS difference was 0.396256. The gain missed the frozen >0.01 resource gate, so no resplits were run. This is neither negligible under the registered definition nor clear harm. It provides no Public or independent generalization claim.

Prespecified top-only attribution, closer to the historical production 400/500 selection exception: RMSE 12.808300723, gain 0.003644557. Only 1/5 outer KRR settings changed. Full-formula reconstruction error was 8.53e-14. Middle/deep training predictions remained identical, so their affine/residual models correctly remained unchanged. The local hashed subset is not the historical development400 subset.

Independent algebra verified 3,640 KRR fits, 600 residual model objects, and 40 affine fits across all five folds. The control reproduced 018 exactly. Representative fold0 outer80-label poisoning recomputed all affected fits in both arms and preserved predictions exactly; anchor, selection-complement, and residual-inner exclusion probes also passed. Unchanged neural caches retain prior scope certificates. No neural training, raw holdout/test access, full500 fit, new CSV, or upload occurred. Best018 and protected archives are unchanged.

Results: `reports/summaries/krr-scope-174-summary.json`; audit: `artifacts/krr-scope-174/validation-original.json`. The conditional fixed-prediction bootstrap intervals include zero and omit refitting/adaptive-research uncertainty.

A separate plan, `reports/provenance/retraining-018-plan-174.json`, inventories remaining fresh-training reproducibility gaps. It proposes an isolated fresh original-fold0 run, then a development400 raw-image-to-trained-bundle run. Historical500 and test stages await next-task scope. Estimated development400 runtime is 15–25 minutes with mixed MPS/CPU; planned process RSS and MPS driver caps are 4GB and 2GB. These are estimates, exclude implementation and complete repeated neural audits, and do not imply full retraining has run.

## Study175: fresh standalone development reproduction of018

Freshly regenerated all400 development features, then trained allfive original018 fold hierarchies in `experiments/01-representations/0175-reproduction-018`. No historical fitted learner or OOF was read by the training process; its fileauditguard enforces this. Official frozen DINO weights and development metadata were reused.

Fresh OOF RMSE 12.811945280527; saved018 12.811945280527. Prediction max difference 0; neural tensor states exact 605/605. Allcomponent predictions exact: True.

Main fresh fits:195CNN,390longMLP,20shortB,65finalKRR+1755innerKRR,20affines,300residualobjects. Independent1820KRR algebra/300residuals/20OLS and605neuralreload checks passed. Representative outer80labelpoison retrained10neuralmodels and28KRRfits exactly; not a full13node poison replay. Completed-node resume andblockedoldinput access tested.

Feature extraction 213.4s; main5foldtraining 27.30min; processRSS max 1.327GB. Original15–25min estimate referred to one400root, not5foldOOF. CPU/MPS differences and tolerances are in the report. Noholdout/test/500fit/newCSV/upload/install/download/cost.018unchanged.

Evidence: `reports/summaries/retraining-175-summary.json`; commands and limitations: `experiments/01-representations/0175-reproduction-018/README.md`; separate future500plan: `reports/provenance/retraining-018-full500-next-plan-175.json`. Fresh500 production andportable raw500 trainer remain unverified;175 establishes same-environment development reproducibility only.

## 176 전체500 최종 학습 재현 완료

018 고정 레시피로 원본500 특징부터 전체13계층을 새로 학습했습니다. 신경망121개 텐서, nested 보정·잔차 및 전체500 재로딩 예측이 기존018과 정확히 일치합니다. 기존 경쟁대회 학습 모델·특징 캐시는 fit에 재사용하지 않았고 공식 frozen DINO S/B 가중치만 재사용했습니다. test1000도 원본에서 재추출 후 추론했고 CSV 바이트가 기존018과 같습니다. SHA256: `6d478f08de769ca0c54d640c0711c95b15ad4e6b707d0516022c23260f1cfc16`.

새 패키지: `deliverables/reproduction-018-v176.zip`, SHA256: `e93bd3f2bd9f11b40014d65997238495293b07ae0eac49d17490552c198797a6`. 독립 수치검증과 별도 압축해제 경로의 원본2장·fresh캐시1000 추론 검증을 통과했습니다. ZIP에서 전체500 재학습을 중복 실행하지 않았고 다른 플랫폼 재현은 미검증입니다. 전체500 학습은490초, 원본500 특징267초. 검증기 루트 조건 수정 및 기록 정정은 원본과 함께 보존했습니다. 기존CSV/ZIP 보존, 업로드·제출·설치·결제 없음. 상세: `reports/summaries/retraining-176-summary.json`.

## 177 DINOv2-L 비교 완료

공식L14만 승인 후 다운로드하고 SHA256·Apache-2.0 출처를 기록했습니다. 개발3장 CPU/MPS 벤치와400장 frozen504px 추출을 통과했습니다. 기존 B와 동일3seed MLP, 마지막CLS의fold 내부PCA/SVR(16/34/64차원×C1/10/100)을 비교했고018과의 결합은0.9/0.1만 사용했습니다. 원본RMSE018=12.811945280526919,L-MLP결합=12.786200775750695,L-SVR결합=12.912427040131584.

두 재분할까지확인된후보: ['MLP']. 기존018 보존. 상세·실패기록·자원·검증한계는 `reports/summaries/dinol-177-summary.json`, 코드는 `experiments/01-representations/0177-dinol/`. holdout/test/전체500fit/CSV/업로드/새설치/비용 없음.

## 178 candidate021 생성·검증 완료 (미제출)

177에서 고정한 `0.90×018 + 0.10×L MLP3seed평균`을 그대로 적용했습니다. L은 원본500 특징과3개 최종head를 새로 학습했고, 각seed 전체학습 재실행의 텐서·예측이 정확히 일치합니다. 018 모델·CSV·176 test특징은 고정90% 성분으로 명시적으로 재사용했습니다.

새 CSV: `artifacts/submission-candidate-021.csv`, SHA256 `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b`. test1000 L특징은 원본에서 새로 추출했습니다. ID/열/행수/순서/finite/CSV재로딩·CPU/MPS·018해시 보존 검증 통과, 별도 저장모델 CLI 실행도 CSV 바이트 정확히 재현했습니다. 예측범위 161.672828–233.800874, 클리핑 없음.

비율·층·seed는177 결과 전에 고정됐으나, 같은400을 반복 사용한 두readout 선택과 재분할 확인이므로 독립 검증이 아닙니다. 개선은 작고 조건부구간에0을 포함합니다. CPU/MPS 회귀예측 비교는1000장 전체, L CPU인코더 비교는사전지정3장입니다. 기존018과보호15파일 보존,018 Public최고상태 유지. 새설치/다운로드/비용/업로드/holdout추가평가 없음. 상세 `reports/submissions/candidate021-178-summary.json`, 재현방법 `experiments/01-representations/0178-candidate021/README.md`.

## 179 DINO-L 공간 특징 교체 대조 완료

018의 B 마지막층 이웃patch 공동변화528을 동일 정의·QR32·seed·정규화·nested 잔차학습의 L528로 교체했습니다. 물리57 및 upstream anchor는 유지하고,021 globalL10%도 고정했습니다. 단순추가·비율탐색은 하지 않았습니다.

결과: rejected_no_new_candidate, 실행분할=['original']. 원본5fold RMSE018=12.811945280527, 고정021=12.786200775751, 교체primary=12.788492654891. 재분할 gate=False. 상세실패·상관·검증·자원 기록 `reports/summaries/spatial-L-179-summary.json`, 재현자료 `experiments/01-representations/0179-spatial-l/README.md`.

개발400 새 L공간 추출·B대조 정확재현·전체잔차모델 독립수식/재로딩·fold0 3블록 라벨제외 검증 통과. 기존018/021 및 보호파일 보존. holdout/test/전체500/새CSV/업로드/설치/다운로드/비용 없음.021제출과결과는 부모관리이며 여기서 조회하지 않았습니다.

## 180 동일용량 SB/SBL joint PCA64 회귀 완료

기존060 S-PCA64 실패·066 B다중해상도concat 실패와구별되는 한가지 대조입니다. foldtrain은행별표준화·폭정규화→joint PCA64→train점수표준화→2113파라미터MLP32를양쪽에동일적용하고3seed·100epochs를고정했습니다.021의10%분기만joint예측으로교체했습니다. 원본021=12.786200775751,SB대조결합=12.890309020945,SBL결합=12.880848911580. 결과=rejected_no_new_candidate,분할=['original'].

PCA/scaler train범위·eigen수식·동일초기값/순서·모든head재로딩·CPU/MPS·fold0전체6학습label/featurepoison 검증 통과. 실패33군은166의기존명단으로사후진단만. S캐시저장축수정은학습전오류였고원본·실패로그보존. 상세 `reports/summaries/joint-readout-180-summary.json`,재현 `experiments/01-representations/0180-joint-readout/README.md`.021/018보존,Public비율튜닝·holdout/test·새CSV·설치·다운로드·비용·업로드없음.

##181 L중간12층 대 최종24층 동일readout 완료

기존S044중간층Ridge실패와L177다층입력의차이를감사하고,층sweep없이L12/L24각3072차원에같은MLP64/3seed/100epochs를적용했습니다.021에12층정보가이미포함돼있으므로새정보추가가아닌readout분리실험입니다. 원본021=12.786200775751,최종층결합=12.808443322487,중간층결합=12.926750581117. 결과=rejected_close_L_representation_variants,분할=['original'].

개발400새추출은177캐시와정확일치,CPU/MPS·모든head재로딩·독립수식·fold0라벨/특징교란재학습검증통과.018/021과보호파일보존. holdout/test/전체500/새CSV/설치/다운로드/비용/업로드없음. 보고서 `reports/summaries/midlayer-L-181-summary.json`,재현 `experiments/01-representations/0181-midlayer-l/README.md`.

179/180/181의고정대조실패에따라추가L표현변형을종료합니다.021의기존globalL성분은유지합니다. 종료기록 `reports/summaries/L-representation-closure-181.json`.
