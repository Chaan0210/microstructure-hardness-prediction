> 기존 연구 흐름과 결론을 보존한 문서다. 아래의 실험 ID에 대응하는 현재 폴더는 [실험 분류](README.md)에서 찾는다.

# experiments/

study 177 이후의 실험은 폴더 단위로 진행했다. 폴더마다 코드, 계획(`plan*.json`), 로그, 결과 요약이 있고, 대부분 `REPORT.md`·`NOTES.md`·`HANDOFF.md` 중 하나에 결론을 적어 두었다. 문서가 없는 폴더는 `reports/<폴더명>-summary.json`에 결론이 있다.

- ★는 최종 제출 #62의 계보에 들어간 단계다.
- 모델 가중치, 특징 캐시(`.npz`), 합성 열화 이미지 같은 대용량 산출물은 저장소에 없다. 최종 #62에 필요한 자산은 `final/package/project/`에 따로 보존했다.
- 스크립트는 `ROOT = HERE.parents[1]`처럼 저장소 루트를 기준으로 경로를 잡는다. 초기 루트 모듈(현재 `studies/`)을 import하는 일부 실험은 `PYTHONPATH=studies`가 필요하다.

## 1. 이미지 모델 탐색과 정체 (study 177–219, 10/06–10/07)

best021(Public 11.488)을 기준으로 사전학습 CNN, DINOv2 변형, 학습 정책을 하나씩 바꿔 대조했다. 모두 사전 등록한 채택 기준을 넘지 못했고, 일반 이미지 모델들의 오차가 서로 강하게 상관(0.85–0.89)한다는 진단으로 이어졌다.

| 폴더 | 내용 | 결론 |
|---|---|---|
| `dinol-177` | DINOv2-L 특징 비교 | 021의 L 성분으로 채택 |
| `candidate021-178` | 후보 021 = 018 × 0.9 + DINOv2-L MLP × 0.1 재현 | ★ 채택 (#17) |
| `spatial-L-179` | B 공간 통계를 L로 교체 | 기각 |
| `joint-readout-180` | SB/SBL 고정 용량 결합 readout | 기각 |
| `midlayer-L-181` | L 중간층 대 최종층 | 기각 |
| `wph-182` | wavelet phase-harmonic 공분산 | 기각, 계열 종료 |
| `tessellation-diagnostic-183` | 학습 없는 분할 안정성 진단 | 기각 |
| `pretrained-cnn-185` | ResNet50·EfficientNet-B0 사전학습 CNN | 021 유지 |
| `cnn-optimizer-186` | optimizer·LR 제한 튜닝 | 021 유지 |
| `cnn-scale-floor-187` | 특징 RMS 스케일 하한(수치 안정성) | 안정성 복구, 021 유지 |
| `cnn-resolution-188` | ResNet 입력 해상도 | 소폭 개선, 021 유지 |
| `cnn-unfreeze-189` | ResNet layer3 추가 미세조정 | 계열 종료 |
| `modern-backbones-190` | ConvNeXt-Tiny·Swin-V2-T | 021 유지 |
| `spatial-moments-191` | mean+std·PCA128 pooling | 021 유지 |
| `pca-ridge-192` | PCA + Ridge readout | 기각 |
| `auxiliary-cnn-193` | 경도 구간 보조 CE loss | 기각 |
| `matched-replacement-194` | 고정 출력 치환 | 기준 미달 |
| `pooling-extrema-195` | 극값 pooling | 기준 미달 |
| `multihead-decoder-196` | 다중 헤드 decoder | 기준 미달 |
| `query-normalization-197` | query 정규화 | 기준 미달 |
| `candidate022-198`, `submission022-diagnostic-199` | 후보 022 제출(#18)과 사후 진단 | 021보다 나쁨 |
| `attention-counterfactual-200`, `auxiliary-repartition-201` | 193 효과의 재분할 재현성 | 분기 종료 |
| `simplification-202` | residual·calibration 제거 단순화 | 021 유지 |
| `cnn-seed-expand-203`, `cnn-nested-204` | CNN seed 3→6 고정·중첩 대조 | 021 유지 |
| `micronet-frozen-205` | 동결 MicroNet 대 ImageNet ResNet50 | 기준 미달 |
| `contrast-augmentation-206` ~ `resolution-repartition-217` | TinyCNN 학습 정책 단일 대조(대비·밝기·감마·블러 증강, checkpoint 평균, EMA, LLRD, Huber, 해상도) | 모두 기준 미달 |
| `grain-validity-218` | 입자 측정 유효성·결측 통제 | 021 유지 |
| `residual-cnn-219` | 잔차 직접 학습 CNN 구현·검증 | 구현 검증 |
| `reproduction-018-175`, `reproduction-018-176`, `reproduction-018-176-isolated` | 후보 018 독립 재현 패키지 | ★ 재현 완료 |
| `tabpfn-v2`, `tabicl-v2`, `tabular-fm`, `tabpfn-v3` | 테이블 파운데이션 모델 환경·벤치마크 | TabPFN 채택 |

## 2. 결정립 측정으로 정체 돌파 (10/07–10/13)

일반 이미지 표현에 없던 정보, 즉 결정립 단위의 크기 불균일성과 2상 분율을 직접 측정하면서 Public이 11.49에서 9.04까지 내려갔다.

| 폴더 | 내용 | 결론 |
|---|---|---|
| `grain-cellpose-220` | Cellpose-SAM 제로샷 결정립 분할 → 특징 36/64개 → KRR·TabPFN | ★ 023·025·027 (#19–#21) |
| `cellpose-panel-221` | 분할 평가 계약 테스트 | 도구 |
| `push-300` | 열화한 train 이미지로 Cellpose 미세조정(잡음 이미지 분할 개선) | ★ 028–030 (#22–#24) |
| `push-400` | 라벨 없이 학습한 화소 U-Net으로 2상·기공·경계 지도 | ★ 031–037 (#25–#30) |
| `push-500` | 블록 이질성 + U-Net darkness z | ★ 038 (#31) |
| `push-600`, `push-700` | 신장 방향 각도 특징, 4차 방향 모멘트 | ★ 039 (#32) |
| `push-800` | 독립 점검, 음영 보정 분율 | ★ 044·045 (#33–#34) |
| `push-900` | 측정 잡음 줄이기, Ridge·logRidge 파트너 결합 | ★ 048–051 (#37–#38) |
| `probabilistic-grains-900` | 확률적 측정, GMM, GrainSet(객체 집합 학습) | ★ 053 (#40) |
| `push-1000`, `push-1100` | 053 이후 정체 원인 진단 | 정보 바닥 확인 |

## 3. 잡음 이미지와 정보 바닥 (10/14–10/18)

남은 오차의 대부분이 잡음이 큰 이미지에서 나왔다. 측정 손실을 줄이는 쪽은 효과가 있었고, 재가중이나 계수 조정은 Public으로 이어지지 않았다.

| 폴더 | 내용 | 결론 |
|---|---|---|
| `push-1200` | 053 감사, 원본 열화 사본으로 Ridge 파트너 증강 | ★ 061 (#47) |
| `push-1300` | 061 감사, 저잡음 행 특징 분리, cumulant·contrast 특징 | ★ 073·076 (#52–#53) |
| `rmse-research-20261015` | RMSE 정체 조사, 잡음 구간 잔차 CNN 보정 | ★ 082 (#56) |
| `076-complement-20261016`, `078-080-update-20261016`, `080-followup-20261016` | 076 보완, 078·080 원인 분석 | 076·082 유지 |
| `push-1400`, `push-1400v2` | 공식 Q&A에 맞춘 규칙 준수 재구축 | 원래 076 파이프라인이 허용 범위로 확인되어 중단 |
| `push-1500` | EfficientNetV2-S 경도 + 측정값 동시 회귀 | 기각 |
| `infomatched-copies-20261016`, `infomatched-pilot-20261016` | 정보량을 맞춘 열화 사본 | 분기 종료 |
| `highnoise-relational-20261016`, `highnoise-ssl-20261016-2312` | 고잡음 이미지 관계 학습·자기지도 학습 | 082 유지 |
| `plateau-break-20261016-2035`, `stepchange-20261016` | 정체 돌파 가설 6개 | 모두 기각 |
| `rescnn-retest-20261017` | 082 잔차 CNN 보정 재검정 | 082 지지 |
| `degradation-pilot-audit-20261017-0030` | 열화 사본 파일럿 감사 | 신호 없음 |
| `infogain-20261017` | 정보 이득 진단(평균 밝기, 1차 모멘트, 스펙트럼) | 기각 |
| `structure-context-20261017-1415` | 원시 화소 전역 문맥 Transformer | 기각 |
| `boundary-connectivity-20261017-1835` | 결정립 경계·연결 관계 학습 | 작은 OOF 개선, 제출 보류 |
| `model-audit-repair-20261017-2335` | 모델 전체 재점검과 수정 3건 | 082 유지 |
| `submission-083-20261017-231722`, `submission-083-20261018-060706` | 후보 083 제출 기록 | 083 Public 8.9179 |
| `082-fusion-20261018-0155` | 082 기반 결합 | 채택 조건 미달 |

## 4. 최종 단계 (10/18–10/19)

| 폴더 | 내용 | 결론 |
|---|---|---|
| `supervised-renewal-20261018-0625` | Cellpose flow·확신도 연속 특징 38개(F38), train 저잡음 250장으로 학습한 source encoder 2개, 마지막 HGB readout | ★ #58–#62, 최종 #62 |

`supervised-renewal-20261018-0625/FINAL-REPORT.txt`와 `EXPERIMENTS.md`에 시도한 가설과 기각 사유가 모두 있다. 최종 파일 `candidate-source-flow38-boostedreadout.csv`는 `final-nonlinear-readouts-20261019-0945/`에 있다.
