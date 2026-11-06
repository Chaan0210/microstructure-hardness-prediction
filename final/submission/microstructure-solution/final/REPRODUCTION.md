# 최종 모델과 실행 경로

최종 제출 파일은 `candidate-source-flow38-boostedreadout.csv`다.

## 고정된 최종 구성

```text
G021 = .9*G018 + .1*mean(DINO-L MLP 3 seeds)
base = .16*G021 + .384*TabH9s2 + .128*Ridge + .128*logRidge
       + .05*GrainSet + .15*GMM + noise-gated residual CNN correction
final = base + eligible*.384*(mean(SourceTab)-ControlTab)
           + eligible*.256*(mean(SourceHGB)-ControlHGB)
```

실제 dtype·연산 순서·실패 fallback은 `raw.py`에 있다. 잡음 12 gray 이상인 정상 분할 행과 나머지 행에 서로 다른 회귀 모델 집합을 적용한다. 분할 실패는 G021로 되돌린다. GrainSet 세 seed의 raw 예측 평균은 float32로 계산한다.

## 재학습 단계

`train.py init`은 새 작업 폴더에 코드, 고정 설정, 공식 사전학습 가중치만 준비한다. 이전 학습 가중치·완료 marker·특징·예측을 복사하지 않는다. `train.py all`은 다음 단계를 순서대로 실행한다. 완료된 단계만 같은 작업 폴더에서 건너뛰며, 중간에 실패한 단계의 부분 산출물을 완료로 간주하지 않는다.

| 단계 | 실제 학습 또는 생성 내용 |
|---|---|
| `g018` | 원본 train의 compact26/physics57, CNN 입력, DINO-S/B/L 특징; G018의 13개 nested base node, calibration, residual heads; DINO-L MLP 3개 |
| `auxiliary` | 원본 train zero-shot Cellpose mask·잡음; a1 800-step schedule의 250-step snapshot; H300 300-step; u1/m1/m2/h1/m3/z1 U-Net |
| `physical` | 재학습 auxiliary 모델로 train 원본 물리 특징·객체 은행·H9s2 특징 생성 |
| `augment` | train 원본에서 deg1~4 열화본과 대응 특징 생성 |
| `heads` | 고정 원본 recipe로 low/noisy Ridge·logRidge, GMM, CNN context용 이전 Ridge·logRidge, GrainSet 3 seeds 학습 |
| `source` | source encoder seed 2026101811/2026101823으로 각각 300회, 원본 train F38, 고정 HGB 3개 학습 |
| `teachers` | 원본 train 500개를 seed 2026101607의 3개 fold로 교차 적합한 CNN teacher 생성 |
| `cnn` | 원본 이미지·6개 context·교차 적합 residual로 seed 2026101501/2026201501로 각각 40 epochs 학습 |
| `finish` | 필요한 새 가중치 존재 확인과 추론용 파일 해시 목록 생성 |

혼합 계수는 고정값을 사용한다. KRR의 alpha와 GrainSet의 학습 종료 epoch는 학습 데이터 안에서 선택한다. Source encoder는 원본 학습 이미지 중 잡음이 낮은 절반을 사용하며, 보조 학습 손실에는 경도 라벨을 사용하지 않는다.

HGB 설정은 `loss=squared_error`, `learning_rate=0.05`, `max_iter=300`, `max_leaf_nodes=7`, `max_depth=3`, `min_samples_leaf=20`, `l2_regularization=10`, `max_bins=63`, `early_stopping=False`, `random_state=2026101951`이다. `frozen-recipe.json`의 `boostedreadout`에서 읽는다. Source/control TabPFN은 estimators 8개와 seeds 0/100/200을 사용한다.

TabH9s2 72개, CNN context용 36개, source/control 9개, 총 117개의 최종 TabPFN context는 `raw.py`가 공식 train 특징·라벨로 적합한다. TabPFN 사전학습 모델 자체는 대회 데이터로 처음부터 학습하지 않는다.

## 실행 경로

`raw.py`는 제공된 학습 가중치로 원본 이미지의 특징·예측을 계산하고 참조값과 비교한다. `trained/raw.py --retrained`는 새로 학습한 가중치로 예측을 출력한다.

사전학습 5개 파일의 정확한 버전과 다운로드 위치는 `model-provenance.json`, 환경은 `requirements-*.txt`에 있다. 공식 대회 데이터는 별도로 제공받아야 한다.
