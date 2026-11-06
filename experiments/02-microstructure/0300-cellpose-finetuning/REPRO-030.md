# 후보 030 재현 절차 (결정립 경로 + best021 결합)

최종 예측 = 0.25 × best021 + 0.75 × 결정립 모델. 결정립 모델은 TabPFN-v2와 TabPFN-3.5 예측의 평균이다. 이 문서는 결정립 경로를 설명하고, best021은 기존 재현 패키지를 그대로 쓴다.

## 1. 환경
- macOS(Apple M1 Pro, 16GB), Python 3.13.3. 분할·미세조정은 MPS GPU, TabPFN은 CPU.
- 두 가상환경을 쓴다.
  - `experiments/02-microstructure/0220-grain-cellpose/.venv`: cellpose 4.2.1.1, torch 2.14.0, scikit-image 0.26.0, scipy 1.18.1, numpy 2.5.3. 분할·특징 단계용.
  - `experiments/frameworks/tabpfn-v2/.venv`: tabpfn 9.1.0, torch, scikit-learn. 예측 단계용.
- 사전학습 가중치(로컬, 네트워크 미사용):
  - Cellpose-SAM `cpsam`(BSD-3-Clause), SHA256 `e1440429…c29a81`
  - TabPFN-v2 회귀(Prior Labs License 1.1), TabPFN-3.5(`tabpfn-v3.5-20260909.safetensors`, TabPFN-3.5 License v1.0, 비상업적 사용 허용)
- 데이터: `data/train.csv`, `data/sample_submission.csv`, `data/train/*.png`, `data/test/*.png`. test 이미지는 추론에만 쓴다(학습·정규화·임계값 계산에 쓰지 않음).

## 2. 미세조정 분할기 (라벨 미사용)
```sh
cd experiments/02-microstructure/0300-cellpose-finetuning
../../02-microstructure/0220-grain-cellpose/.venv/bin/python ft_cellpose.py a1 800 1 2e-5 4 0
```
- train 이미지 중 잡음 추정값 순위 81~250번째(가장 깨끗한 80장은 충실도 검증용으로 제외, 170장)에 무작위 열화(blur σ 0~1.6, contrast 0.55~1.0, 밝기 이동 ±15, 가우스 잡음 σ 0~24 gray, 15%는 거의 원본)를 입력으로 하고, 원본에 대한 제로샷 Cellpose-SAM 마스크를 정답으로 학습한다. D4 대칭 증강, AdamW lr 2e-5, 배치 4, 블록별 gradient checkpointing.
- 스텝 250의 스냅샷 `ft-a1s250.pt`를 쓴다. MPS 연산은 비트 단위로 결정적이지 않으므로, 제출 결과 재현에는 동봉한 가중치 파일을 쓴다.

## 3. 결정립 경로 실행 (`pipeline_grain.py`)
```sh
cd experiments/02-microstructure/0300-cellpose-finetuning
CP=../../02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=../../frameworks/tabpfn-v2/.venv/bin/python; D=../../../data; O=../../../outputs/repro-030
for s in train test; do $CP pipeline_grain.py zs_segment  $s --data $D --out $O; done
for s in train test; do $CP pipeline_grain.py zs_features $s --data $D --out $O; done
for s in train test; do $CP pipeline_grain.py ft_segment  $s --data $D --out $O --ckpt ft-a1s250.pt; done
for s in train test; do $CP pipeline_grain.py assemble    $s --data $D --out $O; done
$TP pipeline_grain.py predict --data $D --out $O --best021 ../../../artifacts/submission-candidate-021.csv
```
1. `zs_segment`: 제로샷 Cellpose-SAM(cp −2, flow 0.8). 커버리지 0.95 미만이면 cp −3, flow 1.0으로 다시 분할하고 커버리지가 늘면 교체한다.
2. `zs_features`: 결정립 특징 64개(`features_v3.py`: 크기 분포·공간 크기장·2상 분율·기공·신장·정렬 등).
3. `ft_segment`: 잡음 추정값이 **train** 중앙값보다 큰 이미지를 미세조정 분할기로 D4 4뷰(원본, 전치, 상하반전, 90° 회전)에서 분할하고 특징을 추출한다.
4. `assemble`: 잡음 절반 행은 4뷰 평균으로 바꾸고, 핵심 특징 16개의 뷰 간 표준편차(분할·특징 추출 변동성 지표, 깨끗한 행은 NaN)와 원본 이미지 구조텐서 정렬도 8개(`stcoh.py`)를 붙인다(88열).
5. `predict`: train 500장 전체로 TabPFN-v2(n_estimators 32, 시드 0/100/200)와 TabPFN-3.5(기본 설정, 시드 0/100/200)를 학습하고 test를 예측한다. 두 모델을 평균한 뒤 best021과 0.25:0.75로 섞는다. 분할 실패(결정립 20개 미만) 이미지는 best021 값을 쓴다.

## 4. 검증 기록
- `assemble` 결과가 후보에 쓴 특징 파일(`feat-a1s250n4vstd-stcoh-{train,test}.npz`)과 값·열·ID 모두 일치.
- `predict`가 `artifacts/submission-candidate-030.csv`를 바이트 단위로 재현(SHA256 `046102fb82120c51206cdb588f897f8ddc43b27034c8fbe468935d8c80a7bb75`).
- 원본 이미지부터의 전체 재실행(`repro-full-20261009/`, 2026-10-09): 제로샷 마스크 train 500/500·test 1000/1000 완전 일치, features-v3 train/test 일치, 미세조정 분할기 4뷰 특징 일치, 최종 88열 특징 train/test 일치(`repro-full-20261009/compare.txt`). 소요: 제로샷 분할 train 37분·test 53분, 특징 2+4분, 미세조정 분할 train 45분·test 약 85분, 조립 2분, 예측(CPU) 15분.

## 5. best021
`deliverables/reproduction-018-v176/`(후보 018 재현 패키지)와 `experiments/01-representations/0178-candidate021/`(DINOv2-L MLP 10% 추가, `infer.py`) 절차로 `submission-candidate-021.csv`(SHA256 `7cf4ce3b…`)를 재현한다.
