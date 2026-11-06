# 후보 057 재현 절차 (Public 8.9963)

057 = 053과 같고, **화소 잡음 ≥ 12 gray인 test 행에서만** Ridge·logRidge 성분(각 0.128 = 0.8 × 0.16)을
"train 500행 + 깨끗한 train 이미지의 라벨 보존 열화본(deg2, deg3)"으로 학습한 같은 모형으로 바꾼다.

```
057 = 053 + 1[noise ≥ 12 & 분할 성공] × 0.128 × ((ridge_aug − ridge) + (logridge_aug − logridge))
```

## 0. 전제
- 053 재현(`experiments/02-microstructure/0900-partner-ensembles/make_053.py`, 성분: 021·038·039·044·Ridge·logRidge·집합 모델·gmm; REPRO-045, push-900 NOTES §7).
- 환경: REPRO-045와 같은 3개 가상환경(루트 `.venv`, `grain-cellpose-220/.venv`(cellpose, torch, torchvision), `tabpfn-v2/.venv`).
- 사용 데이터: train 500장(이미지·라벨)과 test 1000장(추론만). 열화본은 **train 이미지의 라벨 없는 변환**(대회 답변: train 증강 허용)이며,
  열화 매개변수는 우리 train 이미지 통계로만 보정했다(push-900 NOTES §3, `calib_own.py`). test 이미지는 어떤 학습·적합에도 쓰지 않는다.

## 1. 라벨 보존 열화본 생성 (깨끗한 train 이미지 249장, 잡음 < train 중앙값)
```sh
cd experiments/02-microstructure/0400-unet-maps
TP=../../frameworks/tabpfn-v2/.venv/bin/python
$TP make_synth2.py     # synth/train_deg2/*.png  (결정립 밝기 산포·음영·흐림·대비·잡음, 잡음 절반 분포에 맞춤)
$TP make_synth3.py     # synth/train_deg3/*.png  (push-900/realdeg2: 보정 잡음 프리셋 + 경계선 전부 제거)
```

## 2. 열화본에 운영 잡음 행 특징 파이프라인 적용
```sh
CP=../../02-microstructure/0220-grain-cellpose/.venv/bin/python
$CP synth_feats2.py; $CP synth_feats3.py                 # ft-a1s250 4뷰 분할 → v3 특징 + vstd + stcoh (synth/feat-ftB.npz, feat-ftC.npz)
$CP unet_feats_s2.py u1 synth; $CP unet_feats_s3.py u1 synth   # u1 2상 U-Net 열 (unet-u1-synth2.npz, unet-u1-synth3.npz)
cd ../../02-microstructure/0500-heterogeneity
for st in train_deg train_deg2 train_deg3; do for n in m1 m2 h1 m3 u4z1; do $CP synth_unet.py $st $n; done; done   # 다중과제/z U-Net 지도·특징
$TP build_synth_rows.py && $TP build_synth_rows3.py    # 038형 H2Z 93열 (synthrows-train_deg{,2,3}-<v>H2Z.npz)
cd ../../03-robustness/1200-degradation-augmentation
for st in train_deg2 train_deg3; do for n in m1 m2 h1 u4z1; do $CP synth_unet.py $st $n; done; done   # 같은 지도를 push-1200에도(각도·sh_cor15용)
for st in deg2 deg3; do $CP synth_anghcor.py seg $st; $TP synth_anghcor.py build $st; done   # + ang_c4/ang_s4, 상호작용 2열, sh_cor15 → 98열
```
(`synth_anghcor.py`: ft-H300 1뷰 분할 + z1 지도로 운영 `shade_feat.py`와 같은 정의의 sh_cor15, 운영 `angle_feats.py`와 같은 각도 특징.)

## 3. 최종 짝 모델과 후보
```sh
cd experiments/03-robustness/1200-degradation-augmentation
$TP final_aug.py none fin-none        # 운영 Ridge/logRidge test 예측 재현 (최대 차 1e-13 확인)
$TP final_aug.py deg2,deg3 fin-d23    # 500행 + 열화본 498행으로 같은 설정 학습 → test 예측
$TP make_057.py                       # ../../../artifacts/submission-candidate-057.csv (SHA256 971cf686…)
```
설정: RidgeCV(α ∈ {0.1, 1, 10, 100, 1000}, 중앙값 대체, 표준화) on feat-u1dmavgH2Z-anghcor;
log-Ridge α 2(4개 변형 세트 각각, 기하평균). 게이트는 test 특징의 `noise` 열 × 255 ≥ 12 (318행).

## 4. 검증 기록
- dev(Study900 400행 × 3분할, 같은 게이트·가중치): 053 9.7665 → 9.6861, 분할 +0.088/+0.063/+0.090. 무증강 대조는 053을 정확히 재현,
  라벨을 섞은 열화본 대조 −0.29. dev와 무관한 train 100행(잡음 35행): 증강 짝 평균 11.80 → 10.77.
- Public 8.9963 (053 9.0381).

## 5. 061 / 062 / 이후 (열화본 확대)
같은 구조에서 증강 집합만 바꾼다. 새 열화본은 push-1200의 일반화 스크립트로 만든다(원본·열화 매개변수만 다르고 특징 파이프라인은 같음).
```sh
cd experiments/03-robustness/1200-degradation-augmentation
# deg4/5: realdeg2 기본(보정) 프리셋, deg6: 경계선 전부 제거형, deg7: 중간 잡음(train 중앙값 ≤ 잡음 < 12) train 이미지를 원본으로; 모두 목표 잡음은 train ≥12 풀
for n in 4 5 6 7; do
  $TP make_synth$n.py && $CP synth_feats_g.py train_deg$n && $CP unet_feats_sg.py u1 synth train_deg$n
  for m in m1 m2 h1 m3 u4z1; do $CP synth_unet.py train_deg$n $m; done
  $TP build_rows_g.py train_deg$n && $CP synth_anghcor.py seg deg$n && $TP synth_anghcor.py build deg$n
done
NOISEMIN=12 $TP final_aug.py deg,deg2,deg3,deg4 fin-d1234n12 && $TP make_061.py        # 061 (Public 8.9831)
NOISEMIN=12 ALPHAS=0.001,0.01,0.1,1,10,100,1000 LOGALPHAS=0.01,0.1,0.3,1,2,3,10,30,100 \
  $TP final_aug.py deg,deg2,deg3,deg4,deg5,deg6,deg7 fin-d1to7n12-wal                 # 062 후보 (미제출)
$TP make_aug.py 062 d1to7n12-wal "..." 9.6448
```
`NOISEMIN=12`: 측정 잡음이 12 gray 이상인 열화본만 학습에 사용. `ALPHAS`/`LOGALPHAS`: 증강 학습 집합 안의 내부 CV(GCV)로 규제 선택.
dev: 061 9.6636, 062 9.6448 (053 9.7665).

## 6. 063 / 064 / 065 / 066 (TabPFN 슬롯 + 증강 열, CNN 열화 증강)
```sh
cd experiments/03-robustness/1200-degradation-augmentation
# TabPFN 슬롯 + 'sim-augmented log-Ridge 예측' 열 (학습 행은 내부 5겹 OOF, test는 전체 적합); 슬롯 특징 = 038 H2Z / 039 angh / 044 anghcor
$TP final_ta.py deg,deg2,deg3,deg4,deg5,deg6,deg7 mps                       # fin-ta.npz (044)
$TP final_ta.py deg,deg2,deg3,deg4,deg5,deg6,deg7 mps '../../02-microstructure/0700-directional-features/feat-{v}-angh-{split}.npz' fin-ta39    # fin-ta39.npz
$TP final_ta.py deg,deg2,deg3,deg4,deg5,deg6,deg7 mps '../../02-microstructure/0500-heterogeneity/feat-{v}-{split}.npz' fin-ta38    # fin-ta38.npz
$TP make_063.py; $TP make_064.py; $TP make_065.py                           # 064: 잡음 ≥12 행 세 슬롯 교체, 065: 슬롯 게이트 ≥9.5
# CNN 열화 증강 (066): TinyCNN128 대조/증강 쌍, 시드 8, train 500장
../../../.venv/bin/python final_cnn_dta.py 1,2,3,4,5,6,7,8 && $TP make_066.py 9.5623
```
dev: 063 9.6316, 064 9.5951, 065 9.5842, 066 9.5623 (dev 재현: `tabaug_dev.py`, `dev065.py`, `cnn_dta.py`, `eval_cnn_dta.py`).
