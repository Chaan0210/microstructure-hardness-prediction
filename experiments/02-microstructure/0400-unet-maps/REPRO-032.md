# 후보 031·032 재현 절차 (push-400: 화소 U-Net 2상 특징)

최종 예측 = 0.25 × best021 + 0.75 × 결정립 모델. 결정립 모델은 TabPFN-v2(n_estimators 32, 시드 0/100/200)와 TabPFN-3.5(기본, 시드 0/100/200) 예측의 평균이다. 030까지의 결정립 경로는 `experiments/02-microstructure/0300-cellpose-finetuning/REPRO-030.md`를 따른다. 이 문서는 그 위에 더한 부분만 적는다.

## 1. 화소 U-Net (라벨 미사용, test 미사용)
```sh
cd experiments/02-microstructure/0400-unet-maps
../../../.venv/bin/python phase_unet.py u1 3000 0        # -> unet-u1.pt (MPS, 약 18분)
```
- 학습 입력: train 이미지 중 잡음 추정값이 train 중앙값보다 작은 249장(검증 30장 제외, 219장)에 무작위 열화를 입힌 것.
  열화: 결정립별 밝기 오프셋 N(0, σ), σ~U(0,7) · 2차 음영(진폭 U(0,8)) · 가우스 블러 σ~U(0,2.2) · 대비 U(0.6,1) · 밝기 ±15 · 가우스 잡음 σ~U(0,24). 15%는 거의 원본.
- 정답: 깨끗한 원본의 제로샷 Cellpose 마스크(`grain-cellpose-220/masks-v3-train.npz`)와 `features.py`의 2상 규칙(z=(hi−I)/(hi−lo), 기공 z>0.75·조밀, 2상 z>0.3)으로 만든 화소 클래스 {기지, 2상, 기공, 경계}.
- 경도 라벨과 test 이미지는 쓰지 않는다. MPS 연산은 비트 단위로 결정적이지 않으므로 제출 재현에는 동봉한 `unet-u1.pt`(SHA256은 `evidence-031.json`)를 쓴다.

## 2. U-Net 특징
```sh
../../../.venv/bin/python unet_feats.py u1 train     # -> unet-u1-train.npz
../../../.venv/bin/python unet_feats.py u1 test      # -> unet-u1-test.npz
```
D4 8뷰 확률 평균에서 `u_ph`(2상 확률 면적 분율), `u_ph_hard`(argmax 2상 분율), `u_ph_f32`(32px 평활 2상 분율의 공간 표준편차) 등을 계산한다.

## 3. 특징 조립
- 031: `addcols.py {train,test} feat-u1ph-{split}.npz unet-u1-{split}.npz:u_ph,u_ph_hard,u_ph_f32` (030의 88열 + 3열)
- 032: 031 특징에서 분할 임계값 기반 2상 열 12개 제거(`evidence-032.json`의 목록) → `feat-u1phdrop-{train,test}.npz` (79열)

## 4. 예측
```sh
cd ../../02-microstructure/0300-cellpose-finetuning
../../frameworks/tabpfn-v2/.venv/bin/python finalize_ft.py 032 ../../02-microstructure/0400-unet-maps/feat-u1phdrop-train.npz ../../02-microstructure/0400-unet-maps/feat-u1phdrop-test.npz 0.75 v2,v3.5 "<evidence>" mps
```
031은 같은 명령을 `feat-u1ph-*`와 `cpu`로 실행했다. 결과: `artifacts/submission-candidate-03{1,2}.csv`.

## 5. 검증 기록
| 후보 | dev 고정 W0.75 통합 | 세 분할 | Public |
|---|---:|---|---:|
| 030 | 10.779 | 10.713 / 10.789 / 10.835 | 9.8102 |
| 031 | 10.431 | 10.439 / 10.416 / 10.438 | 9.6088 |
| 032 | 10.382 | 10.389 / 10.360 / 10.398 | 9.5891 |
