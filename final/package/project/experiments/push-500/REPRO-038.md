# 후보 038 재현 절차

최종 예측 = 0.25 × best021 + 0.75 × 결정립 모델(분할 실패 n_obj<20 이미지는 best021).
결정립 모델 = TabPFN-3.5(기본 설정, 시드 0/100/200)를 4개 특징 변형(u1dm1sz, u1dm2sz, u1dmavg, u1dmavg4) + 4열(H2Z)에 각각 학습해 평균.

## 1. 035까지의 특징 (기존 절차)
`experiments/push-400/REPRO-035.md`의 1~3단계로 `push-400/feat-{u1dm1sz,u1dm2sz,u1dmavg,u1dmavg4}-{train,test}.npz`를 만든다.
het 계산에는 U-Net 확률 지도가 필요하다: `unet2_feats.py`를 NOMAPS 없이 실행하면 `push-400/unet2maps-<net>-<split>.npz`가 생긴다(m1, m2, h1, m3; m2·m3는 `LA_MU=5.6343302726745605 LA_SD=0.9692867994308472`).

## 2. z 채널 U-Net (라벨 미사용) — 제출 재현은 동봉 가중치 사용
```sh
cd experiments/push-400
../grain-cellpose-220/.venv/bin/python phase_unet4.py z1 4000 0          # unet4-z1.pt (학습, MPS 비결정적)
../grain-cellpose-220/.venv/bin/python unet4_feats.py z1 train z1         # unet4-z1-train.npz (+ unet4maps-z1-train.npz)
../grain-cellpose-220/.venv/bin/python unet4_feats.py z1 test z1          # unet4-z1-test.npz
```
`phase_unet4.py` = `phase_unet2.py`(다중 과제: 4클래스 + log면적 + 형상텐서) + 8번째 채널(소속 결정립의 어두움 z, features.py 규칙의 연속값) 회귀.
입력은 깨끗한 train 이미지(잡음 < 중앙값)의 합성 열화, 정답은 원본의 제로샷 마스크에서 계산. 경도 라벨·test 이미지 미사용.

## 3. 4열 추가 (`experiments/push-500`, tabpfn venv)
```sh
cd experiments/push-500
TP=../tabpfn-v2/.venv/bin/python
for n in m1 m2 h1 m3; do $TP unet_het.py $n; done                         # uhet-<net>-{train,test}.npz (het4_u, het4_u_mx 등)
$TP make_het_sets.py het4_u,het4_u_mx H2                                    # feat-<v>H2-{split}.npz (변형별 U-Net 지도에서 평균)
$TP add_z.py z1 z_f25,z_f50 H2 H2Z                                          # feat-<v>H2Z-{split}.npz
```
- het4_u: 4×4 블록(64px)마다 b = −2·log(Σ g·a^−1/2 / Σ g), a = exp(예측 log면적), g = p기지 + p2상; 블록 간 표준편차. het4_u_mx는 g = p기지.
- z_f25 / z_f50: g 가중 시그모이드((z − t)/0.02) 면적비, t = 0.25 / 0.5.
- 검증(2026-10-10): 스크래치 디렉터리에서 다시 만든 결과가 u1dm1szH2Z·u1dm2szH2Z(train·test)는 바이트 단위 일치,
  u1dmavgH2Z·u1dmavg4H2Z는 het 2열에서 float32 반올림 차(최대 4e-8)만 있다.

## 4. 예측 (`experiments/push-300`)
```sh
cd experiments/push-300
../tabpfn-v2/.venv/bin/python finalize_multi.py 038 0.75 ../push-500/evidence-038.json cpu \
  v3.5@../push-500/feat-u1dm1szH2Z-train.npz@../push-500/feat-u1dm1szH2Z-test.npz \
  v3.5@../push-500/feat-u1dm2szH2Z-train.npz@../push-500/feat-u1dm2szH2Z-test.npz \
  v3.5@../push-500/feat-u1dmavgH2Z-train.npz@../push-500/feat-u1dmavgH2Z-test.npz \
  v3.5@../push-500/feat-u1dmavg4H2Z-train.npz@../push-500/feat-u1dmavg4H2Z-test.npz
```
CPU 실행(결정적). 제출 CSV SHA256 `e6eb78afdd5e2383258345a425c70225672a5f79cf38759ad6e0f472be283023`, Public 9.1817.

## 5. 보관할 가중치
`push-400/unet4-z1.pt`(SHA256은 `evidence-038.json`의 unet_weights). 기존 `unet-u1.pt`, `unet2-{m1,m2,h1,m3}.pt`, `ft-a1s250.pt`와 함께 보관.
