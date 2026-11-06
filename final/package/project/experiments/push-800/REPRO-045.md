# 후보 045 재현 절차 (원본 이미지 → 제출 CSV)

최종 예측 = 0.20 × best021 + 0.80 × 평균(결정립 모델 038, 039, 044). 분할 실패(결정립 20개 미만) 이미지는 best021 값을 쓴다.
045는 038·039·044 제출 CSV(각각 0.25 × best021 + 0.75 × 결정립)에서 고정된 혼합식으로 결정립 예측을 분리한 뒤 다시 조합한다(재학습 없음).
제출 CSV SHA256 `7a9dbb257f9c599596769cfcade50f8dd3d988637879ff9896113c62138725c1`, Public 9.1590.

## 0. 환경 (REPRO-035와 같음)
- macOS(Apple M1 Pro 16GB), Python 3.13. 가상환경 3개:
  - `.venv`(루트): torch 2.14 — U-Net 학습·추론
  - `experiments/grain-cellpose-220/.venv`: cellpose 4.2.1.1 — 분할·결정립 특징
  - `experiments/tabpfn-v2/.venv`: tabpfn 9.1.0 — TabPFN, 조립·예측
- 사전학습 가중치(로컬): Cellpose-SAM `cpsam`(BSD-3-Clause), TabPFN-v2(Prior Labs License 1.1), TabPFN-3.5(`tabpfn-v3.5-20260909.safetensors`, TabPFN-3.5 Non-Commercial License).
- 경도 라벨은 TabPFN 학습(예측 단계)에서만 쓴다. test 이미지는 추론에만 쓴다. U-Net·Cellpose 미세조정은 train 이미지만, 라벨 없이 학습했다.
- 결정성: cellpose 분할(MPS)·U-Net 추론·TabPFN CPU 예측은 재실행 시 비트 단위로 같다(2026-10-12 확인: ft-H300 분할 6장, z1 재학습 가중치, 044 CPU 예측). MPS U-Net **학습**도 같은 seed에서 같은 가중치를 냈다(z1). 그래도 제출 재현에는 동봉 가중치를 쓴다.

## 1. best021
`deliverables/reproduction-018-v176/` + `experiments/candidate021-178/infer.py` → `artifacts/submission-candidate-021.csv`(SHA256 `7cf4ce3b…`).

## 2. 038 (`experiments/push-500/REPRO-038.md`)
REPRO-030(결정립 경로: 제로샷 cpsam 분할, 미세조정 cellpose `ft-a1s250.pt` 4뷰, 88열) → REPRO-035(U-Net u1·m1·m2·h1·m3 특징, 4개 변형 세트)
→ REPRO-038(z1 U-Net, 블록 이질성 het4_u·het4_u_mx, z_f25·z_f50) → `push-500/feat-{u1dm1sz,u1dm2sz,u1dmavg,u1dmavg4}H2Z-{train,test}.npz`
→ `finalize_multi.py 038 0.75 … cpu` → `artifacts/submission-candidate-038.csv`.

## 3. 039 (`experiments/push-700`)
```sh
cd experiments/push-700
TP=../tabpfn-v2/.venv/bin/python
for s in train test; do $TP angle_feats.py $s; done        # angle-{split}.npz: unet2maps-{m1,m2,h1} + unet4maps-z1 (D4 TTA 지도)의 이중각 벡터 → ang_c4, ang_s4
$TP build_sets.py angh u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z   # + het4_u×m_el_pix, sizefield_std_s32×m_el_pix
cd ../push-300 && $TP finalize_multi.py 039 0.75 ../push-700/evidence-039.json cpu \
  v3.5@../push-700/feat-u1dm1szH2Z-angh-train.npz@../push-700/feat-u1dm1szH2Z-angh-test.npz  (… 4개 변형)
```
U-Net 지도가 필요하다: `push-400/unet2_feats.py`(NOMAPS 없이, m2·m3는 `LA_MU=5.6343302726745605 LA_SD=0.9692867994308472`), `push-400/unet4_feats.py z1 {split} z1`.

## 4. 044 (`experiments/push-800`) — 039 + 음영 보정 2상 분율 1열
```sh
cd experiments/push-300
CP=../grain-cellpose-220/.venv/bin/python; TP=../tabpfn-v2/.venv/bin/python
# (학습, 라벨 미사용) $CP ft_cellpose.py H300 300 0 2e-5 4 0 250 heavy   → ft-H300.pt (SHA256 4bd566e5…; 동봉 가중치 사용)
for s in train test; do $CP seg_ft.py ft-H300.pt $s noisy 1; done      # ftseg-ft-H300-{split}-noisy-n1-masks.npz (잡음 > train 중앙값 행만)
cd ../push-800
for s in train test; do $CP shade_feat.py $s; done                       # shade-{split}.npz: sh_cor15, sh_raw15
for s in train test; do $TP add_cols.py cor $s cor; done                 # feat-<v>-anghcor-{split}.npz (+ sh_cor15)
cd ../push-300 && $TP finalize_multi.py 044 0.75 ../push-800/evidence-044.json cpu \
  v3.5@../push-800/feat-u1dm1szH2Z-anghcor-train.npz@../push-800/feat-u1dm1szH2Z-anghcor-test.npz  (… 4개 변형)
```
- sh_cor15: 결정립(깨끗 = 제로샷 masks-v3, 잡음 = ft-H300 1뷰)별 내부 평균 밝기 I, z1 U-Net 기공 확률 ≥ 0.3 결정립 제외,
  기지 결정립(z1 2상 확률 < 0.15) 중심에서 가우스 커널(σ 48px, 자기 제외) 회귀로 기지 밝기장을 만들고, 장 − I > 15 gray인 결정립의 면적 비율.
- 2026-10-12: 위 절차를 저장된 특징으로 CPU 재실행 → `artifacts/submission-candidate-044r.csv`가 044와 일치(§6).

## 5. 045
```sh
cd experiments/push-800 && ../tabpfn-v2/.venv/bin/python make_045.py   # artifacts/submission-candidate-045-repro.csv, SHA 비교 출력
```

## 6. 검증 기록
- make_045.py: 저장된 021·038·039·044 CSV로 다시 만든 파일의 SHA256이 제출 045와 일치(2026-10-12).
- ft-H300 분할: test 3장·train 3장(10/09 생성 파일 포함) 재분할 → 마스크 완전 일치.
- 044 CPU 재예측: §4 마지막 명령을 cid 044r로 재실행 → `submission-candidate-044r.csv` SHA256이 044와 동일(f9fff6f3…, 2026-10-12).

## 7. 보관할 가중치(용량 때문에 git 제외 항목 포함)
`push-300/ft-a1s250.pt`, `push-300/ft-H300.pt`(각 1.2GB), `push-400/unet-u1.pt`, `unet2-{m1,m2,h1,m3}.pt`, `unet4-z1.pt`, `grain-cellpose-220/weights/cpsam`, TabPFN 가중치 2종, best021 패키지의 모델들.
