# 후보 035·036 재현 절차 (원본 이미지 → 제출 CSV)

최종 예측 = (1−W) × best021 + W × 결정립 모델. 결정립 모델은 TabPFN-3.5(기본 설정, 시드 0/100/200)를 4개 특징 변형에 각각 학습해 평균한 것이다.
035는 W=0.75, 036은 W=0.85. 분할 실패(결정립 20개 미만) 이미지는 best021 값을 쓴다.

## 0. 환경
- macOS(Apple M1 Pro 16GB), Python 3.13. 가상환경 3개:
  - `.venv`(루트): torch 2.14 — U-Net 학습·추론
  - `experiments/grain-cellpose-220/.venv`: cellpose 4.2.1.1 — 분할·결정립 특징
  - `experiments/tabpfn-v2/.venv`: tabpfn 9.1.0 — TabPFN, 조립·예측
- 사전학습 가중치(로컬): Cellpose-SAM `cpsam`(BSD-3-Clause), TabPFN-v2(Prior Labs License 1.1), TabPFN-3.5(`tabpfn-v3.5-20260909.safetensors`, 비상업 허용).
- 경도 라벨은 TabPFN 학습(예측 단계)에서만 쓴다. test 이미지는 추론에만 쓴다. U-Net·Cellpose 미세조정은 train 이미지만, 라벨 없이 학습했다.

## 1. 결정립 경로 (후보 030의 88열 특징) — `experiments/push-300/REPRO-030.md`
`pipeline_grain.py zs_segment / zs_features / ft_segment / assemble` (train·test). 원본부터 재실행 시 특징이 완전히 일치함을 2026-10-09에 확인했다.
결과: `feat-{train,test}.npz`(= `push-300/feat-a1s250n4vstd-stcoh-{split}.npz`).

## 2. 화소 U-Net (라벨 미사용) — 학습은 재현용 참고, 제출 재현은 동봉 가중치 사용
```sh
cd experiments/push-400
../../.venv/bin/python phase_unet.py u1 3000 0                     # unet-u1.pt (2상/기공/경계 분류)
../../.venv/bin/python phase_unet2.py m1 4000 0 0 32                # unet2-m1.pt (다중 과제: 분류 + log면적 + 형상텐서)
../../.venv/bin/python phase_unet2.py m2 4000 1 1 32                # unet2-m2.pt
HEAVY=0.5 ../../.venv/bin/python phase_unet2.py h1 6000 3 0 32      # unet2-h1.pt
../../.venv/bin/python phase_unet2.py m3 4000 5 1 32                # unet2-m3.pt
../../.venv/bin/python phase_unet2.py m4 4000 6 1 32                # unet2-m4.pt
```
입력은 깨끗한 train 이미지(잡음 < train 중앙값)를 결정립 밝기 산포·음영·블러·대비·잡음으로 열화한 것, 정답은 원본의 제로샷 Cellpose 마스크와 `features.py` 2상 규칙에서 만든 화소 지도.
MPS 학습은 비트 단위로 결정적이지 않으므로 제출 재현에는 동봉한 가중치를 쓴다(각 7.7MB, git 포함).

## 3. U-Net 특징 → 조립 → 예측 (`pipeline_final.py`)
```sh
cd experiments/push-400
TP=../tabpfn-v2/.venv/bin/python; O=<새 출력 디렉터리>
for s in train test; do $TP pipeline_final.py unet $s --out $O; done          # U-Net D4 TTA 특징(6개 모델)
for s in train test; do $TP pipeline_final.py assemble $s --out $O --base-pattern <1단계 출력>/feat-{split}.npz; done
$TP pipeline_final.py predict 035 --out $O --W 0.75 --device cpu               # 035 구성(4개 변형 평균)
$TP pipeline_final.py predict 036 --out $O --W 0.85 --device cpu               # 036
```
- `assemble`: 030 특징 + U-Net 2상 3열(u_ph, u_ph_hard, u_ph_f32) − 분할 임계값 2상 12열 = `u1phdrop`(79열),
  여기에 다중 과제 U-Net 크기·신장·2상 10열을 붙인 변형 `u1dm1sz`, `u1dm2sz`, `u1dmavg`(m1·m2·h1 평균), `u1dmavg4`(m1·m2·h1·m3 평균) 등.
- `predict` 기본 성분: `v3.5@u1dm1sz, v3.5@u1dm2sz, v3.5@u1dmavg, v3.5@u1dmavg4`.

## 4. 검증 기록 (2026-10-10)
- `unet` 단계: 동봉 가중치로 6개 U-Net 특징(train·test)을 다시 계산 → 기존 파일과 **완전 일치**(MPS 추론은 결정적).
- `assemble` 단계: 8개 특징 세트(train·test) **완전 일치**.
- `predict` 단계: CPU 실행은 결정적이다(036은 CPU로 생성, SHA256 8e8af1c5…). 035는 MPS로 만들었고, 재실행 시 1000행 중 10행에서 최대 1.3e-7 차이(점수 영향 없음).
- best021: `deliverables/reproduction-018-v176/` + `experiments/candidate021-178/infer.py` (SHA256 7cf4ce3b…).
