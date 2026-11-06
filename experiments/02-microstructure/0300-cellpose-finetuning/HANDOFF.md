# Cellpose 미세조정 실험 (2026-10-08)

## 1. 대회 현황 (2026-10-08 15:10 확인)
- 마감: 제출 **10/19 10:00**, 평가 10/28, 종료 10/29. RMSE. Public 30% / Private 70%. 하루 5회 제출. 최종 선택 1개.
- 우리: candidate027, Public **10.3957**, **11위**(123명 중).
- 상위: 1위 8.8662, 2위 8.9630, 3위 9.6509, 4위 9.7090, 5위 9.8804, 6위 9.9768, 7위 10.1591, 8위 10.1868, 9위 10.2932, 10위 10.3082.
- 게시판 글 0개. 참고 PDF(3쪽)는 `reports/summaries/official-reference-review-018.json`에 정리돼 있다.
- 027 구성: 0.5×best021 + 0.5×TabPFN-v2(n_est 32 × seed 3), 입력은 결정립 특징 v3. 잡음이 train 중앙값을 넘는 이미지는 4-뷰 분할 TTA 평균(`features-v3tta`). 만드는 법: `experiments/02-microstructure/0220-grain-cellpose/finalize_tabpfn.py`, 상세는 `experiments/02-microstructure/0220-grain-cellpose/REPORT.md`.

## 2. 평가 하네스 (`harness.py`)
- train 500장 전체, 5-fold × 3회 반복(`folds(n, rep)` = `RandomState(1000+rep)`), 잡음 3분위별 RMSE를 함께 출력한다.
- 기준선은 TabPFN-v2 n_est 8, seed 0, MPS, `features-v3tta`. 캐시는 `oof_base_tabpfn.npy`(shape 3×500)다.
  **11.7549** (반복별 11.530 / 11.740 / 11.994), 3분위 깨끗 **8.855** / 중간 **10.576** / 잡음 **14.543**.
- 실험 스크립트는 모두 `experiments/02-microstructure/0300-cellpose-finetuning`에서 상대경로로 실행한다. TabPFN과 LightGBM은 `../../frameworks/tabpfn-v2/.venv/bin/python`, cellpose는 `../0220-grain-cellpose/.venv/bin/python`, CNN은 `../../../.venv/bin/python`을 쓴다.
- 주의: 이 하네스는 TabPFN 단독 500장 CV다. 제출 판단에 쓰던 공식 dev 절차(400장, 3개 재분할, best021과 0.5 결합, `experiments/02-microstructure/0220-grain-cellpose/validate_v3.py` 등)와는 다르다. 하네스에서 이긴 변경은 공식 절차로 한 번 더 확인하고 제출한다.

## 3. 이번 세션 결과 (모두 같은 fold로 기준선과 짝비교)

| # | 실험 | 파일 | 결과 | 판정 |
|---|---|---|---|---|
| 1 | 잡음 3분위별 오차 분해 | – | 오차 대부분이 잡음 상위 1/3(14.5)에서 나온다. 잡음 구간 corr(pred,y) 0.862 / 0.785 / 0.647. 잡음 구간은 예측이 평균 쪽으로 수축한다(잔차~y 기울기 −0.59, 깨끗 구간 −0.29). | 진단 |
| 2 | 열화 충실도(깨끗한 80장에 blur 1.0 / noise 17 / contrast 0.85 적용) | `fidelity_exp.py`, `fidelity.npz`, `analyze_fidelity.py` | 80장 RMSE: 원본 9.06, 열화 11.69, NLM 잡음제거 20.11, 2배 업샘플 14.12, NLM+업샘플 15.39. `grain_logA_std`, `solidity_wmean`, `sizefield_range_s32`는 열화에 약하다. | 전처리(NLM, 업샘플) **기각** |
| 3 | 분할 없이 쓰는 자기상관 특징 20개 | `robust_feats.py`, `robust-train.npz` | v3tta에 추가: 11.7422(−0.013, 잡음 범위 안). 8개만 추가: 11.8116. 단독: 14.18. cellpose 12개와 함께: 13.92. | **기각**(효과 없음) |
| 4 | best021과 결정립 TabPFN의 결합 비중을 fold별로 고르기(**nested 아님, 2026-10-09 리뷰로 폐기**)(고정 비중 / 잡음에 따라 바꾸는 비중) | `blend_weight_nested.py` | 0.5 고정 → nested 고정 비중: original 11.610 → **11.530**, 20261031 11.569 → **11.536**, 20261101 11.756 → **11.620**(세 분할 모두 0.03~0.14 개선). fold마다 고른 결정립 비중은 **0.6~0.8**. 잡음별 비중: 11.468 / 11.563 / 11.724 | 잡음별 비중 **기각**(불안정). 고정 비중 0.6~0.7은 **채택 후보**(§5-C) |
| 5 | 열화 증강 행 추가(깨끗한 절반 × 2 = 562행, 라벨 동일) | `degrade.py`, `make_aug.py`, `aug-v3-k2.npz`, `eval_aug.py` | 실제 행만 11.7549 → 증강 추가 **12.6027**(+0.85 악화) | **기각** |
| 6 | 라벨 없는 특징 복원 CNN(열화 이미지 → 깨끗한 v3 특징 34개 예측, D4 8뷰 평균) | `distill_cnn.py`, `distill-a.npz`, `distill-a.pt`, `distill-a.log` | 홀드아웃 평균 R² 0.584. v3tta에 붙이면 **11.7229(−0.032)**, 3분위 8.733 / 10.616 / 14.499 | **약한 개선, 확인 필요**(§5-C) |
| 7 | 2배 축소 후 분할(잡음 절반) | `seglib.seg_down2`, `seg_variant.py`, `var-seg_down2-train-v1.npz`, `eval_variant.py` | 교체: 12.6037(악화). 기존 TTA와 평균: 11.8427(v3 11.8561보다 약간 낫지만 v3tta 11.7549보다 나쁨) | **기각** |
| 8 | 경계 대비, 결정립 명암 퍼짐, 조명 등 후보 특징 | `cand_feats.py`, `cand-train.npz` | 깨끗·중간 구간 잔차와의 상관 \|ρ\| < 0.06 | **기각** |
| 9 | 누출 점검(ID 순서, 파일 크기, PNG 메타데이터) | 인라인 | 모두 상관 없음. train/test 파일 크기 분포 같음 | 누출 없음 |
| 10 | LightGBM 깊이 비교 | 인라인 | 전체: depth1 12.82, depth2 12.55, full 12.73. TabPFN 11.75보다 나쁨 | 참고 |
| 11 | LightGBM 전진 특징 선택 | 인라인 | 팬 소음 때문에 4단계에서 중단(sp_frac_grains → sizefield_range_s32 → f_dark_20 → f_dark_50 → solidity_wmean) | 미완 |
| 12 | **8-뷰(D4 전체) 분할 TTA, 잡음 절반** | `seg_variant.py train seg 8 4 noisy` → `var-seg-train-noisy-v4to8.npz` | 125/249장에서 중단, 결과 없음 | **다음 세션 첫 작업**(§3-1) |

기타 관찰:
- 깨끗한 절반에서 선형 모델의 계수가 큰 특징은 aw_eqd_mean(−) / grain_eqd_mean(+) 쌍, sp_frac_grains(+), sizefield_std_s32(+)다. 깨끗한 절반 ridge CV는 11.06, LightGBM depth2는 10.50이다.
- 잡음과 hardness의 상관은 ρ −0.04로 사실상 없다. 그런데 n_obj, coverage, px_pore_n은 잡음과 강하게 상관한다. 이 측정값들은 잡음에 오염돼 있다.
- 잡음 구간 최악 사례: TRAIN_000467(y 152, 예측 192), 000209(230 / 190), 000186(184 / 216), 000120(155 / 186).

### 3-0. 세션 최종 결론 (원래 세션의 마지막 보고 요약)
- **입상권(Public 9.65)까지 갈 개선은 아직 찾지 못했다.**
- **오차 대부분이 잡음 큰 이미지에서 나온다.** 하네스 3분위 RMSE는 깨끗 8.86 / 중간 10.58 / 잡음 14.54다. 깨끗한 이미지는 이미 1위 Public 점수(8.87)와 비슷한 수준이다(CV와 Public은 직접 비교할 수 없으니 참고만 한다).
- **원인:** 잡음 이미지에서는 결정립 사이 경계선이 사실상 사라진다. 그래서 cellpose가 명암이 비슷한 이웃 결정립을 하나로 합친다. 결정립 수가 깨끗한 이미지의 절반 정도로 잡히고, 경도에 중요한 크기 불균일성 측정이 망가진다. best021(CNN·DINO)도 잡음 구간 RMSE가 약 15.2로 똑같이 실패한다(`resolution-216` fixed_control 잡음 3분위: original 15.16, 재분할 15.34).
- **누출은 없다**(#9). 상위권은 정당하게 더 나은 모델을 가진 것으로 본다.
- 분할 쪽에서 유일하게 확인된 개선은 기존 4뷰 TTA다. 1뷰(v3) 11.856 → 4뷰(v3tta) 11.755로 전체 0.10 개선, 잡음 3분위는 14.86 → 14.54.
- **추천 순서와 기대치:**
  1. TabPFN-3.5로 교체(§4). 모델만 바꾸면 되는 가장 싼 큰 시도다. 3위까지 필요한 dev 약 0.85 개선을 기대해 볼 만한 거의 유일한 후보지만, 이 데이터에서는 아직 측정하지 않았다.
  2. 8뷰 TTA 마무리와 깨끗한 절반 TTA. 기대 이득은 작다(약 0.03~0.08).
  3. 021과 결정립 모델의 고정 비중을 0.5에서 0.6~0.7로 조정(#4).
  - 2와 3을 합쳐도 dev 0.1~0.2 정도로 10위권을 지키는 수준이다. 3위(9.65)에 가려면 1이 필요하다.
- **최종 선택:** 순위는 Private 70%로 정해진다. 마지막 1개는 Public 점수가 아니라 dev 근거로 고른다.

### 3-1. 8-뷰 TTA 결과
기존 `features-v3tta`는 원본 + 전치(T) + 상하반전(UD) + 90° 회전(R90)의 4뷰를 쓴다. 이번에는 D4의 나머지 4뷰(R180, R270, 좌우반전, 반대각 전치)를 train 잡음 절반 249장에 대해 같은 masks-v3 규칙으로 분할했다(`seg_variant.py train seg 8 4 noisy`). 두 결과를 평균하면 D4 8뷰 평균이 된다(`eval_variant.py ...:avg_tta`).
**미완료.** 마무리하느라 125/249장에서 중단했다(속도 제한 상태로 약 9초/장). 당시 스크립트는 끝에서만 저장해서 중간 결과는 남지 않았다.
이후 `seg_variant.py`에 10장마다 `<이름>.partial.npz`로 저장하고 다시 실행하면 이어서 진행하는 기능을 넣었다.
다음 세션에서 바로 할 일(속도 제한 없이 MPS, 대략 10~15분 예상):
```bash
cd experiments/02-microstructure/0300-cellpose-finetuning
../../02-microstructure/0220-grain-cellpose/.venv/bin/python -u seg_variant.py train seg 8 4 noisy > var-seg8.log 2>&1
../../frameworks/tabpfn-v2/.venv/bin/python -u eval_variant.py var-seg-train-noisy-v4to8.npz:avg_tta
```
기준선 `oof_base_tabpfn.npy`(11.7549, 잡음 3분위 14.543)와 같은 fold로 비교된다. 잡음 3분위가 낮아지면 §5-B의 test 적용으로 진행한다.

## 4. TabPFN 최신 가중치 (준비만 함, 실행 안 함)
- 라이선스 검토: TabPFN-2.5 License v1.1과 TabPFN-3.5 License v1.0은 비상업적 사용(연구·평가)을 허용한다. 수정(Derivative)과 재배포도 라이선스 사본을 함께 주는 조건으로 허용한다. 금지되는 것은 호스팅/SaaS뿐이다. 대회 규정의 "CC BY-NC 등 비상업 허용 라이선스 가능, 사용·수정·재배포 제한 라이선스 불가"에 맞으므로 **사용 가능하다**(사용자 확인).
- 설치된 `experiments/frameworks/tabpfn-v2/.venv`의 tabpfn **9.1.0**이 v2.5, v2.6, v3, v3.5, v3.5-fast를 모두 지원한다. 새로 설치할 것은 없다.
- 가중치는 HF 게이트 저장소에 있다. **사용자가 본인 계정으로 약관에 동의한 뒤 받아야 한다.** 받은 파일은 `experiments/frameworks/tabpfn-v3/weights/`에 둔다(git 제외 대상 아님. 용량이 커서 커밋하지 말 것).
  - `Prior-Labs/tabpfn_3_5` → `tabpfn-v3.5-20260909.safetensors`(876MB), `tabpfn-v3.5-fast-20260909.safetensors`(334MB)
  - `Prior-Labs/tabpfn_3` → `tabpfn-v3-regressor-v3_default.ckpt`(233MB)
  - (선택) `Prior-Labs/tabpfn_2_5`, `tabpfn_2_6` 회귀 체크포인트. `harness.tabpfn_new`의 `files` 표에 파일명을 추가한다.
  - 예: `huggingface-cli login` 후 `huggingface-cli download Prior-Labs/tabpfn_3_5 tabpfn-v3.5-20260909.safetensors --local-dir experiments/frameworks/tabpfn-v3/weights`
- 출처: [TabPFN-2.5 LICENSE](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE), [TabPFN-3.5 모델 카드](https://huggingface.co/Prior-Labs/tabpfn_3_5), [TabPFN-3.5 LICENSE](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE), [TabPFN README](https://cdn.jsdelivr.net/gh/priorlabs/tabpfn@main/README.md), [TabPFN-2.5 논문](https://arxiv.org/pdf/2511.08667)
- 실행 준비 완료: `harness.tabpfn_new(version, seed, device, n_est)`, `eval_new.py`
  ```bash
  cd experiments/02-microstructure/0300-cellpose-finetuning
  ../../frameworks/tabpfn-v2/.venv/bin/python -u eval_new.py v3.5 features-v3tta
  ../../frameworks/tabpfn-v2/.venv/bin/python -u eval_new.py v3 features-v3tta
  ```
  기준선 11.7549와 짝비교한다. 이기면 n_est·seed를 늘리고, 공식 dev 절차(0.5×021 결합, 3개 재분할)로 확인한 뒤 `finalize_tabpfn.py`처럼 후보를 만든다(가중치 경로·버전만 바꾼다).

## 5. 남은 작업 (우선순위 순. §3-0의 추천 순서를 따른다)
A. **TabPFN-3.5 / 3 평가**(§4). 사용자가 가중치를 받아야 시작할 수 있다. 가중치가 아직 없으면 B부터 하고, 받는 즉시 A로 돌아온다. 하네스에서 이기면 n_est 32 × seed 3로 늘린다. 그다음 공식 dev 절차(3개 재분할, best021과 결합)로 확인하고 후보를 만든다.
B. **8-뷰 TTA 마무리**(§3-1의 명령 두 줄). 개선이면 test 잡음 절반(~500장)도 같은 방식으로 분할한다:
   `../grain-cellpose-220/.venv/bin/python -u seg_variant.py test seg 8 4 noisy` → `var-seg-test-noisy-v4to8.npz`.
   그다음 4-뷰 TTA 특징과 평균한 `features-v8tta-{train,test}.npz`를 만든다(`eval_variant.py`의 `avg_tta` 계산과 같게). evidence json에 채택 근거를 먼저 기록하고 후보를 만든다(`finalize_tabpfn.py 028 features-v8tta ...`).
   **깨끗한 절반에도 TTA**: `seg_variant.py train seg 4 1 clean`(뷰 1~3 = T, UD, R90. 출력 `var-seg-train-clean-v1to4.npz`) 또는 `seg 8 1 clean`. 원래 남은 작업 목록에 있던 항목인데 실행하지 않았다.
   주의: `eval_variant.py`의 `avg_tta`는 기존 행이 4뷰 평균이라고 가정하고 가중치 4를 준다. 깨끗한 행은 1뷰(v3)이므로 `(1×Xv3 + nv×Xm)/(1+nv)`로 계산하도록 모드를 하나 추가해야 한다.
C. ~~**결합 비중 0.5 → 0.6~0.7**(#4, `blend_weight_nested.py`).~~ **폐기(2026-10-09 리뷰): `blend_weight_nested.py`의 fold별 비중 선택은 nested가 아니다(바깥 훈련 fold 행의 OOF 예측을 만든 모델이 바깥 검증 fold 정답을 봤다). 재사용 금지. 결정립 비중은 0.75로 고정하고(후보 028/029), 후속 후보도 0.75 고정으로 비교한다(`dev_combine.py`). 아래 원문은 기록용.** 하네스 OOF(n_est 8, 500장 fold)로 얻은 결과다. 실제 후보 구성(n_est 32 × seed 3)의 공식 dev OOF로 같은 nested 선택을 다시 확인한 뒤 고정한다. `finalize_common.write_candidate`의 `W` 인자로 바꾼다. A나 B로 결정립 모델이 바뀌면 비중도 그 모델로 다시 고른다.
D. **특징 복원 CNN(distill) 확인**: −0.032는 잡음 범위다. seed 3개로 학습해 평균하거나 에폭과 폭을 바꿔 보고, 하네스 반복 수를 5로 늘려 확인한다. 이기면 공식 dev 절차로 확인한다. test 예측(`distill-a.npz`의 `test`)은 이미 있다.
E. **027(또는 최종 후보) 재현 패키지**(10위권이면 10/19~10/22 제출 대비): 018 패키지 + DINO-L 추가분 + cellpose 분할·TTA + 특징 + TabPFN + 결합. 여러 환경을 하나로 합쳐 같은 결과가 나오는지 검증한다. 시작하지 않았다. 점수 작업 뒤에 한다. 참고: `deliverables/reproduction-018-v176/`, `build_reproduction_package_155.py`.
F. (선택) LightGBM depth2와 TabPFN 결합, 전진 특징 선택 재실행(중단된 #11), TabPFN n_est 32 × seed 3를 하네스로 평가.

## 6. 파일 목록 (git에 없는 것은 인계 묶음에 들어 있음)
- git에 있음: `*.py`(`blend_weight_nested.py` 포함), `*.sh`, `*.log`, `*.npz`(`fidelity.npz`, `robust-train.npz`, `aug-v3-k2.npz`, `distill-a.npz`, `cand-train.npz`, `var-*.npz`), 이 문서
- 인계 묶음에만 있음(`.gitignore`의 `*.npy`, `*.pt`): `noise_train.npy`(train 잡음 추정값), `oof_*.npy`(OOF 캐시, **`oof_base_tabpfn.npy`는 모든 eval 스크립트가 쓴다**), `rps_train.npy`, `distill-a.pt`(복원 CNN 가중치)
- `noise_train.npy`는 다시 만들 수 있다(`harness.noise_of(ids)`와 같은 식). `oof_base_tabpfn.npy`는 `cv_predict(lambda: tabpfn(8,0,'mps'), X, y, reps=(0,1,2))`로 다시 만들 수 있다(MPS에서 약 1분).

## 7. 2026-10-08 18:30 ~ 10-09 세션: 미세조정 분할기(돌파구)와 후보 028·029
### 7-1. 결과 요약
- **Cellpose-SAM 미세조정**(`ft_cellpose.py`): 깨끗한 train 이미지(잡음 순위 80~250, 170장)에 무작위 blur 0~1.6 / contrast 0.55~1 / noise 0~24 gray 열화를 입력으로, 원본의 제로샷 masks-v3를 정답으로 학습(라벨 미사용, test 미사용). 16GB에서 OOM이 나서 블록별 gradient checkpointing을 쓴다. 250스텝 체크포인트 `ft-a1s250.pt`가 기준 모델.
  - 충실도(깨끗한 80장 열화): 제로샷 11.68 → **9.79**(원본 9.06). 하네스(잡음 절반 교체, 1뷰): 11.755 → **11.130**.
  - 더 긴 학습(s500 10.01), 전체 250장 학습 F300(하네스 11.271), 실제 잡음 분포 맞춤 heavy 학습 H300(11.169), 분할기 앙상블(11.162)은 모두 개선 없음. 세 모델 모두 잡음 이미지 결정립 수를 깨끗한 쪽의 ~0.90배로 잡는다(정보 한계로 판단).
  - 임계값(flow/cellprob) 조정: 영향 없음(결정립 수 비율 0.873~0.876).
- **TabPFN-3.5**: 하네스 단독 −0.03, v2와 평균 −0.085. dev에서 결정립 단독은 v3.5가 v2보다 0.1~0.2 좋다. 후보는 v2(n32×3)+v3.5(기본×3) 평균.
- **4뷰 TTA 평균만**은 하네스 −0.075 / dev(v3.5) +0.10으로 엇갈림(평가 잡음 ±0.1 수준).
- **4뷰 + 뷰 간 표준편차 16열**(분할·특징 추출 변동성 지표, 깨끗한 행은 NaN): 하네스 v2 10.909, v3.5 10.975, dev 고정 0.75 풀링 10.846(세 분할 모두 개선).
- 기각: 잡음 수준별 평균·분산 보정(`noise_correct.py`, −0.04), 제로샷 특징 병기(+0.03), 같은 잡음 절반끼리만 학습(+0.44), 인코더 임베딩(리지 단독 15.0, TabPFN 보조 +0.06), 객체 평균 기반 z 정규화(`zrobust.py`, 잡음 구간 상관 0.43→0.28), 히스토그램 GMM 2상 분율(잡음 구간 상관 0.28).
### 7-2. 제출
| 후보 | 구성 | dev(고정 0.75 풀링) | Public | 순위 |
|---|---|---:|---:|---|
| 027 | 기존 | 11.662(0.5) | 10.3957 | 11 |
| 028 | ft-a1s250 1뷰 + TabPFN v2+v3.5, 결정립 0.75 | 10.895 | **9.9286** | 7/125 |
| 029 | 028 + 4뷰 평균 + 뷰 표준편차 16열 | 10.846 | **9.8430** | 5/125 |
- 기록: `reports/candidate02{8,9}-public-submission.json`, 근거 `evidence-02{8,9}.json`, 특징 출처 `feat-*.provenance.json`, dev 결과 `dev-*.json`(입력 파일·SHA256·열 목록 포함).
### 7-3. 검증 규칙(2026-10-09 리뷰 반영)
- 결합 비중은 **0.75로 고정**해서 후보를 비교한다. dev OOF로 fold별 비중을 고르는 것은 nested가 아니다(`blend_weight_nested.py` 폐기). 0.75 자체도 dev를 본 뒤 정했으므로 선택 편향이 약간 남아 있다.
- 하네스·dev 차이 0.1 미만은 평가 잡음 범위로 본다. 두 평가 방향이 같고 세 분할 모두 개선일 때만 채택한다.
### 7-4. 다음 작업
1. 8뷰(D4 전체) 뷰 표준편차: train 잡음 절반 뷰 4~7 분할 중(`ftseg-ft-a1s250-train-noisy-n8-v4.npz`). 개선이면 test도 분할.
2. 깨끗한 절반 제로샷 4뷰(`seg_variant.py train seg 4 1 clean`)로 모든 행에 뷰 표준편차를 만드는 실험(대기열).
3. **재현 패키지**(Private 10위권이면 10/19~10/22 제출): best021(018 패키지+DINO-L) + 제로샷 분할·특징 + `ft_cellpose.py`(MPS 비결정성 때문에 가중치 `ft-a1s250.pt` 동봉) + `seg_ft.py` 4뷰 + `build_feat.py --vstd` + `finalize_ft.py`.
4. 최종 선택은 dev 근거로 한다(현재 029).

### 7-5. 공간 이질성(het) 잔차 보정 검증(2026-10-09, 사용자 요청)
- 출처: 공개 저장소 jinw00ch01/microstructure-hardness-prediction(notes/RESUME.md)의 아이디어만 참고(코드 미사용, Public 11.73→11.08 보고). 대회 규칙의 공유 조항을 고려해 코드·PPT 제출 시 출처를 명시할 것.
- 구현(`het_blocks.py`, `make_het.py`, `het_diag.py`): 비기공 결정립을 중심 좌표로 4×4(64px)·8×8(32px) 블록에 배정, 블록별 b=−2·log(mean(a^−1/2)), het=블록 간 b 표준편차(결정립 3개 이상 블록), N_eff=내부 결정립+0.5×경계 결정립. 마스크는 깨끗한 절반 제로샷 masks-v3, 잡음 절반 H300 1뷰(a1s250 마스크 미보관, 진단용 대체).
- (1) 030 하네스 OOF(v2 n8 시드 0·1·2 평균) 잔차와의 Spearman, 잡음 3분위(깨끗/중간/잡음):
  - het4 −0.047/−0.019/0.021, het8 −0.007/−0.057/−0.040, N_eff^0.25 0.032/0.055/−0.014 (유의 기준 n≈165에서 ~0.15)
  - 부분상관(tile_logA_std, tile8_logA_std, sizefield_std_s32, n_obj, 예측 통제): het4 −0.108/−0.060/−0.004, N_eff −0.060/−0.133/0.066 — 일관된 신호 없음
  - 깨끗한 3분위 표본 내 OLS 잔차~1+het4+N_eff^0.25: R² 0.006(n=133)
  - het4는 기존 tile8_logA_std와 ρ 0.89(N_eff는 −0.88): 이미 모델에 있는 정보다. 그 참가자의 이득은 이 정보가 없던 모델에서 나온 것으로 본다.
- (2a) 특징 열 추가(het4, het8, N_eff): 하네스 10.909/10.906/10.909 vs 030 10.883/10.895/10.901 → +0.015, **기각**. (2b) nested 잔차 offset: 표본 내 R² 0.006이라 표본 밖 이득이 불가능해 실행하지 않음.
- 추가 확인(다른 세션 요청, 원 저장소 게이트 raw noise < 9.5 gray = train 249장): 잔차 상관 het4 −0.014, het8 −0.039, N_eff^0.25 0.036. 표본 내 OLS R² 0.004. 하네스 fold 기준 cross-fit offset(다른 fold의 게이트 행으로 적합해 해당 fold에 적용, 3회 반복) 적용 시 게이트 구간 RMSE 8.773 → 8.881로 **악화**. 결론 동일: 우리 모델은 이 정보를 이미 tile/sizefield 특징으로 쓰고 있다.

### 7-6. 그 밖의 030 이후 실험(2026-10-09)
- 030 = 029 + 구조텐서 정렬도 8열(`stcoh.py`, 분할 없이 원본 이미지에서 계산). dev 고정 0.75 풀링 10.779(세 분할 모두 개선), **Public 9.8102, 5위**.
- 기각: 자기상관 특징 20개 추가(하네스 10.886 vs 10.893), 2배 해상도 미세조정 분할기(`ft_cellpose_x2.py`, 1뷰 11.397 vs 11.130), DINO-L PCA32 추가(10.936), H300 구성 전체(4뷰+표준편차+정렬도, 11.013) 및 a1s250 구성과의 예측 평균(10.892).
- 하네스 시드 잡음: 같은 특징에서 TabPFN 시드 0/1/2 차이 ~0.03. 030 기준 10.883/10.895/10.901.
- 진행 중: TabPFN-3.5 n_estimators 32(하네스 단일 시드 10.997→10.936) 공식 dev(`dev-a1s250n4vstd-stcoh-v35n32`).
- `augment=True`(Cellpose 내장 뒤집기 흐름장 평균) 1뷰: 하네스 11.061, 기존 4뷰 특징 평균(11.054)과 같은 수준. 4뷰×augment는 비용 16배라 미채택.
- TabPFN-3.5 n_estimators 32: dev 고정 0.75 풀링 10.777 vs 기본(8) 10.769 → 차이 없음, 기각(하네스 −0.06은 단일 시드 잡음).
- **잡음 이미지의 2상 정보 한계(진단)**: 합성 열화(가장 깨끗한 80장, blur 1/noise 17/contrast 0.85)에서는 2상 분율의 경도 상관이 0.619→0.613으로 유지되지만, 실제 잡음 3분위에서는 미세조정 분할 측정 0.443, 특징 복원 CNN(distill-a) 0.394로 같이 떨어진다(깨끗한 3분위 0.63). 크기 이질성·정렬 특징은 잡음 3분위에서도 상관이 유지된다. 결정립 밝기 z 분포에서 기지/2상 두 봉우리가 뭉개진다(기지 z 퍼짐 0.072→0.090, z 90% 분위 0.405→0.352). 합성 열화의 대비/잡음 비는 실제보다 오히려 낮다(3.62 vs 4.28). 분할 방법과 무관한 정보 손실로 보고 이 방향은 종료.
- 재현: `pipeline_grain.py`(단계별 구동) + `REPRO-030.md`. 조립·예측 단계가 030 특징과 CSV(SHA256 046102fb…)를 바이트 단위로 재현. 원본 이미지부터 전 단계 재실행(`repro-full-20261009/`)도 마스크(train 500·test 1000), v3 특징, 미세조정 4뷰 특징, 최종 88열 특징이 모두 완전 일치. 예측 단계 CSV 재현과 합치면 후보 030 결정립 경로는 `ft-a1s250.pt`와 best021 CSV가 주어지면 정확히 재현된다.

### 7-7. 커밋 상태(2026-10-09)
- git 제외(용량): 미세조정 가중치 `ft-*.pt`/`ftx2-*.pt`(각 1.2GB, `*.pt` 규칙), `emb-dinoL-train.npz`(candidate021-178/train-features.npz에서 재생성), `*.partial.npz`(중단된 H300 test 분할), `repro-full-20261009/*.npz`(재현 검증 중간물, 원본과 동일함이 compare.txt에 기록).
- **재현 패키지 제출 시 `ft-a1s250.pt`(SHA256은 아래 명령으로 기록)를 반드시 따로 보관·첨부할 것.**
- 최종 선택 미지정(현재 Public 최고 = 030, 권장도 030).
17f432a3f48bcc5c896751f94a9fe4b4343307123b46a0dd660878baae24fcf9  experiments/02-microstructure/0300-cellpose-finetuning/ft-a1s250.pt
