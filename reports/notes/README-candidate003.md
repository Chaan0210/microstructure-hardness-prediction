# 합성 미세구조 경도 예측 — candidate003

고정한 최종 모델은 **기존 특징 모델 50% + TinyCNN 25% + 26차원 물리 특징 커널 25%**입니다.
모델 선택은 development400의 CV로 했고, 설정을 고정한 뒤 공식 train500으로 최종 학습했습니다.
기존 holdout100의 새로운 성능 점수는 계산하지 않았습니다. test는 최종 추론·무결성 검증에만 사용합니다.

## 측정 결과

| 개발 분할 | 기존 candidate002 레시피 | candidate003 레시피 |
|---|---:|---:|
| 기존 5-fold | 13.982731 | 13.705696 |
| 고정 재분할 seed20261011 | 14.422193 | 13.777971 |
| 고정 재분할 seed20261012 | 14.303547 | 13.889479 |

새 두 분할의 10개 fold 모두 개선했습니다. 모든 성분을 각 외부 학습 fold에서 다시 학습했고,
커널 설정은 그 안의 3-fold에서 선택했습니다. 과거 OOF 예측을 새 fold에 재사용하지 않았습니다.
다만 같은 400장을 재사용한 **개발 분할 안정성 검사**입니다. 독립 holdout 검증, 유의성 또는 입상 보증이 아닙니다.
여러 가설을 비교한 선택 편향이 남아 있습니다. 두 재분할의 관측을 800개 독립 표본처럼 합치지 않았습니다.

첫 제출 candidate002의 Public RMSE는 **12.8862710295**, 당시 36명 중 32위였습니다.
2026-10-01 00:35 KST, 제출 ID `b3eef90d-3d91-4887-97bb-d9a4ff1be782`.
이는 부모 작업에서 확인한 기록이며 candidate003의 공개 점수가 아닙니다.
Public 점수는 특징·가중치·모델 설정을 고르는 학습 신호로 사용하지 않았습니다.

## 산출물

- 최종 모델: `artifacts/final-model-017/` — baseline.joblib, kernel.joblib, cnn.pt, decision.json, report.json
- 새 추론 파일: `artifacts/submission-candidate-003.csv`
- 추론 해시·성분·시간: 같은 이름의 `.manifest.json`, `.components.csv`
- CSV 검증: `reports/validation/submission-candidate-003-validation.json`
- 모델 재로딩 검증: `reports/submissions/candidate003-reload-verification.json`
- 종합 결과: `reports/submissions/final-candidate003-summary.json`
- 모델 선택 결정: `reports/summaries/research-decision-017.json`
- 개발 재분할 검증: `artifacts/frozen-validation-016/report.json`, 각 seed의 OOF·분할·모델
- 이전 best: `artifacts/run-005/best_model.joblib`, `artifacts/submission-candidate-002.csv` — 원본 보존
- 소스 묶음: `deliverables/source-code-candidate003.zip` — 데이터·가중치 제외, 로컬 보관용

candidate001은 음수 외삽이 있었던 진단용 원본입니다. 제출용으로 사용하지 마세요.
이 작업의 candidate003 생성·검증 코드는 업로드하지 않습니다.

## 저장 모델로 추론 재현

프로젝트 루트에서 기존 uv 가상환경을 사용합니다. 출력 파일은 새 이름이어야 합니다.

```sh
.venv/bin/python -I -B candidate_pipeline.py predict --model artifacts/final-model-017 --output artifacts/submission-reproduced-003.csv
.venv/bin/python -I validate_submission.py --submission artifacts/submission-reproduced-003.csv --report reports/submission-reproduced-003-validation.json
.venv/bin/python -I -B verify_candidate003.py
.venv/bin/python -I -m unittest discover -s tests -v
```

추론은 공식 test PNG, sample_submission.csv와 `artifacts/test-prediction-features.npz`를 사용합니다.
이 캐시는 기존 모델의 795개 특징을 담은 **추론 전용** 캐시이며 학습에는 사용하지 않습니다.
모든 이미지의 해시와 ID·순서·특징 스키마를 확인합니다. 새 물리 특징은 원본 PNG에서 다시 추출합니다.
모델·소스·패키지 버전이 고정 기록과 다르면 중단합니다. 신뢰하는 로컬 joblib만 로드하세요.
CPU로도 추론할 수 있습니다. MPS 수치 연산은 완전한 비트 단위 재현을 보장하지 않습니다.

## 학습 재현

동일 공식 데이터와 아래 선행 산출물이 있는 프로젝트 사본에서, 새 출력 경로로 실행합니다.

```sh
.venv/bin/python -I -B candidate_pipeline.py fit --panel run-012 --output artifacts/final-model-reproduced
```

필수 선행 산출물은 run001 split_manifest, run005 모델·OOF·report 및 candidate002,
run012 개발 특징·report, frozen-validation-016 report입니다. 최종 커널 설정은 development400에서만 선택해
`decision.json`에 먼저 고정한 뒤 train500 라벨을 읽습니다. 기존 baseline은 이미 train500으로 학습된 파일을
해시 검증 후 복사합니다. CNN과 커널만 고정 설정으로 전체500에 다시 학습합니다.

빈 작업 사본에서 연구 과정까지 재현할 때의 주요 순서:

```sh
UV_CACHE_DIR=.uv-cache UV_PYTHON_DOWNLOADS=never uv venv --python /opt/homebrew/bin/python3 --no-managed-python .venv
UV_CACHE_DIR=.uv-cache UV_PYTHON_DOWNLOADS=never uv pip sync --python .venv/bin/python requirements-cnn.lock
.venv/bin/python -I run.py train --data data --output artifacts/run-001 --seed 20260930 --jobs 2
.venv/bin/python -I run_v2.py train --data data --output artifacts/run-003 --seed 20260930 --jobs 2
.venv/bin/python -I finalize.py
.venv/bin/python -I stabilize.py
.venv/bin/python -I run_v2.py predict --data data --run artifacts/run-004 --output artifacts/submission-candidate-001.csv
.venv/bin/python -I diagnose_and_predict.py --output artifacts/submission-candidate-002.csv
.venv/bin/python -I research_grains.py
.venv/bin/python -I cnn_experiment.py bench
.venv/bin/python -I cnn_experiment.py cv
.venv/bin/python -I cnn_controls.py 128x80
.venv/bin/python -I compact_regression.py
.venv/bin/python -I phase_regression.py
.venv/bin/python -I -B frozen_validation.py --feature-runs run-012 run-014 --output artifacts/frozen-validation-016
.venv/bin/python -I -B candidate_pipeline.py fit --panel run-012 --output artifacts/final-model-017
```

설치는 승인된 환경에서만 실행하며 프로젝트 `.venv`와 `.uv-cache`로 한정합니다.
연구 재현 명령 중 초기 baseline 스크립트는 당시의 holdout 진단도 재현합니다. 현재 후보 모델 선택에는
그 진단을 사용하지 않았습니다. 정확한 역사·실패·측정값은 `reports/notes/research-history-through-016.md`에 보존했습니다.
출력 폴더는 덮어쓰지 않습니다. 재학습된 MPS 체크포인트의 바이트 해시는 달라질 수 있습니다.

## 모델과 가설

기존 모델은 밝기·텍스처·스펙트럼·형태 특징의 ExtraTrees/Ridge 결합입니다.
Ridge는 fold-train MinMax 범위 제한과 표준화를 사용해 불안정한 외삽을 줄입니다.
TinyCNN은 128px BOX 입력, 4개 합성곱 단계, mean/std pooling, 약9.2만 파라미터입니다.
80epoch AdamW/cosine, 마지막5epoch 가중치 평균, seed20261003, batch32, train-only D4 augmentation을 사용합니다.
입력과 타깃 평균/표준편차는 해당 학습 fold에서만 계산합니다. 사전학습 가중치를 사용하지 않았습니다.

26개 물리 특징은 결정립 크기·분산, 상/기공 비율, 종횡비·정렬, 내부 대비의 proxy12개,
가로·세로·대각선 경계 사이 거리의 6개 기술통계에 대한 다중 설정 중앙값/범위12개, 밝기 평균/표준편차2개입니다.
이는 휴리스틱 분할이며 실제 결정립·기공 또는 합성 생성기의 잠재값을 복원했다고 보장하지 않습니다.
RBF KernelRidge는 특징군별 크기를 보정하고 train 거리 중앙값으로 bandwidth를 설정합니다.
9개 bandwidth/alpha 조합은 내부 CV에서만 비교합니다. 효과 방향을 강제하지 않습니다.

128px80epoch, 256px40epoch, 다중 단계 moment pooling, 진폭 보존 residual CNN,
26차원 커널, 상분리 추가46차원, 기존795특징 블록 균형 커널을 비교했습니다.
상분리와 더 큰 CNN이 일관되게 좋아지지는 않았습니다. `reports/summaries/research-stage-results.json` 참고.
CNN 단독 성능보다 기존 특징 모델과의 상보성이 유용했습니다.
양끝 경도에서 평균 쪽 수축이 남아 있으며, 이번 결과가 기술적 성능 한계를 입증하지는 않습니다.

## 규정·데이터·환경

[공식 대회 및 규정](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai).
개인전, RMSE, Public30%/Private70%, 하루5회. 확인된 제출 마감2026-10-19 10:00 KST,
코드·자료 마감2026-10-22 10:00 KST. 외부 데이터·test 학습·원격 모델 API 금지.
AI 코딩 보조의 명시적 금지는 확인되지 않았으며 제출 예측 모델은 로컬에서 실행됩니다.
등록과 추가 동의가 완료되어 공식 데이터에 접근했습니다. 추가 개인정보는 사용자가 직접 제출했습니다.
공식 open.zip에는 train500/test1000 256×256 흑백 PNG와 ID,hardness CSV가 있습니다.
ZIP SHA256 `69c96ff9412e96eeb05d69ebeea050f65c7d03274406fa904f3929aef1b1b881`.
아카이브·이미지 무결성은 `reports/validation/data-audit.json`에 기록했습니다.

공식 설명 PDF의 Hall–Petch·상분율·기공 메커니즘은 확인된 설명을 참고했습니다.
이 환경의 직접 PDF 요청은403이어서 원문 파일을 보유하지 않습니다. 해당 효과의 부호를 단정하지 않습니다.

M1 Pro, RAM16GiB, Python3.13.3, uv0.12.10, torch2.14.0 MPS, CPU 작업수2.
승인된 무료 오픈소스만 사용했고 외부 가중치·원격 모델·유료 서비스·결제는 없습니다.
의존성 및 라이선스: `requirements-cnn.lock`, `reports/provenance/software-licenses.json`.
원본 코드는 이 프로젝트에서 작성했습니다. Git commit/push·공개 배포는 하지 않았습니다.
