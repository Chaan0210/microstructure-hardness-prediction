# Historical research notes (superseded by root README)

# 합성 미세구조 경도 예측 — 로컬 CPU 제출 후보

공식 참가 완료와 공식 데이터 다운로드를 확인했습니다. 모델 개발 및 검증을 완료하고 첫 로컬 제출 후보를 준비했습니다.
**후보 002의 첫 제출 완료: Public RMSE 12.8862710295, 당시 36명 중 32위.**
2026-10-01 00:35 KST, 제출 ID `b3eef90d-3d91-4887-97bb-d9a4ff1be782`. 별도 복구 작업에서 검증한 결과를 부모 지시로 기록했습니다. 추가 제출은 승인되지 않았습니다. 상세 `reports/submissions/first-public-submission.json`.

## 현재 결과

| 단계 | 개발 OOF RMSE | holdout RMSE | CPU 소요 |
|---|---:|---:|---:|
| 평균값 예측 | 18.0939 | 16.1871 | 기준 실험에 포함 |
| 최초 Ridge + Extra Trees | 14.0453 | 13.2374 | 110.4초 |
| 추가 모델 11개 탐색 | 기존 후보 유지 | 변경 없음 | 25.3초 |
| 형태 특징 795개 | 14.2325 | 평가 보류 | 222.6초 |
| 원본 75:25 결합 (외삽 문제로 대체) | 13.9821 | 13.1224 (반복 진단) | 6.0초 |
| 최종 입력 안정화 결합 | **13.9827** | **추가 평가 보류** | 3.6초 |

최초 holdout은 개발용 모델 선택에서 분리한 100장이었습니다.
13.1224는 안정화 이전 원본 모델이 같은 holdout을 재사용한 진단이며, 현재 최종 모델의 평가값이 아닙니다.
최종 안정화 모델에 대해서는 요청대로 holdout을 추가 열람하지 않았습니다.
run-003/004에서 holdout 계산을 보류했고 개발 OOF로 최종 설정을 고정한 뒤 한 번 진단했습니다.
개발 OOF 개선은 약 0.45%로 작으며, 여러 설정을 비교한 선택 편향이 있을 수 있습니다.
안정화 이전 반복 holdout의 조건부 부트스트랩 95% 구간은 11.35–14.84입니다. 작은 개선의 통계적 유의성을 주장하지 않습니다.

## 제출 후보 및 재현

후보: artifacts/submission-candidate-002.csv
원본 candidate-001은 음수 외삽이 발견된 진단용 파일이며 제출 대상으로 추천하지 않습니다.
최종 모델: artifacts/run-005/best_model.joblib
최종 선택: artifacts/run-005/selection.json
파일 검증: reports/validation/submission-validation.json
검증 명령: .venv/bin/python -I validate_submission.py
전체 환경: reports/provenance/environment.json / requirements.lock

현재 프로젝트 루트에서 저장 모델로 새 파일에 추론:

```sh
.venv/bin/python -I run_v2.py predict --data data --run artifacts/run-005 --output artifacts/submission-reproduced.csv
```

출력 파일이 이미 있으면 덮어쓰지 않고 중단합니다.
입력/출력 경로는 프로젝트 기준 상대 경로만 받습니다. 파일 밖 심볼릭 링크를 사용하지 마세요.
소스 해시와 패키지 버전이 학습 당시와 다르면 추론을 중단합니다.
joblib은 코드 실행 가능 형식이므로 직접 생성한 신뢰 가능한 모델만 사용하세요.

새 작업 사본에서 같은 공식 데이터와 소스를 사용한 전체 재현 순서:

```sh
UV_CACHE_DIR=.uv-cache UV_PYTHON_DOWNLOADS=never uv venv --python /opt/homebrew/bin/python3 --no-managed-python .venv
UV_CACHE_DIR=.uv-cache UV_PYTHON_DOWNLOADS=never uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/python -I -m unittest discover -s tests -v
.venv/bin/python -I run.py train --data data --output artifacts/run-001 --seed 20260930 --jobs 2
.venv/bin/python -I run_v2.py train --data data --output artifacts/run-003 --seed 20260930 --jobs 2
.venv/bin/python -I finalize.py
.venv/bin/python -I stabilize.py
.venv/bin/python -I run_v2.py predict --data data --run artifacts/run-005 --output artifacts/submission-candidate-002.csv
```

빈 작업 사본이 필요합니다. 실험 디렉터리를 덮어쓰지 않습니다.
설치는 사용자가 승인한 환경에서만 하며 전역 Python을 변경하지 않습니다.
improve.py의 run-002는 추가 회귀기 탐색 기록이고 최종 후보 생성의 필수 단계는 아닙니다.

## 규정과 데이터

[공식 규정·동의사항](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai), 2026-09-30 확인.
- 개인전, Python, 무료 참가. 사용자가 추가 개인정보 동의를 직접 완료한 뒤 참가 완료를 검증했습니다.
- train 500장 / test 1,000장. 256×256 8비트 흑백 PNG.
- train.csv, sample_submission.csv: ID,hardness.
- 평가 RMSE = sqrt(mean((prediction - target)^2)), 낮을수록 좋음.
- Public 30%, Private 70%. 최종 평가는 Private 기준. 하루 최대 5회 제출.
- 제출 마감 2026-10-19 10:00 KST (01:00 UTC).
- 코드·자료 마감 2026-10-22 10:00 KST (01:00 UTC). 설명자료는 자유 형식 PDF로 안내되어 있습니다.
- 외부 데이터, test 학습, 원격 API로만 접근 가능한 예측 모델 금지.
- AI 코딩 보조의 명시적인 허용/금지는 확인되지 않았습니다. 제출 모델은 외부 API 없이 로컬 실행합니다.
- 대회별 외부 데이터 금지 규정을 일반 규정의 외부 데이터 문구보다 우선 적용했습니다.
- 데이터/코드의 비공식 공유, 비참가자 데이터 제공 금지에 유의하세요.

공식 다운로드 버튼으로 받은 open.zip(73,877,373바이트)은 downloads/open.zip에 보존했습니다.
브라우저 기본 다운로드 위치에 받은 단일 파일을 이 프로젝트로 옮겼으며 다른 다운로드 파일은 읽지 않았습니다.
SHA256: 69c96ff9412e96eeb05d69ebeea050f65c7d03274406fa904f3929aef1b1b881
CRC와 예상 파일 1,502개, 모든 CSV/PNG 형식 검사 통과. 상세 reports/data-audit.json.
공식 자료의 압축 구조:

```text
data/train.csv
data/sample_submission.csv
data/train/TRAIN_000001.png ... (500장)
data/test/TEST_000001.png ... (1,000장)
```

## 방법과 검증

기본 특징 519개: 밝기, 다중 스케일 기울기/라플라시안, 방향성, FFT, 자기상관, GLCM, LBP,
어두운 연결영역의 크기·형태·정렬·거리 변환.
추가 특징 276개: 밝은 연결영역과 경계 내부의 크기·분포, 다중 크기 열기/닫기 차이, 국소 대비.
이 영역들은 영상 휴리스틱이며 실제 결정립·기공을 정확하게 분할했다고 보장하지 않습니다.

최종 예측 가중치:
- 기본 특징의 Extra Trees: 37.5%
- 기본 특징의 MinMaxScaler(clip=True) + StandardScaler + Ridge(alpha=100): 37.5%
- 확장 특징의 MinMaxScaler(clip=True) + StandardScaler + Ridge(alpha=1000): 25%

train만 사용해 exact pixel 및 90도 회전/반사 동일 영상을 그룹화했습니다(500개 그룹).
20% 고정 holdout과 나머지 5-fold GroupKFold, seed=20260930.
학습형 전처리는 각 학습 fold 내부에서만 fit합니다.
train 단계는 test 폴더/sample_submission을 읽지 않습니다.
test는 다운로드 무결성 검사 및 최종 선택 이후 추론에만 사용했습니다.
test 기반 통계 정규화, 의사라벨, 튜닝, 학습은 없습니다.

학습 영상의 32×32 정규화 축소영상 유사도 검사에서 상관 0.95 이상 쌍이 없었습니다.
이는 정렬된 근접 중복의 보조 검사이며, 알려지지 않은 생성 원본 계열의 독립성을 증명하지 않습니다.

## 실패·오류 분석 및 해석

SVR, Kernel Ridge, PCA, 특성 선택, 강한/약한 정규화, 추가 Extra Trees 설정은 기본 앙상블을 넘지 못했습니다.
형태 특징만 늘리면 Ridge가 악화됐고 강한 정규화(alpha=1000)가 필요했습니다.
상호작용 다항식과 분위수 변환도 개선하지 못했습니다.
실패 결과까지 각 run의 report.json, progress.json과 reports/run-*.log에 보존했습니다.

최종 개발 OOF의 경도 사분위별 RMSE는 약 16.56 / 8.53 / 8.53 / 19.03입니다.
낮은 경도는 과대, 높은 경도는 과소 예측하는 평균 쪽 수축이 주요 약점입니다.
이 분석은 안정화 이전 후보 기준입니다. 이후 입력 외삽 안정화만 개발 CV로 추가 검증했습니다.
상세 artifacts/run-004/development-error-analysis.json.

[공개 제출 리더보드](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai?section=leaderboard&tab=submission)
확인 당시 35명, 1위 10.8711, 10위 12.1101, 34위 14.0892, 마지막 195.0305였습니다.
Public과 로컬 표본이 다르므로 이 결과로 우리 순위를 추정하지 않습니다.

## 환경·검증 기록

Apple M1 Pro, CPU 8코어, RAM 16GiB, macOS arm64, Python 3.13.3, uv 0.12.10.
사용자 승인 후 무료 numpy/scipy/Pillow/scikit-learn/scikit-image 및 필수 의존성만 프로젝트 .venv에 설치했습니다.
캐시도 .uv-cache로 한정했습니다. CPU 병렬 작업 수 2. PyTorch/MPS는 설치·검증하지 않았습니다.
8개 공학적 테스트 통과: 지표, 경로, 특징 경계조건, 중복 그룹 분리, 회귀기 API/저장·복구.
공학용 fixture는 대회 모델 학습이나 성능 주장에 사용하지 않았습니다.

보고서와 모델은 각 run에 고유 경로로 저장하고 기존 best를 보존했습니다.
개인/회사 자료·인증파일·전역 기억을 탐색하지 않았고, 결제·전역 설치·Git commit/push·배포를 하지 않았습니다. 리더보드에는 별도 승인된 후보 002를 1회 제출했습니다.
추가 개선이나 MPS 실험은 이 첫 후보와 구분해 기록해야 합니다.

## 외삽 진단과 최종 안정화

원본의 TEST_000013 예측은 -66.161이었습니다. 공식 문구는 합성 proxy의 허용 범위를 명시하지 않지만, train 경도는 모두 양수(151.9798–264.1213)였습니다.
영상의 Otsu 연결요소 최소 면적은 train에서 4–5였으나 이 이미지에서는 78로, 학습 표준편차의 743.6배였습니다.
동일 영상의 재계산은 원본 예측과 일치하여 전처리 불일치가 아니라 선형 외삽 문제로 확인했습니다.
해당 원본 성분은 ET 202.788 / 기본 Ridge -354.327 / 확장 Ridge -37.335였습니다.

각 학습 fold에서만 fit한 MinMaxScaler(clip=True)로 Ridge 입력 범위를 제한했습니다.
개발 OOF: 원본13.982113, train-label 출력 clipping13.982113(실제 잘린 값0), 입력 안정화13.982731.
출력 clipping은 개발 검증에서 근거를 얻지 못해 사용하지 않았습니다. 입력 안정화는 CV 차이가 0.000618로 미미하면서 원인을 직접 제한합니다.
제한값은 test에서 추정하지 않습니다. 다만 원래 학습 범위를 벗어난 유효한 신호를 줄일 수 있는 절충이 있습니다.
전체 test 진단은 형식·수치 안정성 확인에만 사용했으며 모델 fit이나 임계값 선택에는 사용하지 않았습니다.
상세: reports/summaries/extrapolation-diagnostic.json, reports/summaries/full-test-stability-diagnostic.json, artifacts/run-005/report.json.

진단용 test 특징 캐시 artifacts/test-prediction-features.npz는 추론 재현 전용입니다. 학습에 사용하지 마세요.

최종 파일 검증 완료: 1,000행, ID/순서 일치, NaN/Inf/음수 0개. 예측 범위163.6834–237.0800.
모델 재로딩 후 전체1,000개 예측 차이는 최대5.69e-14로 허용오차1e-10 이내입니다.
안정화 후보의 전체 test 특징 재계산은 CPU2작업으로192.16초였고, 현재 실행 중인 학습/추론 작업은 없습니다.

## 첫 제출 이후 제한된 특징 연구: run-006

공개 점수는 기록에만 사용했고 선택 기준·특징 조정에 사용하지 않았습니다.
개발 400장에만 watershed 결정립 후보 크기/종횡비/정렬, 3단계 밝기 상분율,
작고 조밀한 어두운 객체의 기공 후보, 국소 방향성, 평활화 변화량 등 157개 특징을 추출했습니다.
공식 페이지의 생성 세부 설명은 이번 세션 웹 도구에서 접근하지 못했습니다.
따라서 실제 생성 공식을 역추정했다고 주장하지 않으며 모든 분할은 휴리스틱입니다.
Holdout 100장의 추가 평가와 테스트 데이터 접근은 하지 않았습니다.

| 개발 CV | 기존 candidate002 | 기존 75% + 새 특징 Ridge 25% |
|---|---:|---:|
| 기존 고정 folds | 13.982731 | 13.871297 |
| 반복 seed 20261001 | 14.656745 | 14.409842 |
| 반복 seed 20261002 | 14.490506 | 14.239970 |

새 특징 단독 Ridge는 14.334263, 기존+새 특징을 한 Ridge로 학습하면 14.005825였습니다.
이는 큰 파라미터 탐색 없이 고정한 4개 가설 후보를 비교한 결과입니다.
고정 folds의 개선 0.111434는 실행 전 진급 기준 0.15에 못 미칩니다.
기존 folds의 조건부 짝지은 bootstrap 개선 구간은 -0.02073~0.25235로 0을 포함합니다.
이 구간은 모델 선택 후 불편 추정 구간이 아니며, 반복 CV도 같은 400장을 공유합니다.
반복 CV에서는 일관된 작은 개선이 있으므로 유망한 신호지만 확정적인 성능 향상으로 주장하지 않습니다.

**기존 run-005와 제출된 candidate002를 유지했습니다. 새 제출 파일이나 추가 업로드는 없습니다.**
기존 모델과 CSV 해시 보존 확인. 추가 설치·비용 없음. CPU 2작업, 특징 40.69초, 전체 실험 61.65초.
공학 검증 10개 통과(1.207초), 실제 대회 성능 검증과 구분합니다.

분할 진단: `reports/figures/grain-segmentation-qa.png`. 흐린 경계를 놓치거나 길쭉한 결정립을 나누는 문제가 있습니다.
예를 들어 개발 TRAIN_000002의 조각 수는 평활화 수준에 따라 260에서 141로 변합니다.
양끝 경도 구간 오차도 약 16.55/18.85로 여전히 크고, 새 특징-잔차 상관은 최대 절댓값 약 0.123으로 약합니다.
후속 진행 근거는 공식 생성기 세부사항(제공될 경우) 또는 더 안정적인 경계 추정입니다.
다른 접근인 MPS 소형 CNN/표현학습은 현재 없는 PyTorch 등의 무료 패키지 설치 승인과 별도 실행시간 검증이 필요합니다.
500장뿐인 훈련 데이터에서 성공을 보장하지 않으므로 이번 작업에서 설치·실행하지 않았습니다.

재현: 빈 `artifacts/run-006` 경로가 있는 같은 프로젝트 사본에서
`.venv/bin/python -I research_grains.py`.
이미 존재하는 run은 덮어쓰지 않습니다. `grain_features.py`가 특징 구현이며
`artifacts/run-006/plan.json`에 사전 가설·후보·seed·진급 기준이 있고,
`report.json`, `development_oof.csv`, `repeat-*.npz`에 결과가 있습니다.
테스트: `.venv/bin/python -I -m unittest discover -s tests -v`.

## 승인된 MPS 소형 CNN 실험: run-007

사용자 승인 후 공식 PyPI의 torch 2.14.0 및 필수 의존성만 기존 uv `.venv`에 설치했습니다.
새 패키지: filelock 4.0.7, fsspec 2026.9.0, jinja2 3.1.6, markupsafe 3.0.3,
mpmath 1.3.0, setuptools 84.0.0, sympy 1.14.0, torch 2.14.0, typing-extensions 4.16.0.
기존 `requirements.lock`은 보존했고 확장 환경은 `requirements-cnn.lock`에 고정했습니다.
`.venv` 약 768MB, `.uv-cache` 약 761MB이며 하드링크에 의해 실제 중복 사용량과 다를 수 있습니다.
PyTorch/MPS 미설치라고 적힌 이전 절은 최초 CPU 실험 당시 기록입니다.
설치/캐시는 프로젝트 내부, 추가 비용·외부 GPU·외부 데이터·사전학습 가중치 사용 없음.

128×128 흑백, 파라미터 91,905개 CNN을 무작위 초기화했습니다.
Conv/GroupNorm/SiLU 4단계와 특징 평균·표준편차 pooling을 사용합니다.
학습 fold에서만 이미지·라벨 평균/표준편차를 추정하고, 학습 배치에만 회전/반사를 적용합니다.
개발 400장의 기존 동일 folds, batch32, AdamW lr0.002, weight decay0.01,
cosine LR, 고정40 epoch 및 마지막5 epoch 가중치 평균.
검증 점수로 checkpoint나 epoch를 선택하지 않았고 holdout/test를 읽지 않았습니다.
MPS는 seed를 고정해도 장치/버전 간 비트 단위 재현을 보장하지 않습니다.

짧은 학습 벤치마크: CPU 0.06197초/단계(3회), MPS 0.01483초/단계(20회),
동일 batch32에서 약4.18배. MPS 약2,158이미지/초는 반복된 한 배치의 짧은 측정이며 전체 학습 처리량은 아닙니다.
프로세스 peak RSS는 벤치마크 약615MB, CV 약592MB; fold 종료 시 MPS driver allocation 약95MB.
실제 온도 센서는 읽지 못했고 pmset에는 열/성능 경고 기록이 없었습니다. 장시간 발열 안전성을 보증하지 않습니다.

| 동일 개발 CV | RMSE |
|---|---:|
| 기존 candidate002 | 13.982731 |
| CNN 단독 | 14.471544 |
| 기존75% + CNN25% | 13.874365 |
| 기존50% + CNN50% | 13.921562 |

5-fold 총36.01초. 25% 결합은5개 중4개 fold에서 개선됐지만 개선폭0.108366이 사전 기준0.15에 못 미칩니다.
조건부 bootstrap 개선 구간 -0.03755~0.25805는0을 포함합니다. 선택 후 불편한 신뢰구간은 아닙니다.
CNN/기존 모델 잔차 상관0.915로 서로 보완하는 정보가 제한적이었습니다.
CNN 단독은4개 fold에서 열세였고 train RMSE12.30~13.78 대 validation13.68~15.80으로 일반화 격차가 있습니다.
128픽셀 축소가 미세 경계를 손실했을 가능성은 가설이며 이번 실험으로 입증하지 않았습니다.
**기존 run-005/candidate002를 유지하고 한 번의 합리적 실험 세트에서 종료했습니다.**
추가 제출·테스트 추론·새 후보 파일 생성 없음. 기존 best/CSV 해시 보존 확인.

재현(새 사본의 결과 경로가 비어 있어야 함):
```sh
UV_CACHE_DIR=.uv-cache UV_PYTHON_DOWNLOADS=never uv pip sync --python .venv/bin/python requirements-cnn.lock
.venv/bin/python -I cnn_experiment.py bench
.venv/bin/python -I cnn_experiment.py cv
.venv/bin/python -I -m unittest discover -s tests -v
```
환경 설치는 승인된 환경에서만 실행합니다. 설치를 제외한 실험은 네트워크 없이 실행됩니다.
새 실험 경로는 덮어쓰지 않습니다. 12개 공학 검증 통과(1.612초).
MPS OOF와 저장 체크포인트의 CPU 재로딩 비교는 `reports/validation/cnn-reload-validation.json`에 기록했습니다.
수동 진단 첫 시도에서 ID list 인덱싱 오류가 있었고 배열 변환으로 해결했으며 학습 결과에는 영향이 없습니다.
공식 구현 참고: https://docs.pytorch.org/docs/2.14/notes/mps.html

## 후속 진단 1회: diagnostic-008 (새 학습 없음)

기존 개발 OOF·로그·CNN 체크포인트·소스만 읽었습니다. 이미지/holdout/test 접근, 추가 학습,
가중치 다운로드, 새 제출 후보와 업로드가 없습니다. 기존 run-005 및 candidate002는 그대로 유지합니다.
이전 RMSE0.15 기준은 임의의 운영상 선별 기준이며 통계적 유의성이나 기술적 성능 상한을 뜻하지 않습니다.

기존 모델 B, 형태 특징 단독 Ridge G, CNN C를 세 성분으로 정의했습니다.
두 기존 개선모델은 각각0.75B+0.25G와0.75B+0.25C입니다.
고정 가중치 세 가지를 계산 전 `artifacts/diagnostic-008/plan.json`에 기록했습니다.

| 조합 B:G:C | 개발 OOF RMSE | B 대비 개선 | 조건부 개선95%구간 |
|---|---:|---:|---:|
| 1:0:0 (기존) | 13.982731 | — | — |
| 1/3:1/3:1/3 | 13.807141 | 0.175590 | -0.13316~0.48091 |
| 0.50:0.25:0.25 | 13.792532 | 0.190199 | -0.04319~0.41950 |
| 0.75:0.125:0.125 | 13.844061 | 0.138670 | 0.02192~0.25365 |

형태/CNN 단독 잔차 상관0.8778, 기존 두 개선 결합모델 잔차 상관0.9918.
그러나 기존 모델 대비 보정값(G-B, C-B) 상관은0.2122로 낮아 동시에 결합할 근거가 있습니다.
낮은 보정 상관만으로 성능 향상을 보장하지는 않습니다.
세 조합 모두 점추정 개선이 있으나, 성분 모델 자체와 과거 설정을 같은400장 OOF에서 골랐습니다.
따라서 지금 가중치를 사전 고정했어도 전체 과정은 중첩된 선택을 포함한 사후 분석입니다.
표의 bootstrap은 선택 편향/학습 불확실성을 반영하지 않습니다. 가장 좁은 구간만 골라
유의성을 주장하거나 독립적인 개선 확인으로 해석해서는 안 됩니다.

CNN 마지막10 epoch 학습 RMSE 기울기는 모든fold에서 음수(-0.0396~-0.0681/epoch)였습니다.
21~30 대비31~40 epoch 평균 손실도 각fold에서 약0.79~0.98 감소했습니다.
최종 train/validation RMSE 차이는 -0.10~2.95입니다. 학습부족과 일반화 문제 모두 가능하며,
검증값은 최종 가중치 평균 모델에서만 기록돼 epoch별 검증곡선이나 과적합 시작점을 알 수 없습니다.
온라인 학습 손실은 augmentation/dropout/변하는 가중치로 측정돼 최종 eval 손실과 직접 같지 않습니다.

타깃 평균/표준편차는5개 저장 체크포인트 모두 해당320개 train-fold 라벨 재계산과 일치했습니다.
입력 평균/표준편차도 코드상train-fold에서만 fit했고 저장 값이 유한/양수임을 확인했으나,
이번에는 이미지를 읽지 않아 픽셀 통계를 다시 계산하지 않았습니다.
CNN 예측std11.66 대 타깃std18.03만 보고 표준화 오류/과소적합이라 단정할 수 없습니다.
조건부 평균 회귀도 예측 분산이 작을 수 있고 진단용 calibration slope는 약0.926입니다.

256→128 BOX 축소는2×2 평균으로 일부 미세 정보를 비가역적으로 버립니다.
이후4번 stride2 합성곱으로8×8 특징맵을 만들고 전역pooling합니다.
얇은 경계나 작은 기공 정보 손실 가능성은 있으나, 실제 손실량과 성능 영향은
이번 로그/모델만으로 측정할 수 없습니다. 픽셀 비교나128/256 대조실험을 아직 하지 않았습니다.

**기술적 성능 한계를 입증한 것이 아니라 소형 CNN 기준선 하나를 확인한 상태입니다.**
다음 유효한 가설은 (A) 고정128픽셀80epoch로 최적화 예산을 늘리거나,
(B) 고정256픽셀40epoch로 미세구조 해상도 손실을 검증하는 것입니다.
둘을 동시에 바꾸지 않고 하나씩, 고정 앙상블을 유지해야 합니다.
A는 기존36초에서 약72초(실무상1~2분), B는 연산량 추정약108~216초(약2~4분)입니다.
B의 실제 속도/메모리는 미측정이므로 먼저 짧게벤치마크해야 합니다.
80epoch cosine schedule은 학습길이뿐 아니라 LR경로도 바꾸므로 순수epoch효과라고 해석하지 않습니다.
가설 선택 후 두개의 사전 지정된 개발 전용 groupedfold seed에서 같은 설정을 검증하면
분할 민감도를 볼 수 있으나 독립 데이터 검증은 아닙니다. 추가CNN시간은 선택 실험약2배+기존모델refit.
새 패키지/외부가중치가 필요한 실험은 아닙니다. 이번 진단에서는 실행하지 않았습니다.

재현: 새 사본에서 `.venv/bin/python -I diagnose_ensemble.py`.
기존 `artifacts/diagnostic-008`을 덮어쓰지 않습니다.
400개 ID·유한값·가중치 식·표준 라이브러리 RMSE 독립 재계산·CNN소스해시 검증 통과.
상세 `artifacts/diagnostic-008/report.json`, `development_ensemble_oof.csv`.

## 경쟁력 개선 연구와 고정 재분할 검증 (run-009~016)

기존 코드를 독립 검토했습니다.8개 OOF 파일의400개 ID/라벨이 일치하고 점수 재계산 오차는
3.6e-15이내였습니다. 직접적인 라벨누수·타깃 역변환·RMSE 오류는 발견되지 않았습니다.
다만 여러 모델을 같은 OOF에서 선택하는 낙관편향, source-family 메타데이터 부재,
조건부 bootstrap의 모델학습/선택 불확실성 누락은 여전합니다.
CNN 후속 실험은 epoch별 검증곡선을 저장하되 사전 고정 endpoint/마지막5epoch 평균만 사용했습니다.

| 가설 | 단독 RMSE | 고정 결합 RMSE | 실험 초 |
|---|---:|---:|---:|
| TinyCNN128px80epoch | 15.0323 | 13.7287 | 78.2 |
| TinyCNN256px40epoch | 14.6215 | 13.8581 | 99.6 |
| 모든 스케일 mean/std pooling | 14.5977 | 13.8221 | 57.0 |
| 26차원 물리proxy+중첩RBF | 14.4829 | 13.7057 | 9.1 |
| 진폭보존 잔차momentCNN | 15.0491 | 13.8666 | 91.1 |
| 경계/기공분리 상혼합 추가46차원 | 14.5368 | 13.7478 | 9.8 |
| 기존795특징의8개블록 균형RBF | 14.2968 | 13.8560 | 2.7 |

CNN행 결합은 기존50%+형태Ridge25%+해당CNN25%입니다.
RBF행 결합은 기존50%+TinyCNN128x80 25%+해당RBF25%입니다.
서로 성분이 다름에 유의하세요. 상세 `reports/summaries/research-stage-results.json`.
26차원 패널은 기존물리proxy12개, 경계intercept6종의6설정 중앙값/범위12개, raw밝기평균/표준편차2개입니다.
RBF의9개 bandwidth/alpha 설정은 매 외부학습fold 내부3-fold에서만 선택했습니다.
기존 run002와 달리 물리proxy의 작은 패널과 train-fold median거리, 특징군 균형화를 사용했습니다.
상혼합추가는 밝기GMM의 어두운상/밝은상 추정 및 겹침·모호함·마스크지원 비율입니다.
수렴했다고 물리적상 분리를 입증하지는 않으며 추가정보의 이득은 확인되지 않았습니다.

256벤치마크는0.03921초/학습단계(batch32), RSS약701MB, MPS드라이버약1.14GB.
40→80실험은 cosine LR경로도 바뀌므로 단순 추가epoch효과가 아닙니다.
256실험은 원본대비 수용영역과pooling맵도 바뀝니다.
모든스케일pooling실험은head파라미터와dropout RNG경로도 바뀌므로 순수단일요인 증명은 아닙니다.
진폭보존 잔차구조도 명시된하나의구조가설이며 각block의효과를 분리하지 않았습니다.

두 후보(run012,run014)의 전체레시피를 고정한 뒤 GroupKFoldseed20261011/20261012로
기존모델·CNN·커널 **모든성분을 각새외부학습fold에서 다시 fit**했습니다.
다른fold로 만들어진 과거 OOF예측을 재사용하지 않았습니다. 커널은 다시inner3-fold 선택했습니다.

| 새분할 seed | 기존후보002 | 소형26차원조합 | 상혼합46차원조합 |
|---|---:|---:|---:|
| 20261011 | 14.422193 | 13.777971 | 13.806526 |
| 20261012 | 14.303547 | 13.889479 | 13.893242 |

소형조합은두분할10개fold모두개선했습니다. 조건부개선구간은각각0.3711~0.9233,
0.1607~0.6606입니다. 이는 **개발분할안정성** 증거이며 독립holdout검증이나 입상성능 보증이 아닙니다.
두분할은같은400관측을공유하므로800개독립관측처럼 합쳐 신뢰구간을계산하지 않았습니다.
검증173.21초. 단순한run012조합을최종학습레시피로선택하고가중치50/25/25를고정했습니다.
`reports/summaries/research-decision-017.json`에 전체학습전선택을기록했습니다.

기존100장holdout은이번실험선택/평가에사용하지않았습니다.
최종고정후 전체공식train500장 refit에서는 훈련샘플로합칠수있으나 별도holdout점수는계산하지않습니다.
커널 최종hyperparameter도 development400 내부에서만정하고 전체500장 fit 전에고정합니다.
테스트는모델고정후추론용으로만사용합니다. Public점수는어떤특징/가중치조정에도사용하지않았습니다.

15개공학테스트통과(2.996초), 새두분할점수와가중치식을별도표준라이브러리로재계산했습니다.
`reports/validation/frozen-validation-error-analysis.json`에서모든경도사분위의오차가줄었지만양끝경도수축은남았습니다.
원본모델·후보002를보존했습니다. 소스/라이선스는 `reports/provenance/software-licenses.json`.
새소스는프로젝트에서작성했으며외부pretrained모델/가중치/데이터를사용하지않았습니다.
공식설명PDF는확인된 Hall–Petch/상분율혼합/기공 메커니즘을참고했으나
이환경직접다운로드는403으로실패했습니다. 부호를강제하거나생성공식을복구했다고주장하지않습니다.

재현(각결과경로가없는작업사본):
```sh
.venv/bin/python -I cnn_controls.py bench256
.venv/bin/python -I cnn_controls.py 128x80
.venv/bin/python -I cnn_controls.py 256x40
.venv/bin/python -I cnn_moments.py
.venv/bin/python -I compact_regression.py
.venv/bin/python -I cnn_residual.py
.venv/bin/python -I phase_regression.py
.venv/bin/python -I block_kernel.py
.venv/bin/python -I -B frozen_validation.py --feature-runs run-012 run-014 --output artifacts/frozen-validation-016
```
