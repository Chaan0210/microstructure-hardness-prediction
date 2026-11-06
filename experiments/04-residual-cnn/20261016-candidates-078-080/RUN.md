# 재현 명령

프로젝트 루트에서 실행한다. 출력은 이 실험 폴더 안에 저장한다.

## 권장 제출 파일 생성 또는 동일성 확인

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/make_submission.py
```

표준 라이브러리만 사용한다. 기존076·078·080 입력 해시,1000개ID와
비중복 변경을 확인하고 `candidate-076-plus-frozen-cnn.csv`를 생성한다.
기존 출력이 다른 내용이면 덮어쓰지 않고 중단한다. 자동 업로드하지 않는다.

## 구성 요소 + 원본 테스트 이미지로 추론 재현

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/infer_frozen.py
```

보존된073 및 Tab/Ridge/GMM/grain-set 예측 캐시로074·076을 재현하고,
보존된 full500 CNN 가중치2개와 테스트 이미지로 다시 예측한다.
`verification/inferred-from-models.csv`와 권장 CSV의 일치를 검증한다.
원래500 학습 라벨·픽셀을 읽지 않으며 새로운 모델을 적합하지 않는다.
의존 파일·해시는 `frozen-inference-verification.json`에 기록된다.

## 선택한 CNN 학습 재현

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/retrain_selected.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/infer_frozen.py --models experiments/04-residual-cnn/20261016-candidates-078-080/retrained-selected
```

기존의 검증된3fold full073 교사 캐시를 사용하여500행 전체로2seed를
40epoch씩 학습한다. 기존 파일은 읽기 전용이며 새 모델은
`retrained-selected/`에 저장한다. 원본 가중치와 bitwise 동일성을 검사한다.
교사 학습 원본 코드는 `../rmse-research-20261015/round4-full073/full_branch.py`
및 `final_teachers.py`에 보존되어 있다. 이것은 추가 성능 실험이 아니라
선택한 모델의 재현이며 전체학습 RMSE를 검증 점수로 쓰지 않는다.

## 이번 새 가설 및 검증 재실행

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/study.py audit
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/study.py train
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/evaluate.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/verify.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/clipping_check.py
experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/calibration_stress.py extend
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/calibration_stress.py evaluate
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/finalize.py
```

학습은CPU2스레드·프로세스1개를 사용한다. 완료된 모델은 meta/seed를
검증하여 재사용한다. 새로 재학습하려면 이 폴더를 새 경로에 복제하여
실험 출력만 비운 독립 사본을 사용하며, 공유 원본 캐시를 삭제하지 않는다.
소프트웨어 버전과 원본 데이터 SHA256은 `data-integrity.json`에 있다.
