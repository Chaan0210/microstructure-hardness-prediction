# 219 D2 코드 독립 감사

**판정: 현재 잠금본에서 A/B/outer 라벨 누수나 완전 anchor 단계 누락처럼 즉시 학습을 중단시킬 확정 결함은 발견하지 못했다.** 다만 이 판정은 구현/CPU-unit 단계의 코드 스냅샷에 한정된다. 실제 5fold 실행·검증 driver와 결과 provenance는 아직 이 스냅샷에 없어서 확인할 수 없었다. 평가 전에 아래 3항목을 확인해야 한다. 기존 학습 프로세스나 코드는 건드리지 않았다.

## 우선 확인 사항

### 1. 평가 함수의 fold list 입력 오류 — 확정된 조건부 결함, 학습 중단 사유는 아님

`experiments/01-representations/0219-residual-cnn/evaluation.py:7`은 y/예측만 NumPy array로 변환하고 `fold`는 변환하지 않는다. `:10`에서 `fold==k`를 mask로 쓴다. fold가 data-contract 등 JSON에서 온 Python list라면 비교 결과는 원소별 mask가 아니라 단일 False이므로 fold RMSE 계산이 빈 배열/NaN으로 바뀐다. list는 `:7`의 `set(fold)` 검사를 통과한다. 실제 실행이 list를 전달했다는 증거는 없다. ndarray를 전달하면 이 특정 문제는 없다.

최소 처리: 평가 입구에서 `fold=np.asarray(fold, dtype=np.int64)` 후 shape==(400,), fold별80행/정수0–4를 검사한다. list와 ndarray 입력의 결과가 동일하고 모든 metric이 finite인지 작은 합성 fixture로 확인하면 충분하다. 이번 감사에서는 함수 실행·학습·수정 모두 하지 않았다.

### 2. outer-safe incumbent 및 완료5fold 검증은 caller에 아직 의존 — 검증 경계 미완성

`evaluation.py:5–8`은 400길이 finite incumbent와 호출자가 제공한 `completed_folds`만 확인한다. ID 순서, original204 fold 일치, incumbent 출처·해시, 5개 완료 bundle과 각 V coverage를 스스로 확인하지 않는다. 변수명을 `outer_safe_best021`로 붙이는 것은 provenance 검증이 아니다. 올바른 caller가 이 조건들을 검증하면 문제가 없으며, 현재 잘못된 incumbent가 사용됐다고 단정하지 않는다.

열람 당시 root의 Python 코드에서 `evaluate_all` 호출은 `unit_checks.py:60`의 미완료-fold 거부 fixture 하나뿐이었다. 실전 scorer의 직접 호출은 없었다. 따라서 **점수 수용 전** 실행 driver가 다음을 제공해야 한다: 검증된 original 204 R 또는177 best021 OOF의 hash/ID/fold, graph V를 기준으로 정확히 한 번씩400행 조립, 각 bundle의 training completion/recipe/code-lock 검증. full500 예측이나 같은 행에 fit한 anchor를 incumbent로 넣어서는 안 된다. 추가 학습 없이 source binding과 metadata assertion으로 막을 수 있다.

### 3. full production poison 및 true/permutation 실행동일성은 아직 검증되지 않음 — 결과 승인 blocker

`unit_checks.py:43–65`의 poison은 작은 합성 자료와 `FixtureBackend`(선형 least-squares, :17–27)로 수행한다. 실제 자료 검사는 cached node별 **2행** CPU replay(`:67–81`)와 과거204 저장모델 수식 replay(`:83–99`)다. 이는 유용한 단위/회귀 검증이지만 새 MPS CNN·신규 deep anchor 전체를 label-poison하여 다시 학습한 검증은 아니다.

`implementation-contract.json`도 `pending_before_main`에 complete neural poison/replay 미실행을 명시한다. 프로토콜의 예정 CNN 검증34개를 완료 수치로 인용해서는 안 된다. 실제 첫 anchor A 고정+B/V 오염에서 새 deep/S/B/L/short/affine/classical residual의 상태·타깃·예측 불변, residualCNN B 고정+outerV 오염에서 최종 상태/예측 불변을 실행 driver가 확인해야 한다. cached root/C 노드는 재학습하지 않았다는 구분과 train-ID/target/cache 해시도 유지해야 한다.

true/permuted는 코드상 같은 seed·order·D4·80epochs·shared true-B normalization을 사용한다(`orchestrate.py:13–15`, `backend.py:65–88`). 그러나 반환 curve에는 initialization/order/D4/dropout trace hash와 update count의 arm 간 equality assertion이 없다. 현재 이를 불공정하다고 판단하지는 않는다. 실행 검증에서 같은 초기 상태·샘플 순서·augmentation draws·실제 optimizer step 수를 대조하고 최종 state/CPU reload를 확인해야 공정 비교가 실측으로 입증된다. 추가 정책이나 seed 탐색은 필요하지 않다.

## 요청된 항목별 코드 판정

| 항목 | 판정과 코드 근거 |
|---|---|
| A/B/outer label isolation | 구조상 적절하다. `orchestrate.py:12`가 T=A∪B와 V의 disjoint coverage를 검사한다. anchor에 `train_labels.restrict(A)`, residual에 restrict(B)를 전달한다(:13). `contracts.py:13–21`은 label subset을 복사하고 허용 ID 밖 take를 거부한다. Features에는 target key가 없다(:22–32). |
| nested A/C/D/E/F 범위 | `anchor.py:48–58`에서 A→C→E로 label capability가 축소된다. D target의 anchor는 C-only deep E/F OOF로 보정하며 D/B/V를 upstream fit에 넣지 않는다. 각 D의 residual learner selector는 해당 D 안에서만 학습/선택한다(:23–31). 동일 anchor_C가 D 전체에서 고정되어 있으므로 D 내부 CV가 타깃을 통해 다른 D 라벨을 upstream에 전달하지 않는다. |
| 완전 anchor 단계 | `.9*(.9*affine(raw)+.1*short+mean(residual3))+.1*L`가 `anchor.py:41–44`, `:59`에 있다. raw는 KRR.3+CNNmean3.3+Smean3.2+Bmean3.2 (`backend.py:63`, `:105–115`, `:126–134`). root A-only short와 L도 포함한다(:49). legacy26/physics585 및 5 residual candidates·512 bootstrap이 유지된다. 218 교정은 들어 있지 않다. |
| residual target/deployment 일치 | `bundle.py:7–13`에서 anchor를 먼저 저장·hash 고정 후 같은 객체를 load하여 yB−G_A(B)를 만든다. residual predictor에 anchor hash를 넣고 bundle 저장/로드 시 검사한다(:15–24). `PairedBundle.predict`는 저장된 anchor+residual 그대로다(:25–29). `orchestrate.py:15–17` 역시 G_A(V)+CNN_B(V)를 사용하며 기존P021_T에 잔차를 붙이지 않는다. 본학습에서는 in-memory sum 대 reloaded bundle의 전체80행 일치 확인이 추가로 필요하다. |
| true/permutation 공정성 | B 순서 고정 상태에서 yB−G_A(B)를 fixed permutation한다. 정규화는 두 군 모두 trueB 통계 공유; residual표준편차 floor1e−6. 각 fit의 NumPy RNG·torch seed 초기화와 loop 횟수 동일, epoch76–80 평균, outer metric 기반 중단 없음. 위3의 실행 trace 검증은 별도 필요하다. |
| outer-safe incumbent | scorer 함수만으로는 출처를 보장하지 못한다. 위2 참조. R1 vs R0와 Rperm이 mechanism, P021은 별도 competition benchmark로 나눈 해석은 올바르다. 약한 A213 anchor를 이겼다는 이유만으로 best021 승격하지 않는다. |
| metric aggregation | `evaluation.py:8–11`의 전체400 RMSE는 SSE를 먼저 평균하여 sqrt한다. 각 비교에서 동일 ID draw를 사용한 paired10000 bootstrap, 5개 fold는 설명용으로 따로 보고하여 평균foldRMSE를 주점수로 쓰지 않는다. 같은 개발400 재사용 편향과 conditional CI 한계 명시. 단 bootstrap은 group이 아닌 ID 기준이므로 새 데이터에 duplicate group이 생기면 별도 검토가 필요하다(이번 개발400은 앞선 감사에서 unique group400 확인). |
| cache membership/targets | `backend.py:96–104`는 train ID 순서+offset으로 unique registry를 찾고 label hash·file hash·저장 train IDs를 검사한다. `data.py:12–20`은 데이터 contract hash와 feature ID/schema를 확인한다. 캐시 node가 단지 크기만 같은 다른 집합으로 대체되는 경로는 발견하지 못했다. |
| recipe 차이의 명시 | protocol은 root KRR를 각 paired A 내부에서 선택한다고 기록한다. 기존 full500에서400 선택을 재사용하던 예외와 다르며, 새 A213 anchor가 그대로 기존 full500021과 동일하다는 주장은 하지 않는다. CPU inference 변경도 implementation-contract에서 고정 정책으로 공개한다. |

## 독립 metadata 검사 결과

데이터 배열·모델·성능 파일을 열지 않고 `data-contract.json`의 인덱스 집합과 `code-lock.json`을 표준 라이브러리로 검사했다.

- 5fold 각각 T320/V80, A213/B107. V의 합집합은 개발400이고 중복 coverage 없음.
- 각 A에서 C142/D71 세 개, D가 A를 정확히 한 번 덮는다.
- 각 C에서 E94 또는95/F48 또는47 세 개. F가 C를 정확히 한 번 덮으며 E/F는 해당 D·B·V와 겹치지 않는다.
- registry20개, 각 root A+offset 일치 entry 하나. development ID400개 고유.
- root source15개 모두 읽은 code-lock SHA256과 동일했다.

여기서 group disjointness를 원시 group 배열로 다시 검사하지는 않았다. 실행 코드 `Features.partition`이 검사하고, 기존400 groupunique 정보와도 모순되지 않는다.

## 실행 단계와 이번 리뷰의 한계

열람 시 `backend.py:14–18`은 `MAIN_TRAINING_ENABLED=False`, protocol.status는 IMPLEMENTATION_AND_CPU_UNIT_TEST_ONLY_NO_MAIN_TRAINING, code-lock.main_training_enabled도 false였다. implementation-contract는 새 recorded code lock과 probe/production 검증을 요구한다. 이것은 ‘현재 프로세스가 실행 중이 아니다’라는 OS 관찰이 아니라 **읽은 파일 스냅샷의 상태**다. 이 리뷰는 앞으로 생성될 실행 driver, runtime monkeypatch, 새로운 source lock을 승인하지 않는다. 새로운 실행 코드가 따로 있으면 해당 artifact만 추가 감사하면 된다.

프로세스 조회·조작, 학습/추론/unit-test 실행, 데이터 전체읽기, 성능 결과 확인, 외부 호출, 기존 파일 수정은 하지 않았다. 요청대로 점수가 생기기를 기다리지 않고 코드 감사만 반환한다. 결과 제출을 막는 미확인 사항은 실전 scorer의 incumbent/ID binding과 production poison/replay 증거이며, 현재 확인한 핵심 학습 수식만으로 중단을 권할 근거는 없다.
