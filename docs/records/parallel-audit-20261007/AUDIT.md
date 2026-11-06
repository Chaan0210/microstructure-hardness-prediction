# 독립 실험·검증 감사 — 2026-10-07

판정: best021 유지. 이번 열람에서 현재 예측을 무효화할 확정 누수·결합 계수 오류·캐시 오염은 찾지 못했다. 가장 구체적인 미실시 대조는 **218의 결측 flag 없이 유효한 scale 평균만 교정하는 조건**이다. 고정 2차 scattering은 096에서 동일 논문·동일 라이브러리로 이미 시행한 접근이다.

## 범위

프로젝트 내부 코드, 보고서, 개발400 OOF의 ID/fold/group 메타데이터와 저장 점수만 열람했다. 새 학습·모델 재추론·원시 이미지/라벨 파일 열람·test/holdout 내용 열람·설치·외부 통신·제출·공유 캐시 변경은 하지 않았다. 첫 파일 목록 검색에 data/test 파일명이 포함되었으나 해당 파일을 열지는 않았다. D2 완전 anchor+잔차CNN 설계/구현은 다른 작업의 범위로 두었다. 결과 파일은 이 디렉터리의 문서 두 개뿐이다.

감사는 001–218 모든 중간 모델을 전수 재현한 것이 아니다. 096/054/055/159/168/182 보고서, 최근 로드맵과 L1–L4, 177/204/217/218 및 현재 결합 경로를 우선 확인했다. 미실시는 이 범위에서 실행 근거를 찾지 못했다는 뜻이다.

## 제안 1 — 218에 빠진 ‘값 교정만’ 대조 한 개 (우선순위 1)

**근거와 구분.** 218은 A=원본26+zero padding, B=A+결측 flag 2열, C=B+valid-scale 평균이다. 따라서 A와 C 사이에는 값 교정과 희소 flag라는 두 변화가 있다. flag 없는 valid-scale 평균 D가 없다. `experiments/01-representations/0218-grain-validity/run.py:34–36`, `:54`, `:63–71`에서 세 조건과 동일 선택 grid를 확인했다. 이번 제안은 segmentation 재조정, 218 C의 반복, 후단 전체 재학습, D2 잔차CNN이 아니다.

저장 결과의 RMSE 개선량(대조−후보)은 다음과 같다. `experiments/01-representations/0218-grain-validity/results.json`에서 가져온 수치이며 새 예측을 만들지 않았다.

| 분할 | flag B−A: KRR 개선 | 값교정 C−B: KRR 개선 | flag B−A: 고정 결합 개선 | 값교정 C−B: 고정 결합 개선 |
|---|---:|---:|---:|---:|
| original | −0.061692 | +0.018599 | −0.011664 | +0.005385 |
| 20261031 | −0.069416 | +0.012413 | −0.015343 | +0.003724 |
| 20261101 | −0.236954 | +0.003538 | −0.045155 | +0.001546 |
| ID별 3분할 평균 SE | −0.122669 | +0.011507 | −0.024233 | +0.003536 |

C−A는 세 분할 모두 악화했고 pooled 고정 결합 개선은 −0.020697이다. 그 판정은 유지한다. 다만 C−B는 세 분할 모두 양수이므로 **218의 악화를 valid 평균 교정 자체의 실패로 단정할 수 없다.** C−B는 flag가 있는 조건의 효과이며 D−A를 대신하지 않는다. 효과 크기도 작고 조건부·사후 관찰이다.

**구체적 메커니즘.** `compact_regression.py:28–35`는 모든 열에 MinMax/StandardScaler를 적용하고 block 폭의 제곱근으로 나눈다. 218 flag block은 두 열이고 sigma1 결측은 0개, sigma2 결측은 3개이다. 학습 비율 p인 binary flag 하나가 두 행 사이 RBF 입력 제곱거리에 기여하는 값은 **1/[2p(1−p)]**다. 실제 outer320에 결측이 1/2/3개이면 약 **160.5016/80.5031/53.8381**이다. ‘작은 flag 2개 추가’가 결측 행 주변의 거리 구조에는 큰 변경이다. bandwidth에 따라 실제 kernel 영향은 달라지므로 이 수치만으로 성능 원인을 확정하지 않는다.

45개 inner 분할 메타데이터 중 2개에는 결측 학습 예가 없고 검증에만 있다: 20261031 outer2/inner2(검증 결측2개), 20261101 outer4/inner0(검증 결측1개). 이 경우 constant-column scale의 동작이 비희소 fold와 달라진다. 이것은 누수가 아니라 **희소 indicator의 fold별 거리 척도 차이**다. flag가 무조건 지워진다고 해석하면 안 된다.

**최소 판별 실험(이번에는 실행하지 않음).** D는 A의 0번 HP 열만 C와 같게 바꾸고 두 padding 열은 그대로 0으로 둔다. 기존 218의 3분할·내부3fold·9개 grid·seed·blocks·선택 규칙을 전부 고정한다. 이미 검증된 A/B/C를 재학습하지 않고 D만 수행하면 outer15+inner405=420 KRR fit이며 CNN fit은 없다. all400과 397개 비결측 행을 함께 보고, primary는 D−A의 pooled untrimmed MSE, C−B와의 차이는 상호작용 진단으로만 둔다. 최소 구현 검사: 원래26열 중 HP의 알려진3행만 변경, 나머지 값/flag/ID 정확 일치, fold-local 통계, outer-label poison, 저장모델 수식 검산. 고정 .243×slope 치환은 조건부 screening이며 실제 최종 pipeline 개선·승격을 뜻하지 않는다. 음성이면 교정 분기를 닫고 flag scaling/비중 sweep으로 확장하지 않는다.

## 제안 2 — 타깃 층화 여부의 검증 프로토콜 점검 (우선순위 2, 정확도 후보 아님)

원본은 `src/microstructure/pipeline.py:112–116`, 반복분할은 `frozen_validation.py:305`의 shuffled GroupKFold다. y를 전달해도 타깃 층화는 아니다. 현재 점검한 217/218 OOF는 모두 400행, unique ID/group 각각400이며 세 분할은 같은 ID를 재사용한다. `docs/records/roadmap-audit/roadmap-30-after-215.md`도 층화를 미실시로 분류한다.

**차이.** 기존 반복 random group CV·L1 paired-loss·L2 이웃·L3 tail 진단과 달리, 고정된 label-bin/group 보존 분할이 fold 타깃 분포를 얼마나 균형 있게 하는지 검증하는 작업이다. 성능 개선이나 독립 검증 확보라고 부르지 않는다.

**최소 판별 단계.** 학습 없이 개발400 metadata만으로 현행 fold별 타깃 평균/분산/고정 분위구간 빈도와 하나의 사전 고정 층화 분할을 비교한다. 결과 점수로 split seed를 선택하지 않는다. 충분한 불균형이 없으면 종료한다. 추후 학습이 별도 승인될 때만 best021 레시피와 단 한 후보를 같은 새로운 outer/inner 분할에서 비교한다. 새로운 분할에 기존 OOF를 단순 재배열하는 행위는 금지한다. 이 제안이 217 실패를 구제하거나 누적 선택편향을 없앤다는 주장은 하지 않는다.

다른 새로운 모델 제안은 보류한다. per-batch trainable-CNN noise는 110의 frozen-DINO noise와 다르지만, 실제 잡음 모형/라벨 불변성 근거가 부족하여 ‘미실시’만으로 유망 후보에 넣지 않았다.

## 추가 요청: scattering·filter bank·연결성 중복 판정

| 후보 | 실행 근거 | 판정 |
|---|---|---|
| 고정 second-order wavelet scattering | `wavelet_scattering_096.py:13`, `:58–66`, `:74–87`; `reports/summaries/wavelet-scattering-096-summary.json:2–11`, `:40` | **중복.** Kymatio0.3.0, native256, J6/L8/order2, 1009개 공간 평균 특징, order1 49개 matched control, train-only scaler+RBF KRR. 전달된 PAMM DOI `10.1002/pamm.202300169`가 이미 :60에 들어 있다. |
| scattering 결과 | `reports/summaries/wavelet-scattering-096-summary.json` | order2 단독15.923348, 011 고정10% 결합13.150322 대 기준13.053998. 결합5fold 모두 악화, 재분할 미실행. order1보다 좋아도 기존 모델 보완성에는 실패했다. 원본 OOF의 어떤 양의 추가 비중도 악화한다는 진단까지 기록되어 있다. |
| 환경 | `.venv/lib/python3.13/site-packages/kymatio-0.3.0.dist-info/METADATA` | 현재 파일의 Name/Version을 읽어 **Kymatio0.3.0 설치 존재 확인**. import/새 추출/설치 없음. 과거 `kymatio.numpy`는 3D의 scipy sph_harm 문제로 실패했고 직접 NumPy2D frontend가 동작했다는 기록. 설치됨과 모든 frontend가 호환됨을 구분한다. |
| multiscale filter-bank energy | `native_texture_features.py:83–99`; `reports/summaries/wph-182-summary.json:19–22` | LoG fine/coarse 제곱 에너지·tile 상관은 기존 구현. 182는 complex wavelet phase harmonics까지 별도 시행하여 종료. 단순 energy bank 도입을 새 접근으로 부르지 않는다. |
| lineal path | `lineal_features_054.py:44–54`; `reports/summaries/lineal-054-summary.json:2–15` | 120개 raw-quantile 특징·Ridge·10% 결합 시행, 13.241597→13.403404, 개선0/5fold. |
| Euler/Betti | `betti_features_055.py:19–25`, `:28–41`; `reports/summaries/betti-055-summary.json` | Euler에서 b1=b0−Euler를 계산, signed-distance 132열 시행. 고정 결합 악화. Euler 첫 도입은 중복. |
| connectivity/C2 | `cluster_pair_features_159.py:13–20`, `:43–51`; `reports/summaries/cluster-pair-159-summary.json` | 같은 component에 속하는 확률 C2와 같은-mask S2 대조 시행. C2 pooled12.850728 대018 12.836545, 재분할도 미개선. |
| full grayscale H1 | `reports/summaries/cubical-h1-168-summary.json`; `README.md:1195–1197` | 055 threshold Betti와 다른 정확 cubical H1도 168에서 검증·학습. 원본 미세 개선 후 재분할 악화. ‘아직 topology를 안 했다’도 틀리다. |

이들은 각 고정 레시피의 실패이지 해당 수학적 표현의 모든 가능성을 부정하는 결과는 아니다. 하지만 기존 실패 뒤 head/scale/ratio만 바꾼 접근을 새 전략으로 포장할 근거는 없다. 외부 문헌 URL은 열지 않았고 프로젝트에 저장된 인용과 시행 여부만 대조했다.

## 검증·캐시·결합 코드 검토

1. **고정 결합 계수: 오류 발견 안 됨.** 021=.9×018+.1×L, 018에서 calibrated branch=.9, raw CNN/KRR 비중=.3. CNN 첫 seed는 다시 /3이므로 .081×fold slope, KRR은 .243×fold slope가 맞다. `experiments/01-representations/0217-resolution-repartition/common.py:91–101`, `score.py:6`, `experiments/01-representations/0218-grain-validity/run.py:43–45`, `validate_score.py:24–27`가 독립 수식/대조 재현을 검사한다. 이번에는 저장모델 재추론을 반복하지 않았다.

2. **후단 고정의 해석 범위.** 217/218은 후보 출력을 바꿔도 기존 affine/anchor/residual을 유지한다. 완전 모델 효과가 아닌 conditional counterfactual이라는 제한이 코드와 보고서에 명시된다. 이것을 누수라고 단정하지 않는다. 반대로 이 수치를 ‘후단 재학습도 완료한 정확도’로 인용하면 오류다. 204는 별도로 각 nested 단계와 후단을 새로 구성한다(`experiments/01-representations/0204-cnn-nested/pipeline.py:43–64`).

3. **캐시 교환/OOF 인덱스: 현재 오염 근거 없음.** 204 cache는 file hash를 plan에 잠근다(`common.py:49–62`); middle/deep 범위와 outer 제외를 검사한다(`pipeline.py:23–32`); residual targets와 저장 모델의 train/validation scope까지 검증한다(`pipeline.py:95–104`). 217도 ID/y/fold 일치와 보호 input hash를 확인한다(`common.py:44–61`, `:91–101`). 새 분할·타깃 변형에 기존 cache를 이름만 바꿔 붙이는 것은 안전하지 않지만, 이번 코드에서 그런 사실은 확인하지 못했다.

4. **poison test가 보장하는 범위.** 204의 후단 poison은 캐시된 upstream 예측을 입력으로 두고 y만 오염한다(`pipeline.py:57–59`, `preflight.py:13–14`). 따라서 이 검사 하나만으로 upstream 전체를 새로 학습해 누수를 배제했다고 주장할 수 없다. upstream hash·scope 증거와 결합해야 한다. 실제 preflight는 `cached_old540_inner_predictions_reused_not_retrained`를 기록하여 이 구분을 지킨다. 218 검증도 원 neural branch는 재학습하지 않는다고 명시한다(`validate_score.py:29`).

5. **bootstrap의 ID 단위: 현재 구현과 표본에 적합.** 217은 동일 ID의 3분할 SE를 먼저 평균한 뒤 400 ID bootstrap을 한다(`score.py:12–15`, `:36–38`). 이번 OOF에서 400개 group이 모두 unique인 것을 확인했다. 1,200 독립 표본으로 취급하지 않는다. conditional CI는 training/역사적 탐색 불확실성을 포함하지 않으며, 양의 하한·모든 fold·top5 제거는 사전 의사결정 규칙이지 일반적 성능 개선의 필수 정리는 아니다. 사후 기준 완화는 제안하지 않는다.

6. **실질 개선 가능성이 있는 구현 선택은 218 희소 flag 거리 효과.** 수식과 입력 빈도는 확인된 사실, 이것이 악화의 주원인이라는 설명은 아직 가설이다. 기존 validation이 표준화를 수식 그대로 재현해도 선택한 metric이 유용하다는 증명이 되지는 않는다. 제안1은 그 차이를 분리하는 최소 대조다.

## 프로젝트 내부 규칙 기록과 제한

`README.md:945–948`는 2026-10-04 공식 규정 확인 기록으로 공개 사전학습 가중치의 허용 라이선스/로컬 실행, 외부 학습 데이터·test 학습 금지를 명시한다. 최신 규정 원문을 이번에 재조회하지 않았으므로 새 법적/규정 적합성 판단을 하지 않는다. 제안1·2는 기존 개발400만 필요하다.

잔여 blocker는 실행 승인이 없는 학습 자체이며, 이번 audit 산출에는 blocker가 없다. 프로젝트 전체/미래 코드 무결성, 실제 생성 family 독립성, 진짜 결정립 정답, 독립 외부 일반화 성능은 이 읽기 전용 감사로 확정할 수 없다. D2의 결과는 별도 작업을 기다린다.
