# 204 검토 권고와 기존 실험 범위 대조

진행 중인204의 리뷰 권고와 과거 실제 수행 범위를 대조. 새 로드맵 실험은 별도 세션으로 이관.

## 데이터 증강

실제 통제 연구 및 제한 파일럿이 있으나 전체 증강 공간을 검증한 것은 아님

- 129: 원본 5fold에서 D4 제거 시 CNN RMSE 15.03232→16.33843, raw 결합도 악화.
- 110: 고정 잡음 조건의 matched 비교에서 최종 결합 13.05400→13.05727 악화.
- 061 feature mixup과151 D4 추론 평균은 일관된 최종 승격 근거를 만들지 못함.
- 172 D4 등변 CNN은 첫 fold의 제한 파일럿만 실행; 전체5fold 완료로 표시하지 않음.

한계: 물리적 라벨 불변성은 입증되지 않았다. crop/zoom/광도 변환의 미실행을 유해성의 실험적 증거로 취급하지 않는다.

근거: `reports/summaries/cnn-orientation-129-summary.json`, `reports/summaries/input-noise-110-summary.json`, `reports/summaries/cnn-view-mean-151-summary.json`, `reports/summaries/mixup-061-summary.json`, `reports/summaries/cnn-dropout-171-summary.json`, `reports/summaries/d4-cnn-172-summary.json`

## 데이터 분리

고정5fold·기존2재분할·중첩 분리는 실행; 새 독립 검증은 아님

- 같은400개를 세 번 나눈 결과이며1200개 독립 표본으로 계산하지 않는다.
- 160의 제한된 유사성 감사에서 확정 family를 찾지 못해 근거 없는 새 그룹을 만들지 않았다.
- 204에서는 모든 outer T 내부의 middle/deep OOF를 사용한다.

한계: 생성 family의 독립성 또는 OOF/Public 차이의 단일 원인은 확인되지 않았다. 이미 사용한100 holdout을 새 검증으로 다시 열지 않는다.

근거: `artifacts/nested-fusion-112/validation-replay.json`, `reports/summaries/layout-dependency-160-summary.json`, `reports/validation/domain-risk-audit-126-summary.json`, `reports/notes/research-history-through-016.md`

## 손실 함수 / ASL

복수 손실 통제 연구 있음; ASL 자체의 실행 근거는 찾지 못함

- 058의 MSE 대체,120의 순위 목적,131의 RNC,169의 라벨 밀도 가중은 해당 고정 채택 기준에 미달.
- 193/194/201의 보조 CE는 true/detach/permuted로 검사했고 특정 고정 분기는 종료.
- 보고서 및 루트 Python 소스 검색에서 ASL/focal loss의 실행 증거는 없었다.

한계: ASL을 직접 scalar 회귀 손실로 옮길 대상이 정의되지 않았다. 고정 CE 실패를 모든 보조 분류 손실의 무용성으로 일반화하지 않는다.

근거: `reports/summaries/optimization-057-058-summary.json`, `reports/summaries/ranking-objective-120-summary.json`, `reports/summaries/rnc-131-summary.json`, `reports/summaries/label-density-169-summary.json`, `reports/summaries/research-193-201-summary.json`

## CNN / ViT / MHA 구조

실제 구현·학습·matched 대조 있음; 일부는 제한 예산 또는 파일럿

- 동결 DINO S/B/L과 TinyCNN은 현재 조합에 포함된다.
- 196 MHA와 mean 대조,197 Q 정규화 단일 변경,200 가중치 고정 개입이 실행됐다. Q 정규화는 단독 성능을 개선했지만 최종021을 넘지 못했다.
- 190의 Swin last-stage는 같은 seed 재학습의 엄격한 bitwise 기준에 실패했고 실패 상태를 보존했다.

한계: 제한 학습 예산의 실패는 architecture의 최적 성능 한계가 아니다. scratch ViT 또는 모든 fine-tuning 정책을 끝냈다는 주장은 하지 않는다.

근거: `reports/summaries/cnn-update-budget-128-summary.json`, `reports/summaries/dinol-177-summary.json`, `reports/summaries/modern-backbones-190-summary.json`, `reports/summaries/research-193-201-summary.json`, `reports/summaries/d4-cnn-172-summary.json`

## 앙상블 / 선택 정책

고정 결합·성분 제거·seed 평균·TTA·nested 후단 검증 있음

- 202 단순화 A/B는 모두 악화했으며 Claude의 직접 제안이 아닌 자체 진단이다.
- 203은 CNN 3→6의 후단 고정 실험이다. L 3→9는 실행하지 않았다.
- 204가 outer/middle/deep의 seed 수를 일치시키고 양 arm에 같은 후단 정책을 재적합하는 누락 대조를 채운다.

한계: 과거 개발을 보기 전에 현재 모든 가중치가 고정됐다고 인증할 수 없다. seed 분산 감소는 실현된 최종 RMSE 개선을 보장하지 않는다.

근거: `reports/summaries/cnn-seed-mean-133-summary.json`, `reports/summaries/bootstrap-selector-144-summary.json`, `reports/summaries/simplification-202-summary.json`, `reports/summaries/cnn-seed-expand-203-summary.json`, `experiments/01-representations/0204-cnn-nested/plan.json`

## 결과 분석 / 학습곡선

Claude가 새로 제안한 일부 분석은 이미 수행됐음

- 127은 학습 크기142/143,213/214,320을 같은 outer80에 평가했다. 새 학습0, 저장 leaf840개 재추론; 성분별 오차 공분산·상관도 포함.
- 127의 CNN RMSE는15.56077→14.98564→14.62192였으나 데이터 수·업데이트 수·seed 차이가 섞여 인과 곡선이 아니다.128은 이를 따라 데이터 수와 업데이트 예산을 통제했다.
- 073/076은 prediction-conditioned bias와 성분 차이를 분석했고199는 타깃 분위·물리 통계·상위 기여 표본 민감도를 분석했다.

한계: 낮거나 평평한 학습곡선만으로 bias 한계와 잡음 한계를 식별할 수 없다. 상위 이득 제거는 ESS 숫자를 추정하는 방법이 아니다. 현재021 전체 파이프라인을 모든 작은 학습 크기에서 재구축한 곡선과는 구분한다.

근거: `reports/summaries/base-learning-curve-127-summary.json`, `reports/summaries/base-learning-curve-127-analysis.json`, `reports/summaries/cnn-update-budget-128-summary.json`, `reports/summaries/bias-calibration-073-075-summary.json`, `reports/summaries/post009-research-076-078-summary.json`, `reports/summaries/submission022-diagnostic-199-summary.json`

## 주요 CV 대회 전략의 전이

출처 조사와 일부 대응 실험 있음; 우승 전략 전체를 재현한 것은 아님

- NDSB2015 cyclic pooling/형태 특징 후단 결합은129/151/172와121/140/144에 부분 대응한다. 해당 회전 pooling 구현 전체를 재현한 것은 아니다.
- DSB2018 팀 저장소는 확인했지만 강한 증강 관련 Kaggle 본문의 세부 주장은 독립 확인되지 않았다는 기록이 있다.
- RSNA bone age의 상보적 모델 결합 교훈은 고정 치환·오차 분석과 연결된다. MAD와 hardness RMSE, 예측 상관과 오차 상관은 동일하지 않다.

한계: 외부 대회의 성능 향상을 현재 대회의 개선량으로 환산할 수 없다. 새 문헌을 근거로204 밖의 구조·증강·loss 실험을 자동 실행하지 않는다.

근거: `reports/registers/strategy-experiment-register.csv`

## 공식 규정 기록 정정

앞선 답변의 public/private 비율 일괄 미확인은 부정확했다. 로컬에 공식 페이지 확인 이력이 있다.
2026-09-30 확인 기록에는 Public 30%, Private 70%, 하루 최대5회 제출이 적혀 있다. 이번 작업에서 현재 공식 페이지를 다시 조회하거나 계정의 최종 선택을 변경하지 않았다. 최종 선택 가능 개수는 기존 자료와 대조할 항목이다.

192개 연구/제작/진단 디렉터리와448개 메타데이터 출처를 색인했다. 연구 수·seed·fold·replay·최종 arm을 합산한 독립 시행 수는 제시하지 않는다.204의R/F/N은 모두 명시 등록한다.
