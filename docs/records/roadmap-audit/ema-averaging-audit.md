# EMA와 checkpoint 평균 중복 감사

**현재 마지막 5 epoch 평균은 EMA와 다르다. 동일 학습 경로에서 평균 정책만 바꾼 기존 대조는 찾지 못했으며, 207에서 저장 checkpoint만 사용한 한 가지 후반 지수 평균 대조를 완료했다.** 206 대비 감소는 종료 상태로 유지하고 그 후보 checkpoint는 사용하지 않았다.

| 항목 | 상태·근거 | 중복 판단과 남은 범위 |
|---|---|---|
| 마지막 5 epoch 동일 평균 | 완료. `frozen_validation.py`의 `epoch >= 75`에서 `state/5`를 누적. 76–80 epoch 각각 20%. 009·129·133·193·204·206에 적용. | 기존 기본 정책이다. 지수 감쇠가 없어 EMA가 아님. |
| 여러 seed의 예측 평균 | 완료. 133은 각 seed 안의 마지막 5 state 평균 후 3개 seed의 예측을 동일 평균. 204는 3→6 seed와 후단 재학습으로 pooled 개선 0.005785, CI [-0.026274, 0.037320] 및 두 재분할 악화로 기각. | 서로 다른 학습 경로의 예측 평균이며 한 경로의 checkpoint EMA와 다름. 반복하지 않음. |
| 평균 시점과 학습 예산 | 완료. 128은 5개 정규화된 후반 시점을 동일 평균하되 update 예산을 변경. 작은 표본 400→800 update RMSE 15.418684→15.597598. | 평균 정책 단독 대조가 아님. |
| 최종 state 대 평균 state | 부분. 131/132는 마지막 state와 기존 평균을 사용한 경로가 다르지만, 2-view batching·LR 계단·readout·seed 등도 달라짐. `mse_readout_132.py`에 단일 schedule 대조가 아님을 명시. | 평균 방식만의 효과로 해석할 수 없음. 동일 대조를 했다고 간주하지 않음. |
| 같은 마지막 5 state의 지수 평균 | **207 완료.** 206 control의 동일 경로 checkpoint 25개를 재사용. 학습 0. 단독 개선 +0.0101814895, 고정 앙상블 개선 +0.0021584665; 두 CI 모두 0 포함, 상위 5개 이득 제외 시 둘 다 악화. | 하나의 사전 정책만 검증했고 기준 미달로 종료. decay·구간 탐색 없음. |
| 매 update 또는 전체 학습 EMA | 미실시. 현재 원래 경로의 모든 800 update state와 epoch1–75 state가 저장되지 않음. | 207을 전체 EMA 검증으로 부르지 않음. 이 범위의 정확한 대조에는 별도 재실행이 필요. |

최상위 및 실험별 Python 코드 697개를 명칭·API로 검색하고 관련 학습 루프와 계획을 읽었다. 명칭 검색만으로 임의의 수제 구현 부재를 증명하지는 않으므로 조사 범위를 명시한다. 기본 코드의 동일 평균, 133·128·132의 실제 대조 내용과 206의 저장 자료를 함께 확인했다.

현재 원본 경로의 `artifacts/run-009/fold-*.pt`에는 평균 state만 있다. 206 control에는 fold마다 `epoch-76.pt`부터 `epoch-80.pt`까지와 최종 평균 모델이 있다. 207에서 25개 state hash를 검증하고 동일 평균을 재구성하여 206 및 run009와 bitwise exact임을 확인했다. 따라서 마지막 5 state 범위의 비교에는 새 학습이 필요하지 않았다.

전체 update EMA를 별도로 요청하면 최소한 첫 seed의 원본 5fold를 동일 800 update씩 재실행해야 한다. 필요량은 **5회 학습·4,000 update·128,000 image presentation**이다. 재실행 중 EMA를 별도 누산하고 마지막 5 state 동일 평균도 함께 보관하면 학습 경로를 같게 유지할 수 있다. 전처리·seed·D4·loss·optimizer·80 epoch를 고정하고 decay 하나를 결과 전에 정하며 early stopping·epoch 선택을 하지 않는 설계가 필요하다. 이번에는 실행하거나 예약하지 않았다. 기존 3개 seed 전체를 바꾸는 질문은 최소 15개 outer 학습이 필요하며, 후단 전체 재학습은 또 다른 범위다.

[207 전체 보고서](../../../experiments/01-representations/0207-checkpoint-averaging/REPORT.md), [중복 조사 상세](../../../experiments/01-representations/0207-checkpoint-averaging/duplicate-audit.json), [기존 평균 구현](../../../studies/baseline/frozen_validation.py), [133](../../../studies/125-149/cnn_seed_mean_133.py), [128](../../../studies/125-149/cnn_update_budget_128.py), [132](../../../studies/125-149/mse_readout_132.py).
