# 209 완료 관리표

고정 EMA의 분할 안정성 기준을 충족하지 못했다. 이번 정책의 확대를 중단하고 best021을 유지한다.

| 항목 | 상태 | 결과·남은 질문 |
|---|---|---|
| 208 고정 전체 update EMA | 재분할 검증 완료 | Decay 0.99, warmup 0, E0 = θ0, 매 update 직후 누산, 최종 E800. 설정 변경 없음. |
| 기존 두 재분할 | 완료 | 각 5fold, 신규 주 학습 총 10회·8,000 update. 원본 208은 재학습 없이 포함. |
| 기존 CNN 및 optimizer 재현 | 통과 | 10개 가중치·OOF·최종 main optimizer·scheduler exact. 과거 RNG는 시작·종료만 직접 비교 가능. |
| CNN 단독 세 분할 pooled | 완료 | 개선 +0.1076517245, ID cluster CI [0.051597907528502775, 0.1638772944047099], 상위 5개 제외 +0.0856857215. |
| 고정 첫 seed 치환 | 진단 완료 | 개선 -0.0020413716, ID cluster CI [-0.007153032305276063, 0.0030923111362676447], 상위 5개 제외 -0.0040422754. 추가 blend가 아님. |
| 전체 nested 재학습 | 미실시 | 기존 보정·잔차는 고정했다. 전체 재적합의 효용은 아직 알 수 없으며 별도 판단 대상이다. |
| 최종 평가·제출 | 미실시 | Best021과 기존 CSV·모든 보호 파일 보존. 비용 0. |

같은 400개 ID를 세 번 사용했다. 독립 holdout 또는 1,200개 독립 표본이 아니며, 조건부 CI는 누적 선택 편향과 재학습·seed 변동을 보정하지 않는다. 새로운 decay·seed·epoch 탐색이나 후단 학습을 시작하지 않았다.

[전체 보고서](../../../experiments/01-representations/0209-ema-repartition/REPORT.md) · [구조화 결과](../../../reports/summaries/ema-repartition-209-summary.json) · [사전 계획](../../../experiments/01-representations/0209-ema-repartition/plan.json)
