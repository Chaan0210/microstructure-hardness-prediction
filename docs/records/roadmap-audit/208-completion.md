# 208 완료 관리표

사전 기준 충족 여부만 보고하고 정지. 자동 채택 또는 후속 학습 없음.

| 항목 | 상태 | 결과·범위 |
|---|---|---|
| 전체 update EMA | 완료 | Decay 0.99, warmup 0, E0=θ0, 매 update 직후 누산, 최종 E800. 한 정책만 실행. |
| 기존 평균 대조 | 정확히 재현 | 같은 5fold 학습 경로에서 마지막 5 epoch 동일 평균 재구성. 이전 state·OOF·학습 스트림 exact. |
| Optimizer·RNG 통제 | 통과 | 동일 gradient의 AdamW 복제본 및 EMA 갱신 전후 상태 검사 각각 4,000회. 별도 gradient 학습 없음. |
| BatchNorm·buffer | 없음 | BatchNorm 0, GroupNorm 4, buffer 0. Running statistics 재적합 없음. |
| 단독 OOF | 완료 | 15.0323178431 → 14.8642102551, 개선 +0.1681075880, CI [0.09173510093220538, 0.24156584601754474], 상위 5개 제외 +0.1245456020. |
| 고정 앙상블 | 비선택 진단 | 12.7862007758 → 12.7823960772, 개선 +0.0038046986, CI [-0.0028598152190292487, 0.010351216068238632], 상위 5개 제외 +0.0001158200. |
| 후속 | 정지 | 기준 충족 여부 True. 기존 평균·best021 유지, decay·구간 재탐색 없음. |

[전체 보고서](../../../experiments/01-representations/0208-update-ema/REPORT.md) · 구조화 결과 · [사전 계획](../../../experiments/01-representations/0208-update-ema/plan.json)

207은 후반 checkpoint 지수 평균 대조였다. 기존 EMA 감사의 전체 update EMA 미실시 상태는 당시 기록으로 보존하고, 이번 208에서 위 한 정책을 검증 완료했다. 206·207은 재개하지 않았다.

주 gradient 학습 5회·4,000 update, 검증용 optimizer replay 4,000 step. 기존 133개 파일과 best021 해시 유지. 새 설치·다운로드·holdout/test·CSV·제출·비용 0.
