# 205 관리 기록

완료: MicroNet 동결 특징 비교. 기존 [로드맵 대조표](audit.md)는 계획 시점 기록으로 보존합니다. 당시 미실시였던 MicroNet은 205에서 실행됐으며 사전 기준에 미달했습니다.

| 실험 | 대조 | 결과 | 판단 | 남는 한계 |
|---|---|---|---|---|
| 205 | 동일 native256 ResNet50 mean2048 + Ridge EDF32, original 5-fold | 단독 ImageNet 15.46900080 / MicroNet 15.87450101. 고정 10% 결합 12.86846411 / 12.88770281. 기준021 12.78620078. | 기술 검증 통과, 성능 기준 미달. 해당 분기 종료, best021 유지. | 같은400 재사용. 특정 checkpoint만 평가. v1.0/v1.1 대응 미확인. |

[완료 보고서](../../../experiments/01-representations/0205-micronet-frozen/REPORT.md) · 수치 요약 · [재현 검증](../../../experiments/01-representations/0205-micronet-frozen/validation.json)

다운로드 1개, 설치 0, Ridge 본 학습 10회, 신경망 gradient 학습 0회. holdout/test·전체 500·제출 CSV·업로드·최종 선택 변경 없음. 비용 0. 기존 로드맵·등록부·best 파일을 수정하지 않았습니다.
