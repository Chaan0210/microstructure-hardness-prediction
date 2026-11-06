# 실험 결과와 실행 기록

결론은 `summaries`·`notes`, 검증 근거는 `validation`, 제출 점수는 `submissions`부터 본다. 원시 실행 로그와 개별 샘플의 정답·예측·잔차 기록은 공개 대상에서 제외한다.

| 폴더 | 내용 | 파일 수 |
|---|---|---:|
| [summaries](summaries/) | 실험 결론·점수·최종 의사결정 | 161 |
| [validation](validation/) | 독립 검증·감사·재현 확인 | 96 |
| [submissions](submissions/) | 후보별 제출 점수·생성 정보·최종 검증 | 104 |
| [provenance](provenance/) | 계획·설계·환경·보존 정보·출처 | 47 |
| [registers](registers/) | 실험 관리표와 결과 집계 CSV | 3 |
| [notes](notes/) | 사람이 읽는 실험 종합 설명 | 3 |
| [source-snapshots](source-snapshots/) | 과거 오류·시도의 실행 소스 사본 | 1 |

## 시작점

- [실험 관리표](registers/strategy-experiment-register.csv)
- [초기 실험 결과 집계](registers/experiment-summary.csv)
- [증강·잡음 연구 종합](notes/push-1200-report.md)
- [초기 최종 결과](summaries/final-summary.json)
- [후보 021 제출 기록](submissions/candidate021-public-submission.json)

## 규칙

기존 보고서는 파일명을 그대로 유지하고 역할별 폴더만 바꿨다. 새 실험의 결과·검증·로그는 해당 `experiments/<분류>/<실험>/` 안에 함께 두고, 여러 실험을 모은 자료만 이곳에 추가한다. 

[실험 목록](../experiments/README.md) · [초기 study 소스](../studies/README.md) · [최종 모델](../final/package/README.md)
