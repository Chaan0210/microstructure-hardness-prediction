# artifacts/

`studies/`와 `experiments/` 스크립트가 쓰는 산출물 폴더다.

- `submission-candidate-NNN.csv` / `.manifest.json`: 후보 파일 001–083과 생성 정보(구성식, SHA256). 실제 제출 여부와 점수는 [`docs/submissions.md`](../docs/submissions.md)에 있다.
- `<이름>-<study 번호>/`: study별 집계 결과와 설정. 개별 샘플의 정답·예측·잔차, 특징 캐시와 모델 가중치는 로컬에 보관한다.
- `final-model-NNN/`, `run-NNN/`: 후보 생성 실행 기록.
