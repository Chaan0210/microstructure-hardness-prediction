# final/

대회 **최종 3위**를 기록한 #62 `candidate-source-flow38-boostedreadout.csv`(Public 8.8196, Private 9.6967)의 재현 패키지다. 리더보드 점수와 최종 수상 순위는 구분한다. 제출 Markdown과 코드 ZIP에는 `package/submission-files.json`에 지정한 최종 학습·추론 코드와 의존성만 포함한다.

| 폴더 | 내용 |
|---|---|
| [`package/`](package/) | 재현 코드와 문서. [`README.md`](package/README.md) → [`REPRODUCTION.md`](package/REPRODUCTION.md) 순서로 읽는다. `project/`에 최종 계보의 원본 소스·설정·모델·특징 캐시가 있다 |
| [`submission/`](submission/) | 솔루션 PDF, 코드 Markdown, 코드·자산 ZIP, 제출 CSV와 [파일 안내](submission/DELIVERY.md) |
| `work/` | 패키지 구성·검증 도구 |

코드·설정·라이선스는 Git과 Markdown에 포함한다. 자산 ZIP과 검토용 자산 디렉토리는 로컬에 보관한다. 자산에는 학습 라벨과 이미지 입력 캐시가 있으므로 공개 저장소에 포함하거나 공개 다운로드 링크로 연결하지 않는다. 별도로 보유한 두 ZIP의 내부 경로는 `final/...`이며, 같은 위치에 풀어 하나로 합친다.

제출 당시 가중치로 원본 이미지를 추론하려면 `raw.py`, 대회 학습 데이터로 가중치를 새로 만들려면 `train.py`를 사용한다. 실행 환경과 데이터 배치는 [패키지 README](package/README.md)에 있다.
