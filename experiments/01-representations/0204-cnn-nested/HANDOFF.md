# 204 완료 인계

- 범위: `.`만. 기존 `.venv/bin/python -I -B`, CPU 2개 스레드/MPS, 메모리 6GB, 무료 로컬 실행.
- 현재 best: candidate021, 기록된 Public RMSE 11.4878517800.
- 파일: `artifacts/submission-candidate-021.csv`. SHA256: `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b`.
- 분할: 개발 400개, 원본 5-fold와 기존 20261031/20261101 재분할. 같은 400개이며, 독립 표본 1,200개가 아닙니다.

모든 중첩 단계에 CNN 6개 seed를 적용하고 같은 후단 정책으로 재적합한 후보가 사전 개발 기준에 미달했습니다. best021을 유지하며, 이번 작업에서 추가 seed·가중치·구조 탐색은 하지 않습니다.

| 분할 | R | F | N |
|---|---:|---:|---:|
| original | 12.78620078 | 12.74550800 | 12.74088174 |
| 20261031 | 12.62190242 | 12.64422148 | 12.64583680 |
| 20261101 | 13.03244921 | 13.03444330 | 13.03656928 |

N 대 R pooled gain +0.00578492, CI [-0.026273519539496214, 0.03731951544595677]. 실패 gate: ['all3_gains_nonnegative', 'conditional_CI_lower_positive', 'pooled_top5_removal_positive'].

- 실행: 신규 CNN 540회, 검증 48회(clean 6/NaN 6/finite 36). 모든 신규 모델과 중간 후단 검증 완료. 같은 학습을 다시 실행하지 말 것.
- 구현 오류 이력: 기존 root bytecode 및 불필요한 DINO provenance import를 접근 제한기가 학습 전에 차단. source-only import와 해시 고정 수학 함수 분리로 수정; 원본과 실패 로그 보존.
- 주요 해석: `interpretation.json`, `summary.json`, `review-verification.json` 참조. 이 후보의 기준 미달을 모든 6개 seed 조합의 무용성으로 일반화하지 말 것.
- AI 추가 검토: `docs/records/reviews/nested-204/response-complete.md`. 원 CLI 결과 `response.md`는 두 번째 구간만 담으므로 완전본을 읽을 것. 원문·events 모두 보존.
- 외부 검토는 비식별 설계/집계만 전달한 1회 CLI 실행. CLI 내부 응답 2개 구간, 도구/MCP/hooks 0회. 리스트 가격 표시는 청구서가 아님. 계정·비밀정보를 새 프롬프트/인계에 복사하지 말 것.
- 7범주 이력·공식 규정 정정: `coverage-audit.json`. 기존 확인 기록에는 Public 30%/Private 70%가 있음. 최종 선택 개수는 공식 기록과 대조 필요; 계정 최종 선택은 건드리지 않음.
- 과거 arm 출처 색인: `historical-arm-source-inventory.json`.192개 디렉터리/448개 메타데이터가 독립 시행 수 또는 확정된 최종 arm 수라는 뜻은 아닙니다. 204의 R/F/N은 전부 등록했습니다.

## 읽기 전용 완료 검증

```sh
.venv/bin/python -I -B experiments/01-representations/0204-cnn-nested/verify_artifacts.py
```

- 학습 명령은 이미 완료됐으며 출력 존재 시 중단하도록 구성돼 있음. `train.py`, `preflight.py`, `finish.py`, `finalize.py`를 중복 실행하지 말 것.
- 완료 직전 프로세스 점검: `process-status.json` 참조. 새 세션 시작 시 먼저 프로젝트 프로세스와 완료 파일을 확인할 것.
- 남은 일: 사용자가 제공한 새 로드맵을 별도 세션에서 검토. 이번 세션에서 새 실험·새 AI 호출을 예약하지 않았음.
- 데이터 권한: 새 설치/다운로드/결제/회사 자료/프로젝트 밖 파일 접근 금지. 100개 holdout 추가 평가·선택 금지. test는 별도 최종 모델 고정 이후 승인된 추론만 수행. 204에서는 holdout/test/전체 500개 재학습/새 제출 CSV/업로드가 없었습니다.
- 다음 실험은 새 번호·새 폴더로, 기존 best와 실패 기록 보존. 추가 외부 리뷰는 필요한 비식별 정보만 전달하는 독립 호출로 구성.
