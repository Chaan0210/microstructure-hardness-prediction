# Mean-only control: reject_mean_only_stop_no_tuning_keep_best021

218 A/B/C 확인 뒤 추가한 사후 탐색이다. 독립 확증이 아니다. best021 유지.

| 분할 | KRR D-A gain | 고정 결합 D-A gain | 고정 결합 CI95 |
|---|---:|---:|---|
| original | +0.051141665 | +0.017456060 | [-0.004569763497731483, 0.054804557336160666] |
| 20261031 | +0.019665695 | +0.007463340 | [-0.006339156946017121, 0.02714133986763797] |
| 20261101 | +0.025632630 | +0.008742785 | [-0.0025554954173026886, 0.02402782836906647] |
| pooled_ID_meanSE | +0.032190514 | +0.011219171 | [-0.0038215444893382688, 0.035197574671483295] |

| pooled 고정 결합 D-A | n | baseline RMSE | mean-only RMSE | gain | improved/worsened IDs |
|---|---:|---:|---:|---:|---:|
| all400 | 400 | 12.814628145 | 12.803408973 | +0.011219171 | 193/207 |
| empty3 | 3 | 22.644461386 | 21.922989509 | +0.721471876 | 1/2 |
| other397 | 397 | 12.711442207 | 12.709608826 | +0.001833381 | 192/205 |

D는 A의 HP 열 3행만 바꾸고 flag 2열은 0으로 유지했다. 동일 모델·9개 grid·inner seed·ID·fold를 사용했다. A/B/C는 재학습 없이 저장 OOF를 재사용했고 기존 점수와 CI를 재현했다.

새 본실험 KRR 420회, 검증 KRR 168회. 15개 모델 exact reload 및 독립 kernel solve, 모든 fit의 train-only 통계, 각 분할 fold0 전체 nested label/feature poison 검증 통과. 세부 수치는 validation.json 참조.

사전 practical gate: pooled fixed-ensemble gain>=0.02, 세 분할 양수, remaining397 비악화, 조건부CI 하한>0. {"practical_gain_at_least_0.02": false, "all_split_positive": true, "remaining397_nonnegative": true, "conditional_CI_lower_positive": false}

3장/397장과 A/B/C/D 전체 비교는 results.json 및 empty3-effects.json/ID-effects.json에 있다. 평균 squared-error 기준 factorial interaction도 기록했다. 결과는 후단을 고정한 proxy이며 전체 재학습 효과가 아니다.

추가 threshold/feature/weight 탐색 없이 종료한다. rawdata/test/holdout/전체500/새 제출 CSV/설치/외부통신/비용 없음. 기존 소스·입력 해시 및 best021 파일 stat 보존은 completion.json에 기록한다.

타깃 층화 문제는 기존 fold 분포만 읽기 진단했다. 새로운 분할이나 점수 기반 seed 선택은 없다.

## 해석 및 종료

고정 결합의 pooled SSE 개선 중 TRAIN_000206 한 장의 순기여는 86.60%다. 이는 사후 영향 진단이며 그 행을 제거해 새 기준을 만들거나 학습에서 제외하지 않았다. 알려진3장 중 pooled 기준 개선1장/악화2장이고, 나머지397장의 개선은 RMSE 0.001833381에 그친다.

D의 선택된 bandwidth/alpha는 세 분할 15fold 모두 A와 같았다. 이번 개선을 hyperparameter 선택 변경 때문이라고 해석할 근거는 없다. 218의 flag 포함 조건 악화와 값 교정만의 효과는 구분되지만, 현재 값 교정의 이득도 작고 조건부 CI가0을 포함하여 채택하지 않는다.

계산·검증 구간 2.574초(프로세스 시작 및 문서 작성 제외), peak RSS 457.1MiB. CPU thread 최대2. 기존 입력/소스 해시와 best021 파일 stat 동일, 차단된 접근0. best021 내용은 읽지 않았고 쓰기 guard가 이 디렉터리 밖의 변경을 금지했다.
