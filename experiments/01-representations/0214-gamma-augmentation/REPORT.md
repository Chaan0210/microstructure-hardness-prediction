# 214: 작은 감마 변동의 단일 비교

**사전 단독 기준 미달. 이번 단일 감마 정책은 채택하지 않고 best021 유지. CNN 단독과 고정 첫 seed 치환을 구별해 아래에 보고한다. 추가 강도 탐색·재분할·다른 증강 학습 없이 이번 비교를 완료했다.**

| 비교 | 대조 RMSE | 감마 RMSE | 이득(대조−감마) | 조건부 paired 95% CI | 상위 5 ID 제외 이득 | 개선 fold |
|---|---:|---:|---:|---:|---:|---:|
| CNN 단독 | 15.03231784 | 14.96353623 | +0.06878162 | [-0.12126641, +0.25142103] | -0.02476630 | 3/5 |
| 고정 첫 seed 치환 | 12.78620078 | 12.78508851 | +0.00111227 | [-0.01937287, +0.02044726] | -0.00710592 | 3/5 |

사전 단독 기준: 이득 ≥0.02, 개선 ≥3/5 fold, paired95%CI 하한 >0, 상위5개 squared-error 이득 ID 제외 후 이득 >0, 기술 검증 통과. 실제 판정: `{'technical_validation': True, 'gain_at_least_0_02': True, 'at_least3_folds': True, 'conditional_CI_lower_positive': False, 'top5_removed_positive': False}`. 상위5개는 두 비교에서 각각 선정했으며 목록과 fold별 이득은 summary.json에 있다. 모든400 ID의 예측·잔차·제곱오차 이득은 ID-residuals.json, OOF.npz, ensemble-OOF.npz로 제공한다.

## 변경한 정책과 원본 화질 확인

213의 밝기는 단독 평균 개선에도 앙상블 이득이 작고 상위5개 제외 시 악화했다. 지시에 따라213 재분할은 하지 않았다. 기존206 대비 감소,213 덧셈 밝기와 구분되는 `x**gamma` 한 종류를214로 시험했다. gamma 커널 계수나 기존 대비 증강의 재실행이 아니다.

원본 data/train에서 기존 개발400 ID만 읽어 모두256×256 grayscale임을 확인했다. 나머지100 holdout과 test는 열지 않았다. 원본 및 개발128캐시의 화질·밝기 통계를 확인했으며 원본에서 기존 BOX128/uint8 → float32/255를 재계산해 캐시와 exact 일치를 확인했다. **원본256과 기존 모델 입력128 해상도는 그대로 유지했다.** 추가 resize·crop·종횡비·좌표 변환은 없다. 실제 학습 감마는 기존 BOX128 후 float 입력에서 적용한다. native256 감마는 수치 진단만 수행하며 새 입력해상도 실험이 아니다.

전처리는 각 fold train320 전체에서 얻은 고정 평균·표준편차로 정규화한다. per-image 표준화가 없어 감마 변화는 상쇄되지 않으며 TinyCNN readout의 입력 mean/std도 바뀐다. 최종 모델은 기존 `cnn_controls.TinyCNN` 91,905파라미터 그대로다.

강도 산식과 노출은 성능을 보기 전에 잠갔다. 각 train영상의 `sensitivity=-mean(x*log(x))`를 계산하고 `tau=min(0.02,0.1*IQR(train image mean)/median(sensitivity))`, `gamma_low=exp(-tau)`, `gamma_high=exp(tau)`로 정했다. 0의 xlogx 항은0이다. 평균 밝기 변동의1차 근사가 train 평균 IQR의10% 정도가 되게 하되 log-gamma를0.02로 제한하는 작은 단일 정책이다. 타깃·validation·OOF 성능은 강도 선택에 쓰지 않았다.

fold별 [low,high]는 [[0.985768199, 1.014437318], [0.985544503, 1.014667511], [0.98522681, 1.01499474], [0.984977067, 1.015252113], [0.986069918, 1.014126897]]다. ID hash와 epoch의4주기로 원본40회, high감마(어둡게)20회, low감마(밝게)20회 노출한다. 즉 전체 노출50%, 방향별25%이며 추가 RNG를 소비하지 않는다. 다른 강도·확률·감마와 밝기 조합은 실행하지 않았다.

## 범위·클리핑·양자화 및 측정 변화

native256과 model128 각각 모든 fold train영상에서 float32 범위[0,1], finite값, 양끝값, float64 reference 오차<1e-6, 동일 shape, unique intensity의 순위/동률 보존 및 영상별 Q10/Q50/Q90 상대 mask exact를 검사했다. 최대 픽셀 이동은 0.00556850, 클리핑0, 실제 재양자화0이다. 정규화 후 RMS 변화는 모델입력에서 약0.0630–0.0679 표준편차로0이 아니다.

가상 uint8 재양자화에서는 최대 오차 0.00196075와 영상당 최대1개 unique level의 합쳐짐이 관찰됐다. 실제float 학습에서는 이런 양자화를 적용하지 않았고 순위/동률이 보존됐다. 기존 BOX 처리의 uint8 양자화는 두 군 공통이다.

그러나 **단조변환이어도 고정 임계값의 어두운 상 분율과 대비·gradient energy는 보존되지 않았다.** 아래는 fold train영상에서 관측한 절대 분율 변화의 최대값(%p)이다. 임계값0.25/0.35/0.50은 모든 영상에 고정했고 Q10은 각 fold train 픽셀에서만 계산해 원본/변환 모두 같은 값을 적용했다. 영상별 변화·평균 및 분위수는 preflight.json에 모두 기록했다.

| 해상도 | threshold0.25 | threshold0.35 | threshold0.50 | train pixel Q10 |
|---|---:|---:|---:|---:|
| native256 진단 | 0.1099 | 1.0330 | 2.5360 | 4.0497 |
| model128 실제 입력 | 0.1160 | 0.8179 | 3.1677 | 5.8411 |

영상별 gradient energy 비율(변환/원본)은 native 0.983011–1.017086, model128 0.983554–1.016518다. 좌표·스케일·공간 배열은 그대로지만 물리적 상 분율이나 결정립 경계 측정의 정확성이 보장되지 않는다. 특히 이 dark fraction은 고정 intensity threshold 기반 대리 측정이며 실제 상 정답이 아니다. 감마의 물리적 라벨 보존을 입증했다고 주장하지 않으며, 이 작은 정책의 일반화 효과만 통제 비교했다. 변환 뒤 중심화·대비 복원·역감마 등 별도 보정을 추가하지 않았다.

## 정확한 대조와 검증

213과 같은 원본400의5fold(320/80), seed20261003+fold, D4, MSE, batch32, AdamW lr0.002/weight_decay0.01, cosine80epoch/eta_min0.0001, gradient clip5, 마지막5checkpoint 동일 평균을 유지했다. validation에는 감마를 적용하지 않았다.

- 실제 새 학습은 **7회/5,600update**: 감마 후보5회, 대조fold0 80epoch 재현1회, 감마fold0 validation타깃 NaN 교란 재현1회. 기존 대조5개를 재사용했다. 강도탐색·재분할·밝기 재학습0회다.
- 206 당시 소스·대조54개 hash를 검증했다. 대조fold0의 마지막5개state·평균state·OOF·800회 order/D4/LR/loss/gradient norm/CPU·MPS RNG가 exact 재현됐다. 대조5개 state는 run009와 exact이고 전부 재추론했으며206의 ID/타깃/fold/controlOOF도 exact다.
- 양 군은 fold별 초기 state 및 모든800 order/D4/LR/CPU·MPS RNG가 동일하다. 각 train영상의 노출20/20/40을 검증했다.
- 대조·후보10개 모두 독립5state 평균 및 CPU/MPS 재로드 통과. MPS 최대오차0, CPU 최대오차 0.0000305176다.
- 강도 산식은 validation이미지 NaN 변경에 불변, 정규화는 validation타깃 NaN 변경에 불변이다. 감마fold0 전체80epoch에서 validation타깃 NaN 교란 후 state/OOF/stream/최종optimizer/scheduler exact를 확인했다.
- 고정 첫seed치환은 `Q=P021+0.081*slope_fold*(new_first_CNN-old_first_CNN)`이다. 계수와 best021을194/177 계보에 대조해 exact임을 확인했다. 나머지2seed·보정·잔차 고정이며 전체 후단 재학습이나 새로운 가중치 탐색이 아니다.
- 동일400 ID paired bootstrap10,000회(seed202610214), CI와 상위5개 제외 및 ID별 이득을 직접 재계산해 검증했다. 소스/plan은 학습 전 잠갔고 학습·평가 정책은 변경하지 않았다.

중복 train1,600관측 RMSE는 대조 9.885685, 감마 10.101977다. 같은400장을 여러 실험에 누적·적응적으로 재사용했으므로 독립검증이 아니며 조건부 CI는 선택편향·seed·재학습 변동을 보정하지 않는다.

## 범위와 다음 상태

사전 단독 기준 미달. 이번 단일 감마 정책은 채택하지 않고 best021 유지. 자동 재분할·다른 강도·추가 증강은 실행하지 않았다. 213 밝기 재분할도 진행하지 않는 최신 지시를 유지한다. 감마214는 이번 고정 정책의 수행 완료이며 모든 감마 설정이나 화질 증강의 결론으로 일반화하지 않는다. Blur는 다음 미수행 항목으로만 남기며 실행 대기열을 만들지 않는다.

기존파일 755개와 원본400 hash를 보존했고 best021 SHA256 `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b`가 유지됐다. 기존 uv .venv, 무료CPU2 threads/MPS를 사용했으며 측정 최대 RSS+MPS 1.791GB, swap증가0, 비용0이다. 설치·다운로드·외부Claude·test·이전holdout·전체500학습·CSV생성/수정·제출은 없었다. 새 관리표는 docs/records/roadmap-audit/214-management.json 및 roadmap-30-after-214.md에 저장하며 과거 스냅샷을 덮어쓰지 않는다. 차단 사항 없음.

근거: plan.json/code-lock.json, preflight.py/preflight.json, engine.py, training-complete.json, validation.json, final-audit.json, summary.json, photometric-diagnostics-summary.json, ID-residuals.json 및 fold별 state/streams.
