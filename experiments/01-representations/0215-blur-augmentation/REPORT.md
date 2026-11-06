# 215: 작은 블러 한 정책의 비교

**사전 단독 기준 미달. 이번 블러 정책을 채택하지 않고 best021 유지. 추가 강도·재분할·다른 증강은 실행하지 않고 이번 비교를 완료했다.**

| 비교 | 대조 RMSE | 블러 RMSE | 이득(대조−블러) | 조건부 paired 95% CI | 상위 5 ID 제외 이득 | 개선 fold |
|---|---:|---:|---:|---:|---:|---:|
| CNN 단독 | 15.03231784 | 15.11121241 | -0.07889457 | [-0.24846616, +0.09653134] | -0.15519860 | 2/5 |
| 고정 첫 seed 치환 | 12.78620078 | 12.79482955 | -0.00862878 | [-0.02370560, +0.00599409] | -0.01528429 | 1/5 |

사전 기준은 단독 이득≥0.02, 개선fold≥3/5, CI하한>0, 상위5개 squared-error 이득ID 제외 후 이득>0 및 기술검증 통과였다. 결과: `{'technical_validation': True, 'gain_at_least_0_02': False, 'at_least3_folds': False, 'conditional_CI_lower_positive': False, 'top5_removed_positive': False}`. 상위5개는 각 비교에서 따로 제거했다. ID목록·fold이득은 summary.json, 모든400 ID의 예측·잔차·제곱오차 이득은 ID-residuals.json/OOF.npz/ensemble-OOF.npz에 있다.

## 사전 고정과 화질 EDA

블러 학습 ablation은 기존 로드맵에서 미실시였다. 071/072의 복원·098 NLM 측정 대조와 다른 정책이다. 개발400 원본256×256 grayscale만 읽었고 기존 BOX128 uint8→float32/255를 다시 계산해 캐시와 exact를 확인했다. 원본256과 모델128 해상도, 좌표·종횡비는 그대로이며 추가 crop/resize를 하지 않았다. 실제 블러는 기존 모델128 입력에서 적용한다. native256은 같은3×3픽셀 필터의 별도 진단이며 물리적 길이가 같다는 의미는 아니다.

train320의 gradient energy E를 `mean(diffX²)+mean(diffY²)`로 정의하고, `a=min(0.04,0.1*max(0,1-Q10(E)/Q50(E)))`를 float32로 고정했다. 작은 train 선명도 변동을 근거로 이웃 평균의 강도를 제한한 한 산식이며 결과·validation·타깃을 보지 않았다. 각 축의 필터는 `[a,1-2a,a]`, 경계처리는 reflect, 두 축으로 차례 적용한다. 정규화된 양의 국소 평균이고 Gaussian filter라고 부르지 않는다.

fold별 a는 [0.038620289, 0.039083432, 0.037343509, 0.038004223, 0.039117061]다. 축별 kernel RMS반경 sqrt(2a)는 약0.273–0.280픽셀이며, 이 작은 공간범위가 측정 영향도 작다는 보장은 없다. ID hash와 epoch의2주기로 원본40회/블러40회, 노출50%를 고정했다. 다른 강도·확률·감마/밝기와의 조합은0회다. 블러는 fold train 고정정규화 이전에만 적용하고 validation은 원본을 사용했다.

## 경계·선명도·결정립/상 측정에 미친 영향

float범위[0,1], finite, shape동일, kernel합1, float64연산 대조 오차<1e-6, 평균오차<1e-6, 추가 clipping0/재양자화0을 확인했다. 고정 train정규화 뒤에도 입력차이는 남았다. 가상uint8반올림 오차는 최대 0.00196081이나 실제 학습에서는 반올림하지 않았다. D4와 블러 순서의 수치차이도1e-6미만이었다.

| 진단 | model128 실제 입력 | native256 진단 |
|---|---|---|
| gradient energy 비율(변환/원본) | [0.679688036441803, 0.8534396290779114] | [0.6671547889709473, 0.8227242231369019] |
| Laplacian 분산 비율 | [0.6333963871002197, 0.7602642774581909] | [0.6303726434707642, 0.7224947810173035] |
| threshold0.50 어두운 분율 최대 절대변화 | 2.5208%p | 3.0106%p |
| threshold0.50 원래1–4픽셀 연결요소 완전소실 최대수/영상 | 166 | 960 |
| watershed 영역수 변화 범위(면적≥16) | -66~+5 | -21~+8 |

고정threshold0.25/0.35/0.50와 fold train pooledQ10에서 분율,8연결요소수,1–4픽셀 작은 영역수·완전소실을 전train에서 진단했다. 원래 component의 모든 픽셀이 변환 후 threshold밖으로 벗어나면 완전소실로 정의한다. 합쳐짐/분리의 물리적 추적이나 실제 상 정답은 아니다. 각 fold·해상도·ID별 원본/변환 측정값은 phase-*.npz, sharpness-*.npz, 전체요약은 preflight.json에 있다.

결정립 대리값은 fold마다 라벨과 무관한 IDhash로 train12장을 먼저 선택하고, 기존 `grain_features.segment(image_float*255,sigma=1)`의 watershed를 그대로 사용했다. 총120쌍(60train관측×2해상도)에서 면적≥16 영역수, 평균등가직경, 평균d^−1/2, 경계분율을 비교했다. 재양자화 없이 float를 넘겼으며 기존 측정코드의 자체Gaussian1 평활화는 양쪽에 동일하다. 상세 before/after는 grain-measurement-deltas.json과 measurement-diagnostics-summary.json에 있다. 기존097/098/183에서 검증되지 않은 측정법이므로 실제 결정립 변화를 측정했다고 주장하지 않는다.

**블러는 좌표와 해상도를 유지해도 선명도·경계와 작은 어두운 영역·결정립/상 대리 측정값을 바꿨다. 구조나 hardness 라벨 보존은 입증되지 않았다.** 이번 실험은 원래타깃을 유지한 작은 증강 정책의 경험적 효과를 평가한 것이며, 진단을 보고 강도를 다시 선택하지 않았다.

## 학습과 검증

213/214와 같은 원본400의5fold(320/80), seed20261003+fold, TinyCNN91,905파라미터/128입력, D4, MSE, batch32, AdamW0.002/weight_decay0.01, cosine80epoch/eta_min0.0001, gradient clip5, 마지막5state동일평균이다. 바뀐 학습변수는 입력 블러뿐이다.

- 새 학습 **7회/5,600update**: 후보5회, 대조80epoch재현1회, 블러validation타깃NaN교란80epoch재현1회. 기존 대조5개 재사용. 탐색·재분할0회.
- 206 역사적 소스·대조54개 hash 확인. fold0대조의 state/epoch76–80/OOF 및 모든800 order/D4/LR/loss/gradientnorm/CPU·MPS RNG exact. 대조5개는 run009state와 exact이며 재추론과206OOF/ID/타깃/fold도 exact.
- 후보/대조 초기state 및 모든800 order/D4/LR/RNG 동일. 각 train영상 blur40/original40회 노출 확인. 전fold train통계/강도는 validation변경에 불변.
- 10개모델의5state독립평균 및 CPU/MPS재로드 통과. MPS오차0, CPU최대오차 0.0000305176. fold0라벨NaN교란 뒤80epoch state/OOF/stream/optimizer/scheduler exact.
- 고정치환 `Q=P021+0.081*slope_fold*(new_first_CNN-old_first_CNN)`을194/177계보와 대조했다. 나머지2seed·보정·잔차 고정이다. **약한 첫 seed 치환 이득을 전체 모델 개선으로 취급하지 않는다.**
- 400ID paired bootstrap10,000회(seed202610215), CI/상위5제외/ID잔차를 직접재검산했다. 같은400을 반복선택한 조건부CI로서 누적선택편향·seed·재학습 불확실성을 보정하지 않는다.

중복train1,600관측 RMSE는 대조 9.885685, 블러 9.931937. 측정 최대 RSS+MPS 1.876GB, swap증가0, CPU2threads/MPS, 기존uv.venv, 비용0이다.

기존 870파일·원본400hash·best021 SHA256 `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b` 보존. 설치/다운로드/AI/test/holdout/전체500/CSV/제출 없음. 비교 후 정지하며213–215요약·남은부분8개·다음우선가설1개는 docs/records/roadmap-audit/handoff-after-215/HANDOFF.md에 정리한다. 미실시블러를 완료로 바꾼 최신범위는 완료22·부분8·미실시0이다. 차단사항 없음.
