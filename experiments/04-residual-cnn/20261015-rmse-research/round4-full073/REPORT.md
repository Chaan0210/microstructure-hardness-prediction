# 073 전체 구성: 내부 가중치 선택을 적용한 CNN 검증

기준은 원래400개·5-fold의 수정된073이다. 3개 partition을 합친 기존9.58169와 이 표의 단일 partition 점수를 직접 비교하면 안 된다.
기준 RMSE: **9.54724515**. 외부 예측은 기존 구성 요소를 그대로 사용하고, 증강 원본은 각 학습 fold 안으로 제한했다.

정상 샘플에서 `073 = .16 × 021 + .84 × T`이다. 추가는 `073 + λr`, 대체는 `T + λr`이며, λ는 내부B107에서만 선택했다. 분할 실패2개는 모든 방식에서 기존021 예측을 유지했다.

| 모델 | 방식 | RMSE | 073 대비 개선 | 대응 대조군 대비 보정 효과 |
|---|---|---:|---:|---:|
| 073 | reference | 9.547245 | +0.000000 | — |
| remove021 | reference | 9.493439 | +0.053807 | — |
| aux-mean2 | addition | 9.445883 | +0.101362 | +0.101362 |
| aux-mean2 | replacement | 9.375906 | +0.171339 | +0.117533 |
| aux-seed0 | addition | 9.429668 | +0.117577 | +0.117577 |
| aux-seed0 | replacement | 9.354567 | +0.192678 | +0.138871 |
| aux-seed1 | addition | 9.458337 | +0.088908 | +0.088908 |
| aux-seed1 | replacement | 9.395518 | +0.151727 | +0.097920 |
| cnn-mean2 | addition | 9.457536 | +0.089709 | +0.089709 |
| cnn-mean2 | replacement | 9.390307 | +0.156938 | +0.103132 |
| cnn-seed0 | addition | 9.463887 | +0.083358 | +0.083358 |
| cnn-seed0 | replacement | 9.400638 | +0.146607 | +0.092801 |
| cnn-seed1 | addition | 9.458016 | +0.089229 | +0.089229 |
| cnn-seed1 | replacement | 9.391283 | +0.155962 | +0.102155 |
| context-mean2 | addition | 9.496568 | +0.050677 | +0.050677 |
| context-mean2 | replacement | 9.472176 | +0.075069 | +0.021263 |
| context-seed0 | addition | 9.497953 | +0.049292 | +0.049292 |
| context-seed0 | replacement | 9.473266 | +0.073979 | +0.020173 |
| context-seed1 | addition | 9.495208 | +0.052037 | +0.052037 |
| context-seed1 | replacement | 9.471227 | +0.076018 | +0.022212 |
| stats-mean2 | addition | 9.516544 | +0.030701 | +0.030701 |
| stats-mean2 | replacement | 9.455920 | +0.091325 | +0.037519 |
| stats-seed0 | addition | 9.517406 | +0.029839 | +0.029839 |
| stats-seed0 | replacement | 9.456923 | +0.090323 | +0.036516 |
| stats-seed1 | addition | 9.515692 | +0.031553 | +0.031553 |
| stats-seed1 | replacement | 9.454934 | +0.092311 | +0.038505 |
| adaptive021 | weight-only control | 9.489906 | +0.057340 | — |
| aux-mean2 | joint inner weights | 9.383527 | +0.163718 | +0.106379 |
| aux-seed0 | joint inner weights | 9.365816 | +0.181429 | +0.124089 |
| aux-seed1 | joint inner weights | 9.398528 | +0.148717 | +0.091377 |
| cnn-mean2 | joint inner weights | 9.401927 | +0.145318 | +0.087978 |
| cnn-seed0 | joint inner weights | 9.411306 | +0.135939 | +0.078599 |
| cnn-seed1 | joint inner weights | 9.393940 | +0.153306 | +0.095966 |
| context-mean2 | joint inner weights | 9.468095 | +0.079150 | +0.021811 |
| context-seed0 | joint inner weights | 9.469331 | +0.077914 | +0.020574 |
| context-seed1 | joint inner weights | 9.467008 | +0.080237 | +0.022897 |
| stats-mean2 | joint inner weights | 9.456096 | +0.091149 | +0.033810 |
| stats-seed0 | joint inner weights | 9.456956 | +0.090289 | +0.032950 |
| stats-seed1 | joint inner weights | 9.455252 | +0.091993 | +0.034653 |
| imagenet-direct | addition | 9.558789 | -0.011543 | -0.011543 |
| imagenet-direct | replacement | 9.519213 | +0.028032 | -0.025774 |
| imagenet-residual | addition | 9.551000 | -0.003755 | -0.003755 |
| imagenet-residual | replacement | 9.490914 | +0.056331 | +0.002525 |
| micronet-direct | addition | 9.545247 | +0.001998 | +0.001998 |
| micronet-direct | replacement | 9.469923 | +0.077322 | +0.023515 |
| micronet-residual | addition | 9.579368 | -0.032123 | -0.032123 |
| micronet-residual | replacement | 9.529275 | +0.017970 | -0.035836 |
| cnn-noise | addition | 9.409595 | +0.137650 | +0.137650 |
| cnn-noise | replacement | 9.333580 | +0.213665 | +0.159859 |
| aux-noise | addition | 9.319650 | +0.227596 | +0.227596 |
| aux-noise | replacement | 9.258787 | +0.288458 | +0.234651 |
| stats-noise | addition | 9.489403 | +0.057842 | +0.057842 |
| stats-noise | replacement | 9.444878 | +0.102368 | +0.048561 |
| context-noise | addition | 9.501336 | +0.045909 | +0.045909 |
| context-noise | replacement | 9.450545 | +0.096700 | +0.042894 |

대체 방식의 대응 대조군은021을 제거한T이다. 공동 가중치 방식은 내부에서021 비중만 정한 대조군과 비교한다. 이 비교가 음수라면 전체073보다 좋아도 CNN이 개선을 만들었다고 해석할 수 없다.

| 모델·방식 | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---:|---:|---:|---:|---:|
| 073 | 9.895929 | 9.142149 | 10.533605 | 8.324342 | 9.694826 |
| remove021 | 10.032998 | 9.197346 | 10.319304 | 8.322369 | 9.467037 |
| aux-mean2-addition | 9.729608 | 9.132036 | 10.301005 | 8.244020 | 9.694826 |
| aux-mean2-replacement | 9.801313 | 9.171768 | 10.100108 | 8.228853 | 9.467037 |
| aux-seed0-addition | 9.617207 | 9.114067 | 10.345006 | 8.247881 | 9.694826 |
| aux-seed0-replacement | 9.664222 | 9.165221 | 10.139120 | 8.228889 | 9.467037 |
| aux-seed1-addition | 9.819714 | 9.142149 | 10.249534 | 8.261501 | 9.694826 |
| aux-seed1-replacement | 9.917287 | 9.190514 | 10.028713 | 8.268059 | 9.467037 |
| cnn-mean2-addition | 9.498007 | 9.135047 | 10.536365 | 8.362424 | 9.623857 |
| cnn-mean2-replacement | 9.599519 | 9.208642 | 10.316447 | 8.346008 | 9.373075 |
| cnn-seed0-addition | 9.486456 | 9.136022 | 10.528448 | 8.338357 | 9.694826 |
| cnn-seed0-replacement | 9.593757 | 9.209095 | 10.305054 | 8.318258 | 9.467037 |
| cnn-seed1-addition | 9.515752 | 9.137263 | 10.545246 | 8.389749 | 9.572935 |
| cnn-seed1-replacement | 9.607873 | 9.216387 | 10.329568 | 8.377571 | 9.319030 |
| context-mean2-addition | 9.827338 | 9.142149 | 10.373819 | 8.316418 | 9.694826 |
| context-mean2-replacement | 9.986216 | 9.197346 | 10.210620 | 8.391266 | 9.467037 |
| context-seed0-addition | 9.831807 | 9.142149 | 10.375156 | 8.317376 | 9.694826 |
| context-seed0-replacement | 9.987017 | 9.197346 | 10.211848 | 8.394970 | 9.467037 |
| context-seed1-addition | 9.823286 | 9.142149 | 10.372545 | 8.315031 | 9.694826 |
| context-seed1-replacement | 9.985896 | 9.197346 | 10.209458 | 8.387705 | 9.467037 |
| stats-mean2-addition | 9.823528 | 9.142149 | 10.456990 | 8.313363 | 9.709883 |
| stats-mean2-replacement | 9.948712 | 9.201207 | 10.231203 | 8.293846 | 9.485127 |
| stats-seed0-addition | 9.824330 | 9.142149 | 10.458548 | 8.314692 | 9.710479 |
| stats-seed0-replacement | 9.949604 | 9.200991 | 10.232631 | 8.296054 | 9.485929 |
| stats-seed1-addition | 9.822736 | 9.142149 | 10.455475 | 8.312050 | 9.709267 |
| stats-seed1-replacement | 9.947832 | 9.201425 | 10.229826 | 8.291678 | 9.484304 |
| adaptive021 | 9.922236 | 9.189832 | 10.319304 | 8.322369 | 9.572769 |
| aux-mean2-joint | 9.736361 | 9.170172 | 10.100108 | 8.228853 | 9.572769 |
| aux-seed0-joint | 9.614417 | 9.165221 | 10.139120 | 8.228889 | 9.572769 |
| aux-seed1-joint | 9.835146 | 9.184649 | 10.028713 | 8.268059 | 9.572769 |
| cnn-mean2-joint | 9.545851 | 9.208642 | 10.316447 | 8.346008 | 9.485456 |
| cnn-seed0-joint | 9.540994 | 9.209095 | 10.305054 | 8.318258 | 9.572769 |
| cnn-seed1-joint | 9.552607 | 9.216387 | 10.329568 | 8.377571 | 9.388970 |
| context-mean2-joint | 9.872340 | 9.189832 | 10.210620 | 8.391266 | 9.572769 |
| context-seed0-joint | 9.873849 | 9.189832 | 10.211848 | 8.394970 | 9.572769 |
| context-seed1-joint | 9.871358 | 9.189832 | 10.209458 | 8.387705 | 9.572769 |
| stats-mean2-joint | 9.858659 | 9.201207 | 10.231203 | 8.293846 | 9.579561 |
| stats-seed0-joint | 9.859272 | 9.200991 | 10.232631 | 8.296054 | 9.579946 |
| stats-seed1-joint | 9.858055 | 9.201425 | 10.229826 | 8.291678 | 9.579157 |
| imagenet-direct-addition | 9.951490 | 9.142149 | 10.533605 | 8.324342 | 9.694826 |
| imagenet-direct-replacement | 10.079742 | 9.197346 | 10.319304 | 8.322369 | 9.546456 |
| imagenet-residual-addition | 9.870340 | 9.131352 | 10.579617 | 8.390742 | 9.642093 |
| imagenet-residual-replacement | 9.979472 | 9.199420 | 10.363375 | 8.367841 | 9.420603 |
| micronet-direct-addition | 9.886286 | 9.142149 | 10.533605 | 8.324342 | 9.694826 |
| micronet-direct-replacement | 9.867903 | 9.197346 | 10.319304 | 8.322369 | 9.522642 |
| micronet-residual-addition | 9.867976 | 9.142149 | 10.608782 | 8.445695 | 9.694826 |
| micronet-residual-replacement | 9.991892 | 9.197346 | 10.408144 | 8.464753 | 9.467037 |
| cnn-noise-addition | 9.504630 | 9.274767 | 10.401219 | 8.355212 | 9.399370 |
| cnn-noise-replacement | 9.606429 | 9.285218 | 10.179326 | 8.336877 | 9.162839 |
| aux-noise-addition | 9.447869 | 9.127364 | 9.938312 | 8.303533 | 9.694826 |
| aux-noise-replacement | 9.561178 | 9.173375 | 9.731666 | 8.290276 | 9.467037 |
| stats-noise-addition | 9.752732 | 9.142149 | 10.492578 | 8.330317 | 9.594830 |
| stats-noise-replacement | 9.896975 | 9.212837 | 10.276416 | 8.327202 | 9.394417 |
| context-noise-addition | 9.811932 | 9.142149 | 10.308998 | 8.441502 | 9.694826 |
| context-noise-replacement | 9.951846 | 9.197346 | 10.109340 | 8.432677 | 9.467037 |

95% 구간은 저장된 외부 예측에 대한 ID bootstrap 조건부 구간이다. 이번 비교군 안의 다중 비교를 고려한 동시 구간도 `simultaneous-bands.json`에 별도로 계산했다. 기존에 탐색한400개 샘플이며, 과거 연구 선택 편향과 재학습 변동을 모두 포함하지 않는다. 다른seed도 같은 데이터이므로 독립 검증셋으로 세지 않는다.

확실한 채택 여부는 `results.json`의 대조군 비교와 반복seed, fold별 방향, 상위오차 사례 의존성을 함께 검토해야 한다. 자동으로 기존061/073을 교체하거나 제출하지 않는다.

재현 명령은 [RUN.md](RUN.md), 사전 계획은 [PLAN.md](PLAN.md)와 [AMENDMENT-01.md](AMENDMENT-01.md), 전체 수치와 대조군 비교는 [results.json](results.json)에 있다. 각 모델 디렉터리의 `verification.json`에 누수·재로딩·정답 변조 검증이 기록된다.
