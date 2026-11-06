# 제출 이력

최종 수상 순위는 **3위**다. 리더보드 점수는 제출별 예측 성능의 기록이며 최종 평가 순위와 구분한다.

플랫폼 제출 62건의 Public·Private RMSE다. 사용자 제공 전체 제출 이력과 파일명·점수를 대조했다(원본: `final/package/verification/submission-score-ranking.json`).

- 플랫폼 제출 번호(#1–#62)와 후보 파일 번호(`submission-candidate-NNN.csv`)는 서로 다른 번호 체계다.
- ▼는 그 시점까지의 Public 최고 갱신이다. Private 점수는 대회 종료 후 공개됐으므로 제출 당시에는 볼 수 없었다.

![제출별 RMSE](assets/score-progression.svg)

| # | 파일 | Public | Private | 변경 내용 |
|---:|---|---:|---:|---|
| 1 | `submission-candidate-002.csv` | 12.8863 ▼ | 13.0302 | 첫 제출 |
| 2 | `submission-candidate-003.csv` | 12.4480 ▼ | 12.8511 |  |
| 3 | `submission-candidate-004.csv` | 12.5258 | 12.7367 |  |
| 4 | `submission-candidate-005.csv` | 12.2764 ▼ | 12.6872 |  |
| 5 | `submission-candidate-006.csv` | 12.2224 ▼ | 12.6519 |  |
| 6 | `submission-candidate-007.csv` | 11.9686 ▼ | 12.4505 |  |
| 7 | `submission-candidate-008.csv` | 11.8898 ▼ | 12.4137 |  |
| 8 | `submission-candidate-009.csv` | 11.8407 ▼ | 12.3674 |  |
| 9 | `submission-candidate-011.csv` | 11.7353 ▼ | 12.3016 |  |
| 10 | `submission-candidate-010.csv` | 11.8391 | 12.3665 |  |
| 11 | `submission-candidate-015.csv` | 11.5818 ▼ | 12.3315 |  |
| 12 | `submission-candidate-016.csv` | 11.5829 | 12.3472 |  |
| 13 | `submission-candidate-017.csv` | 11.5648 ▼ | 12.3147 |  |
| 14 | `submission-candidate-018.csv` | 11.5231 ▼ | 12.2924 |  |
| 15 | `submission-candidate-019.csv` | 11.5451 | 12.2583 |  |
| 16 | `submission-candidate-020.csv` | 11.5290 | 12.2914 |  |
| 17 | `submission-candidate-021.csv` | 11.4879 ▼ | 12.2231 | best021: 018 앙상블 0.9 + DINOv2-L MLP 0.1 |
| 18 | `submission-candidate-022.csv` | 11.5436 | 12.2393 |  |
| 19 | `submission-candidate-023.csv` | 10.5397 ▼ | 12.0403 | Cellpose-SAM 결정립 측정 특징 + KRR, 021과 0.5 결합 (study 220) |
| 20 | `submission-candidate-025.csv` | 10.4939 ▼ | 11.3345 | 결정립 특징 v3 + TabPFN v2 |
| 21 | `submission-candidate-027.csv` | 10.3957 ▼ | 11.2697 | TabPFN v2 seed 3개 + 잡음 이미지 4-view 분할 TTA |
| 22 | `submission-candidate-028.csv` | 9.9286 ▼ | 10.8633 | 열화 이미지로 미세조정한 Cellpose (push-300) |
| 23 | `submission-candidate-029.csv` | 9.8430 ▼ | 10.6966 |  |
| 24 | `submission-candidate-030.csv` | 9.8102 ▼ | 10.6800 |  |
| 25 | `submission-candidate-031.csv` | 9.6088 ▼ | 10.3153 | 라벨 없이 학습한 화소 U-Net 상(phase) 지도 (push-400) |
| 26 | `submission-candidate-032.csv` | 9.5891 ▼ | 10.3218 |  |
| 27 | `submission-candidate-033.csv` | 9.4900 ▼ | 10.2831 |  |
| 28 | `submission-candidate-035.csv` | 9.4519 ▼ | 10.2954 | TabPFN-3.5 x U-Net 변형 4개 평균 |
| 29 | `submission-candidate-036.csv` | 9.4915 | 10.3318 |  |
| 30 | `submission-candidate-037.csv` | 9.4616 | 10.3045 |  |
| 31 | `submission-candidate-038.csv` | 9.1817 ▼ | 10.1362 | 블록 이질성 + U-Net darkness z (038) |
| 32 | `submission-candidate-039.csv` | 9.1950 | 10.0540 |  |
| 33 | `submission-candidate-044.csv` | 9.1987 | 9.9195 |  |
| 34 | `submission-candidate-045.csv` | 9.1590 ▼ | 9.9984 | 038·039·044 평균, 결합 비중 0.8 |
| 35 | `submission-candidate-046.csv` | 9.1722 | 9.9128 |  |
| 36 | `submission-candidate-047.csv` | 9.2659 | 10.1031 |  |
| 37 | `submission-candidate-048.csv` | 9.0998 ▼ | 9.9443 | Ridge·logRidge 파트너 결합 (push-900) |
| 38 | `submission-candidate-050.csv` | 9.0847 ▼ | 9.9542 |  |
| 39 | `submission-candidate-052.csv` | 9.0408 ▼ | 9.8556 |  |
| 40 | `submission-candidate-053.csv` | 9.0381 ▼ | 9.9032 | 051 + GrainSet 0.05 + GMM 0.15 (053) |
| 41 | `submission-candidate-055.csv` | 9.1151 | 9.9699 |  |
| 42 | `submission-candidate-054.csv` | 9.0397 | 9.8793 |  |
| 43 | `submission-candidate-056.csv` | 9.0664 | 9.8750 |  |
| 44 | `submission-candidate-059.csv` | 9.0592 | 9.9714 | 최종 계보에서 제외 |
| 45 | `submission-candidate-060.csv` | 9.0053 ▼ | 9.9040 | 최종 계보에서 제외 |
| 46 | `submission-candidate-057.csv` | 8.9963 ▼ | 9.8948 |  |
| 47 | `submission-candidate-061.csv` | 8.9831 ▼ | 9.8681 | 원본 열화 사본으로 증강한 Ridge 파트너 (push-1200) |
| 48 | `submission-candidate-064.csv` | 9.0166 | 9.9120 |  |
| 49 | `submission-candidate-069.csv` | 9.0020 | 9.8928 |  |
| 50 | `submission-candidate-071.csv` | 9.0238 | 9.8490 |  |
| 51 | `submission-candidate-072.csv` | 9.0288 | 9.8563 |  |
| 52 | `submission-candidate-073.csv` | 8.9864 | 9.8794 | 저잡음 행은 이질성·밝기 열 제외 (073) |
| 53 | `submission-candidate-076.csv` | 8.9647 ▼ | 9.8867 | TabPFN 12개에 cumulant·contrast 특징 추가 (076) |
| 54 | `submission-candidate-080.csv` | 8.9998 | 9.9462 |  |
| 55 | `submission-candidate-078.csv` | 9.0299 | 9.9117 |  |
| 56 | `submission-candidate-082.csv` | 8.9343 ▼ | 9.9213 | 076 + 잡음 구간 잔차 CNN 보정 (082) |
| 57 | `submission-candidate-083.csv` | 8.9179 ▼ | 9.9175 | 082 + TabPFN 블록에 sigma 2·4 cumulant 열 추가 (083) |
| 58 | `candidate-flow38-v1.csv` | 8.8706 ▼ | 9.7186 | Cellpose flow·확신도 연속 특징 38개 (F38) |
| 59 | `candidate-source-flow38.csv` | 8.8626 ▼ | 9.7727 | train 저잡음 250장으로 학습한 source encoder 2개 |
| 60 | `candidate-source-flow38-tabreplace.csv` | 8.8761 | 9.7159 |  |
| 61 | `candidate-source-flow38-hybridkernel.csv` | 8.8647 | 9.7818 |  |
| 62 | `candidate-source-flow38-boostedreadout.csv` | 8.8196 ▼ | 9.6967 | **최종 선택**: 마지막 readout을 HGB 3개로 교체 |
