# 212 고정 손실 재분할 안정성 사전 계획

211의 MSE 대 2 × Huber(delta 1), 중앙 gradient scale 정의, seed 20261003 + fold, 타깃 표준화·D4·AdamW·80 epoch·마지막 5 epoch 평균을 유지한다. 기존 20261031/20261101 분할 각각 5-fold에서 두 군을 학습한다. 원본 211은 다시 학습하지 않고 평균 개선과 CI 기준 미달 판단을 보존한다.

주 학습 20회, 두 재분할 fold 0 양 군 validation label poison 검증 학습 4회다. 기존 209의 MSE state·OOF·800 update RNG·최종 optimizer를 직접 비교하고 양 군의 초기 state·입력·RNG 경로를 일치시킨다.

세 분할 각각과 두 재분할/세 분할 pooled를 보고한다. Pooled는 squared error의 평균에서 RMSE를 계산한다. 같은 ID의 분할 반복을 함께 재표집하는 400 ID cluster bootstrap 10,000회를 사용하며, 211의 draw(seed 202610211)를 재사용해 원본 CI를 보존한다. 1,200개 독립 표본이나 예측 평균 앙상블로 해석하지 않는다. 이득 상위 5개 ID 및 원본 211의 상위 5개 고정 제외 감도를 보고한다.

새 재분할 각각에는 기존 단독 기준(gain≥0.02, 3 fold 이상 개선, CI 하한>0, 상위 5개 제외 이득>0)을 적용한다. 두 개 및 세 개 pooled에는 gain/CI 하한/상위 5개 제외 이득이 모두 양수인지 확인한다. 결과와 관계없이 안정성 판단 후 정지한다. 원본 211의 기준 미달 사실은 소급 변경하지 않는다.

고정 앙상블은 기존 첫 seed 출력 치환이다. 추가 blend나 후단 전체 재학습이 아니다. 다른 seed·보정·잔차·계수는 유지한다. beta/delta/배율·손실 탐색은 없다. 기존 uv·무료 CPU 2 threads/MPS 6 GB 이하, 원본 보존, 프로젝트 내부 새 기록만 허용한다. 설치·다운로드·외부 AI·holdout/test·전체 500장·CSV·제출을 금지한다.
