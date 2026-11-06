# 211 사전 고정 계획

058은 동결 DINO head 실험이며 현재 TinyCNN과 다르다. 동일 대조가 확인되지 않아 MSE 대 2 × Huber(delta 1) 한 조건만 실행한다.

전이값 1은 fold train 타깃 표준편차 1개다. 배율 2는 중앙 구간의 r², 기울기 2r, 곡률 2를 MSE와 맞춘다. 전체 gradient norm이 같다는 주장은 하지 않는다. LR·clip·decay를 유지하고 실제 꼬리 비율 및 clipping 빈도를 기록한다.

원본 400장 5-fold, seed 20261003 + fold, BOX 128, train 전용 표준화, D4, AdamW, 80 epoch/800 update, 마지막 5 epoch 동일 평균을 유지한다. 기존 state·OOF·RNG 재현 및 optimizer를 검증한다. 주 학습 10회, fold 0 양 군 validation label poison 검증 재학습 2회다. 모든 기술 검증 뒤 train/validation 원단위 RMSE와 고정 첫 seed 앙상블 치환을 계산한다.

단독 개선 ≥ 0.02, 개선 fold ≥ 3, paired bootstrap 10,000회(seed 202610211)의 95% CI 하한 > 0, 제곱 오차 이득 상위 5개 ID 제외 개선 > 0을 사전 기준으로 둔다. 미달이면 추가 loss/delta/배율 탐색 없이 종료한다. 통과해도 보고 후 정지하고 자동 채택·재분할·전체 nested 재학습은 없다.

기존 파일과 best021을 보존한다. 비용·설치·다운로드·외부 AI·holdout/test·전체 500장·새 CSV·업로드를 금지한다. 기존 uv CPU 2 threads/MPS, RSS+MPS 6 GB 이하 및 swap 증가 0을 감시한다. 상세 계획은 plan.json, 실행 전 소스 잠금은 code-lock.json에 저장한다.
