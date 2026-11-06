# 205 완료 — 동결 MicroNet 대 ImageNet ResNet50

**사전 기준 미달로 이번 고정 분기를 종료합니다. best021을 유지합니다.** 새 재분할·미세조정·pooling/EDF 탐색·제출은 진행하지 않았습니다.

| 개발 original 5-fold RMSE | ImageNet | MicroNet |
|---|---:|---:|
| 동결 mean2048 + Ridge EDF32 단독 | 15.46900080 | 15.87450101 |
| best021 90% + 해당 모델 10% | 12.86846411 | 12.88770281 |

기존 best021의 같은 분할 RMSE는 **12.78620078**입니다. MicroNet 결합의 개선량(기존−신규)은 **-0.10150204**, 조건부 95% CI **[-0.19244214, -0.00984844]**입니다. best021보다 개선한 fold는 **0/5**이고, 이득 상위 5개 ID를 제외해도 개선량 **-0.14358964**로 악화합니다. ImageNet 결합보다도 0.01923870 나쁘며 해당 paired CI는 0을 포함합니다. 기술 검증은 통과했으나 성능 관련 채택 기준은 모두 미달했습니다.

## 다운로드와 호환성

- [NASA 공식 저장소](https://github.com/nasa/pretrained-microscopy-models/tree/9b7c4abc1321e81eca7a68d548e5371676fa74fa)의 pretrained encoder MIT 설명 및 [HF 모델 카드](https://huggingface.co/jstuckner/microscopy-resnet50-micronet/tree/c810b294acda2461775542841e6d0ba824ec46a9)의 해당 가중치 MIT를 재확인했습니다. 코드·가중치 출처를 각각 보존했으며 대회 규정에 관한 새로운 법적 판단을 내린 것은 아닙니다.
- 승인된 공개 checkpoint **한 개**, `resnet50_micronet_weights.pth`, **94,344,986바이트**를 로그인 없이 받았습니다. SHA256: `4c906df94c224efa267e59311244900a5633549aeda5e11f5ecfade6f30bafd2`. HF revision: `c810b294acda2461775542841e6d0ba824ec46a9`. 카드 예시 파일명 `encoder_weights.pth`와 실제 파일명의 차이를 기록했습니다.
- 기존 uv의 torch 2.14.0 / torchvision 0.29.0에서 `weights_only=True`를 사용했습니다. encoder 텐서 **318개**가 이름 변환 없이 `strict=True`로 로드됐습니다. MicroNet에는 제외할 head가 없고, ImageNet의 사용하지 않는 `fc.weight`와 `fc.bias`만 명시적으로 제외했습니다. 두 arm encoder 파라미터는 각각 **23,508,032개**입니다.
- 공식 SMP ResNet50은 torchvision ResNet/Bottleneck을 상속합니다. 원문은 실행하지 않고 정적으로 읽었으며, 구버전과 현재의 핵심 계산 AST 및 실제 stage/stride/BN 속성을 비교했습니다. [상세 대응](architecture-review.json), [호환성 검사](compatibility.json), [텐서 목록](micronet-tensor-manifest.json).
- HF 파일이 논문의 MicroNet v1.0인지 후속 v1.1인지는 확정하지 않았습니다. 따라서 결과는 특정 revision·hash 파일의 결과로만 해석합니다. 구버전 전체 런타임의 수치 동등성을 주장하지 않습니다.

## 비교 조건과 검증

양쪽 모두 같은 native256 전체 이미지, RGB 복제·ImageNet 정규화, 동결 eval ResNet50, 최종 2048차원 공간 평균을 사용했습니다. D4·resize·추가 증강·중간층·PCA·미세조정은 없습니다. 각 outer train 320에서 RMS floor 전처리·타깃 표준화를 적합하고, y와 무관하게 X의 고유값으로 Ridge EDF32를 맞췄습니다. 기존 ImageNet 캐시는 같은 입력의 재추론 exact를 확인해 재사용했습니다. 두 가중치의 사전학습 자료·목적·절차가 함께 달라지므로 데이터 도메인만의 인과 효과라고 부르지 않습니다.

- 실제 본 학습: **Ridge 10회**. 검증: held-out NaN/finite 치환 10회, train-label 순열 확인 10회, 독립 solver 10회. 신경망 gradient 학습은 **0회**입니다. 모든 기술 검증과 OOF 저장을 마친 뒤 점수를 계산했습니다.
- 3장 smoke: 두 arm 반복 MPS 추론 exact, ImageNet 기존 캐시 exact. CPU/MPS feature 최대 오차는 ImageNet 5.72e-06, MicroNet 3.81e-06로 사전 1e-3 한도 이내입니다. 400개 원시 이미지의 해시도 이전 native256 추출 기록과 일치했습니다.
- 10개 fold 모델의 저장·재로딩과 validation X/y 치환 후 state·원본 query 예측이 exact였습니다. 독립 Ridge 예측 최대 차이 **2.84e-13**, EDF 최대 오차 **7.11e-14**, 별도 count 방식 bootstrap 계산 차이 **7.11e-15**입니다.
- 관측 최대 RSS+MPS driver는 **0.851 GB**, swap 증가 **0**, 감시한 호환성·smoke·특징·Ridge/검증 단계 합 **16.13초**입니다. 다운로드 11.64초 및 코드 작성·최종 보고 시간은 이 합계에서 제외했습니다. 새 설치·유료 자원 사용은 없고 추가 비용은 **0**입니다.

## 보존·범위·다음 판단

기존 보존 대상 **74개 파일**의 SHA256이 모두 일치합니다. best021 CSV SHA256은 `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b`입니다. holdout/test 평가, 전체 500 재학습, 새 제출 CSV, 업로드 및 최종 선택 변경은 없습니다. 기존 204와 기존 보고서·등록부도 수정하지 않았습니다.

같은 개발 400을 반복 사용했으므로 이 결과는 독립 holdout 결과가 아닙니다. 조건부 CI는 누적 연구 선택 편향·재학습 불확실성을 보정하지 않습니다. 이번 실패는 **이 가중치의 동결 mean2048 + EDF32 회귀 및 고정 10% 결합**의 실패이며, MicroNet 전체·현미경 사전학습 전체의 무용성이나 물리 측정의 오류를 증명하지 않습니다.

이 분기를 종료하고 best021을 유지합니다. 205의 계획에는 다른 설정으로 구제하거나 다음 실험을 자동 시작하는 단계가 없습니다. 주요 증빙: [사전 계획](plan.json), [검증](validation.json), 수치 요약, [완료·보존 기록](completion.json).
