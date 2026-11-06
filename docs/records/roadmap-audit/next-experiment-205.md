# 다음 실험 205 제안 — 동결 ImageNet ResNet50 대 MicroNet ResNet50

**상태: 대조 설계만 완료. 다운로드·설치·호환성 검사·특징 추출·학습·실행 예약 모두 미착수.** 205는 현재 없는 번호의 제안이며 실제 실험 폴더는 만들지 않았습니다. 실행 전에 번호 충돌을 확인해야 합니다. 다운로드 승인을 아직 받지 않았으므로 이 문서만으로 파일을 받지 않습니다.

## 질문과 선택 이유

동일한 ResNet50 구조와 native256 입력, 동일한 2048차원 평균 특징·회귀 복잡도에서 **현미경 사전학습 체크포인트가 현재 ImageNet 체크포인트보다 유용한 정보를 제공하는가?**

185–189는 ImageNet ResNet50의 head·optimizer·해상도·부분 미세조정 비교입니다. 191은 ImageNet 최종층 mean/std, 192는 ConvNeXt의 PCA 절단을 비교했습니다. 로컬 기록에서 MicroNet 실행은 찾지 못했습니다. 따라서 새 architecture를 추가하지 않고 가중치 출처를 바꾸는 이 질문은 중복이 아닙니다. 이미지 도메인의 차이에 관한 가설이지, 사전학습 데이터만의 인과 효과를 분리하는 대조는 아닙니다. 두 가중치의 사전학습 자료·목적·학습 절차가 함께 다를 수 있습니다.

Microscopy 가중치의 공개·라이선스 정보가 특정됐다는 새 근거를 반영했습니다. MicroNet 원논문은 segmentation 결과이며 경도 회귀 이득을 보장하지 않습니다. [127](../../../reports/summaries/base-learning-curve-127-summary.json)의 학습곡선·오차 상관, [097–098](../../../reports/validation/grain-visual-audit-097-nlm098-summary.json)·[183](../../../reports/summaries/tessellation-diagnostic-183-summary.json)의 물리 측정 진단, 166–167의 화질·형태별 잔차는 재실행하지 않습니다. EMA·LLRD·증강·seed 확대를 이 비교에 끼워 넣지 않습니다.

## 단계 0 — 승인과 호환성 관문

| 구분 | 고정 대상·확인 사항 |
|---|---|
| A: 기존 ImageNet | [191 계획](../../../experiments/01-representations/0191-spatial-moments/plan.json)의 `experiments/01-representations/0185-pretrained-cnn/weights/resnet50-11ad3fa6.pth`, SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. 기존 로컬 파일 사용, 재다운로드하지 않음. |
| B: 제안 MicroNet | [카드](https://huggingface.co/jstuckner/microscopy-resnet50-micronet), [실제 파일 페이지](https://huggingface.co/jstuckner/microscopy-resnet50-micronet/blob/main/resnet50_micronet_weights.pth). 실제 이름 `resnet50_micronet_weights.pth`, 표시 크기 약 94.3 MB, 전달된 SHA256 `4c906df94c224efa267e59311244900a5633549aeda5e11f5ecfade6f30bafd2`. 카드 예시의 `encoder_weights.pth`와 혼동하지 않음. |
| 근거의 상태 | 상위 세션은 NASA README의 pretrained encoder MIT와 해당 HF 카드 MIT를 확인함. 여기서는 로컬 파일을 아직 받지 않았으며 해시·내부 tensor·라이선스 파일을 새로 실측하지 않음. HF 가중치의 MicroNet v1.0/v1.1 대응은 미확인으로 보존. |
| 허용할 다운로드 | 이후 사용자가 정확한 체크포인트 다운로드를 승인한 경우에만 새 실험 폴더에 한 파일을 저장. 승인 시 원문·불변 revision/commit·배포 조건을 근거에 고정하고 실측 크기·SHA256을 확인. hash 불일치 시 다른 파일·revision으로 바꾸지 않고 중단. 설치는 별도이며 현재 계획에 없음. |
| 로컬 환경 | 기존 uv 환경 `experiments/01-representations/0185-pretrained-cnn/.venv`의 Python을 사용. 191 소스가 기록한 torch 2.14.0 / torchvision 0.29.0 환경을 실행 전에 확인. 새 venv·SMP·timm·구버전 torch를 설치하지 않음. |
| 구조 대응 | 전달된 공식 `encoder.py`의 SMP ResNet50 구현을 버전·출처와 함께 확인하고, 현재 torchvision encoder와 stage·bottleneck·stride 위치·padding·BN eps/momentum·eval 동작을 대조. tensor 이름과 shape 일치만으로 계산 구조 일치를 인증하지 않음. 원본 배포 구현을 기존 환경에서 검증할 수 없다면 불확실성을 보존하고 성능 비교로 넘어가지 않음. |
| 엄격한 로드 | `torch.load(..., map_location='cpu', weights_only=True)`로 읽을 수 있어야 함. 공통 encoder-only 모듈에 모든 conv·BN 파라미터/버퍼를 완전하게 매핑하고 `strict=True`로 로드. 키 접두사 제거는 기록한 일대일 매핑만 허용. 양쪽 모두 사용하지 않는 fc 또는 문서로 식별한 segmentation head만 명시적으로 제외. 무작위로 남긴 층·누락 BN buffer·shape 강제 수정·일괄 strict=False는 허용하지 않음. |
| 실패 시 | 다운로드·형식·구조·로드 관문 실패는 **기술적 차단**이며 MicroNet의 학습 또는 예측 성능 실패가 아님. 구버전 설치·다른 backbone·다른 가중치로 우회하지 않음. 이후 필요 조치는 별도 판단. |

두 encoder가 계산 구조까지 같다는 관문을 통과한 뒤에만 아래 비교를 수행하는 조건부 계획입니다. 안전한 로드가 실패하면 `weights_only=False`로 재시도하지 않습니다. 이 관문은 현재 실행되지 않았습니다.

## 단계 1 — 가장 작은 matched 대조

| 항목 | 두 arm 공통 고정 조건 |
|---|---|
| 데이터 | 기존 개발 400개 ID와 **original 5개 group fold**만. 100 holdout·test·전체500 모델 학습 없음. 라벨은 기존 개발 전용 캐시에서 ID·그룹·해시를 대조해 읽으며 전체 label CSV를 다시 평가하지 않음. |
| 입력 | native256 전체 이미지, crop/resize 없음. 흑백을 RGB로 복제, float32 [0,1], ImageNet mean [0.485,0.456,0.406] / std [0.229,0.224,0.225]. 두 arm의 동일 입력 tensor 해시 확인. D4·광도·noise·blur 증강/TTA 없음. |
| Encoder | 같은 ResNet50 encoder, 모든 파라미터 고정, eval/inference_mode, BN running statistics도 고정. 미세조정·재정규화·중간층 추가 없음. |
| 특징 | 최종 layer4의 2048×8×8 공간 맵을 global average pooling한 **2048차원 mean만**. CLS/std·다층 concatenation·PCA·특징 선택 없음. |
| A 캐시 재사용 | [191](../../../experiments/01-representations/0191-spatial-moments/features.py)의 `resnet50/features.npz` mean은 native256 ImageNet mean2048이며 [188](../../../reports/summaries/cnn-resolution-188-summary.json) 캐시와 exact를 확인한 기록이 있음. ID·이미지·가중치·소스 해시와 아래 3장 재추론을 먼저 대조. 원본 캐시는 수정하지 않음. 캐시 불일치 시 유리한 구현을 택하지 말고 원인 해결 전 중단. |
| B 추출 | 호환성 관문 뒤 같은 개발400에 한 번만 추출. 정답을 보지 않는 고정 encoder이므로 특징 계산은 공유할 수 있으나 회귀의 모든 전처리·통계는 각 train fold에서 따로 적합. |
| 전처리 | 191/192의 RMS floor 규칙을 **단일 2048차원 bank**로 사용: train feature 평균을 빼고 `max(train std_j, sqrt(mean_j(train var_j)))`로 나눔. 전부 상수면 단위 scale, 이후 rank 관문 적용. validation 값은 통계에 사용하지 않음. |
| Head | deterministic float64 full Ridge. 타깃은 train 평균·표준편차로 표준화. 192의 방법처럼 train-X의 고유값으로 `sum(lambda_i/(lambda_i+alpha)) = 32`가 되도록 각 arm의 alpha를 결정. 절편은 비규제이며 전체 조건부 EDF는 33. alpha·EDF를 y로 탐색하지 않음. centered train rank≤32이면 중단, EDF 변경 없음. |
| Head 비교의 의미 | 같은 입력 차원·Ridge EDF로 용량을 제한하지만 bias·분산이 완전히 같다는 보장은 없음. 191의 PCA128 MLP 결과를 A 대조 점수로 대신 쓰지 않음. A/B 모두 새 고정 Ridge로 학습해야 함. |
| 학습 규모 | 2개 arm × 5개 fold = **10개 Ridge fit**, encoder gradient 학습 **0회**. deterministic head이므로 신경망 seed 반복 없음. 추가 원본 대비군·회귀기·정규화 격자 없음. |

같은 소프트웨어 구조의 A/B가 성립하지 않으면 이 실험은 진행하지 않습니다. SMP/torchvision 구현 차이가 섞인 비교를 가중치만의 효과라고 보고하지 않습니다.

## 단계 2 — 평가·종료 기준

- 주 대조는 **B 대 A 단독 OOF RMSE**입니다. 실용성 확인은 각 arm에 고정식 `Q = 0.9*P021 + 0.1*P_arm`을 적용한 OOF입니다. P021은 해당 outer fold의 기존 저장 예측을 사용하고 ID·분할·해시를 잠급니다. 이는 새 가중치를 탐색하지 않는 추가 결합이며 **021 내부 CNN 교체나 후단 재적합**이 아닙니다.
- B가 A보다 단독 RMSE와 Q의 RMSE 모두 낮고 Q의 paired fold 개선이 **최소 3/5**이어야 합니다. 동시에 `Q_B`가 기존021보다 **RMSE ≥0.02 개선**, 최소 **3/5 fold 개선**, 조건부 paired ID bootstrap의 개선 **95% CI 하한 >0**, 021 대비 순 SSE 이득 상위 **5개 ID 제외 후에도 개선 >0**이어야 합니다. 기술 검증을 모두 통과해야 합니다. 0.02는 연구 진행의 실용 문턱이며 통계적 최소 검출 효과가 아닙니다.
- Bootstrap은 **10,000회**, seed **202610205**, 400개 ID 단위의 A/B/021 예측을 함께 재표집합니다. 점수는 모든 예측·검증 파일을 저장한 뒤 확인합니다. conditional CI는 누적 모델·특징 선택 및 재학습 불확실성을 보정하지 않습니다.
- 이 기준은 **다음 검토로 진행할 근거**만 판단합니다. 미달 시 현재 체크포인트+mean2048+EDF32 가설을 종료합니다. 낮은 단독 오차나 특정 subgroup의 이득만으로 우회 승격하지 않고, EDF·pooling·미세조정·가중치·다른 checkpoint를 탐색하지 않습니다.
- 통과해도 이 최소 계획은 original 5-fold에서 종료하고 보고합니다. 같은 400장의 기존 두 재분할 검토는 별도 후속 범위이며 독립 holdout이 아닙니다. 이번 계획에는 최종 모델 선택·제출·Public 점수 확인이 없습니다.

## 변경 경로 검증과 자원 상한

1. 191에서 사용한 고정 smoke ID `TRAIN_000001`, `TRAIN_000173`, `TRAIN_000339`만 먼저 사용합니다. 두 encoder 입력과 출력 shape·finite·반복 추론·state 불변을 확인합니다. CPU/MPS feature 최대 절대 오차와 상대 L2 오차 각각 1e-3 미만, MPS 재추론 exact, A 기존 캐시 exact를 관문으로 고정합니다. 결과를 본 뒤 오차 한도를 완화하지 않습니다.
2. 각 10개 Ridge fit의 train-only 통계·alpha·계수·ID·그룹 분리를 독립 확인합니다. outer feature와 label을 NaN/큰 finite 값으로 치환해도 train state가 같고 원래 validation query의 예측이 같은지 **10개 검증 재적합**으로 확인합니다. 훈련 y 변경에 alpha가 변하지 않는지, EDF residual <1e-8, 독립 Ridge solver 예측 차이 <1e-8을 검사합니다. 수치 조건 문제가 있으면 미해결로 기록하고 채택 판단을 중단합니다.
3. 원본 10개 모델의 저장·재로딩, feature/weight 해시와 best021 SHA256을 확인합니다. 독립 수치 재계산은 검증으로 따로 세고 독립 실험 arm으로 합산하지 않습니다. 이미 완료한 127/175/204 전체 학습·재현 검증은 재실행하지 않습니다.
4. 기존 uv 환경에서 CPU 2 thread, 로컬 MPS, 한 번에 encoder 하나를 사용합니다. RSS+MPS driver 합 6 GB 이하, swap 증가 없음. smoke 5분·개발 특징 추출 10분·head+검증 10분의 합 25분 상한을 사전 고정합니다. 첫 smoke 처리량으로 상한 초과가 예상되면 전체 추출 전에 중단합니다. 다운로드 시간과 승인 대기는 계산 시간 상한 밖이며 처리량을 보장하지 않습니다.
5. 이후 승인된 파일 다운로드 외에는 네트워크를 차단합니다. 새 설치·유료 compute·외부 AI/API·회사 자료·다른 프로젝트·credentials 접근은 하지 않습니다. 기존 코드·보고서·가중치·CSV를 수정하지 않고 신규 실험 폴더만 사용합니다.

## 예상 산출물·남는 질문

승인 후 실행된다면 체크포인트 출처·실측 hash, SMP↔torchvision 구조 및 strict key 대응표, 3장 smoke 결과, 400×2048 특징 증빙, 10개 fold Ridge 및 paired OOF, 사전 관문 결과·조건부 CI·상위 ID 민감도, 검증 재적합 수·실제 자원·best021 보존 hash를 남깁니다. 정확한 train-only 범위와 실패 이유를 구분합니다.

현재 남은 실행 전제는 **정확한 파일의 다운로드 승인**, **배포 가중치와 원 구조의 검증 가능한 대응**, **기존 환경의 안전한 strict load 호환**입니다. MIT 확인만으로 이 전제가 해소되지는 않습니다. 호환성 실패는 성능 실패가 아니며, OOF 실패도 물리적 측정이 틀렸다는 증거가 아닙니다. 양성 결과 역시 unseen 성능이나 DINO보다 우월함을 확정하지 않습니다.
