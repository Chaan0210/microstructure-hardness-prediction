# 210 완료 후 다음 세션 인계

**이 세션은 210에서 실험을 종료했다. 새 실험 번호나 실행 대기열을 추가하지 않았다.** 최신 로드맵은 [30개 항목 관리표](roadmap-30.md), 기계 판독용 상태는 [management.json](management.json)이다. 이전 감사·실험 보고서·등록부는 수정하지 않았다.

## 현재 기준과 실행 상태

- **best021 유지.** 사용자 확인 Public RMSE **11.4878517800**. 후보022는 **11.5435627849**로 악화했다. 021 원본 개발 OOF는 **12.786200775750695**다. 이 값들은 서로 다른 평가 대상이다.
- CSV: `artifacts/submission-candidate-021.csv`. SHA256: `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b`. 이번에는 바이트 해시만 확인했다.
- 205–210에서 **새 제출 CSV·업로드·test/holdout 평가·최종 평가 선택 없음**. 이 인계에서도 학습·추론·점수 재계산·설치·다운로드·외부 AI 호출·웹 호출을 하지 않았다. 비용 0.
- 205에는 당시 승인된 MicroNet 가중치 다운로드 **1회**가 있었다. 이를 “205–210 전체 다운로드 0회”로 바꾸어 기록하지 않는다. 206–210 및 이번 인계의 설치·다운로드는 0회다.
- 알려진 실험 실행은 완료됐다. **2026-10-07 00:47:48 UTC** 점검에서 프로젝트 cwd를 사용하는 Python/torchrun/uv 프로세스가 없었다. [범위와 결과](process-status.json). 이 스냅샷 이후의 상태까지 보증하지는 않는다.

## 205–210 결과와 현재 판단

아래 개선은 대조 RMSE−후보 RMSE다. 원본 5-fold는 같은 개발 400장이며, 209의 두 재분할도 그 400장을 재사용한다. CI는 누적 선택 편향·재학습·seed 불확실성을 보정하지 않는다.

| 실험 | 시행한 정책 | 결과와 현재 판단 |
|---|---|---|
| [205](../../../../experiments/01-representations/0205-micronet-frozen/REPORT.md) | 동결 ImageNet/MicroNet ResNet50, native256 mean2048+Ridge EDF32 | 단독 15.469001/15.874501. MicroNet 10% 결합 12.887703, 021 대비 −0.101502, CI [−0.192442, −0.009848]. 이 고정 조건 종료. 모든 MicroNet 활용법에 일반화하지 않음. |
| [206](../../../../experiments/01-representations/0206-contrast-augmentation/REPORT.md) | 평균 보존·최대 5% 대비 감소 | 단독 개선 −0.016734, 고정 첫 seed 치환 −0.002577. 이 정책 종료. 구조 검사 통과가 라벨 불변성이나 실제 물리 측정의 정확성 증명은 아님. |
| [207](../../../../experiments/01-representations/0207-checkpoint-averaging/REPORT.md) | 저장된 마지막 5개 epoch의 지수 가중 평균 | 새 학습 0회. 단독 +0.010181, CI [−0.027854, 0.047671], 고정 치환 +0.002158이나 CI가 0 포함·상위 5개 제외 이득 음수. 이 후반 평균 정책 종료. 전체 update EMA와 구별. |
| [208](../../../../experiments/01-representations/0208-update-ema/REPORT.md) | 전체 800 update EMA, β=0.99, warmup 0, 초기값 θ0 | 단독 +0.168108, CI [0.091735, 0.241566]. 고정 첫 seed 치환 +0.003805, CI [−0.002860, 0.010351]. 이후 209에서 같은 정책을 검증함. |
| [209](../../../../experiments/01-representations/0209-ema-repartition/REPORT.md) | 208 그대로 두 기존 재분할 추가; 원본은 재학습 안 함 | 3분할 pooled 단독 +0.107652, CI [0.051598, 0.163877]. 고정 첫 seed 치환 −0.002041, CI [−0.007153, 0.003092]. 한 재분할의 단독 CI가 0 포함. **현재 상태는 보류**, 단독 평균 개선과 앙상블 미채택을 함께 보존. [최신 상태](../209-status-after-review.md). |
| [210](../../../../experiments/01-representations/0210-convnext-llrd/REPORT.md) | 기존 ConvNeXt 마지막 세 블록만 LLRD ratio 0.8; 평균 LR·실제 decay 고정 | 단독 16.764185→16.768129, 개선 −0.003944, CI [−0.013212, 0.005392]. Train 오차는 낮아졌으나 train/OOF 격차는 0.046014 확대. 10% 추가 결합은 021 대비 −0.172312. 이번 고정 정책 종료, LLRD 전체 실패로 일반화하지 않음. |

206–209의 고정 앙상블은 기존 CNN 세 seed 중 첫 seed의 출력 치환이다. 205·210의 10% 추가 결합과 다르며, 어느 쪽도 신규 성분을 반영한 전체 nested 후단 재학습은 아니다. **204의 3→6 seed 전체 재학습(신규 540회+검증 48회) 분기는 반복하지 않는다.** 204 pooled 개선은 +0.005785, CI [−0.026274, 0.037320], 두 재분할 악화였다. 209 때문에 이를 자동 반복하지 않는다.

## 남은 질문을 다루는 순서

로드맵 상태는 **완료 18개, 부분 9개, 미실시 3개**다. 완료는 수행 범위 표시이며 문제 해결 판정이 아니다. 다음 세션에는 관리표에 적은 두 검토만 권고한다.

1. 현재 TinyCNN의 동일 경로 MSE–Huber 대조가 정말 누락됐는지 확인하고, 필요하면 단일 정책 설계까지만 검토한다. 058의 동결 DINO head 결과를 그대로 적용하지 않는다. 큰 오차를 라벨 오류로 단정하지 않는다.
2. 결정립/상 정답·물리적 픽셀 길이 또는 검증 가능한 생성 자료를 확보할 수 있는지 확인한다. 새로운 근거가 없으면 097/098/183의 측정 실패 경로를 반복하지 않는다.

화질·형태별 잔차 진단은 068/126/166/199/206에서 수행했지만 원인이나 성능 문제를 해결하지 못했다. 물리 측정 정확성도 미해결이다. 기존 `train.csv` 헤더 확인은 `ID,hardness`로, 추가 독립 타깃은 없다. 타깃 층화는 미실시이며 그 목적은 fold별 경도 분포 균형이다. 같은 400장의 새 분할을 독립 holdout으로 부르지 않는다. GBM은 초기 HGB 및 027 HGB/GBR 비교 이력이 있다. Extra Trees를 GBM으로 분류하거나 서로 다른 특징 패널 비교를 순수 회귀기 대조로 부르지 않는다. **127의 learning curve·오차 상관 진단을 다시 제안하지 않는다.**

## 필요한 파일과 기존 실행 환경

검토를 시작할 때 이 HANDOFF → `roadmap-30.md` → 해당 실험의 `REPORT.md`, `plan.json`, `completion.json`, `artifact-manifest.json` 순으로 읽는다. 오래된 계획의 “미착수”나 207의 “전체 EMA 미평가”는 작성 당시 상태다. 이후 완료 기록과 209의 검토 후 상태를 함께 읽으며 원본을 덮어쓰지 않는다.

| 용도 | 필요한 경로 |
|---|---|
| 초기 근거 | `experiments/01-representations/0204-cnn-nested/HANDOFF.md`, `coverage-audit.md`, `reports/registers/strategy-experiment-register.csv`, `reports/summaries/cnn-nested-204-summary.json`, `docs/records/roadmap-audit/audit.md` |
| 205–210 결과 | `experiments/{micronet-frozen-205,contrast-augmentation-206,checkpoint-averaging-207,update-ema-208,ema-repartition-209,convnext-llrd-210}/`와 각 `reports/*-summary.json` |
| 공통 개발 표본·기준 OOF | `artifacts/run-012/development_features.npz`, `experiments/01-representations/0177-dinol/{original,20261031,20261101}/predictions.npz` |
| TinyCNN 입력·기존 경로 | `artifacts/nested-calibration-074/development_features.npz`, `cnn_controls.py`, `frozen_validation.py`, `artifacts/cnn-seed-mean-133/audit/`, `artifacts/frozen-validation-021/` |
| 207/208의 대조 | 206의 fold별 control 체크포인트와 마지막 5개 epoch state; 경로·해시는 각 `plan.json`과 `preservation.json`에 고정 |
| 209의 직접 재현 근거 | `experiments/01-representations/0201-auxiliary-repartition/`의 detach 대조 state/trace, 기존 021 체크포인트 및 208 원본 결과 |
| 210의 직접 재현 근거 | `experiments/01-representations/0190-modern-backbones/convnext_tiny/`의 `prefix.npy`, `features.npz`, fold별 기존 head/tail·trace·가중치; 정확한 입력 목록은 210 `preservation.json` |
| 205 가중치 | `experiments/01-representations/0185-pretrained-cnn/weights/resnet50-11ad3fa6.pth`, `experiments/01-representations/0205-micronet-frozen/weights/resnet50_micronet_weights.pth`; 다운로드 반복 불필요 |
| 기존 uv 환경 | 206–209 및 읽기 전용 기록 검증: `.venv/bin/python`. 205/210: `experiments/01-representations/0185-pretrained-cnn/.venv/bin/python` (torch 2.14.0 / torchvision 0.29.0) |

## 읽기 전용 확인 명령과 재실행 경계

프로젝트 루트에서 아래 명령으로 인계 시점의 원본 해시와 best021 보존 상태를 다시 확인할 수 있다. 표준 라이브러리만 사용하며 학습 코드·모델·원시 이미지·OOF 배열을 실행하거나 역직렬화하지 않는다. 이는 **기록 무결성 재현**이며 재학습 성능 검증을 대신하지 않는다.

```sh
.venv/bin/python -I -B - <<'PY'
from pathlib import Path
import hashlib, json
p = Path.cwd()
v = json.loads((p/'docs/records/roadmap-audit/handoff-after-210/verification.json').read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()
bad = [n for n,h in v['protected_files_sha256'].items()
       if not (p/n).is_file() or sha(p/n) != h]
assert not bad, bad
assert sha(p/'artifacts/submission-candidate-021.csv') == v['best021_sha256']
print('원본 해시 일치:', len(v['protected_files_sha256']))
print('best021 유지. 모델 실행·점수 재계산 없음.')
PY
```

학습의 재현 근거는 기존 실행 소스·계획·잠금 해시·환경·검증 로그에 있다. 아래는 **완료된 실행의 단계 순서 기록**이며 이번 인계에서 실행하지 않는다.

| 실험 | 보존된 주요 진입점 순서 |
|---|---|
| 205 | `compatibility.py` → `features.py` → `study.py` → `score.py` |
| 206 | `preflight.py` → `study.py` → `validate.py` → `score.py` → `finalize.py` |
| 207 | `study.py` → `score.py` → `finalize.py` |
| 208 | `study.py` → `validate.py` → `score.py` → `finalize.py` |
| 209 | `preflight.py` → `study.py` → `validate.py` → `score.py` → `finalize.py` |
| 210 | `preflight.py` → `control.py` → `study.py` → `validate.py` → `score.py` → `finalize.py` |

기존 210 실행 형태의 예는 다음과 같다. **완료 디렉터리에 재실행하지 않는다.**

```sh
experiments/01-representations/0185-pretrained-cnn/.venv/bin/python -I -B -c 'import sys,runpy; sys.path.insert(0,"experiments/01-representations/0210-convnext-llrd"); runpy.run_path("experiments/01-representations/0210-convnext-llrd/study.py",run_name="__main__")'
```

이 소스들은 출력 경로를 코드 위치에 묶고 기존 파일에 배타적으로 쓰므로 일반적인 `--output` 재실행 명령이 없다. `validate.py`도 쓰기나 검증 재학습을 할 수 있어 읽기 전용 확인 명령이 아니다. 실제 재학습이 필요해지면 원본을 보존하는 새 출력 경로와 코드 잠금·예산을 별도로 설계해야 한다. 이번에는 재학습·설치·가중치 다운로드·외부 호출·test/holdout 평가·제출을 시작할 권한을 추가하지 않는다.

인계 원본 보존 증거는 [verification.json](verification.json)에 있다. 미해결 항목은 과학적 판단과 다음 검토의 근거 부족이며, 인계 파일 작성 자체의 실행 차단 사항은 없다.
