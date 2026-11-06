"""Report actual architecture and matched CNN objective experiments."""
import csv
import json
from pathlib import Path
import numpy as np
import complement_cnn as c


def table(lines,rows):
    lines.extend(['| 모형 | OOF RMSE | 기준 대비 개선 | 95% 조건부 CI | 개선 fold |','|---|---:|---:|---|---:|'])
    for name,q in rows:
        lo,hi=q['conditional_id_bootstrap_gain_ci95'];n=len(q['fold_gains'])*5
        lines.append(f"| {name} | {q['candidate']['pooled']:.6f} | {q['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {q['improved_folds']}/{n} |")
    lines.append('')


def main():
    h=c.HERE;read=lambda name:json.loads((h/name).read_text())
    a=read('architecture-audit.json');s=read('summary.json');q=read('confirmation-summary.json');ctrl=read('mechanism-controls.json');v=read('verification.json')
    names={'direct':'경도 직접 예측 CNN, 16% 혼합','residual':'이미지 전용 잔차 CNN','context':'문맥을 받는 잔차 CNN','aux':'문맥 + 경도 보조 과제 CNN','context_only':'이미지 없는 문맥 대조'}
    lines=['# 021의 실제 구성과 표 기반 모델을 보완하는 CNN 실험','',
        '**현재 구조의 모든 부분이 최적이라는 근거는 없다. 021을 영구적으로 유지할 이유도 없다.** '
        '보존한다는 것은 비교 기준·재현 가능한 제출물을 지킨다는 뜻이다. 새 구성요소는 독립적인 정보를 주는지와 최종 앙상블 RMSE로 교체를 결정해야 한다.','',
        '**021은 CNN 하나가 아니다.**','',
        '코드와 저장 예측을 재조립한 구조는 다음과 같다.','',
        '```text',
        '원영상 → 작은 CNN 3개 + DINOv2-S/B 특징 MLP + 예전 compact KRR',
        '       → affine 보정 + short-B + 고전 잔차 모델 → 018',
        '021 = 0.9 × 018 + 0.1 × DINOv2-L 특징 MLP 3개 평균',
        '061/073 = 0.16 × 021 + TabPFN/Ridge/logRidge/GMM/결정립 집합망',
        '```','',
        '021 내부의 raw 조합은 KRR .3, TinyCNN .3, DINO-S .2, DINO-B .2이며 그 뒤 보정 단계가 있다. '
        f"061/073 안에서 TinyCNN 예측의 직접 선형 경로 계수는 중앙값 {100*a['TinyCNN_actual_linear_coefficient_median']:.2f}% "
        f"(fold별 {100*a['TinyCNN_actual_linear_coefficient_range'][0]:.2f}~{100*a['TinyCNN_actual_linear_coefficient_range'][1]:.2f}%)다. "
        '이는 전체적인 인과 기여도나 재학습 후 중요도가 아니다. 다만 “CNN이 16%”라는 해석은 틀리다.',
        '',
        '**가중치도 최적이라고 검증된 값이 아니다.** 엄격한 원본 분리로 재조립한 400장 OOF의 설명용 곡선은 다음과 같다. '
        '이 결과를 보고 새 가중치를 고른 것이 아니며 기존 061/073을 변경하지 않았다.',
        '', '| 021 복합 모델 비중 | 통합 OOF RMSE |','|---:|---:|']
    for row in a['diagnostic_weight_curve']:lines.append(f"| {row['weight']:.2f} | {row['pooled']:.6f} |")
    lines.extend(['',
        '이 곡선에서는 0.16이 최적이 아니다. 그러나 같은 400장을 반복 사용한 선택 편향과 기존 dev/Public 불일치가 있어 '
        '이 표만으로 021을 삭제해 제출하면 안 된다. 완전한 외부 fold 안에서 가중치 선택까지 다시 수행하는 비교가 필요하다. '
        '새 CNN이 생기면 예전 0.16을 자동 상속할 이유도 없다.',
        '',
        '**새로운 큰 CNN만 학습하면 해결되는가?**','',
        '별도 실험에서 공식 EfficientNetV2-S 가중치로 직접 경도 예측을 학습했다. 원래 분할의 2-seed 결과는 '
        '단독 약14.40, 021 약12.81이고, 021 자리의 같은 비중 교체는 약0.139 악화했다. '
        'V2-M의 단일 fold도 S보다 낫지 않았다. 그 작업의 파일·학습·가중치를 수정하지 않았다.',
        '',
        '이것은 해당 데이터·crop·해상도·BN·학습률·학습량의 결과다. 모든 ImageNet 백본이나 긴 학습이 무효라는 증거가 아니다. '
        '특히 최근 V2 실험은 비배율 crop을 사용한다. 크기를 재조정하지 않아도 crop은 전체 상 분율·공간 불균일성을 바꿀 수 있다. '
        '이 가정과 BN 통계 적응 여부 등은 별도 대조가 필요하다. 이번 실험은 원본 전체256을 유지하며 crop·밝기 증강을 하지 않았다.',
        '',
        '**이번에 실제 구현한 방향**','',
        '표 기반 모델 T를 기본값으로 두고, 영상 CNN이 δ를 예측한다. 최종식은 `T + 0.25 × δ`다. '
        'CNN의 목표는 `y − T`이며, 학습 행의 T는 그 행과 outer 검증 행을 모두 제외하고 학습한 내부 모델의 예측이다. '
        '단독 경도 예측으로 T가 이미 아는 정보를 다시 배우는 대신 T의 오차를 학습하도록 목표를 바꾼다.',
        '',
        '```text',
        '학습 영상 ─→ 표 기반 내부 교차적합 ─→ T_OOF ─→ 잔차 y−T_OOF',
        '     └────→ 전체 영상 CNN ─────────────────→ δ 학습',
        '추론 영상 ─→ 학습 완료 표 기반 모델 ─→ T ──────┐',
        '     └────→ 학습 완료 CNN ──────────→ 0.25δ ─┴→ 최종 예측',
        '```','',
        '새 CNN은 native256, 첫7×7/stride4 후16/32/48/64 채널, 모든 stage의 mean/std를 읽는 68,788개 파라미터 모델이다. '
        'GroupNorm/SiLU, readout64/dropout.25, 출력 zero initialization, D4, 40epoch, AdamW .002, 마지막5epoch 평균을 고정했다. '
        '모든 대조는 같은 초기화·데이터 순서·D4·학습량을 쓴다. 경도 보조 과제는 이미지에서 원래 y를 예측하는 별도 head로 '
        '잔차의 작은 신호만으로 표현을 배우기 어려운 문제를 시험한다.',
        '',
        '과거 residual-CNN219는 실제로 실패했다. 213장 anchor와107개 잔차 라벨로 학습한 full-strength 보정은 '
        '13.371→14.326으로 악화했다. 이번에는 현대적인 표 기반 anchor, outer-train400개 전체의 honest residual, '
        '고정 .25 shrinkage, 문맥/보조 과제/단순 통계 대조를 쓴다. 과거 실패를 무시하고 같은 실험을 반복한 것이 아니다.',
        '',
        '**검증 범위**','',
        '새500장, 세 5-fold 분할(2026101501/02/03), 총15fold다. 기존 round2 CPU 표 기반 anchor 및 모든 inner 예측을 그대로 사용했다. '
        'anchor는 한 TabPFN 평균 bank와 Ridge/logRidge/GMM이며 021·결정립 집합망을 포함하지 않는다. '
        '따라서 아래 점수는 **현재 완성형073을 교체했을 때의 점수가 아니다.** '
        'CNN이 학습할 잔차는 내부 약267장 모델에서, 검증의 T는400장 모델에서 나와 학습 크기 차이도 남는다.',
        '',
        '첫 분할의 다섯 대조를 모두 완료했다. 이미지 전용 잔차 및 보조 과제 모델이 사전 compute-allocation 조건을 통과해 '
        '다른 두 분할에서 같은 설정으로 확인했다. 실패한 direct/context 모델은 추가 실행하지 않았다. '
        '세 분할은 동일500장의 반복이므로 독립 확인자료는 아니며, CI는 ID 단위의 조건부 bootstrap이다.',
        '',
        '**첫 5fold: 학습 목표와 문맥 대조**',''])
    table(lines,[(names[key],s[key])for key in c.ARMS])
    lines.extend(['**세 분할 확인: 모델이나 가중치 변경 없음**',''])
    table(lines,[(names[key],q[key])for key in ('residual','aux','context_only')])
    lines.extend(['**효과의 출처를 확인한 대조**',''])
    table(lines,[('이미지 평균·표준편차만 사용',ctrl['stats']),('학습 평균 잔차만 더함',ctrl['constant']),
                 ('공간 CNN vs 평균·표준편차 대조',ctrl['image_CNN_vs_stats']),('공간 CNN vs 상수 보정',ctrl['image_CNN_vs_constant'])])
    lines.extend(['',
        '위 마지막 두 행은 각각 다른 대조군 대비 개선량이다. 모든 값은 실제 저장 예측에서 계산했다. '
        '직접 예측 CNN의 성패는 그 전체 혼합의 RMSE로, 잔차 CNN은 같은 T 위의 증가분으로 판정한다. '
        '두 모델의 standalone 점수를 섞어 판단하지 않는다.',
        '',
        '**통합 판단**',''])
    residual=q['residual'];ci=residual['conditional_id_bootstrap_gain_ci95'];vsstats=ctrl['image_CNN_vs_stats']
    lines.extend([f"이미지 전용 잔차 CNN의 확인 결과: 기준 {residual['baseline']['pooled']:.6f} → {residual['candidate']['pooled']:.6f}, "
        f"개선 {residual['gain']:+.6f}, CI [{ci[0]:+.6f}, {ci[1]:+.6f}], {residual['improved_folds']}/15 fold 개선.",
        '분할별 개선: '+' / '.join(f'{x:+.6f}'for x in residual['partition_gains'])+'.',
        f"학습된 공간 CNN의 통계-only 대조 대비 효과: {vsstats['gain']:+.6f}.",
        '',
        '이미지 평균·표준편차만으로 개선의 상당 부분이 설명된다. 전체 CNN의 추가 이득은 약0.0131이고 '
        '통계-only 대비 조건부 신뢰구간은0을 포함한다. 따라서 “TabPFN이 본질적으로 처리할 수 없는 공간 정보를 CNN이 확인해서 배웠다”는 '
        '강한 결론은 아직 내리지 않는다. 입력 표현의 누락과 학습 목표의 부적합도 함께 검토해야 한다.',
        '',
        '이 실험의 목적은 021을 영구 고정하는 전제를 깨고, 새 이미지 모델을 앙상블 오차에 맞춰 훈련하는 절차를 검증하는 것이다. '
        '표 기반 anchor 위에서의 효과와 완성형073에 더해지는 효과는 다르다. '
        '021의 기존 이미지 정보와 겹칠 수 있으므로 현재073과 동일한 fold·완전한 구성에서 추가 효과가 검증되기 전에는 교체하거나 제출하지 않는다. '
        '고정 .25도 최적 가중치라는 주장은 하지 않는다. 성능이 확인된 이미지 경로에 대해서만 내부 검증으로 가중치를 재선택해야 한다.',
        '',
        '**재현 파일**','',
        '- `architecture-audit.json`: 021 재조립, 직접 CNN 계수, 구성별 RMSE·잔차 상관, 설명용 가중치 곡선.',
        '- `complement_cnn.py`: 이미지 로드·honest residual 구성·모형·학습·추론.',
        '- `confirm.py`: 모델 설정을 바꾸지 않은 두 추가 분할.',
        '- `stats_control.py`: 이미지 통계-only 및 상수-only 대조.',
        '- `partition-*/fold*-*/model.pt`: 실제 학습 모델, train/validation IDs, 정규화와 설정.',
        '- `training.json`, `predictions.npz`: 학습 곡선·업데이트 수·모델 재로딩 검증·실제 예측.',
        '- `verification.json`: 45개 확인 fit 독립 검산과 실제40epoch 검증 라벨 교란 재학습.',
        '- `comparison.csv`, `fold-metrics.csv`: 모든 점수.',
        '',
        '원래 .venv의 PyTorch만 사용했다. torchvision 설치나 새 가중치 다운로드를 하지 않았다. '
        'CPU 2스레드로 실행했다. '
        '기존 학습 파일이 있으면 재사용하거나 덮어쓰기를 거부한다. 새로운 실험은 별도 출력 경로로 복제해 수행해야 한다.',
        '',
        '```sh',
        '.venv/bin/python experiments/rmse-research-20261015/round3-cnn/complement_cnn.py probe',
        '.venv/bin/python experiments/rmse-research-20261015/round3-cnn/complement_cnn.py run --partition 0',
        '.venv/bin/python experiments/rmse-research-20261015/round3-cnn/confirm.py',
        '.venv/bin/python experiments/rmse-research-20261015/round3-cnn/stats_control.py',
        '.venv/bin/python experiments/rmse-research-20261015/round3-cnn/verify.py',
        '```',''])
    (h/'REPORT.md').write_text('\n'.join(lines))
    with(h/'comparison.csv').open('w',newline='')as output:
        w=csv.writer(output);w.writerow(['stage','arm','RMSE','gain','ci_low','ci_high','improved_folds'])
        for stage,data,keys in [('screen',s,c.ARMS),('confirmation',q,('residual','aux','context_only')),('controls',ctrl,tuple(ctrl))]:
            for key in keys:
                z=data[key];w.writerow([stage,key,z['candidate']['pooled'],z['gain'],*z['conditional_id_bootstrap_gain_ci95'],z['improved_folds']])
    with(h/'fold-metrics.csv').open('w',newline='')as output:
        w=csv.writer(output);w.writerow(['arm','partition','fold','baseline_RMSE','candidate_RMSE'])
        for key in ('residual','aux','context_only'):
            for part in range(3):
                for k in range(5):w.writerow([key,part,k,q[key]['baseline']['folds'][part][k],q[key]['candidate']['folds'][part][k]])
    c.r.save_json(h/'decision.json',dict(current_components_proven_optimal=False,old_021_is_single_CNN=False,
        residual_CNN_gain=residual['gain'],conditional_ci95=ci,spatial_CNN_vs_stats_gain=vsstats['gain'],
        full073_improvement_validated=False,adopted_in_061_or_073=False,public_upload=False,
        conclusion='Image-complement training is evaluated as a new branch; current full ensemble replacement remains a separate validation question.'))
    print('Wrote architecture/training report and complete comparison tables.')


if __name__=='__main__':main()
