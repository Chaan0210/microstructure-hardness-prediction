"""Write complete continuation tables from executed results, no inferred scores."""
import csv
import json
from pathlib import Path
import numpy as np
from models import HERE,inputs
import research as r


def main():
    read=lambda name:json.loads((HERE/name).read_text())
    d=inputs();a=read('evidence-iter2000.json');p=read('pls.json');c=read('cumulants-eval.json');f=read('full500-summary.json');t=read('h9-summary.json');q=read('h9-three-seeds.json')
    old=[('엄격 073 기준',r.metrics(d['strict_base'],d['y'][d['dev']],d['fold']),None),
         ('H5 특징군별 evidence',a['group']['candidate'],a['group']),
         ('H5 등방 대조',a['isotropic']['candidate'],a['isotropic']),
         ('H5 무작위 특징군',a['permuted_group']['candidate'],a['permuted_group']),
         ('H6 PLS 저랭크 이차항',p['quadratic']['candidate'],p['quadratic']),
         ('H6 같은 격자 선형 대조',p['null']['candidate'],p['null']),
         ('H7 고차 밝기 통계 + GMM',c['cumulants']['candidate'],c['cumulants']),
         ('H7 무작위 18열 대조',c['random']['candidate'],c['random'])]
    fresh=[('500장 표 기반 기준',f['group_all_blend']['baseline'],None)]+[(name,f[key]['candidate'],f[key])for name,key in [
        ('H5 전체 적용','group_all_blend'),('H5 저잡음만 적용(사후 가설)','group_low_blend'),('H8 완전 중첩 선택','nested_blend'),
        ('H7 밝기 통계 + GMM','cumulants_blend'),('H7 무작위 대조 0','random0_blend'),('H7 무작위 대조 1','random1_blend'),('H7 무작위 대조 2','random2_blend')]]
    fresh += [('H9 밝기 분율·대비 + TabPFN',t['physical_blend']['candidate'],t['physical_blend']),('H9 무작위 2열 + TabPFN',t['random_blend']['candidate'],t['random_blend'])]
    replicated=[('동일 조건 3-seed 기준',q['physical']['baseline'],None),('H9 3-seed 실제 특징',q['physical']['candidate'],q['physical']),('H9 3-seed 무작위 특징',q['random']['candidate'],q['random'])]
    with(HERE/'comparison.csv').open('w',newline='')as output:
        w=csv.writer(output);w.writerow(['benchmark','model','RMSE','partition0','partition1','partition2','gain','ci_low','ci_high','improved_folds'])
        for scope,rows in [('400_strict',old),('500_fresh',fresh),('500_fresh_3seeds',replicated)]:
            for name,m,stats in rows:
                w.writerow([scope,name,m['pooled'],*m['partitions'],stats['gain']if stats else 0.,*(stats['conditional_id_bootstrap_gain_ci95']if stats else [0.,0.]),stats['improved_folds']if stats else 0])
    with(HERE/'all-fold-metrics.csv').open('w',newline='')as output:
        w=csv.writer(output);w.writerow(['benchmark','model','partition','fold','RMSE'])
        for scope,rows in [('400_strict',old),('500_fresh',fresh),('500_fresh_3seeds',replicated)]:
            for name,m,stats in rows:
                for part in range(3):
                    for k in range(5):w.writerow([scope,name,part,k,m['folds'][part][k]])
    lines=['# 수정된 검증 기준으로 계속한 연구 — 2026-10-15','',
        '061·073을 보존하고, 새로운 측정·모형·선택 절차를 실제 실행했다. CPU로 수행했다. '
        '첫 연구 결과는 상위 디렉터리에 그대로 남겨 두고 이번 결과는 `round2`에 저장했다.','',
        '**서로 다른 세 검증 점수를 섞지 않는다.**','',
        '1. 기존 400장 × 3분할에서는 073의 실제 성분을 조립하고, 증강 원본을 각 fold의 실제 train 320장으로 제한한다. '
        '엄격 기준은 9.5816850901이다. 역사적 061 9.663650 및 073 직접 조립 9.592886과는 학습 범위가 다르다.',
        '2. 500장 전체에서 seed 2026101501/2026101502/2026101503의 새로운 5-fold CV를 실행했다. '
        '이 비교는 CNN/집합망을 제외한 CPU 표 기반 앙상블이다. TabPFN은 평균 특징 bank 하나, n_estimators=4, fold당 seed 하나이고 '
        '가중치는 TabPFN/Ridge/logRidge/GMM = (0.384,0.128,0.128,0.15)/0.79이다. 기존 073 전체 CV라고 부를 수 없다.',
        '3. H9는 같은 500장/분할에서 기준·실제 특징·무작위 특징 세 모델을 seed offset 0/100/200으로 모두 반복하고 동일 평균했다. '
        '다른 앙상블 성분은 고정했다. 이 비교의 기준도 별도로 보고한다.',
        '',
        '모든 증강 사본은 해당 학습 fold의 원본에 속한다. H8은 outer 15개마다 inner 3개, 총 45개 내부 검증에서도 '
        'TabPFN과 모든 선형/GMM 성분을 다시 학습했다. 과거 OOF를 다시 나누어 메타 학습한 결과가 아니다. '
        '단, 같은 500장의 새 분할은 독립 데이터가 아니며, 기존 표현·과거 모델 선택의 편향을 지우지 못한다. '
        'CI는 이미지 ID별 bootstrap 10000회이며 세 분할을 한 묶음으로 재표집한다.',
        '',
        '**새 가설과 기존 실험과의 차이**','',
        '- H5: 크기·형상·2상·영상 품질 등의 기존 의미 있는 특징군마다 계수의 prior precision을 학습한다. '
        '13열만 임의로 약하게 규제했던 기존 실험과 다르다. 등방 prior 및 같은 크기의 무작위 특징군을 대조로 둔다. '
        '초기 100회는 일부 수렴하지 않아 그 결과를 보존하고, 동일 식·prior·damping을 2000회 한도까지 계산했다. '
        '400장 비교의 적합은 324회 이내, 500장 중첩 검증의 120개 group 적합은 649회 이내에 모두 수렴했고, 성능 결론은 바뀌지 않았다.',
        '- H6: 각 inner train에서 PLS 네 축을 새로 학습하고, 그 열 개 이차항을 원래 특징에 더한다. '
        '일반 spline이나 raw 특징의 모든 곱, 선형 PLS와 다르다. 지도학습으로 만든 축을 전체 데이터에서 먼저 만든 뒤 CV하지 않았다.',
        '- H7: 그림자와 기공의 영향을 줄인 밝기 분포에서 3·4차 cumulant를 측정한다. Gaussian 잡음과 이진 상 혼합이라는 '
        '근사에서 상 분율·대비를 추정하며, 1/2/4px 세 평활도에서 총 18열을 만든다. '
        '공간 상관과 영상 의존 가중치가 있으므로 불편 추정량·정확한 생성기 복원이라고 주장하지 않는다.',
        '- H8: 단독 모델 점수가 좋아도 앙상블에 해로울 수 있다는 H5/H6 결과에 따라, 실제 inner 앙상블 손실로 '
        'group-Ridge 변경의 적용량 0/0.25/0.5/1을 선택한다. 두 기존 잡음 구간마다 선택하되, outer 라벨은 전혀 보지 않는다.',
        '- H9: H7의 18열 선형 추가보다 조건별 비선형 사용이 적합할 수 있다는 가설이다. 물리적 근거로 고정한 '
        'scale-1 상 분율 p34와 대비 추정치 두 개만 TabPFN에 넣고 같은 두 열 수의 무작위 대조와 비교했다. '
        '추가 seed 검증에서도 특징·가중치·잡음 경계·seed 선택을 바꾸지 않았다.',
        '',
        '아이디어 참고: [Tipping의 sparse Bayesian learning 논문](https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf), '
        '[SciPy cumulant 정의](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstat.html). '
        '논문의 모델을 그대로 재현했다는 주장은 하지 않으며, 외부 학습 데이터나 새 가중치를 가져오지 않았다.',
        '',
    ]
    for title,rows in [('400장 엄격 기준',old),('500장 새로운 분할: 1-seed 표 기반 기준',fresh),('500장 같은 분할: 3-seed 반복 검증',replicated)]:
        lines.extend([f'**{title}**','', '| 실험 | RMSE | 기준 대비 개선량 | 조건부 95% CI | 개선 fold/15 |', '|---|---:|---:|---|---:|'])
        for name,m,stats in rows:
            if stats:
                ci=stats['conditional_id_bootstrap_gain_ci95'];lines.append(f"| {name} | {m['pooled']:.6f} | {stats['gain']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {stats['improved_folds']} |")
            else:lines.append(f"| {name} | {m['pooled']:.6f} | 기준 | — | — |")
        lines.append('')
    lines.extend(['**결과 해석**','',
        'H5는 의미 있는 특징군이 무작위 grouping보다 좋고 Ridge 단독 RMSE도 9.8433→9.6900으로 낮아졌지만, '
        '고정 앙상블에는 해로웠다. 사후 발견한 저잡음 전용 적용도 500장 세 분할 모두 악화했다. '
        'H6은 같은 격자의 선형 대조보다 개선됐으나 운영 기준을 넘지 못했다. H8의 완전 중첩 선택도 '
        '평균 0.015272 악화했다. 단독 점수 개선이나 새 내부 선택 절차를 곧바로 최종 개선으로 해석하지 않았다.',
        '',
        'H7은 400장 엄격 비교 +0.008606, 500장 표 기반 비교 +0.008453으로 비슷한 작은 이득을 보였다. '
        '하지만 새 분할의 개선은 7/15 fold, 세 분할 중 하나는 악화했고 개선 상위 다섯 ID를 빼면 −0.000544이다. '
        '원래 400장 무작위 18열 대조도 좋아졌다는 점까지 고려하면 정보 이득 확정은 이르다. '
        '최대 개선/악화 이미지의 원본 비교는 `cumulant-tail-diagnostic.png`에 있다. 네 예시는 모두 고잡음이고, '
        '373번은 p34가 0.564로 U-Net 0.225와 크게 엇갈린다. 고차 통계가 항상 더 옳은 측정은 아니다.',
        '',
        'H9 1-seed는 평균 +0.016606이며 무작위 대조 대비 +0.015665였다. 그러나 분할별 이득은 '
        '+0.022934 / −0.029980 / +0.056686으로 일관되지 않았다. 추가 seed 두 개를 모두 같은 비중으로 평균한 결과는 위 표와 아래 결정을 따른다.',
        '',
        '**재현과 파일**','',
        '- `PLAN.md`, `PLAN-H9.md`, `PLAN-H9-REPLICATION.md`: 실행 전 고정 내용과 후속 가설의 시점.',
        '- `models.py`: 특징군별 evidence/PLS 이차항, strict-source 학습.',
        '- `cumulants.py`: train 원영상에서 새 특징 추출, 수학/D4 검사, GMM 추가 실험.',
        '- `full500.py`: 모든 base를 포함한 500장 완전 중첩 선택 벤치마크. CPU만 사용.',
        '- `cumulant_tab.py`, `replicate_h9.py`: 두 물리 특징의 TabPFN 검증과 seed 반복.',
        '- `comparison.csv`, `all-fold-metrics.csv`: 모든 전체·분할·fold 점수.',
        '- `verification.json`, `verification-h9.json`: 원본 ID 배제, 선택 식, 지표, 보호 파일, 재로딩 검증.',
        '- 모든 `.npz`, `.json`, `.log`: 실제 예측·설정·실패·판정 기록. 초기 full500 구문 오류 로그도 보존했다.',
        '',
        '루트 `.venv`는 CPU 통계 실험에, `experiments/tabpfn-v2/.venv/bin/python`은 기존 로컬 TabPFN에 사용한다. '
        '새 설치는 없었다. 원래 디렉터리에 존재하는 결과를 덮어쓰지 않으려면 소스와 입력 캐시를 새 실험 디렉터리에 복제해서 실행한다. '
        'full500/H9 스크립트는 완성된 fold 파일을 재사용하므로, 다른 설정으로 실행할 때는 새 디렉터리가 필요하다.',
        '',
        '```sh',
        '.venv/bin/python experiments/rmse-research-20261015/round2/models.py evidence --max-iter 2000',
        '.venv/bin/python experiments/rmse-research-20261015/round2/models.py pls',
        '.venv/bin/python experiments/rmse-research-20261015/round2/cumulants.py check',
        '.venv/bin/python experiments/rmse-research-20261015/round2/cumulants.py extract',
        '.venv/bin/python experiments/rmse-research-20261015/round2/cumulants.py evaluate',
        '# full500.py: --partition 0, 1, 2를 차례로 실행',
        'experiments/tabpfn-v2/.venv/bin/python experiments/rmse-research-20261015/round2/full500.py --partition 0',
        'experiments/tabpfn-v2/.venv/bin/python experiments/rmse-research-20261015/round2/cumulant_tab.py',
        'experiments/tabpfn-v2/.venv/bin/python experiments/rmse-research-20261015/round2/replicate_h9.py',
        '```','',
        '기존 061/073 재조립·선형 모델 재학습·제출 CSV 생성은 상위 `reproduce_best.py`와 첫 보고서에 있다. '
        '이번 탐색에서 기존 모델, 기존 제출 CSV, 대회 최종 선택과 Public 제출을 변경하지 않았다.',
        '',
    ])
    physical=q['physical'];vsrandom=q['physical_vs_random'];ci=physical['conditional_id_bootstrap_gain_ci95']
    passed=physical['gain']>0 and all(v>=0 for v in physical['partition_gains']) and physical['improved_folds']>=8 and vsrandom['gain']>0
    decision=dict(three_seed_gain=physical['gain'],three_seed_gain_ci95=ci,three_seed_partition_gains=physical['partition_gains'],
                  physical_vs_random_gain=vsrandom['gain'],screen_pass=passed,adopted=False,originals_preserved=['061','073'],public_submission=False)
    decision['reason']='Positive point estimate remains after seed repetition, but one partition worsens and conditional intervals against both baseline and random control span zero. No proven full-073 or Private improvement.'
    lines.extend(['**이번 반복의 결정**','',
        f"H9 3-seed의 관측 이득은 {physical['gain']:+.6f}, 조건부 95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}]이며, "
        '분할별 이득은 '+' / '.join(f'{v:+.6f}'for v in physical['partition_gains'])+'이다. '
        f"무작위 특징 대조 대비 이득은 {vsrandom['gain']:+.6f}이다.",
        '',
        '**이번에는 새 모델을 채택하지 않는다.** 두 모델 seed를 더 반복해도 한 분할의 악화가 남고, 기준·무작위 대조 양쪽 대비 신뢰구간이 0을 포함한다. '
        '약한 양의 평균이나 단독 모델 향상을 Private 개선으로 표현하지 않는다. '
        '새로운 정보 후보는 고차 밝기 측정이며, 지금까지의 검증으로 확인된 범위와 불확실성을 함께 보존한다. '
        '기존 특징의 규제·내부 선택만 바꾸는 경로는 이번 새 분할에서도 뒷받침되지 않았다.',''])
    r.save_json(HERE/'decision.json',decision)
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    print('Wrote REPORT.md, comparison.csv, all-fold-metrics.csv, decision.json; screen_pass =',passed)


if __name__=='__main__':main()
