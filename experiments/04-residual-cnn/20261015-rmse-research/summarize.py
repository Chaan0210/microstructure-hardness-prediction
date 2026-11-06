"""Generate the research report and complete numerical comparison tables."""
import csv
import json
from pathlib import Path
import numpy as np
import research as r


def main():
    h=r.HERE
    read=lambda f: json.loads((h/f).read_text())
    audit=read('audit.json'); verification=read('verification.json')
    items=[]
    for label,file,arm,status in [
        ('H1 영역별 계수', 'domain.json', 'candidate', '기각'),
        ('H1 동일 격자 pooled 대조', 'domain.json', 'nested_control', '대조'),
        ('H2 원본–열화본 일관성', 'consistency.json', 'candidate', '기각'),
        ('H2 일관성 0 대조', 'consistency.json', 'nested_control', '미세한 선택 효과; 미채택'),
        ('H2 엄격 원본 분리', 'consistency-strict.json', 'candidate', '재검증도 실패'),
        ('H3 잡음 영역 전용 13열', 'nuisance.json', 'candidate', '기각'),
        ('H3 85열 대조', 'nuisance.json', 'control', '대조'),
        ('H4 GMM 16개 부분 표본 평균', 'bagging.json', 'bagged', '기각'),
        ('H4 규제 증가 대조', 'bagging.json', 'penalty_control', '대조')]:
        q=read(file)[arm]; items.append((label,q,status,file,arm))
    scope=read('scope.json')
    items.append(('원본 범위 민감도',scope,'검증 설계 진단; 제출 후보 아님','scope.json',''))
    with (h/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['experiment','pooled_RMSE','original','20261031','20261101','gain_vs_073','gain_CI_low','gain_CI_high','improved_folds','decision'])
        for label,q,status,_,_ in items:
            writer.writerow([label,q['candidate']['pooled'],*q['candidate']['partitions'],q['gain'],*q['conditional_id_bootstrap_gain_ci95'],q['improved_folds'],status])
    baselines=[('061',audit['baseline061']),('073_direct',audit['baseline073_actual_components']),('073_legacy_delta',audit['baseline073_legacy_delta'])]
    with (h/'fold_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['experiment','partition','fold','n','RMSE'])
        for label,m in baselines+[(label,q['candidate']) for label,q,_,_,_ in items]:
            for p,pname in enumerate(['original','20261031','20261101']):
                for k in range(5): writer.writerow([label,pname,k,80,m['folds'][p][k]])
    lines=[
        '# RMSE 정체 조사와 실제 실험 결과 — 2026-10-15',
        '',
        '**새로 채택할 모델 개선은 발견하지 못했다.** 네 개의 새 가설을 실제 구현해 기존 3분할 × 5fold에서 검증했다. '
        '061/073의 모델·가중치·제출물은 유지했다. CPU 2스레드로 검증했다. '
        '',
        '',
        '**1. 기준 모델과 점수**',
        '',
        '| 모델 | 통합 OOF RMSE | original | 20261031 | 20261101 | 저장된 Public |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for label,m in baselines:
        public='8.9831248984' if label=='061' else '8.9863906091'
        lines.append(f"| {label} | {m['pooled']:.6f} | "+' | '.join(f'{v:.6f}' for v in m['partitions'])+f' | {public} |')
    lines.extend([
        '',
        '`073_direct`는 저잡음 행에서 실제 nb GMM 예측을 직접 사용한 기준이다. 기존 `073_legacy_delta`는 '
        '기존 GMM OOF에, 다른 내부 CV seed로 계산한 full→nb 차이를 더했다. 이 차이 전송은 비교 효과와 실제 구성의 절대 점수를 혼동한다. '
        '**이번 모든 새 가설은 실제 성분 기준 9.5928855251과 비교했다.** 073 제출 CSV가 잘못됐다는 뜻은 아니다. '
        '이 작은 불일치는 기존 push-1300 메모 §26에도 언급됐으며, 여기서 독립 재조립해 수치와 fold별 차이를 고정했다.',
        '',
        '061의 구조는 실패 행을 021로 대체하고, 정상 행에서 '
        '`0.16×021 + 0.384×TabPFN + 0.128×Ridge + 0.128×logRidge + 0.05×결정립 집합망 + 0.15×GMM-Ridge`이다. '
        '021 자체는 CNN·DINO 등의 기존 복합 모델이다. TabPFN 블록은 038/039/044 세 특징 집합 × 네 U-Net 변형 × 세 seed의 평균이다. '
        '잡음 ≥12에서는 Ridge/logRidge에 train 열화본 deg~deg4를 쓴다. 073은 잡음 <12에서 TabPFN과 선형/GMM 파트너의 het·intensity 13열을 제외한다. '
        '가중치에는 과거 Public 기반 선택 이력이 있으며, 이번 연구에서는 동결하고 Public으로 새 값을 적합하지 않았다.',
        '',
        '**2. 규칙·데이터·검증 감사**',
        '',
        '[공식 대회 페이지](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai)와 연결된 설명 PDF를 2026-10-15 확인했다. '
        '학습 500장, 평가 1000장, 256×256 합성 미세구조에서 경도 대용값을 예측한다. 지표는 원래 HV 단위의 RMSE, Public 30% / Private 70%이다. '
        '외부 데이터와 test 학습은 금지된다. 이번 실험은 제공 train과 이미 존재하는 train 유래 특징·열화본만 사용했다. '
        '새 외부 모델·다운로드·API 추론·Public 제출은 없었다. 공개 사전학습 가중치의 허용 범위는 공식 규정의 공개 시점·라이선스 조건을 따른다.',
        '',
        '- train은 모두 256×256 흑백, 타깃은 유한하고 범위는 151.979828~264.121303이다. 정확한 원본/D4 중복은 없었다. '
        '숨은 생성기 가족·부분 재사용까지 독립임을 증명한 검사는 아니다.',
        '- 12개 TTA 분산 특징은 50.2%가 결측이다. 이는 영상별로 TTA를 적용한 기존 설계와 연결된 결측 패턴이다. 새 전처리는 각 학습 fold의 중앙값·표준화만 적합했다.',
        '- RMSE는 `sqrt(mean((pred-y)**2))`로 독립 검산했다. 반복 분할 점수의 평균이 아니라 전체 제곱오차를 합한 RMSE다. '
        'logRidge는 지수 역변환 뒤 네 변형을 기하평균하며, 이번 연구에서 역변환이나 clipping을 바꾸지 않았다.',
        '- 세 분할은 동일한 400개 ID의 반복 측정이다. CI는 ID 400개를 재표집하며 세 예측을 함께 보존했다. '
        '10000회, seed 20261015. 1200개의 독립 표본으로 취급하지 않았다.',
        '- 기존 열화본 CV는 outer 검증 원본은 모두 제외했지만, dev 400장 밖의 원본 46장/복사본 135행을 학습에 포함했다. '
        '따라서 고잡음 파트너의 실제 원본 수는 fold당 320+46=366이다. **현재 평가 fold의 정답 누수는 아니지만, '
        '100장 holdout이 전체 개발 과정과 독립이라는 설명은 성립하지 않는다.** 이미 holdout을 여러 번 열람한 선택 편향도 남는다.',
        '- 기존 OOF 위에 다시 fold를 나눈 stacking/잔차 학습은 완전 중첩 CV가 아니다. 메타 학습 행의 기저 예측을 만들 때 메타 검증 행의 라벨이 쓰일 수 있다. '
        '따라서 “앙상블 가중치가 일반화 관점에서 최적”이라는 강한 결론을 낼 수 없다. 이번 새 head 선택은 source-grouped inner CV로 수행했다.',
        '- U-Net/Cellpose 및 오래 사용한 특징 집합을 outer fold마다 새로 학습하지 않았다. 이번 수치는 기존 표현에 조건부인 head 비교이며, '
        '이미 수백 번 재사용한 검증 자료의 선택 편향을 제거한 독립 일반화 점수가 아니다.',
        '',
        '**3. 오차 구조와 병목**',
        '',
        '| 잡음 구간 | 이미지 수 | 073 RMSE | 전체 SSE 비율 | 평균 예측−정답 |',
        '|---|---:|---:|---:|---:|',
    ])
    for name,q in audit['noise_bands'].items():
        lines.append(f"| {name} | {q['n']} | {q['rmse']:.6f} | {100*q['sse_share']:.2f}% | {q['bias']:+.4f} |")
    lines.extend([
        '',
        '고잡음 124장(31%)이 SSE의 51.17%를 만든다. 이 구간 TabPFN 잔차와 Ridge/logRidge/GMM 잔차의 상관은 '
        '각각 약 0.905/0.915/0.906이다. 서로 다른 파트너가 상당 부분 같은 방향으로 틀린다. '
        '최악 10장(2.5%)의 SSE 몫은 15.91%이다. 전체 실패를 한두 개 이상치로 설명할 수 없다. '
        'ID별 예측·잔차·오차 순위는 `residuals.csv`에 저장했다.',
        '',
        '타깃 하위/상위 20%에서 평균 잔차는 +8.94/−8.53이다. 그러나 **정답으로 구간을 나눈 회귀 평균화 현상 자체는 '
        '예측의 calibration 오류 증거가 아니다.** 이를 근거로 예측을 늘이거나 극단값을 잘라내지 않았다. 고잡음 평균 편향은 +0.055 HV로 거의 없다.',
        '',
        '과거 실험은 더 많은 백색 잡음보다 경계 소실·흐림·대비 저하가 측정의 핵심 한계임을 시사한다. '
        '현재 공통 잔차만으로 라벨 잡음의 바닥을 확정할 수는 없다. 공통으로 누락된 미세구조 정보를 모든 모델이 놓쳤을 가능성도 있다. '
        '따라서 기존의 “정보가 모두 소진됐다”는 표현을 이번 결과로 증명했다고 주장하지 않는다.',
        '',
        '**4. 사전 가설과 실행 결과**',
        '',
        '실행 전 우선순위는 H1 영역별 관계 차이 → H2 측정 불변성 규제였다. 둘 다 기존 특징·CPU만으로 검증 가능하고, '
        '실패했던 일반 spline/커널/열화 재가중과 구별되는 구조 변경이다. H1/H2 실패 후에는 H3의 제한된 nuisance 조정, '
        '기저 GMM의 큰 분할 분산을 근거로 H4 표본 평균화를 각각 실행 전에 추가 등록했다. '
        '정확한 후보 격자·seed·중단 조건은 `PLAN.md`에 있다.',
        '',
        '| 실험 | OOF RMSE | 073 대비 개선량¹ | 조건부 95% CI | 개선 fold /15 | 판정 |',
        '|---|---:|---:|---|---:|---|',
    ])
    for label,q,status,_,_ in items:
        lo,hi=q['conditional_id_bootstrap_gain_ci95']
        lines.append(f"| {label} | {q['candidate']['pooled']:.6f} | {q['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {q['improved_folds']} | {status} |")
    lines.extend([
        '',
        '¹ 양수면 개선. 완전한 분할별 결과는 `comparison.csv`, 15개 fold의 모든 결과는 `fold_metrics.csv`에 있다.',
        '',
        '- **H1**: 공유 계수+축소된 clean/noisy/synthetic 영역별 계수. 모든 물리 관계를 영역별로 풀면 파라미터 분산이 늘어난다. '
        '동일 alpha 격자 대조군 대비도 −0.003563으로 실패했다.',
        '- **H2**: 동일 원본/열화본의 예측 차이를 제곱 패널티로 억제. 073 대비 +0.000359만 보면 이득처럼 보이지만, '
        '패널티 0 대조는 +0.003576으로 더 좋다. **일관성 규제 자체는 15개 fold 중 개선 0개**, '
        '세 분할 모두 악화했다. 원본을 엄격히 분리한 조건에서도 실패했다. 정보 손실 뒤 조건부 평균이 원본과 같아야 한다는 가정은 지나치게 강할 수 있다.',
        '- **H3**: 공유 85열에 고잡음 train 행에서만 13개 nuisance 열을 활성화. 동일 85열 대조 대비 +0.000012로 사실상 차이가 없었다. '
        '큰 악화는 대조군에도 나타나므로 이 실험에서 독립적인 nuisance 효과라고 해석하지 않는다. 단독 head RMSE에 맞춘 규제 선택이 '
        '앙상블에 필요한 잔차 상보성을 보존한다는 보장이 없다는 점도 드러난다.',
        '- **H4**: 기존 GMM OOF 세 예측을 평균하면 0.24 RMSE 낮아지는 것은 진단일 뿐이다. 실제 outer train의 90%를 16회 뽑아 평균하면 '
        '최종 앙상블은 오히려 0.006508 악화했다. 표본 감소에 따른 편향이 분산 감소를 상쇄했다. '
        '규제만 1/0.9배 한 대조도 악화해, seed/표본 평균을 계속 늘리지 않았다.',
        '- **원본 범위 민감도**: 실제 outer train에 속한 원본의 복사본만 사용하면 9.581685가 된다. CI가 0을 포함한다. '
        '최종 학습은 500장 전체이므로 이 수치는 그대로 배포할 새 알고리즘의 개선이 아니다. '
        '잘 정의된 학습 범위가 미세한 후보 차이보다 중요하다는 검증 진단으로만 보존했다.',
        '',
        '**5. 우연·선택 편향 평가와 채택 결정**',
        '',
        '073 직접 구성의 061 대비 이득은 0.070764, ID bootstrap 95% CI [−0.006100, +0.152227]이다. '
        '조건부 bootstrap에서 양수인 비율 0.9631은 “일반화 개선 확률 96.31%”가 아니다. '
        '가장 큰 개선 10장을 빼면 이득은 −0.008050으로 바뀐다. 수많은 과거 특징·모델 선택의 낙관은 CI에 포함되지 않는다.',
        '',
        '네 가설 모두 사전 채택 조건을 충족하지 못했다. 한 격자의 아주 작은 이득을 다음 격자의 출발점으로 삼지 않고 중단한다. '
        '**새 모델·앙상블·후처리를 채택하지 않았다.** 061은 저장된 Public 최고, 073은 더 나은 조건부 local 기준으로 각각 보존한다. '
        '최종 대회 선택 상태와 Public 제출은 변경하지 않았다.',
        '',
        '**6. 보존 모델의 fold별 RMSE**',
        '',
        '| 분할 | fold | 061 | 073 직접 구성 |',
        '|---|---:|---:|---:|',
    ])
    for p,pname in enumerate(['original','20261031','20261101']):
        for k in range(5):
            lines.append(f"| {pname} | {k} | {baselines[0][1]['folds'][p][k]:.6f} | {baselines[1][1]['folds'][p][k]:.6f} |")
    lines.extend([
        '',
        '**7. 재현 코드·설정·제출 파일**',
        '',
        '- `research.py`: 입력 스냅샷·기준 재조립·H1/H2·실제 원본에 의한 inner/outer 분리.',
        '- `followups.py`: H3와 네 logRidge 변형의 원본 범위 민감도.',
        '- `bagging.py`: 기존 GMM 코드 재사용, H4 및 규제 대조.',
        '- `verify.py`: 14개 결과표의 독립 scalar 검산, 검증 라벨 +10000 교란, 모델 저장/재로딩, 입력 보호 해시.',
        '- `reproduce_best.py`: 기존 성분을 안전하게 재조립하고, 선택 시 15개 선형 모델을 실제 재학습·저장한다.',
        '- `inputs.npz`, `provenance.json`, 각 실험 `.npz/.json/.log`: ID·타깃·fold·예측·설정·선택된 alpha/패널티·입력 해시.',
        '',
        '루트 디렉터리에서 새 출력 폴더 이름으로 실행한다. 기존 출력 폴더나 기존 제출 CSV가 있으면 덮어쓰지 않고 중단한다.',
        '',
        '```sh',
        '.venv/bin/python experiments/rmse-research-20261015/reproduce_best.py --output-name reproduction-user-run',
        '.venv/bin/python experiments/rmse-research-20261015/reproduce_best.py --refit-linear --output-name refit-user-run',
        '.venv/bin/python experiments/rmse-research-20261015/verify.py',
        '```',
        '',
        '실제 생성·검증한 파일:',
        '',
        '| 파일 | 검증 | SHA256 |',
        '|---|---|---|',
        '| `reproduction/submission-061.csv` | 기존 061과 바이트 동일 | `259d69c4839a1f607e6b08686d16468f7dd5a6af9fc541e2e0449f020d23021d` |',
        '| `reproduction/submission-073.csv` | 기존 073과 바이트 동일 | `6a243cfb5ff5a230d1da7de56be0fe2333db56902dacd3404e95078ceb791c11` |',
        '',
        '`--refit-linear`도 실제 실행했다. 모델 15개의 재로딩 예측은 exact, 최종 CSV의 기존값 대비 최대 차이는 1.5006663e−11이다. '
        '부동소수점 차이 때문에 이 경로의 CSV SHA는 다르다. 행 1000개, ID 순서·중복·유한값·실패 행 fallback을 검증했다. '
        '환경과 alpha·변형·seed·가중치는 `verification.json`, `retrained-linear/fit-validation.json`, 각 출력 `manifest.json`에 고정했다.',
        '',
        '**재현 범위:** 기본 명령은 고정된 성분 예측에서 제출 파일을 재구성하는 회귀 검사다. 이것만으로 raw image에서 시작하는 '
        '독립 코드 심사 패키지를 새로 만들었다는 뜻은 아니다. 선형 모델 외 모든 신경망을 이번에 다시 학습하지 않았다. '
        '원본 이미지→기존 모델의 훈련·추론 경로는 다음 저장소 문서를 따른다. 기존 번호 파일을 보호하려면 별도 복제본에서 수행해야 한다.',
        '',
        '- `experiments/push-800/REPRO-045.md`: 021 + Cellpose/U-Net 특징 + TabPFN 038/039/044, 필요한 로컬 가중치와 환경.',
        '- `experiments/push-1200/REPRO-057.md` §5: 061의 deg~deg4 증강·full500 선형 재학습.',
        '- `experiments/push-1300/final_nb_tab.py`, `final_nb_partners.py`, `make_073.py`: 073의 축소 특징 학습·조립.',
        '- `experiments/probabilistic-grains-900/set_model/finalize500.py`, `experiments/push-900/gmm_partner.py`: 집합망/GMM.',
        '',
        '실험 자체를 새로 돌릴 때는 이 디렉터리의 코드와 `inputs.npz`를 새 `experiments/<별도이름>`으로 복사하고 '
        '`research.py domain`, `research.py consistency`, `research.py consistency --strict`, '
        '`followups.py nuisance`, `followups.py scope`, `bagging.py` 순서로 실행한다. '
        '이 방식은 기존 결과를 삭제하지 않는다. `snapshot`은 원래 저장소 입력을 다시 읽는 선택 단계다. '
        '경량 실험은 CPU에서 완료되며 외부 네트워크/가중치 설치가 필요 없다.',
        '',
        '**8. 검증 결과와 남은 가장 유망한 경로**',
        '',
        '기존 입력/보호 파일 30개의 시작·종료 SHA가 같았다. head의 outer 검증 라벨을 +10000 바꿔도 '
        '내부 선택·계수·예측이 exact였고, 15개 fold의 원본 격리를 확인했다. 해당 증거는 `verification.json`에 있다. '
        '실패한 결과와 대조군도 모두 보존했다.',
        '',
        '지금 남은 후보 중 기대값이 상대적으로 큰 것은 **현재 요약 특징과 다른, 원영상의 추가 정보**를 실제로 학습하는 경로다. '
        '병행한 CNN 실험은 이 범주에 속한다. 평가할 때 독립된 새 정보의 잔차 상보성, '
        '동일 학습량의 CNN 대조, 고정 혼합 비중, 여러 분할에서의 이득을 먼저 확인해야 한다. '
        '그 결과가 나오기 전에 GPU 학습을 중복 실행하지 않았다.',
        '',
        '또 하나는 모든 선택·잔차·가중치 학습까지 outer fold 안에 넣고, 가능하면 500장 전체의 새 분할에서 '
        '후보와 기준을 함께 재학습하는 것이다. 이는 점수 개선법이라기보다 미세한 이득을 믿을 수 있게 하는 기반이다. '
        '현재 네 가설에 양성 결과가 없어 비싼 전체 nested 신경망 재학습은 정당화되지 않았다. '
        '이미 시도한 손쉬운 열화본 조정·일반 spline·비중 sweep을 다시 반복할 근거도 얻지 못했다.',
        '',
        '이번 결론은 “이 데이터의 최적 RMSE를 달성했다”가 아니라, **검증한 네 변경이 보존 모델보다 유리하지 않았고, '
        '새 성능 개선을 주장할 증거가 없다는 것**이다.',
        '',
    ])
    (h/'REPORT.md').write_text('\n'.join(lines))
    r.save_json(h/'decision.json',dict(adopted_model_change=False,preserved_candidates=['061','073'],
        reference_oof_061=audit['baseline061']['pooled'],reference_oof_073=audit['baseline073_actual_components']['pooled'],
        rejected_hypotheses=['domain_partial_pooling','paired_consistency','noisy_domain_nuisance','gmm_subbagging'],
        uploaded=False,other_agent_files_modified=False,new_pretrained_models=False,compute='CPU two threads; no remote jobs',
        reason='No preregistered candidate passed the paired repeated-fold comparison; tiny control gains are inconclusive and not adopted.'))
    print('Wrote REPORT.md, comparison.csv, fold_metrics.csv, decision.json')


if __name__=='__main__': main()
