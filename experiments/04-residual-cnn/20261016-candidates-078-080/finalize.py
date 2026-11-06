"""Assemble the experiment record only after all required checks finish."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv
import json
import numpy as np
import study as s


def main():
    s.protect()
    def read(name):return json.loads((s.HERE/name).read_text())
    audit=read('attribution.json');pilot=read('pilot-results.json')
    stress=read('calibration-stress.json');clipping=read('clipping-check.json');verify=read('verification.json')
    infer=read('frozen-inference-verification.json');trained=read('selected-training-reproduction.json')
    assert verify['all_passed'] and infer['candidate_bitwise_exact'] and trained['complete']
    assert not pilot['advance_by_locked_rule']
    assert not stress['stronger_production_weight_passes_locked_rule']
    for p,sha in infer['source_sha256'].items():assert s.r.digest(_repo_path(s.ROOT, p))==sha,p
    d=dict(np.load(_repo_io(s.R4.parent/'inputs.npz')));y=d['y'][d['dev']];h=_repo_path(s.ROOT, 'experiments/push-1300')
    p76=np.load(_repo_io(h/'dev-combo.npy'));p78=np.load(_repo_io(h/'dev-078.npy'))
    a=np.mean((p76-y)**2,axis=0);b=np.mean((p78-y)**2,axis=0)
    draws=np.random.default_rng(2026101623).integers(0,400,(10000,400))
    gain=np.sqrt(a[draws].mean(1))-np.sqrt(b[draws].mean(1))
    old_partner_uncertainty=dict(gain=float(np.sqrt(a.mean())-np.sqrt(b.mean())),
                                paired_ID_ci95=np.quantile(gain,[.025,.975]).tolist(),
                                partitions_are_not_independent=True)
    z=np.load(_repo_io(s.HERE/'fixed-composition-oof.npz'))
    a=(z['recorded076']-y)**2;b=(z['candidate']-y)**2
    gain=np.sqrt(a[draws].mean(1))-np.sqrt(b[draws].mean(1))
    fixed_uncertainty=dict(gain=float(np.sqrt(a.mean())-np.sqrt(b.mean())),
                          paired_ID_ci95=np.quantile(gain,[.025,.975]).tolist(),
                          conditional_reused400=True)
    p80=p78[0]+z['correction']
    hist080=s.metrics(y,p80,d['fold'][0])
    s.save(_repo_io('uncertainty.json'),dict(partner078_vs076=old_partner_uncertainty,
                                  fixed_CNN_on076=fixed_uncertainty,historical080_OOF=hist080))
    decision=dict(adopted='076+(080-078), equivalently076+the unchanged frozen full500 CNN correction',
        CSV=str((s.HERE/'candidate-076-plus-frozen-cnn.csv').relative_to(s.ROOT)),CSV_sha256=audit['CSV_sha256'],
        production_noise_threshold=12,production_CNN_weights=[0.,.5467904360508202],
        measured_Public_RMSE=None,
        uploaded=False,new_CNN_training_arms_adopted=[],stronger_coefficient_adopted=False,
        preserved_original_count=len(read('protected.json')),all_original_hashes_unchanged=True,
        reason_to_stop='Two new CNN objectives failed matched controls; stronger calibration failed locked stability tests. Fixed composition is reproducible and keeps the pre-existing coefficient.',
        remaining_promising_work='Build full076 nested teachers and learn its residual directly, with more independent internal calibration roles. Existing400/500 labels cannot provide a fresh independent confirmation.',
        selected_training_reproduced_bitwise=True)
    s.save(_repo_io('decision.json'),decision)
    rows=[]
    labels={'strict073':'보존된073, strict-source 재현','old-frozen-noise-mean2':'기존CNN 2seed + 기존 내부 계수',
            'old-seed0':'기존CNN 1seed + 고잡음 내부 OLS','old-mean2':'기존CNN 2seed + 고잡음 내부 OLS',
            'high_loss-seed0':'새 학습H1: 고잡음 주 손실','no_teacher_context-seed0':'새 학습H2: 교사 예측 context 제거'}
    for key in labels:
        q=pilot['results'][key]
        rows.append('| '+labels[key]+f" | {q['rmse']:.6f} | "+' / '.join(f'{v:.6f}' for v in q['folds'])+' |')
    body=f'''# 078 추가 점수 분석과 개선 실험 — 2026-10-16

최종 파일은 `candidate-076-plus-frozen-cnn.csv`이다. 076에 기존 CNN 보정만
더했으며 원래 보정 계수, 모델, noise12 경계는 바꾸지 않았다. 새 CNN 두
학습법과 계수 확대는 비교 실험을 통과하지 못해 채택하지 않았다.

## 동일 fold에서 실제 실행한 비교

기존061의 같은 strict-source original5fold RMSE는9.629271이다.
073의 일반행 구성은 `.16*G021+.384*TabPFN+.128*Ridge+.128*logRidge
+.05*grain-set+.15*GMM-Ridge`이며, 실패행은G021로 돌아간다.
G021은 TinyCNN 하나가 아니라 KRR·TinyCNN·DINO 계열과 보정을 포함한
기존 앙상블이다. 076은073 위에 H9 두 측정값을 TabPFN에 추가하고
고잡음 logRidge의 내부 검증된 alpha 변경을 반영한 버전이다.

아래는 동일한 개발400행·원래5fold·strict-source073 기준이다.
새 CNN은 A213 학습/B107 계수 결정 뒤 T320 재학습/V80 평가를 수행했다.
교사 잔차도 각 학습행을 제외한 내부 OOF 예측이며, 외부 라벨로 학습량이나
계수를 선택하지 않았다. 고잡음 OLS는 내부 고잡음 행의 bounded least squares
한 계수이고, 저잡음 보정은0으로 사전 고정했다.

| 방법 | 전체 RMSE | fold0 / fold1 / fold2 / fold3 / fold4 |
|---|---:|---|
{chr(10).join(rows)}

H1은 잔차 주 손실을 고잡음 행에만 주고 원래 경도 예측 보조 손실(.2)은
전체 이미지에 유지했다. H2는 6개 context 중 T와 Tab-minus-Ridge만 제거했다.
두 방법 모두 기존68,788파라미터 구조, AdamW .002, weight decay .01,
40epoch, batch32, D4, 마지막5epoch 평균, 동일 seed를 사용했다.
seed는 `2026101501+100*fold`이다. 두 방법 합계20개 모델을 실제 학습했다.

H1은 같은 계수 방식의 기존1seed CNN보다0.016396 나빴고 5fold 중2개만
개선했다. H2는0.111828 나빴고 모든 fold에서 악화했다. 따라서 단순히
073보다 좋아졌다는 이유로 새 학습법을 채택하지 않았다. 사전 확장 기준에
미달했으므로 이 두 방법의 두 번째 seed와 full500 학습은 중단했다.

기존CNN의 고잡음 OLS 계수는2seed에서0.9901/0.5000/1.0000/0.5862/1.0000이었다.
기존 계수 방식보다0.091372 좋아 보였지만, 그 조건부95% 구간은
[-0.034096,0.237405]이며 이득이 fold4에 집중됐다.

이를 확인하려고 모델을 A213으로 고정한 세 anchor(0,1,4)의 학습 제외187행을
각각107/80으로5회 나눴다. anchor4는 결과를 본 뒤 선정한 집중 검사이므로
독립 확인이 아니다. 추가로 full073 교사 한 묶음(72TabPFN·3grain-set)을
실제 학습해 제외된80행에 대한 정직한 예측을 확보했다.
총1200예측행을342개 원본ID별 평균 제곱오차로 먼저 합친 결과는 다음과 같다.

| 고정 모델 검증 | RMSE |
|---|---:|
| 보정 전 | {stress['summary']['rmse']['base']:.6f} |
| 기존 계수 선택, 고잡음에만 적용 | {stress['summary']['rmse']['original']:.6f} |
| 고잡음 OLS | {stress['summary']['rmse']['high_ols']:.6f} |

OLS의 추가 이득은0.018944, 조건부95% 구간은[0.000411,0.040824]였지만
anchor1에서는 악화했고, 이득이 큰10개ID를 제외하면−0.007332였다.
미리 정한 안정성 기준에 미달했다. **운영 계수를 키우지 않았다.**

## 076 위 조합과 검증 한계

| 저장된076의 original fold 비교 | 전체 RMSE | fold0 / fold1 / fold2 / fold3 / fold4 |
|---|---:|---|
| 076 | {audit['recorded076']['rmse']:.6f} | {' / '.join(f'{v:.6f}' for v in audit['recorded076']['folds'])} |
| 076 + 기존CNN | **{audit['recorded076_plus_frozenCNN']['rmse']:.6f}** | {' / '.join(f'{v:.6f}' for v in audit['recorded076_plus_frozenCNN']['folds'])} |

이 비교의 조건부 paired-ID 이득95% 구간은
{fixed_uncertainty['paired_ID_ci95']}이다. 076은 저장된 과거 OOF를 사용했으며
이번에 strict-source076 전체를 재학습한 점수는 아니다. 기존 선형 DTA의
개발군 밖 source 사용 문제를 숨기지 않는다. strict073의 별도 비교가 위에 있다.
OOF는 fold별 내부 계수이고 운영 계수는 기존5개 내부 계수의 평균이다.
그 평균 운영 계수 자체에 별도의 정직한 OOF 점수가 있는 것은 아니다.

공정한 비교를 위해 적는다: 저장된080 대응 OOF는 **{hist080['rmse']:.6f}**로
이번 조합보다 조금 좋다. 078의3분할 평균 로컬 이득은
{old_partner_uncertainty['gain']:.6f}, 조건부95% 구간은
{old_partner_uncertainty['paired_ID_ci95']}였다. 따라서076으로 돌아가는 결정에는
이번 Public 대비가 반영되어 있다. 이를 새 독립 검증의 승리로 표현하지 않는다.
세 분할은 같은400라벨을 재사용했고, 이전100행 holdout도 이미 학습·선별에
노출됐다. Private 향상 보장은 없으며 모든 부트스트랩은 반복 탐색 이후의
조건부 요약이다.

## 078 악화 원인 검사

Public은 저잡음 파트너 변경 전체의 악화를 보여주지만 어느 모델이나 행이
원인인지는 알려주지 않는다. 테스트 예측 변화 RMS는0.493763, 개발3분할은
0.349802였다. 저잡음4행은 새 피처가 학습 범위 밖이며 변화 에너지의11.75%가
그 행들에 있다. 이것은 분포·외삽 위험의 단서이지 정답을 이용한 원인 증명이 아니다.

CV는 H9 피처를 자르지 않고 최종 추론은 train 범위로 자른 불일치를 찾았다.
원래15fold의 선형 파트너를 다시 학습한 후 raw와 train-fold 범위 clipping을
같이 예측했다. 저장된 raw 예측을 세 파트너 모두 오차0으로 재현했다.
clipping의 전체 RMSE 효과는 **{clipping['clipping_gain']:.9f}**뿐이었다.
이 불일치는 실제 존재하지만 이번 성능 악화의 충분한 설명으로 채택하지 않는다.

## 재현과 보존

- 새CNN20개의 체크포인트 재추론이 정확히 일치했다.
- 검증 라벨을 바꿔도 모든 학습 입력이 같고, 실제40epoch 재학습한 대조의
  모든 파라미터도 같았다. 원래 학습 코드의 all-ones 대조 재현 오차도0이다.
- 선택한 기존 full500 CNN2개를 별도 폴더에서 재학습했다. 두 seed의 모든
  파라미터가 보존된 원본과 bitwise 동일했다. 전체학습 점수를 OOF로 보고하지 않았다.
- 074·076을 구성 요소 예측으로 재조립하고, 테스트 원본318장과 보존된CNN
  가중치로 다시 추론했다. 권장 CSV와 bitwise 동일하고 재추론 오차0이었다.
- 학습 원본500장·train.csv는 보존된 공식 다운로드 ZIP과 byte 동일했다.
- 보존 대상으로 지정한{len(read('protected.json'))}개 파일의 SHA256은 모두 유지됐다.

권장 CSV SHA256: `{audit['CSV_sha256']}`. 1000행, ID순서 일치,
유한한 hardness 값, 기존 실패행 fallback 유지. 제출은 하지 않았다.
실행법은 `RUN.md`, 전체 수치는 `pilot-results.json`,
`calibration-stress.json`, `clipping-check.json`, `uncertainty.json`에 있다.

이번 실험은 제공 train만 학습에 사용했고 외부 데이터·test 학습·원격 예측 API를
추가하지 않았다. [공식 규정](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai)의
2026-10-16 열람 내용은 제공 train 외 외부 데이터 및 test 학습을 금지하고,
허용 라이선스로 대회 시작 전 공개된 사전학습 가중치만 허용한다.

남은 가장 직접적인 가설은 실제076 전체를 내부 fold마다 교사로 재학습하여
그 잔차를 학습하고, 내부 계수 검증을 여러 역할 분할로 확장하는 것이다.
현재 고정 보정은073의 비021 분기를 대상으로 학습했다. 다만 현재의 수백 개
라벨 재사용으로는 새로운 독립 증거를 만들 수 없으며, 두 새 학습법의 실패와
계수 민감도를 고려하면 대규모 모델 탐색의 당장 기대 이득은 낮다.
'''
    (s.HERE/'REPORT.md').write_text(body)
    (s.HERE/'RUN.md').write_text('''# 재현 명령

프로젝트 루트에서 실행한다. 출력은 이 실험 폴더 안에 저장한다.

## 권장 제출 파일 생성 또는 동일성 확인

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/make_submission.py
```

표준 라이브러리만 사용한다. 기존076·078·080 입력 해시,1000개ID와
비중복 변경을 확인하고 `candidate-076-plus-frozen-cnn.csv`를 생성한다.
기존 출력이 다른 내용이면 덮어쓰지 않고 중단한다. 자동 업로드하지 않는다.

## 구성 요소 + 원본 테스트 이미지로 추론 재현

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/infer_frozen.py
```

보존된073 및 Tab/Ridge/GMM/grain-set 예측 캐시로074·076을 재현하고,
보존된 full500 CNN 가중치2개와 테스트 이미지로 다시 예측한다.
`verification/inferred-from-models.csv`와 권장 CSV의 일치를 검증한다.
원래500 학습 라벨·픽셀을 읽지 않으며 새로운 모델을 적합하지 않는다.
의존 파일·해시는 `frozen-inference-verification.json`에 기록된다.

## 선택한 CNN 학습 재현

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/retrain_selected.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/infer_frozen.py --models experiments/04-residual-cnn/20261016-candidates-078-080/retrained-selected
```

기존의 검증된3fold full073 교사 캐시를 사용하여500행 전체로2seed를
40epoch씩 학습한다. 기존 파일은 읽기 전용이며 새 모델은
`retrained-selected/`에 저장한다. 원본 가중치와 bitwise 동일성을 검사한다.
교사 학습 원본 코드는 `../rmse-research-20261015/round4-full073/full_branch.py`
및 `final_teachers.py`에 보존되어 있다. 이것은 추가 성능 실험이 아니라
선택한 모델의 재현이며 전체학습 RMSE를 검증 점수로 쓰지 않는다.

## 이번 새 가설 및 검증 재실행

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/study.py audit
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/study.py train
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/evaluate.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/verify.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/clipping_check.py
experiments/tabpfn-v2/.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/calibration_stress.py extend
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/calibration_stress.py evaluate
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/finalize.py
```

학습은CPU2스레드·프로세스1개를 사용한다. 완료된 모델은 meta/seed를
검증하여 재사용한다. 새로 재학습하려면 이 폴더를 새 경로에 복제하여
실험 출력만 비운 독립 사본을 사용하며, 공유 원본 캐시를 삭제하지 않는다.
소프트웨어 버전과 원본 데이터 SHA256은 `data-integrity.json`에 있다.
''')
    s.save(_repo_io('execution-notes.json'),dict(inference_check_repair='Initial np.array_equal rejected one matchingNaN. All finite entries andNaN masks were identical; equal_nan=True repairs only the verification assertion. Original log retained.',
                                      full_inference_recheck_exact=True,new_CNN_training_fits=20,additional_verification_refits=2,
                                      selected_full500_reproduction_fits=2,extra_full073_teacher_bundles=1))
    delivery=[s.HERE/'candidate-076-plus-frozen-cnn.csv',s.HERE/'REPORT.md',s.HERE/'RUN.md',s.HERE/'decision.json']
    delivery+=list(s.HERE.glob('*.py'))+list(s.HERE.glob('*results.json'))
    delivery+=[s.HERE/k for k in ('verification.json','selected-training-reproduction.json','frozen-inference-verification.json',
                                'calibration-stress.json','clipping-check.json','data-integrity.json')]
    s.save(_repo_io('delivery-hashes.json'),{str(p.relative_to(s.ROOT)):s.r.digest(p) for p in delivery})
    print(json.dumps(decision,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
