"""Summarize completed experiments without presenting the interrupted pilot as done."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv
import json
from pathlib import Path
import probe as p
H=p.HERE


def main():
    p.protect()
    def read(name):return json.loads((H/name).read_text())
    base=read('baseline076.json');probes=read('probe-results.json');confidence=read('confidence-results.json')
    stream=read('stream-results-0.json');repeat=read('control-repeat-results.json');ensemble=read('ensemble-results.json')
    outer=read('calibration-order-outer.json');stress=read('calibration-order-stress.json')
    collapsed=read('collapsed-control-verification.json');normalization=read('lowmem-reproduction.json')
    assert not probes['advance'] and not confidence['advance'] and not stream['advance']
    assert not repeat['adoption_gate_passed'] and not ensemble['adoption_gate_passed'] and not stress['adopt']
    assert collapsed['all_passed'] and normalization['complete']
    teacher_check=read('teacher076-verification.json')
    assert teacher_check['all_checked_passed'] and teacher_check['n_complete']==3
    assert not (H/'teachers076-complete-01.json').exists(),'076 target pilot progressed; update this historical report before rerunning'
    assert read('stream-seed0-complete.json')['complete'] and read('stream-seed1-complete.json')['complete']
    assert read('calibration-role-prediction-verification.json')['all_passed']
    fits=list((H/'stream').glob('*/*/training.json'));assert len(fits)==30
    prior=json.loads((p.PREV/'decision.json').read_text());candidate=_repo_path(p.ROOT, prior['CSV'])
    assert p.r.digest(candidate)==prior['CSV_sha256']
    best_name='duplicate_post_original';best=outer['results'][best_name];check=stress['summary']['results'][best_name]
    rows=[('Current076+frozenCNN',probes['results']['old4']['candidate']['pooled'],'reference'),
          ('Eight-view mean',probes['results']['old8']['candidate']['pooled'],'interval crosses0'),
          ('Refresh076 context',probes['results']['context4']['candidate']['pooled'],'worse'),
          ('Seed-disagreement shrinkage',confidence['scores']['confidence']['rmse'],'worse'),
          ('Legacy matched seed0',stream['scores']['legacy_matched']['rmse'],'one-seed control'),
          ('Raw+Gaussian seed0',stream['scores']['gaussian_matched']['rmse'],'fails duplicate-channel control'),
          ('Raw+raw seed0',stream['scores']['duplicate_matched']['rmse'],'repeat performed'),
          ('Raw+raw two-seed mean',repeat['scores']['duplicate_mean2']['rmse'],'interval crosses0'),
          ('Fixed family mean',ensemble['scores']['fixed_family_mean'],'interval crosses0'),
          ('Raw+raw: calibrate seeds then mean',best['rmse'],'fails frozen-model role stress test')]
    with (H/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','strict076_original400_OOF_RMSE','decision']);w.writerows(rows)
    with (H/'fold-metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','fold','rmse'])
        for name,values in [('strict076',base['strict076']['folds']),('current',repeat['scores']['current']['folds']),
                            ('duplicate_mean2',repeat['scores']['duplicate_mean2']['folds']),
                            ('duplicate_post_original',best['folds'])]:
            w.writerows((name,k,value) for k,value in enumerate(values))
    decision=dict(status='No additional production promotion from completed studies',
        completed_new_CNN_fits=30,teacher_pilot_complete=False,completed_teacher_bundles=3,
        completed_unique_teacher_Tab_predictions=252,teacher_last_exit_code=137,teacher_exit_cause_unconfirmed=True,
        current_strict076_OOF=probes['results']['old4']['candidate']['pooled'],
        best_fixed_two_seed_exploratory_OOF=best['rmse'],best_role_stress_gain=check['gain'],
        unchanged_recommended_CSV=prior['CSV'],unchanged_CSV_sha256=prior['CSV_sha256'],
        new_full500_variant_trained=False,new_submission_created=False,uploaded=False,
        protected_files=len(read('protected.json')),all_protected_hashes_unchanged=True,
        conclusion='Point-score gains were not stable across the calibration-role check; do not select the best single seed or promote a fragile OOF winner.',
        unresolved='Actual076 residual-target alignment remains untested because the teacher pilot is incomplete; retained caches support later resumption when resources permit.')
    p.save(_repo_io('decision.json'),decision)
    table='\n'.join(f'| {name} | {score:.6f} | {note} |' for name,score,note in rows)
    text=f'''# 추가 개선 연구 — 2026-10-16

**이번에 교체를 권할 만큼 안정적인 추가 개선은 확인하지 못했다.**
새 CNN30개를 실제 학습하고 추론·신뢰도·보정 순서까지 비교했다.
고정2seed 계열에서9.184772까지 낮은 OOF를 얻었지만 추가 역할 검증에서
개선이 유지되지 않았다. 따라서 기존076+잔차CNN 후보를 보존했다.

## 비교 기준

기존061·073은 수정하지 않았다. 동일original400/5fold의 strict073 RMSE는
9.547245이다. 이번에는076도 source 범위를 제한하여 재구성했다.
H9 Tab12개 bank의 기존 OOF는 ID·fold·정답·피처 해시와 생성 코드를 확인해
재사용했고, 고잡음 선형 모델은 실제 학습 source의 열화 사본만 허용한
내부5fold GroupKFold로 규제 강도를 골랐다. G021·GMM·grain-set 등은
같은 strict073 구성 요소를 유지했다.

그 결과 strict076 기준점은 **{base['strict076']['rmse']:.6f}**, 기존 잔차CNN을
더한 현재 후보는 **{probes['results']['old4']['candidate']['pooled']:.6f}**이다.
이전 기록의0769.508303/076+CNN9.268706과의 차이는 검증 재구성에 따른 것이다.
이를 새 운영 모델의 개선으로 세지 않는다. 아래 모든 전체 OOF는 같은400행이다.

| 실험 | 전체 OOF RMSE | 판단 |
|---|---:|---|
{table}

단일seed 실험은 같은seed의 기존 CNN과 비교했다. 새2seed 실험은 기존2seed와
비교했다. 한 seed에서9.172398도 나왔지만, 외부 점수를 본 뒤 가장 좋은seed를
선택하지 않았다. 사전에 고정한 동일 평균과 내부 검증을 유지했다.

이번 작은CNN들의 학습 목표는 현재 보정과 동일한 `y−T073`이다.
여기서 T073은073에서 G021을 제외한 나머지 분기를0.84로 정규화한 예측이다.
전체 앙상블 P073에 대한 잔차와 구분해야 한다. 보정 계수는 전체P073의
내부 B 예측에 맞추고, 실제 외부 비교는 strict076 위에서 했다.

## 실제로 확인한 부분

**입력 표현:** 기존 원본256 CNN에 raw+Gaussian(sigma1,5x5,reflect) 두 채널을
주었다. 동일 파라미터 수69,572개·초기값·학습 순서의 raw+raw 대조군을 함께
학습했다. 기존40epoch/AdamW/D4/경도 보조 손실.2는 유지했다. 평활 입력은
같은seed 기존 CNN보다0.007443만 개선했고 대조군보다0.009405 나빴다.
평활 자체의 효과로 해석할 수 없어 이 방법의 추가seed는 학습하지 않았다.

**예상 밖의 대조군 이득:** raw+raw만 두 번째seed로 반복했다. 두seed 평균은
현재 후보보다0.019633 좋아졌지만95% 조건부 구간은[-0.016359,0.056729]였다.
원본 채널을 두 번 넣는 첫 합성곱은 W0+W1로 접으면 기존 단일채널 구조와
같은 함수가 된다. 실제10개 모델의 접힌 예측과 비교한 최대 차이는
{max(q['collapsed_max_error'] for q in collapsed['records']):.3g}였다.
따라서 새로운 영상 정보가 생긴 것이 아니라 초기값·학습 과정 변화의
가능성이다. 이 둘 중 어느 것이 원인인지는 분리하지 못했다.

**앙상블:** 기존2seed와 새 대조군2seed의 고정50:50 평균은9.245211이었다.
잔차 상관은{ensemble['residual_correlation']:.6f}, 개선95% 구간은
[-0.006601,0.030233]으로 불확실했다. 가중치 검색은 하지 않았다.

**보정의 순서와 선택 안정성:** 예측을 먼저 평균하고 계수를 고르는 것과,
각seed에서 내부 계수를 고른 후 평균하는 것은 결과가 달랐다. 실제 보정을
적용하는 고잡음 구간으로 LOO 손실을 맞춘 경우도 함께 비교했다.
기존 pooling grid와[0,1]범위를 유지했고 모든 계수는 B 내부 라벨만 사용했다.

가장 낮은 고정2seed 계열은 보정 후 평균하는 raw+raw 모델이었다.

| original5fold | 현재 후보 | 보정 후 평균 raw+raw |
|---|---:|---:|
{chr(10).join(f'| {k} | {a:.6f} | {b:.6f} |' for k,(a,b) in enumerate(zip(repeat['scores']['current']['folds'],best['folds'])))}

전체 이득은0.071967이지만 fold4에 집중됐다. 단일 비교의 조건부95% 구간은
[0.000100,0.153040]이고 여러 비교를 함께 고려한 구간은
{outer['simultaneous_bands'][best_name]}로0을 포함한다.
이득이 큰10개ID를 제외하면 원래5fold에서도−0.015927이었다.

추가로 A213에 모델을 고정한 anchor0·1·4에서, 학습에 쓰이지 않은187행을
107/80역할로5번씩 나눠 계수를 다시 골랐다. 1200예측행은342개 원본ID별
평균 제곱오차로 합쳤으며 독립1200표본으로 세지 않았다. 이 검증의 baseline은
기존 strict073 A-only여서076 전체 재현과는 목적이 다르다.

| 고정 모델 역할 검증 | RMSE |
|---|---:|
| 기존 보정 방식 | {stress['summary']['results']['legacy_mean_original']['rmse']:.6f} |
| OOF에서 가장 좋아 보였던 보정 후 평균 | {check['rmse']:.6f} |

이득은 **{check['gain']:.6f}**, 즉 악화였다. anchor0·4에서도 악화했고
확대 채택 기준을 통과한 방법은 없었다. 이는 해당 설정의 안정성이 부족하다는
증거이며, 더 큰 학습 데이터에서의 성능이나 Private 악화를 입증하는 것은 아니다.
학습 규모·표본 구성·보정 선택의 효과는 완전히 분리되지 않았다.

## 실행·검증·보존

- 새CNN30개(2표현×1seed×5fold×2단계20개, 대조군 추가seed10개)를 실제 학습했다.
- 모든 완료 모델은 저장 후 재추론을 검증했다. 보정 계수는 외부 라벨 변조에
  영향받지 않음을 확인했다. 학습 source와 query ID의 분리를 유지했다.
- 첫 반복은1GiB 상한에서 중단했다. 임시 이미지 정규화 배열만 줄여 모든
  픽셀이 기존 계산과 같음을 확인했고, 이미 학습된 모델을 재사용했다.
  이후 최대메모리는약800MiB였다. 모델 학습 알고리즘은 바꾸지 않았다.
- 원본061·073·현재 추천CSV를 포함한{len(read('protected.json'))}개 보호 파일의 해시는 그대로다.
- 외부 데이터·새 사전학습 가중치·test 학습·Public 계수 적합·자동 제출은 없었다.

## 완료하지 못한 실험과 남은 가설

실제076의 비021 분기 T076에 대한 잔차로 CNN을 다시 학습하는 별도 파일럿은 **미완료**다.
두fold용12개 교사 중3개가 완료됐고,864개 Tab 적합·예측 중252개를 완료·보존했다.
메모리 압박이 관측되는 동안 프로세스가137로 종료됐지만 원인을 확정하지는
못했다. 이전의 자발적 중지는 이 작업의 프로세스에만 적용했고 다른 작업은
건드리지 않았다. 완료 캐시의 읽기·유한값·구성 요소를 검증해 보존했다.
새076-target CNN이 학습되었거나 개선됐다고 보고하지 않는다.

기존 교사073과 실제 결합 대상076의 차이는 아직 직접 검증할 가치가 있다.
다만 입력값만076으로 바꾸면 악화했으므로, 목표 잔차까지 일관되게 바꿔야
판단할 수 있다. 충분한 메모리가 확보되면 유지된 캐시에서 이어갈 수 있다.
그 전까지 같은 모델의 작은 변경·계수 탐색을 더 늘릴 근거는 약하다.

모든 결과는 이미 반복 탐색된400/500라벨을 사용했다. 부트스트랩은 조건부
요약이고 공유 학습 집합의 의존성이나 전체 과거 탐색을 제거하지 않는다.
현재 추천은 이전의 `076+(080-078)` 파일이며 새 제출은 생성하지 않았다.
명령과 파일 위치는 `RUN.md`, 수치는 `comparison.csv`·`fold-metrics.csv`와
각 결과JSON에 기록했다.
'''
    (H/'REPORT.md').write_text(text)
    p.save(_repo_io('delivery-hashes.json'),{str(f.relative_to(p.ROOT)):p.r.digest(f) for f in
        [H/'REPORT.md',H/'RUN.md',H/'decision.json',H/'comparison.csv',H/'fold-metrics.csv',*H.glob('*.py'),
         H/'stream-results-0.json',H/'control-repeat-results.json',H/'ensemble-results.json',
         H/'calibration-order-outer.json',H/'calibration-order-stress.json',H/'teacher-pilot-incomplete.json',
         H/'lowmem-reproduction.json',H/'collapsed-control-verification.json',
         *(H/'stream').glob('*/*/model.pt'),*(H/'stream').glob('*/*/predictions.npz'),
         *(H/'stream').glob('*/*/training.json'),*(H/'stream').glob('*/*/verification.json'),
         *(H/'teachers076').glob('*/predictions.npz'),*(H/'teachers076').glob('*/tab-*.npz')]})
    print(json.dumps(decision,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
