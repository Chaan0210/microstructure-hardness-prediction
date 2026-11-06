"""One-off report for this frozen record, not a generic fresh-training reporter."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,datetime,hashlib
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
def read(name):return json.loads((_repo_path(HERE, name)).read_text())
def link(name,label=None):return f'[{label or name}]({_repo_path(HERE, name)})'
def interval(x):return f'[{x[0]:+.5f}, {x[1]:+.5f}]'
def main():
    result=read('results.json');quality=read('quality-policy/results.json');decision=read('delivery-decision.json');diagnostic=read('diagnostics.json');pres=read('preservation.json');repro=read('reproduction/082-inference-verification.json');fix=read('deterministic-replay/poison/verification.json');gradient=read('gradient-repeatability.json');initial=read('starting-state.json');r=result['results']['topology'];high=r['high_noise'];weak=result['weak_structure'];ts=read('target-summary.json')['source_groups']['high_eligible'];numeric=read('final-verification.json');source=read('encoder-verification.json')
    assert not decision['new_candidate_recommended'] and decision['selected_protocol'] is None
    assert r['candidate_rmse']==9.243578769211139 and high['candidate_rmse']==11.473955290831457, 'This narrative belongs to the frozen evaluated record; interpret a fresh training run separately.'
    assert [row['arms']['topology']['beta'] for row in result['selection']]==[0.,.5,0.,0.,0.]
    assert all(row['arms']['topology']['route'] is None for row in quality['selection'])
    assert pres['all_unchanged'] and repro['CSV_byte_identical'] and fix['comparison']['parameter_states_exact']
    for file in ['graph-verification.json','encoder-verification.json','pre-score-verification.json','final-verification.json','quality-policy/preflight.json','quality-policy/verification.json']:assert read(file)['all_passed'],file
    stage_files=['starting-state.json','target-preflight.json','target-audit.json','graph-verification.json','resource-probe.json','resource-probe-b32.json','encoders-complete.json','encoder-verification.json','readout-resource-probe.json','readouts-complete.json','pre-score-verification.json','quality-policy/results.json','quality-policy/verification.json','training-repeatability.json','warmup-replay/original/verification.json','warmup-replay/poison/verification.json','gradient-repeatability.json','deterministic-replay/original/verification.json','deterministic-replay/poison/verification.json','final-verification.json','measurement-shift.json','preservation.json']
    stages={name:read(name)['seconds'] for name in stage_files};total=sum(stages.values());assert total<7200
    # Sum only individual stage timers, never the overlapping total_compute fields.
    logs={p.name:max(0.,p.stat().st_mtime-p.stat().st_birthtime) for p in HERE.glob('*.log') if hasattr(p.stat(),'st_birthtime')};account=dict(individual_recorded_stage_seconds=stages,sum_recorded_stage_seconds=total,log_wall_spans_seconds=logs,sum_log_wall_spans_seconds=sum(logs.values()),budget_seconds=7200,meaning='Individual measured stage wall times, not hardware CPU/GPU utilization. Includes verification/debugging, excludes some imports/process teardown and document/tool overhead. Log spans independently include most process startup; they are not added to stage sums. No overlapping subtotal counted twice.')
    with (_repo_path(HERE, 'compute-accounting.json')).open('x') as f:json.dump(account,f,indent=2)
    labels={'topology':'경계+국소 연결+maximin','local':'경계+국소 연결 대조군','hardness':'일반 경도 지도학습+D4','random':'무학습 초기 특징','raw':'원본 보조 마스크 요약'}
    table=[f"| 082 기준 | {result['baseline082_overall']:.5f} | {result['baseline082_high']:.5f} | — |"]
    for arm,name in labels.items():
        z=result['results'][arm];table.append(f"| {name} | {z['candidate_rmse']:.5f} | {z['high_noise']['candidate_rmse']:.5f} | {z['gain']:+.5f} |")
    qtable=[]
    for arm,name in labels.items():
        z=quality['results'][arm];qtable.append(f"| {name} | {z['candidate_rmse']:.5f} | {z['high_noise']['candidate_rmse']:.5f} | {z['gain']:+.5f} |")
    errors='\n'.join(f"| {z['ID']} | {z['y']:.3f} | {z['baseline082']:.3f} | {z['candidate']:.3f} | {z['SSE_gain']:+.2f} |" for z in diagnostic['worst'][:4]);better='\n'.join(f"| {z['ID']} | {z['y']:.3f} | {z['baseline082']:.3f} | {z['candidate']:.3f} | {z['SSE_gain']:+.2f} |" for z in diagnostic['best'][:3]);connection_reduction=1-weak['topology']['high_BCE_boundary_local_connection'][2]/weak['local']['high_BCE_boundary_local_connection'][2];repair=gradient['cases'][0]['modes'];family=read('decision.json');qdecision=read('quality-policy/decision.json');ms=read('measurement-shift.json');msentropy=next(z for z in ms['results']['topology'] if z['feature']=='affinity_score_entropy')
    text=f'''# 결정립 경계·연결 학습: 작은 OOF 개선, 제출 권고 보류

**연결 관계를 직접 배우는 모델을 구현하고 중첩 검증했지만, 제출할 가치가 있다고 권할 만큼 안정적인 경도 개선은 얻지 못했다. 082를 유지하고076을 롤백으로 보존한다.** 평균 개선은 전체0.01316·고잡음0.03417RMSE다. 그러나 개선한 분할은1/5이고 구간에0이 포함된다. 도움이 된10개를 제외하면 악화된다. 경계 복원 성과와 경도 일반화 성과를 구별해야 한다.

로컬 기록상 Public 최고082는8.9343098075,076은8.9646534604다([기존082 제출 기록]({_repo_path(ROOT, 'reports/candidate082-public-submission.json')})). 이번에 새 제출·공식 최고 변경은 없다. 아래는 Public 점수가 아닌, 기존400개 원본 중 고잡음124개를 포함하는 동일한 조건부 OOF 평가다. 기존082의 CNN 기여를 없다고 해석하지 않았고 그 전체 파이프라인을 유지했다.

| 고정된 전 구간 정책 | 전체400 RMSE | 고잡음124 RMSE | 082 대비 개선량 |
|---|---:|---:|---:|
{chr(10).join(table)}

개선량은082 RMSE−새 RMSE이며 양수가 개선이다. 연결 모델의 전체95% 구간은 **{interval(r['conditional_CI95'])}**, 고잡음95% 구간은 **{interval(high['conditional_CI95'])}**다. 바깥 fold0..4 개선량은 **0,+0.06718,0,0,0**이고 β는 **0,0.5,0,0,0**이었다. 개별 초기화 시드는 **{r['seeds'][0]['gain']:+.5f} / {r['seeds'][1]['gain']:+.5f}**로, 첫 시드는 사실상 변화가 없었다. 가장 많이 도운5개를 제외하면 **{r['removed_largest_gains']['5']['gain']:+.5f}**,10개를 제외하면 **{r['removed_largest_gains']['10']['gain']:+.5f}**다. 상위10개가 전체 순 제곱오차 감소의 **{100*r['removed_largest_gains']['10']['share_net_SSE_gain']:.1f}%**를 차지하므로, 나머지에서는 손해가 났다.

연결 모델은 무학습 대조보다 나았지만, 국소 연결 학습·일반 경도 학습·원본 보조 마스크 요약 대비 우위를 입증하지 못했다. 특히 원본 마스크 요약 대비 개선량 차이는 **{result['topology_vs_controls']['raw']['gain']:+.5f}**,95% 구간 **{interval(result['topology_vs_controls']['raw']['conditional_CI95'])}**다. 일반 경도 학습의0.02344 개선도 구간이0을 포함하고 상위10개 제외 시 악화돼 제출 근거로 삼지 않았다. 대조군 중 가장 좋아 보이는 것을 사후 최종 후보로 바꾸지 않았다.

이전 실패와 중복되는지 코드부터 확인했다({link('AUDIT.md')}). 기존 U-Net은 Cellpose 기반4개 화소 클래스를 예측했고, u1b는 경계 약화 학습 후3개 상분율 열을 교체했으나 고잡음12.31→12.62로 실패했다. K6 그래프는 이미 추출된 결정립 객체 특징에 한 번의 이웃 전달을 적용했고, H0/Betti/C2·지도 PCA·U-Net/DINO 토큰 방식은 주로 고정 표현을 읽었다. 직전 실험은 가린 영역의 평균 명암·대비·기울기를 복원했다. 이번에는 원시 영상에서 경계 지도와 인접 연결을 학습하고, **잘못된 연결 하나 때문에 서로 다른 영역이 합쳐지거나 같은 영역이 끊기는 오류**를 직접 손실로 삼았다. 서로 다른 영상의 경도가 비슷하다고 표현을 묶는 경도 이웃 손실도 쓰지 않았다.

| 사전 가설 | 구별되는 검정 | 실제 관측 |
|---|---|---|
| 전역 연결 손실이 국소 경계 분류보다 유용한 구조를 배운다 | 같은 모델·초기값·원본·D4·40epoch의 국소 학습과 비교 | 보류된 약한 연결 목표는 개선, 경도에서 독자적 우위는 미확인 |
| 마스크 전체의 신뢰성이 낮은 경우 추가 보정이 불안정하다 | 고정된 영상 근거 기준과 B만의 LOO 선택 | 연결 모델은5개 분할 모두 보정 없음 선택 |
| 풀링으로 연결 정보가 사라지는 것이 문제다 |28개 명시적 연결 통계+16개 잠재PC, 무학습·원본 마스크 대조 | 결합 표현의 이득은 불안정; 잠재표현/연결 통계 각각의 효과는 별도 분리하지 않음 |

모델은558372개 파라미터의 작은 화소 피라미드와16×16 위치의2개 전역 attention 층(폭96),128×128 경계 디코더,32×32 인접 연결 그래프다. 입력은256×256 원본을2×2 평균해128부터 처리하므로, 원래 화소 정보를 모두 보존하는 구조는 아니다. 그래프 간격8px도 아주 작은/가는 결정립을 표현하는 데 한계가 있다. 경계·국소 연결 BCE에64개 샘플 쌍의 maximin BCE를 더했다. 연결 점수는 가능한 경로들 중 최소 연결 점수가 가장 큰 경로의 값이다. [MALIS 원 논문](https://arxiv.org/abs/0911.5372)의 개념에 따른 직접 구현이며, 원 논문의 데이터·모델이나 전체 쌍 Rand 목적의 완전한 재현은 아니다.

목표는 기존 학습 원본의 Cellpose 기본/D4 마스크에서 만들었다.0 라벨은 다른 결정립이 아니라 미측정으로 처리하고, 사용 가능한 뷰의75% 이상이 동의하는 내부 화소 쌍만 감독했다. 경계의 영상 내 근거에 위치별 가중치를 줬지만, 이미지·클래스별 정규화 때문에 전체적으로 잘못된 한 이미지의 영향까지 제거하지는 못한다. 상·기공을 하나의 인스턴스로 나눈 마스크가 실제 결정립과 다를 수 있고, 같은 보조 모델의 여러 뷰는 같은 오류에 동의할 수 있다. **보조 마스크는 결정립 정답이 아니다.** 균형 BCE의 sigmoid는 보정된 물리 확률이 아니므로, 점수 엔트로피를 검증된 사후 불확실성으로 해석하지 않는다.

약한 목표를 맞추는 능력은 실제로 달라졌다. 고잡음 보류 원본의 연결 BCE는 국소 학습 **{weak['local']['high_BCE_boundary_local_connection'][2]:.5f} → maximin {weak['topology']['high_BCE_boundary_local_connection'][2]:.5f}**, **{100*connection_reduction:.2f}% 감소**다. 두 방법 차이95% 구간은 **[−0.12148,−0.10145]**다. 반면 경계 BCE는0.35241→0.35915, 인접 연결 BCE는0.21874→0.23122로 조금 나빠졌다. 목표 변경이 전역 연결을 개선하는 대신 국소 분류와 절충한 것이다. 이것은 실제 결정립을 더 정확하게 복원했다는 검증이 아니다.

고잡음124개에서 확실한 국소 연결 목표의 중앙값 비율은 **{100*ts['edge_known_fraction']['median']:.2f}%**다. 마스크 경계의 원본 영상 근거AUC 중앙값은 **{ts['boundary_evidence_AUC']['median']:.3f}**,10백분위는 **{ts['boundary_evidence_AUC']['q10']:.3f}**다. 보류 뷰와의 높은 합의는 이 확실한 부분에 조건부인 값이다. TRAIN_000025/158 같은 예시에서 모델은 약한 마스크의 미측정 영역에도 촘촘한 경계망을 채운다. 이를 실제 경계의 복원으로 확인할 독립 주석은 없다. 아래 지도는 해당 원본을 제외해 학습한 모델의 예측이며, 처음3개는 라벨을 보지 않고 고정한 예시, 나머지는 사후 큰 손해 사례다. 연결 성분 그림은 저장float16 지도에서 그렸으므로 임계 근처의 연결은 실제float32 판독 특징과 다를 수 있다.

![보류 원본의 경계·연결 예측]({_repo_path(HERE, 'heldout-boundaries.png')})

신뢰도 선택 정책은 경도 결과를 보기 **전에**, 이미지별 가중치 한계를 발견하고 별도로 고정했다({link('QUALITY-PLAN.md')}, {link('quality-code-lock.json')}). 영상 근거AUC가 우연 수준0.5를 **초과**할 때만 지원되는 원본으로 정의했다. 임계값 검색은 하지 않았다. 고잡음124개 중94개가 해당했다. 각 B 원본을 제외해 기존082 계수와 새 β를 둘 다 다시 적합하고, 전 구간·지원 원본만·보정 없음 중 LOO 이득과1SE 규칙으로 선택했다. V 라벨은 선택이나 계수에 사용하지 않았다. 같은 절차를 모든 대조군과 두 시드에 적용했다.

| 내부 신뢰도 선택 정책 | 전체 RMSE | 고잡음 RMSE | 082 대비 개선량 |
|---|---:|---:|---:|
{chr(10).join(qtable)}

연결 모델이 유일하게 활성화됐던 원래 fold1에서도 내부 LOO 평균 제곱오차 이득은5.67, 표준오차6.04였다. 지원 원본으로 제한하면3.05 대5.61로 더 약해졌다. 따라서 추가 정책은5개 분할 모두 보정 없음을 선택했고082와 정확히 같았다. 원래 정책의 기존 기준, 두 정책에 대한 더 엄격한97.5% 구간과 고정된 최종 우선순위 모두 통과하지 못했다. 검증 후에 문턱이나 혼합 계수를 바꾸지 않았다.

새 특징을 넣은 단독 물리 Tab 판독기는 전체 **{r['paired_tabular_components']['physical_overall']:.5f}→{r['paired_tabular_components']['augmented_overall']:.5f}**, 고잡음 **{r['paired_tabular_components']['physical_high']:.5f}→{r['paired_tabular_components']['augmented_high']:.5f}**로 악화됐다. 따라서 강한082 위의 작은 보정 이득을 새로운 물리 측정의 전반적 정확도 향상으로 볼 수 없다. 단순 퍼짐만의 효과는 아니다. 고잡음 보정의 선형 위치·퍼짐 성분만 적용한 사후 진단 이득은 **{high['affine_diagnostic']['affine_only_gain']:+.5f}**, 이를 제외한 성분은 **{high['affine_diagnostic']['orthogonal_only_gain']:+.5f}**다. 두 성분은 새 후보나 추가 적합이 아니다.

새 분기의 학습 크기를 A213으로 유지해 B와 V를 예측한 진단도 **{r['fixed_A_branch']['gain']:+.5f}**,95% 구간 **{interval(r['fixed_A_branch']['conditional_CI95'])}**였다. T320 재학습 크기 변화만으로 이득 부족을 설명하지 못한다. 또한 같은 원본을 동일320개 학습 크기의 모델에 포함/제외했을 때의 구조 통계를 비교했다. 연결 점수 엔트로피의 고잡음 평균 차이는 **{msentropy['high_paired_difference']:+.6f}**,95% 구간 **{interval(msentropy['conditional_CI95'])}**로, 큰 평균 신뢰도 이동은 확인되지 않았다. 서로 다른 겹친 분할과 시드의 영향도 포함한 진단이며 한 원인의 부재를 증명하지는 않는다. PCA16은 학습된 잠재표현 분산 평균97.49%를 유지했지만, 이것이 경도 관련 정보의 보존을 보증하지도 않는다.

| 악화 사례 | 실제 경도 | 082 | 연결 모델 추가 | 제곱오차 감소량 |
|---|---:|---:|---:|---:|
{errors}

| 가장 도움된 사례 | 실제 경도 | 082 | 연결 모델 추가 | 제곱오차 감소량 |
|---|---:|---:|---:|---:|
{better}

수치 재현성에서도 실제 구현 문제를 찾았다. 저장된60개 학습 모델·20개 무학습 모델의 대표 배치 특징 재추론은 전부 최대 차이0이었다. 하지만 같은 시드로 처음부터 다시 학습한 원래 maximin 모델은 달랐고, 일반 경도 모델은 정확히 재현됐다. 무학습 특징 계산까지 시작 순서를 맞추고 원래 영상을 그대로 써도 maximin 재학습 차이는 남았다. 따라서 처음의 차이를 보류 영상 변경이나 시작 순서 탓으로 단정하지 않았다.

실제 첫31개 학습 원본의 고정 logits·쌍·목표로 역전파만40번 반복했을 때, 중복 인덱스 gather 경사는 **40개의 서로 다른 해시**를 냈다. 최대 차이는 **{repair['gather']['maximum_repeat_gradient_difference']:.3g}**였다. 샘플 쌍의32.7%가 이미 다른 쌍이 선택한 연결을 재사용했다. 이런 작은 오차가 불연속적인 maximin 경로 선택을 거치면 재학습 경로 차이로 커질 수 있다. 실제 전체 재학습에서 가중치 최대 차이는0.0144~0.0160, 구조 서명 최대 차이는0.504까지 관측했다. 이 값은 **HV 예측 오차가 아니라 가중치·특징 단위의 차이**다.

같은 수학적 BCE를 연결별 양·음 가중치로 먼저 합산하는 {link('deterministic_loss.py','수정 손실')}을 별도로 구현했다. 실제 배치에서는 독립float64 기준과 손실 오차7.04×10⁻⁸ 이하·경사 오차2.29×10⁻¹⁰ 이하였고,40회 반복 경사는 모두 같았다. 별도 수치 스트레스 사례의 경사 기준 차이는2.29×10⁻⁹ 이하로, 이 사례에서도 반복 경사는 같았다. 수정판을 같은 A213·40epoch·시드로 별도 프로세스에서 두 번 학습하되 두 번째 실행의 보류 warm-up 영상187개를 바꿨을 때, **가중치·학습/보류 특징 차이가 모두0**이었다({link('deterministic-replay/poison/verification.json')}). 이는 중복 경사 집계가 재현성 문제의 한 원인이었음을 뒷받침하며 이 사례의 수정 효과를 확인한다.

**수정판으로 전체60모델 중첩 검증을 다시 수행하지 않았다.** 따라서 위 RMSE는 원래 고정된 모델들의 결과이고, 수정판의 성능으로 옮겨 쓰지 않는다. 원래 모델·예측·실패 재현 기록도 그대로 남겼다. 수정은 수치 재현성 개선이며 경도 RMSE 개선을 보장하지 않는다. 이 구분 때문에 새 제출 후보나 공식 최고 모델을 만들지 않았다.

검증은 A213/B107 안쪽 선택과 T320/V80 바깥 평가,5개 분할·2개 시드로 수행했다. 원본과 모든 D4 뷰는 함께 분할했다. 정규화·새 인코더·잠재 표준화/PCA·회귀·β·추가 라우팅은 각 적절한 학습 부분 안에서만 적합했다. 물리100열과 물리100+구조28+PC16(총144열)은 같은 원본 수와 원래 경도 목표로 비교했다. 단독 이미지 모델로082를 교체하지 않고 두 판독기의 차이만 high-noise β로 추가했다. 무학습 대조와 일반 경도+D4 대조도 같은 backbone·초기값·원본 방문 수·40epoch(무학습 제외)·판독 절차를 사용했다. 목적별 벽시계/FLOPs가 같다는 뜻은 아니다.

모든500 라벨은 과거 연구에서 노출됐고, 이번400도 반복 사용된 개발 라벨이다. 기존082의 보조 표현·측정기·과거 선택 과정 전체를 이번에 처음부터 중첩 재학습하지는 않았다. 새 단계는 이 동결된 역사적 표현에 조건부다. 원본 단위10000회 짝 bootstrap도 고정 적합 결과에 조건부이며, 겹친 학습 집합·전체 재학습 변동·누적 선택 편향을 모두 반영하지 않는다. 재분할이나 두 시드를 독립 확인으로 부르지 않고, Public/Private 예상 점수를 만들지 않았다.

계산은 로컬 MPS 한 작업씩, CPU2 threads, 기존 환경으로 수행했다. 본 인코더/특징79.30분·본 판독14.44분, resource probe·수치 재현·보존 감사를 포함한 기록된 단계 합계는 **{total/60:.2f}분**이다({link('compute-accounting.json')}). 일부 프로세스 시작·문서/도식 처리 시간은 별도이며 로그 시간도 함께 남겼다. 회귀에서 기록한 RSS+MPS driver 최대는4.801GiB였고5GiB 상한을 통과했다. 최초 배치16 시간 예상은 실패했고 사전 허용된 배치32로 조정했다. 배치32의 처음 false 기록은 문서보다10분 더 엄격했던 코드 예산 판정을 문서의120분에 맞춰 바로잡은 것으로, 모델·epoch·대조군·총예산은 늘리지 않았다. 이 과정은 {link('RESOURCE-NOTE.md')}와 {link('resource-accounting.json')}에 남겼다. 외부 학습 이미지·새 설치·유료 자원·test 학습·pseudo-labeling·제출·Git 기록 변경은 없다.

보호된 **{pres['files']:,}개 파일**, Git HEAD 및 기존 tracked diff가 모두 그대로다({link('preservation.json')}). HEAD는 `{initial['head'].strip()}`이다.076은 동결 성분에서 다시 조립했고082는 저장 CNN 가중치와318개 기존 추론 영상으로1000행을 다시 조립했다. 두 CSV는 바이트 단위로 같고082 최대 수치 차이는0이다({link('reproduction/082-inference-verification.json')}). 이 보존 검사만 기존 test 영상을 읽었으며 새 모델 적합·선택에는 쓰지 않았다. 이는 저장 산출물 추론/조립 재현이고 전체 역사적 훈련의 재현은 아니다. 새 Tab 문맥의 실제 재현 오차는 최대1.12×10⁻⁶HV, CPU/MPS 비교 최대5.82×10⁻⁵HV였다.

남은 병목은 **보조 마스크의 연결 관계를 잘 맞추는 것과,082가 놓친 실제 경도 정보를 얻는 것이 일치하지 않는 점**이다. 미측정 영역의 그럴듯한 경계망을 정확한 구조라고 검증할 근거가 부족하고, 현재 표현·판독기에서는 작은 이득이 몇 사례에 집중됐다. 수치 불안정 원인 하나는 고쳤지만, 이것을 성능 개선으로 대체해서 주장할 수 없다. 이 결과로 원본 영상의 정보가 불가약적이라고 결론 내리지 않는다. 현재 검증된 선택은082 유지다.

정확한 명령·시드·환경·입력 경로는 {link('RUN.md')}, 원본별 예측은 {link('per-image.csv')}와 {link('quality-policy/per-image.csv')}, 전체 결과는 {link('results.json')}·{link('quality-policy/results.json')}, 최종 사전 기준 판정은 {link('delivery-decision.json')}에 있다. 계산·수정·재현 파일을 모두 현재 디렉터리에 보존했고, 모든 작성이 끝나면 `delivery-hashes.json`으로 닫는다.
'''
    with (_repo_path(HERE, 'REPORT.md')).open('x') as f:f.write(text)
    completion=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,new_candidate_recommended=False,incumbent='082',rollback='076',official_promotion=False,submission_created=False,submitted=False,main_overall_gain=r['gain'],main_high_gain=high['gain'],main_CI95=r['conditional_CI95'],quality_topology_gain=quality['results']['topology']['gain'],protected_files_unchanged=pres['files'],incumbent_CSVs_byte_identical=True,original_saved_weight_features_exact=True,original_topology_training_bit_repeatable=False,aggregated_loss_gradient_repeatable=True,aggregated_loss_full_replay_exact=True,aggregated_loss_nested_accuracy_validated=False,recorded_stage_seconds=total,report_sha256=hashlib.sha256(text.encode()).hexdigest(),limits='Previously exposed development data and historical representations; no independent confirmation. Numerical repair not the model used for the nested scores.')
    with (_repo_path(HERE, 'completion.json')).open('x') as f:json.dump(completion,f,indent=2)
    (_repo_path(HERE, 'STATE.md')).write_text('# Completed boundary/connectivity study\n\nAll60newencoders/270mainTabcontexts and both prospectively registered selection policies are complete. No new candidate meets the frozen criteria; preserve082/076. Main topology OOF9.24358/high11.47396 vs0829.25674/high11.50813: +.01316 overall, uncertain and concentrated. Quality policy selects zero topology correction in allfive folds.\n\nSaved-weight feature inference is exact. Original topology full retraining is not bit deterministic. Repeated-index MPSgather backward was isolated; equivalent per-edge aggregation gives repeatable gradients and two complete40epochA213replays exact despite excluded warmup image changes. This prototype has NOT had full nested accuracy evaluation and does not replace original scores/models. See REPORT.md and completion.json.\n\n153547 protected files and baseline CSVs remain unchanged/exact. No active compute or submissions. Run final integrity checks and manifest.py only after every writer has exited.\n')
    print('REPORT AND COMPLETION WRITTEN',len(text),round(total/60,2),flush=True)
if __name__=='__main__':main()
