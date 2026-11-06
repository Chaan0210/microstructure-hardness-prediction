"""Build the final evidence-backed record after verification has completed."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,datetime,hashlib
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((_repo_path(HERE, name)).read_text())
def link(name,label=None):return f'[{label or name}]({_repo_path(HERE, name)})'
def pair(x):return f'[{x[0]:+.5f}, {x[1]:+.5f}]'
def main():
    a=read('results.json');b=read('alignment-followup/results.json');d=read('diagnostics.json');follow=read('alignment-followup/diagnostics.json');pres=read('preservation.json');repro=read('reproduction/082-inference-verification.json');fit=read('alignment-followup/fit-complete.json');start=read('starting-state.json')
    required=['mathematical-checks.json','loss-audit.json','encoder-verification.json','random-control-verification.json','pre-score-verification.json','readout-verification.json','alignment-followup/pre-score-verification.json','alignment-followup/selection-verification.json','alignment-followup/numerical-replay.json']
    for name in required:
        z=read(name)
        # Some early diagnostic records use a descriptive schema rather than all_passed.
        assert z.get('all_passed',z.get('passed',True)),name
    assert pres['all_unchanged'] and repro['CSV_byte_identical'] and repro['incumbent082_max_difference']==0
    assert not read('decision.json')['advance'] and not read('alignment-followup/decision.json')['advance']
    s=a['results']['spatial'];t=b['results']['spatial'];geom=d['geometry']['spatial']['high'];rows=[]
    rows.append(f"| 082 기준 | {a['baseline082_overall']:.5f} | {a['baseline082_high']:.5f} | — |")
    for label,key in [('공간 문맥 사전학습 + PCA','spatial'),('공간 순서 섞기 + PCA 대조군','permuted'),('동일 초기 무학습 인코더 + PCA','random')]:
        z=a['results'][key];rows.append(f"| {label} | {z['candidate_rmse']:.5f} | {z['high_noise']['candidate_rmse']:.5f} | {z['gain']:+.5f} |")
    for label,key in [('PCA/PLS/0 내부 선택: 공간 모델','spatial'),('같은 선택 절차: 순서 섞기','permuted'),('같은 선택 절차: 무학습 인코더','random')]:
        z=b['results'][key];rows.append(f"| {label} | {z['candidate_rmse']:.5f} | {z['high_noise']['candidate_rmse']:.5f} | {z['gain']:+.5f} |")
    failures='\n'.join(f"| {z['ID']} | {z['y']:.3f} | {z['baseline082']:.3f} | {z['candidate']:.3f} | {z['SSE_gain']:+.2f} |" for z in follow['worst_changes']['spatial'][:5])
    report=f'''# 고잡음 공간 문맥 학습 실험 — 2026-10-17

**제출을 권할 새 후보는 만들지 못했다. 082를 유지하고 076을 롤백으로 보존한다.** 기존 CNN과 다른 원시 화소 기반 전역 문맥 Transformer를 구현해 5개 바깥 분할·2개 시드로 평가했다. 이어 경도 관련 정보를 선택하는 PLS와 PCA, 보정 없음의 선택까지 내부 검증에 넣었으나 두 실험 모두 사전 기준을 통과하지 못했다. 새 제출 파일이나 공식 최고 모델은 만들지 않았다.

현재 로컬 기록으로 확인한 Public 최고는 082의 **8.9343098075**(2026-10-16 제출56), 롤백 076은 **8.9646534604**이다. 이는 기록된 외부 평가이며 이번 실험의 적합·하이퍼파라미터 결정에 사용하지 않았다. 근거는 [082 제출 기록]({_repo_path(_RepoPath(__file__).resolve().parents[3], 'reports/candidate082-public-submission.json')})과 보호된 기존 실행 기록이다. 아래 RMSE는 Public 점수가 아닌 같은400개 원본의 조건부 OOF 비교다.

| 방법 | 전체400 RMSE | 고잡음124 RMSE | 082 대비 개선량(양수가 개선) |
|---|---:|---:|---:|
{chr(10).join(rows)}

원래 공간 모델의 전체 개선량95% 구간은 **{pair(s['conditional_CI95'])}**, 고잡음 구간은 **{pair(s['high_noise']['conditional_CI95'])}**다. 추가 내부 선택 실험의97.5% 구간은 전체 **{pair(t['conditional_CI975'])}**, 고잡음 **{pair(t['high_noise']['conditional_CI975'])}**다. 원본 단위 짝 bootstrap10000회이며 패치·시드를 독립 표본으로 세지 않았다. 고정된 적합 결과에 조건부인 구간으로, 겹치는 학습 집합·기존 개발 적응·전체 학습 불확실성을 모두 반영하지는 않는다.97.5% 구간도 과거의 모든 실험을 보정하는 것은 아니다.

이전 실험을 먼저 점검했다. 경도 이웃 손실은 서로 다른 영상의 경도 거리를 표현 거리에 강제했고, 약한 국소 변환 VICReg는 두 뷰의 표현 일치를 강제했다. 둘 모두 082에 추가되는 신호를 입증하지 못했다. 열화 복제 파일럿도 중복 복제 대조군 대비 이득이 없었고, 잘못 섞인 라벨 대조군을 교정한 감사에서도 추가 신호가 없었다. 당시 작업은 종료되어 이번 계산과 GPU 충돌이 없었다. K6 결정립 그래프 및 기존 U-Net/DINO 토큰·공간맵 실험 역시 검토했다. 순위와 기각 기준은 {link('PLAN.md')}에 고정했다.

이번 모델은 영상256×256의 원시 화소에서 시작한다.8×8 화소 임베딩을 이웃끼리 묶어16×16 토큰을 만들고,4개 전역 attention 층(폭96,335600개 파라미터)으로 멀리 떨어진 구조를 함께 본다. 입력 토큰의 약40.6%를32×32 화소 블록 단위로 내부 마스킹하고, 해당 영역의4×4 평균 화소 격자를 예측한다. 손실은 평균 명암 MSE +4×중심화한 국소 대비 MSE +2×격자 기울기 MSE다. 학습은100epoch,AdamW,사전 고정한2개 시드,마지막5epoch 상태 평균이다. 외부 이미지·외부 인코더 가중치나 생성 미세조직을 사용하지 않았다. 기존 물리 특징 회귀를 위해 이미 설치된 offline TabPFN 가중치만 재사용했다.

이 목적은 같은 원본 안의 관측 가능한 공간 문맥을 배우며, 다른 영상의 경도가 비슷하다는 이유로 표현을 묶지 않는다. 마스킹은 모델 내부 연산이고, 회귀 특징은 가리지 않은 원본에서 추출한다. 따라서 파괴된 영상에도 원본과 같은 경도를 강제하는 학습은 하지 않는다. 회전·반사(D4) 외에 경계를 약화시키는 새 증강을 추가하지 않았다. 다만4×4 평균 목표가 얇은 경계 정보를 약화시킬 수 있으며, 관측 화소를 예측하는 일이 잡음을 제거하거나 실제 결정립 경계를 복원한다는 뜻은 아니다.

각 바깥 분할에서 V80은 새 적합에 사용하지 않고 T320만 사용했다. T 안의 A213으로 정규화·인코더·표준화·PCA 또는 PLS·회귀를 학습해 B107을 예측하고, B에서 새 보정 계수를 결정했다. V를 예측하는 모델은 T320으로 새로 적합했다. 모든 원본과 그 뷰는 같은 분할에 있다. 물리100열 회귀와 물리100열+이미지16열 회귀는 동일한 원본 수와 원래 경도 목표로 각각 적합했다. 두 예측의 차이만082에 추가하며, 물리 모델을 약한 이미지 단독 모델로 바꾸지 않았다. Tab 시드3개와 인코더 시드2개를 고정 평균했다. 고잡음 기준 noise≥12와 기존 실패 마스크는 고정했다.

공간 순서를 원본별로 섞은 동일 학습 모델과, 같은 초기 가중치를 그대로 쓰는 무학습 인코더를 대조군으로 평가했다. D4·학습 순서·마스크 난수 스트림은 두 학습군에서 맞췄다. 다만 순서 섞기 후 동일 토큰 마스크가 가리는 실제 화소 위치는 달라진다({link('CONTROL-NOTE.md')}). 이는 모든 조건을 고정한 순수한 위상 개입이라고 볼 수 없다. 무학습 대조군도 동일한 D4 특징 평균과 회귀를 쓰지만, 이번 Transformer의 경도 지도학습+일반 증강 대조군을 별도로 훈련하지는 않았다. 이전 CNN 증강 대조군은 새 아키텍처의 완전한 일치 대조군이 아니다. 따라서 모든 종류의 사전학습·일반 증강보다 우월하거나 열등하다고 일반화하지 않는다.

첫 실험이 실패한 뒤 한 번의 제한된 후속 실험을 고정했다({link('alignment-followup/PLAN.md')}). PCA가 경도에 유용한 작은 변동을 버렸을 가능성을 시험하기 위해 PLS16을 넣었다. PLS의 목표는 해당 A/T 안에서 각 원본을 제외한 Ridge(alpha100) 교사의 잔차다. 교사마다 결측 대체·표준화·회귀도 원본을 제외하고 적합했다. PLS는 이 잔차로 압축 방향만 선택하고 최종 Tab은 계속 원래 경도를 학습한다. B의 leave-one-out에서 기존082 계수와 새 계수를 모두 해당 라벨 없이 다시 적합한 뒤 PCA·PLS·보정 없음 중 선택했다. 평균 이득이 표준오차를 넘을 때만 활성화하고, 1SE 이내면 PCA를 우선했다. 바깥 라벨로 압축·계수를 선택하지 않았다.

후속 공간 정책은 fold0..4에서 **보정 없음, PCA, 보정 없음, 보정 없음, PCA**를 선택했다. 활성 계수는0.5,0.5였다. PLS만 강제 적용한 진단은 전체RMSE **{t['fixed_PLS_diagnostic']['candidate_rmse']:.5f}**로 더 나빴다. 순서 섞기 PLS의 고정 방식은 사후 진단에서 약0.037 개선됐지만, 그 군의 안쪽 선택은 모든 분할에서 보정 없음을 택했다. 이를 사후 우승 후보로 채택하지 않았다. 두 사전 고정된 주 실험 모두8개 승격 조건이 전부 거짓이다.

실제 학습된 정보와 경도 신호를 분리하면 다음과 같다. 보류된 고잡음124개에서 자기 기준 평균 채움 대비 평균 명암 오차는 **{100*geom['relative_reduction'][0]:.2f}%**, 패치 내부 대비 오차는 **{100*geom['relative_reduction'][1]:.2f}%**, 기울기 오차는 **{100*geom['relative_reduction'][2]:.2f}%** 감소했다. 따라서 문맥을 전혀 배우지 못했다는 결론은 틀리다. 반면 강한 이득은 대부분 명암 수준에 있고 세부 구조의 이득은 작다. 공간 표현의 표준화 후 유효 rank는 평균약6.03이며 PCA16은 분산약97.94%를 유지한다. 분산 보존은 경도 정보 보존의 증명이 아니다. 목표 정렬 PLS도 이득을 입증하지 못했으므로 PCA만이 병목이었다는 설명은 지지되지 않았다.

공간 모델의 바깥 fold별 개선량은 **0,−0.15120,0,0,−0.04478**이다. 개선한 분할은0/5이고, 가장 많이 도운10개를 제외하면 개선량은 **{s['removed_largest_gains']['10']['gain']:+.5f}**다. 원래 실험의 개별 시드는 각각 **{s['individual_encoder_seeds'][0]['gain']:+.5f}, {s['individual_encoder_seeds'][1]['gain']:+.5f}**, 추가 내부 선택 정책은 **{t['seeds'][0]['gain']:+.5f}, {t['seeds'][1]['gain']:+.5f}**다. 일부 좋은 원본이나 한 시드가 평균 이득을 가린 상황이 아니다.

안쪽에서 활성화된 두 분할은 바깥에서 보정 방향과082 잔차의 내적이 음수가 됐다. 같은 V에서 새 분기 학습 크기를 A213으로 유지한 진단도 개선량 **{follow['fixed_A_branch_diagnostic']['gain']:+.5f}**로 악화됐다. A→T 재학습 시 보정 방향 RMS 이동이 해당 고잡음 집합에서약1.32/2.04HV인 것은 실재하지만, 크기 이동만이 실패 원인은 아니다. 전체 보정의 선형 위치·퍼짐 성분만 적용한 사후 진단은 **{s['affine_diagnostic']['affine_only_gain']:+.5f}**, 그것을 제외한 성분은 **{s['affine_diagnostic']['orthogonal_only_gain']:+.5f}**였다. 두 성분도 새 후보로 쓰지 않았으며 이득은 단순 퍼짐 효과로 설명되지 않는다.

| 악화 원본 | 실제 경도 | 082 | 공간 후보 | 제곱오차 감소량 |
|---|---:|---:|---:|---:|
{failures}

이 결과는 현재 목표·풀링 표현이082 잔차와 안정적으로 연결되지 못했고, 적은 고잡음 내부 표본(분할별24~41개)에서의 보정 방향 선택이 이식되지 않았음을 보여준다. 높은 경도를 이미 낮게 예측한 TRAIN_000227에서는 추가 보정이 더 낮췄고, 낮은 경도를 높게 예측한 TRAIN_000274에서는 더 높였다. 경도 범위 압축을 일관되게 고친 것도 아니다. 원인 후보로는 거친 복원 목표에 부족한 미세 경계/연결 구조, 전역 pooling의 정보 손실, 제한된 고잡음 라벨 수가 남는다. 이들은 구분되지 않은 가설이며, 공유 잔차가 비가역적·불가약적이라는 증거가 아니다. 다음 연구에 필요한 것은 이 보정의 가중치 추가 탐색이 아니라, 실제 미세 구조를 더 잘 보존하는 목표가 경도와 연결되는지에 대한 별도 증거다.

검증은 새 단계에 대해 엄격히 분리했으나,082 및 물리 특징의 과거 보조 표현·과거 선택 과정을 이번에 모두 처음부터 중첩 재훈련한 것은 아니다. 결과는 동결된 역사적 표현에 조건부다. 전체500 라벨은 이전 연구에서 노출되었고, 이번400도 반복 사용한 개발 라벨이다. 새 분할이나 bootstrap을 독립 확인 또는 미사용 holdout으로 부르지 않는다. Private 점수나 예상 개선을 추정하지 않았다.

계산은 로컬 MPS 한 작업씩, CPU2 threads로 진행했다. 사전 probe의90분 예산 조건은 실패했고, 주요 학습·경도 평가 전에 {link('AMENDMENT-01.md')}로120분 계산/180분 작업 상한을 한 번 고지했다.40개 인코더,210개 기본/PCA 회귀 문맥,180개 PLS 회귀 문맥,2665개 LOO 교사를 완료했다. 인코더·특징·회귀와100epoch 전체 인코더 재현을 포함한 기록된 계산 시간은 **{fit['total_compute_seconds']/60:.2f}분**이며, 후속 적합은 **{fit['seconds']/60:.2f}분**이다. 별도 수치·해시·기록 검증은 추가 수행했다. 관측 최대 RSS+MPS driver는4.755GiB로5GiB 상한 안이었다. 의존성 설치·갱신, 유료 자원, 외부 데이터 수집이나 새 제출은 없었다.

입력 범위·손실 수식을 독립 검사했다.40개 체크포인트/특징과20개 무학습 대조군,390개 회귀 입력 문맥·평균,60개 PCA 및60개 PLS 변환을 검사했다. PLS60개 재적합은 정확히 일치했고 교사30개는 자기 라벨 및 바깥 라벨을 바꿔도 그대로였다. 전체100epoch 인코더 재현에서 바깥 배열287개를 변경해도 학습 상태·특징의 최대 차이는0이었다. 새 PLS 회귀 재현 오차도0, CPU/MPS 비교 최대 차이는0.00005365HV였다. 모델 경로에 없는 바깥 라벨을 변경하는 검사와 실제 저장 입력 해시 확인을 함께 사용했지만 모든 역사적 적합을 다시 수행한 것은 아니다.

독립 앙상블 산술 검사 중 평균 연산 순서 때문에2.84×10⁻¹⁴HV 차이가 나 원래 exact 단언이 실패했다. 첫 검증 코드는 보존했고, 같은 순서 조립은 정확히 일치하도록, 다른 순서의 독립 계산은1e-10 안인지 확인하도록 검증만 수정했다({link('verification-repair.json')}). 모델·예측·평가 숫자는 바꾸지 않았다.

보호된 **{pres['files']:,}개 파일**의 SHA-256, Git HEAD 및 기존 tracked diff가 시작과 동일하다({link('preservation.json')}). 시작 HEAD는 `{start['head'].strip()}`이며 미커밋 변경·두 Python 환경의 의존성·모든 원본 경로와 해시는 {link('starting-state.json')}, {link('protected.json')}, {link('readout-code-lock.json')}에 기록했다.076은 동결 성분으로 조립하고082는 저장 CNN 가중치를318개 기존 추론 대상에 다시 실행해1000행을 조립했다. 두 CSV는 모두 바이트 단위로 동일했고082 최대 수치 차이는0이었다({link('reproduction/082-inference-verification.json')}). 이는 저장 산출물 추론/조립 재현이며 전체 역사적 훈련의 재현은 아니다. 이 별도 보존 검사만 기존 테스트 영상을 읽었고 연구 적합·선택은 학습 원본만 사용했다.

082 SHA-256: `0fa34cec1626b0887d3d7971197207c135a447a8f4805096b258e77b912f0e9f`.
076 SHA-256: `b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0`.


정확한 명령·시드·환경·동결 경로는 {link('RUN.md')}, 전체 원본별 예측은 {link('per-image.csv')} 및 {link('alignment-followup/per-image.csv')}, 수치 결과는 {link('results.json')}과 {link('alignment-followup/results.json')}에 있다. 새 깨끗한 실험 디렉터리에서 재실행해야 하며 기존 출력은 독점 생성으로 보호한다. `delivery-hashes.json`은 모든 작성 작업 종료 후 생성한다.082/076과 전체 상위 의존성은 그대로 보존되며 새 공식 최고 모델로의 변경은 없다.
'''
    with (_repo_path(HERE, 'REPORT.md')).open('x') as f:f.write(report)
    completion=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,new_candidate_recommended=False,incumbent='082',rollback='076',public_fit=False,submission_created=False,submitted=False,official_promotion=False,parent_advance=False,followup_advance=False,protected_files_unchanged=pres['files'],incumbent082_replay_max_difference=0.,incumbent_CSVs_byte_identical=True,compute_seconds=fit['total_compute_seconds'],limitation='Reused development labels; new nested fits conditional on historical incumbent representations; no independent private confirmation.',report_sha256=hashlib.sha256(report.encode()).hexdigest(),path_typo_cleaned=True)
    with (_repo_path(HERE, 'completion.json')).open('x') as f:json.dump(completion,f,indent=2)
    (_repo_path(HERE, 'STATE.md')).write_text('# Completed experiment\n\nThe spatial-context PCA experiment and the one bounded PLS/PCA/zero nested-selection follow-up are complete. Neither passed its frozen gate. Keep082 as incumbent and076 as rollback; no new submission candidate or official promotion. No active compute remains.\n\nSee REPORT.md, completion.json, RUN.md, preservation.json and reproduction/082-inference-verification.json. All152340 protected files are unchanged and both incumbent CSVs reproduce exactly. New fits are source-nested conditional on historical representations; the reused400 labels are not independent confirmation.\n\nRun manifest.py only after all other writers have closed. Final manifest completion is checked separately.\n')
    print('REPORT AND COMPLETION WRITTEN',len(report),flush=True)
if __name__=='__main__':main()
