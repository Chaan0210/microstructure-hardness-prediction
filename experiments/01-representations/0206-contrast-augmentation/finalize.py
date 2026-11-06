from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
s=read(_repo_path(R, 'summary.json'));v=read(_repo_path(R, 'validation.json'));pre=read(_repo_path(R, 'structure-preflight.json'));pres=read(_repo_path(R, 'preservation.json'));assert v['passed'] and pre['numeric_structure_gate_passed'] and read(_repo_path(R, 'visual-review.json'))['passed']
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
for lock in ['preflight-code-lock.json','training-code-lock.json']:
 for n,h in read(_repo_path(R, lock)).items():assert sha(_repo_path(R, n))==h,n
for k in range(5):assert read(_repo_path(R, f'fold-{k}-control/control-reproduction.json'))['passed'] and read(_repo_path(R, f'paired-streams-fold-{k}.json'))['passed']
resources={};seconds=0.;peak=0;swap=0
for stage in ['preflight','setup','training','validation','score']:
 z=read(_repo_path(R, f'{stage}-resources.json'));r=z['trace'];pk=max(x['conservative_RSS_plus_MPS'] for x in r);sw=max(x['swap']-z['initial_swap'] for x in r);peak=max(peak,pk);swap=max(swap,sw);seconds+=r[-1]['seconds'];resources[stage]={'seconds':r[-1]['seconds'],'peak_RSS_plus_MPS':pk,'swap_growth':sw,'access':z['access']};assert z['access']=={'raw_data_attempts':0,'outside_writes':0}
assert peak<=6e9 and swap==0 and not list(R.glob('*STOP*')) and not list(R.rglob('*.csv'))
write(_repo_path(R, 'resources-summary.json'),{'stages':resources,'monitored_stage_seconds':seconds,'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':swap,'cost':0})
q=s['comparisons']['standalone'];f=s['comparisons']['fixed_ensemble'];rows=[]
for name,d in [('CNN 단독',q),('고정 앙상블 출력 치환',f)]:rows.append(f"| {name} | {d['baseline_RMSE']:.10f} | {d['candidate_RMSE']:.10f} | {d['gain_baseline_minus_candidate']:+.10f} | [{d['conditional_CI95'][0]:+.10f}, {d['conditional_CI95'][1]:+.10f}] | {d['positive_folds']}/5 | {d['top5_removed_gain']:+.10f} |")
quality=[]
for a in s['quality_diagnostics']:
 z=a['scores'];quality.append(f"| {a['proxy']} | {a['group']} | {a['n']} | {z['control']:.6f} | {z['contrast']:.6f} | {a['standalone_gain']:+.6f} | {z['fixed_control']:.6f} | {z['fixed_contrast']:.6f} | {a['fixed_ensemble_gain']:+.6f} |")
limits='\n'.join('- '+x for x in s['limitations'])
report=f'''# 206: 학습 시 대비 감소 단일 대조

**{s['decision']}** 추가 검토 기준 충족 여부는 `{s['further_review_eligible']}`이다. 조건에 관계없이 원본 5fold 결과를 보고하고 정지하며 새 모델·제출을 선택하지 않는다.

## 시작 조건과 실제 수행

기존 206 설계의 추가 공식 원문 확인 전 보류 조건은 사용자의 새 지시로 해제했다. 대비 변동의 공식 기록, 영상 구조 보존 검사, train 전용 대조를 근거로 **타깃 불변성을 미검증 가정으로 둔 제한적 검증**을 허용한 것이다. 기존 보고서를 수정하지 않고 `authorization.json`과 새 `plan.json`에 변경 근거를 보존했다.

원본 개발 400장의 기존 5fold(320/80), seed 20261003 + fold, TinyCNN 91,905개 파라미터, 기존 BOX 128 입력, D4, MSE, train 전용 표준화, AdamW, 80 epoch/800 update, 마지막 5 epoch state 평균을 유지했다. 두 군 차이는 train 입력의 평균을 중심으로 한 양의 affine 대비 감소 하나다. 좌표·해상도 변경, clipping, 학습 입력 재양자화는 없다.

Train의 q95−q05 대비 Q10/Q50에서 사전 규칙으로 정한 감소 상한은 모든 fold에서 5%였다. 유효 감소 대상은 fold별 {[x['effective_images'] for x in pre['folds']]} / 320개였다. 각 ID는 고정 hash로 40회 원본·40회 변환 경로에 노출되며 c=1인 ID의 실질 변환 횟수는 0이다. 낮은 대비 영상은 train 범위를 벗어나지 않도록 감소를 제한했다. 강도와 노출 확률의 결과 기반 조정은 없었다.

## 구조 보존 및 통제 검증

- 5fold train의 총 1,600개 중복 관측에서 픽셀 순서·동률과 q10/q50/q90 rank 마스크가 정확히 일치했다. 유한차분의 부호와 0인 위치도 동일했다. 평균 오차 최대 {max(x['mean_error'] for x in pre['folds']):.3g}, 대비 비율 오차 최대 {max(x['contrast_ratio_error'] for x in pre['folds']):.3g}, 미분 스케일 오차 최대 {max(x['gradient_error'] for x in pre['folds']):.3g}로 사전 1e-6 기준 이내였다. 값과 대비는 train 허용 범위 안에 있었다.
- 사전 선택한 {pre['visual_unique_images']}개 고유 영상의 5개 패널을 검토했다. Clip·공간 이동·처리상 경계 또는 작은 어두운 물체의 소실을 관찰하지 않았다. PNG 표시만 양자화했으며 학습에는 float 캐시를 사용했다. 참 물리 마스크·경도 보존의 증명은 아니다.
- 모든 control의 최종 state와 legacy FP32 OOF가 기존 CNN 첫 seed와 bitwise exact였다. 두 군의 초기 state, 전체 800 update의 샘플 순서·D4·LR·CPU/MPS RNG가 일치했다.
- 10개 모델의 5개 state 재평균, 재로드, train 전용 통계를 검증했다. MPS 재로드 최대 오차 {v['max_MPS_prediction_error']:.3g}, CPU 최대 오차 {v['max_CPU_prediction_error']:.3g}. Validation 타깃을 NaN 또는 1e30으로 바꾼 fold0 두 군의 전체 80 epoch 재실행에서 state·예측·학습 스트림이 정확히 같았다.
- 주 학습 10회와 검증 재학습 2회. 검증을 마친 후 OOF 점수를 계산했다. 단일 seed이므로 seed 분산은 평가하지 않았다.

## 결과

개선량은 기준 RMSE − 대비 감소 RMSE이며 양수가 개선이다. CI는 저장된 학습 결과에 조건부인 400 ID paired bootstrap 10,000회(seed 202610206)다.

| 비교 | 기준 RMSE | 대비 감소 RMSE | 개선량 | 조건부 95% CI | 개선 fold | 상위 5개 이득 제외 개선 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

단독 fold 개선량: {q['fold_gains']}. 고정 앙상블 fold 개선량: {f['fold_gains']}.

앙상블 식은 194에서 검증한 `Q = P021 + 0.081 × 기존 fold 보정 기울기 × (새 첫 seed CNN − 기존 첫 seed CNN)`이다. Coefficient와 다른 seed·보정·잔차는 고정했다. Control은 best021 OOF와 정확히 같았다. 이는 **후단을 고정한 출력 치환 진단**이며 변경된 CNN으로 후단까지 학습한 전체 앙상블 성능은 아니다.

사전 단독 기준은 개선 ≥ 0.02, 개선 fold ≥ 3, CI 하한 > 0, 상위 5개 이득 제외 개선 > 0 및 기술 검증 통과다. 실제 판정은 `{s['criteria']}`이다. 고정 앙상블과 화질 진단은 대체 후보·가중치 선택에 사용하지 않았다.

## 화질 구간별 진단

각 fold의 train 320장만으로 정한 tertile 경계를 validation 80장에 적용했다. 타깃·test·OOF 결과로 경계를 고르지 않았다. Contrast는 128 입력의 q95−q05이며 noise/contrast는 097에 저장된 native 256 대리값이다. 실제 생성 열화량이나 물리 상 분율이 아니다.

| 대리값 | 구간 | n | 단독 control | 단독 contrast | 단독 개선 | 앙상블 control | 앙상블 contrast | 앙상블 개선 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(quality)}

구간별 결과는 설명용이며 소집단 선택·가중치 학습·추가 탐색으로 연결하지 않는다.

## 범위와 보존

{limits}

기존 파일 {len(pres)}개의 SHA256을 전후 확인했다. Best021 CSV SHA256은 `{pres['artifacts/submission-candidate-021.csv']}`로 유지됐다. 무료 CPU 2 threads/MPS 사용, 보수적 RSS+MPS 관측 최대 {peak/1e9:.3f} GB, swap 증가 0, 감시 단계 합계 {seconds:.1f}초였다. 작성 시간을 포함한 전체 벽시계 시간은 아니다.

신규 설치·다운로드·외부 AI·raw train/holdout/test 파일 읽기·전체 500장 학습·새 CSV·제출·비용 발생은 없다. 개발 400장의 기존 이미지·라벨·예측 캐시만 사용했다. 204는 재시도하지 않았고 205의 기각 범위도 변경하지 않았다.

근거: `plan.json`, `structure-preflight.json`, `visual-review.json`, `setup-validation.json`, 각 fold의 `control-reproduction.json`, `paired-streams-fold-*.json`, `OOF.npz`, `ensemble-OOF.npz`, `validation.json`, `summary.json`, `resources-summary.json`.
'''
with (_repo_path(R, 'REPORT.md')).open('x') as out:out.write(report)
top={**s,'report':'experiments/contrast-augmentation-206/REPORT.md','preserved_file_count':len(pres),'best021_SHA256':pres['artifacts/submission-candidate-021.csv'],'peak_RSS_plus_MPS_bytes':peak,'swap_growth_bytes':swap,'training_fits':10,'verification_retraining_fits':2,'structure_gate_passed':True,'control_reproduction_all5_exact':True,'paired_streams_all5_exact':True}
write(_repo_path(P, 'reports/contrast-augmentation-206-summary.json'),top)
completion=f'''# 206 완료 기록

{s['decision']}

| 항목 | 상태 | 결과·판단 |
|---|---|---|
| 로드맵 대비 단일 증강 | 완료 | TinyCNN에 평균 보존 대비 감소만 추가. 원본 개발 5fold, 주 학습 10회. |
| 구조 보존 | 통과 | Train 범위 제한·5% 상한·50% 노출. 픽셀 순서·동률·rank mask·경계 위치 검사 및 {pre['visual_unique_images']}장 시각 검토 통과. 라벨 불변성 증명 아님. |
| 대조·학습 통제 | 통과 | 기존 5개 CNN state·FP32 OOF exact. 초기 state·샘플 순서·D4·LR·RNG 일치. 검증 재학습 2회 exact. |
| 단독 OOF | 완료 | {q['baseline_RMSE']:.10f} → {q['candidate_RMSE']:.10f}, 개선 {q['gain_baseline_minus_candidate']:+.10f}, CI {q['conditional_CI95']}. |
| 고정 앙상블 | 완료 | {f['baseline_RMSE']:.10f} → {f['candidate_RMSE']:.10f}, 개선 {f['gain_baseline_minus_candidate']:+.10f}, CI {f['conditional_CI95']}. 후단 재학습 아님. |
| 화질 구간 | 완료 | Train tertile 기반 contrast·noise/contrast 6개 구간 모두 보고. 구간 선택·구제 탐색 없음. |
| 후속 작업 | 정지 | 기준 충족 여부 {s['further_review_eligible']}. Best021 유지. 같은400·단일seed 한계 보존. 추가 학습·제출 없음. |

[전체 보고서](../experiments/contrast-augmentation-206/REPORT.md), [구조화 결과](../reports/contrast-augmentation-206-summary.json), [실행 승인 및 보류 조건 변경](../experiments/contrast-augmentation-206/authorization.json).

기존 `next-experiment-206.md`와 `quality-augmentation-audit.md`는 당시 설계·보류 기록으로 보존했다. 현재 실행 상태는 이 완료 기록과 새 실험 폴더를 따른다. 기존 관리표·보고서·CSV {len(pres)}개 해시 유지. 신규 CSV·설치·다운로드·holdout/test·제출·비용 0.
'''
with (_repo_path(P, 'docs/records/roadmap-audit/206-completion.md')).open('x') as out:out.write(completion)
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
manifest={str(p.relative_to(P)):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','completion.json','finalize.log']}
for p in [_repo_path(P, 'reports/contrast-augmentation-206-summary.json'),_repo_path(P, 'docs/records/roadmap-audit/206-completion.md')]:manifest[str(p.relative_to(P))]=sha(p)
write(_repo_path(R, 'artifact-manifest.json'),manifest)
write(_repo_path(R, 'completion.json'),{'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'완료','preserved_files':len(pres),'all_protected_unchanged':True,'artifact_manifest_SHA256':sha(_repo_path(R, 'artifact-manifest.json')),'report_SHA256':sha(_repo_path(R, 'REPORT.md')),'cost':0,'remaining_blockers':[],'scientific_limits':s['limitations'],'best021_SHA256':pres['artifacts/submission-candidate-021.csv']})
print(json.dumps({'report':str(_repo_path(R, 'REPORT.md')),'comparisons':s['comparisons'],'criteria':s['criteria'],'peak_GB':peak/1e9,'preserved_files':len(pres)},ensure_ascii=False,indent=2))
