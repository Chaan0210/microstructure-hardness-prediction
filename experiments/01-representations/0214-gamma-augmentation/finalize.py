from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ['GAMMA214_RAW_EDA']='1'
import common as c,engine as e
from common import R,P,np,torch
c.check();s=c.read(R/'summary.json');v=c.read(R/'validation.json');pre=c.read(R/'preflight.json');z=c.load(R/'OOF.npz');ref=c.load(R/'frozen-reference.npz');old=P/'experiments/contrast-augmentation-206'
manifest=c.read(old/'artifact-manifest.json');keys=[n for n in manifest if ('/fold-' in n and '-control/' in n) or n.endswith(('/engine.py','/common.py','/frozen-reference.npz','/training-code-lock.json'))]
for n in keys:assert c.sha(P/n)==manifest[n],n
original=c.load(old/'OOF.npz')
for n in ['ids','y','fold','control']:np.testing.assert_array_equal(z[n],original[n])
f=c.load(P/'experiments/matched-replacement-194/frozen-components.npz');b=c.load(P/'experiments/dinol-177/original/predictions.npz')
np.testing.assert_array_equal(ref['ids'],z['ids']);np.testing.assert_array_equal(ref['best021'],b['blend_L_MLP']);np.testing.assert_array_equal(ref['best021'],f['best021']);np.testing.assert_array_equal(ref['coefficient'],f['coef']);np.testing.assert_array_equal(ref['coefficient'],.081*f['slope'])
for n,h in c.read(R/'raw-image-hashes.json').items():assert c.sha(P/n)==h
# plan and all fixed preflight/training/validation/scoring sources are checked against the pretraining code lock.
# Independently validate stored score and per-ID residuals; no fitting or model selection.
res=c.read(R/'ID-residuals.json');assert len(res)==400 and len({r['ID'] for r in res})==400
boot=c.load(R/'bootstrap.npz');draws=boot['draws'];assert draws.shape==(10000,400)
for name,a,b in [('standalone',z['control'],z['gamma']),('fixed_ensemble',ref['best021'],ref['best021']+ref['coefficient']*(z['gamma']-ref['old_first_CNN']))]:
 ae=(z['y']-a)**2;be=(z['y']-b)**2;gain=float(np.sqrt(np.sum(ae)/400)-np.sqrt(np.sum(be)/400));r=s['comparisons'][name];assert abs(gain-r['gain_baseline_minus_candidate'])<1e-12
 boots=np.array([np.sqrt(ae[ix].mean())-np.sqrt(be[ix].mean()) for ix in draws]);np.testing.assert_allclose(boots,boot[name],rtol=0,atol=1e-12);np.testing.assert_allclose(np.quantile(boots,[.025,.975]),r['conditional_CI95'],rtol=0,atol=1e-12)
 removed=np.argsort(ae-be,kind='stable')[-5:];keep=np.ones(400,bool);keep[removed]=False;assert abs(float(np.sqrt(ae[keep].mean())-np.sqrt(be[keep].mean()))-r['top5_removed_gain'])<1e-12
 for i,row in enumerate(res):assert row['ID']==z['ids'][i] and abs(row[name+'_squared_error_gain']-(ae[i]-be[i]))<1e-12
# Same final optimizer and scheduler on the validation-label poison replay.
pa=torch.load(_repo_io(R/'poison-fold0-gamma/final-live-and-optimizer.pt'),map_location='cpu',weights_only=True);pb=torch.load(_repo_io(R/'fold-0-gamma/final-live-and-optimizer.pt'),map_location='cpu',weights_only=True);e.equal_tree(pa,pb)
train={}
for arm in ['control','gamma']:
 errors=[];grad=[]
 for k in range(5):
  folder=old/f'fold-{k}-control' if arm=='control' else R/f'fold-{k}-gamma';p=c.load(folder/'predictions.npz');st=c.load(folder/'streams.npz');errors.extend(((z['y'][p['train_indices']]-p['train_predictions'])**2).tolist());grad.extend(st['values'][:,1].tolist())
 train[arm]={'train_RMSE_repeated1600':float(np.sqrt(np.mean(errors))),'gradient_clip_fraction':float(np.mean(np.array(grad)>5))}
resources=[c.read(p) for p in R.glob('*-resources.json')];samples=[v for r in resources for v in r['trace']];mem=max(v['conservative_RSS_plus_MPS'] for v in samples);swap=max(max(v['swap']-r['initial_swap'] for v in r['trace']) for r in resources);assert mem<6e9 and swap==0
fits=list(R.glob('*/fit.json'));assert len(fits)==7;assert sum(c.read(p)['updates'] for p in fits)==5600;assert not list(R.rglob('*.csv'))
audit={'passed':True,'historical206_source_control_manifest_hashes_verified':len(keys),'original206_ID_target_fold_control_arrays_exact':True,'fixed_ensemble194_and177_exact':True,'poison_final_state_optimizer_scheduler_exact':True,'score_direct10000bootstrap_and_ID_residuals_verified':True,'pretraining_plan_and_fit_code_unchanged':True,'protected_existing_files':len(c.read(R/'preservation.json')),'raw_development400_SHA256_preserved':True,'training_runs':7,'candidate_runs':5,'verification_runs':2,'updates':5600,'reused_control_runs':5,'best021_SHA256':c.sha(P/'artifacts/submission-candidate-021.csv'),'peak_RSS_plus_MPS_bytes':mem,'swap_growth_bytes':swap,'train_diagnostics':train,'no_new_CSV':True,'no_install_download_network_Claude_holdout_test_full500_submission':True,'cost':0}
c.write(R/'final-audit.json',audit)
rows=[]
for name,label in [('standalone','CNN 단독'),('fixed_ensemble','고정 첫 seed 치환')]:
 q=s['comparisons'][name];lo,hi=q['conditional_CI95'];rows.append(f"| {label} | {q['baseline_RMSE']:.8f} | {q['candidate_RMSE']:.8f} | {q['gain_baseline_minus_candidate']:+.8f} | [{lo:+.8f}, {hi:+.8f}] | {q['top5_removed_gain']:+.8f} | {q['positive_folds']}/5 |")
diag={}
for scale in ['native256','model128']:
 ds=[v for f in pre['folds'] for v in f['diagnostics'][scale]]
 diag[scale]={'max_pixel_shift':max(d['max_pixel_shift'] for d in ds),'gradient_energy_ratio_min':min(d['gradient_energy_ratio']['min'] for d in ds),'gradient_energy_ratio_max':max(d['gradient_energy_ratio']['max'] for d in ds),'max_requantization_error':max(d['if_uint8_requantized_max_error'] for d in ds),'max_if_requantized_unique_levels_lost':max(d['if_uint8_unique_levels_lost']['max'] for d in ds),'max_absolute_dark_fraction_change':{t:max(d['dark_phase_proxies'][t]['abs_fraction_change']['max'] for d in ds) for t in ['fixed_025','fixed_035','fixed_050','train_pooled_Q10']}}
c.write(R/'photometric-diagnostics-summary.json',diag)
decision='사전 단독 기준 충족. 독립 확인 전 채택하지 않으며 best021 유지.' if s['further_review_eligible'] else '사전 단독 기준 미달. 이번 단일 감마 정책은 채택하지 않고 best021 유지.'
report=f"""# 214: 작은 감마 변동의 단일 비교

**{decision} CNN 단독과 고정 첫 seed 치환을 구별해 아래에 보고한다. 추가 강도 탐색·재분할·다른 증강 학습 없이 이번 비교를 완료했다.**

| 비교 | 대조 RMSE | 감마 RMSE | 이득(대조−감마) | 조건부 paired 95% CI | 상위 5 ID 제외 이득 | 개선 fold |
|---|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(rows)+f"""

사전 단독 기준: 이득 ≥0.02, 개선 ≥3/5 fold, paired95%CI 하한 >0, 상위5개 squared-error 이득 ID 제외 후 이득 >0, 기술 검증 통과. 실제 판정: `{s['criteria']}`. 상위5개는 두 비교에서 각각 선정했으며 목록과 fold별 이득은 summary.json에 있다. 모든400 ID의 예측·잔차·제곱오차 이득은 ID-residuals.json, OOF.npz, ensemble-OOF.npz로 제공한다.

## 변경한 정책과 원본 화질 확인

213의 밝기는 단독 평균 개선에도 앙상블 이득이 작고 상위5개 제외 시 악화했다. 지시에 따라213 재분할은 하지 않았다. 기존206 대비 감소,213 덧셈 밝기와 구분되는 `x**gamma` 한 종류를214로 시험했다. gamma 커널 계수나 기존 대비 증강의 재실행이 아니다.

원본 data/train에서 기존 개발400 ID만 읽어 모두256×256 grayscale임을 확인했다. 나머지100 holdout과 test는 열지 않았다. 원본 및 개발128캐시의 화질·밝기 통계를 확인했으며 원본에서 기존 BOX128/uint8 → float32/255를 재계산해 캐시와 exact 일치를 확인했다. **원본256과 기존 모델 입력128 해상도는 그대로 유지했다.** 추가 resize·crop·종횡비·좌표 변환은 없다. 실제 학습 감마는 기존 BOX128 후 float 입력에서 적용한다. native256 감마는 수치 진단만 수행하며 새 입력해상도 실험이 아니다.

전처리는 각 fold train320 전체에서 얻은 고정 평균·표준편차로 정규화한다. per-image 표준화가 없어 감마 변화는 상쇄되지 않으며 TinyCNN readout의 입력 mean/std도 바뀐다. 최종 모델은 기존 `cnn_controls.TinyCNN` 91,905파라미터 그대로다.

강도 산식과 노출은 성능을 보기 전에 잠갔다. 각 train영상의 `sensitivity=-mean(x*log(x))`를 계산하고 `tau=min(0.02,0.1*IQR(train image mean)/median(sensitivity))`, `gamma_low=exp(-tau)`, `gamma_high=exp(tau)`로 정했다. 0의 xlogx 항은0이다. 평균 밝기 변동의1차 근사가 train 평균 IQR의10% 정도가 되게 하되 log-gamma를0.02로 제한하는 작은 단일 정책이다. 타깃·validation·OOF 성능은 강도 선택에 쓰지 않았다.

fold별 [low,high]는 {[[round(f['gamma_low'],9),round(f['gamma_high'],9)] for f in pre['folds']]}다. ID hash와 epoch의4주기로 원본40회, high감마(어둡게)20회, low감마(밝게)20회 노출한다. 즉 전체 노출50%, 방향별25%이며 추가 RNG를 소비하지 않는다. 다른 강도·확률·감마와 밝기 조합은 실행하지 않았다.

## 범위·클리핑·양자화 및 측정 변화

native256과 model128 각각 모든 fold train영상에서 float32 범위[0,1], finite값, 양끝값, float64 reference 오차<1e-6, 동일 shape, unique intensity의 순위/동률 보존 및 영상별 Q10/Q50/Q90 상대 mask exact를 검사했다. 최대 픽셀 이동은 {diag['model128']['max_pixel_shift']:.8f}, 클리핑0, 실제 재양자화0이다. 정규화 후 RMS 변화는 모델입력에서 약0.0630–0.0679 표준편차로0이 아니다.

가상 uint8 재양자화에서는 최대 오차 {diag['model128']['max_requantization_error']:.8f}와 영상당 최대1개 unique level의 합쳐짐이 관찰됐다. 실제float 학습에서는 이런 양자화를 적용하지 않았고 순위/동률이 보존됐다. 기존 BOX 처리의 uint8 양자화는 두 군 공통이다.

그러나 **단조변환이어도 고정 임계값의 어두운 상 분율과 대비·gradient energy는 보존되지 않았다.** 아래는 fold train영상에서 관측한 절대 분율 변화의 최대값(%p)이다. 임계값0.25/0.35/0.50은 모든 영상에 고정했고 Q10은 각 fold train 픽셀에서만 계산해 원본/변환 모두 같은 값을 적용했다. 영상별 변화·평균 및 분위수는 preflight.json에 모두 기록했다.

| 해상도 | threshold0.25 | threshold0.35 | threshold0.50 | train pixel Q10 |
|---|---:|---:|---:|---:|
| native256 진단 | {100*diag['native256']['max_absolute_dark_fraction_change']['fixed_025']:.4f} | {100*diag['native256']['max_absolute_dark_fraction_change']['fixed_035']:.4f} | {100*diag['native256']['max_absolute_dark_fraction_change']['fixed_050']:.4f} | {100*diag['native256']['max_absolute_dark_fraction_change']['train_pooled_Q10']:.4f} |
| model128 실제 입력 | {100*diag['model128']['max_absolute_dark_fraction_change']['fixed_025']:.4f} | {100*diag['model128']['max_absolute_dark_fraction_change']['fixed_035']:.4f} | {100*diag['model128']['max_absolute_dark_fraction_change']['fixed_050']:.4f} | {100*diag['model128']['max_absolute_dark_fraction_change']['train_pooled_Q10']:.4f} |

영상별 gradient energy 비율(변환/원본)은 native {diag['native256']['gradient_energy_ratio_min']:.6f}–{diag['native256']['gradient_energy_ratio_max']:.6f}, model128 {diag['model128']['gradient_energy_ratio_min']:.6f}–{diag['model128']['gradient_energy_ratio_max']:.6f}다. 좌표·스케일·공간 배열은 그대로지만 물리적 상 분율이나 결정립 경계 측정의 정확성이 보장되지 않는다. 특히 이 dark fraction은 고정 intensity threshold 기반 대리 측정이며 실제 상 정답이 아니다. 감마의 물리적 라벨 보존을 입증했다고 주장하지 않으며, 이 작은 정책의 일반화 효과만 통제 비교했다. 변환 뒤 중심화·대비 복원·역감마 등 별도 보정을 추가하지 않았다.

## 정확한 대조와 검증

213과 같은 원본400의5fold(320/80), seed20261003+fold, D4, MSE, batch32, AdamW lr0.002/weight_decay0.01, cosine80epoch/eta_min0.0001, gradient clip5, 마지막5checkpoint 동일 평균을 유지했다. validation에는 감마를 적용하지 않았다.

- 실제 새 학습은 **7회/5,600update**: 감마 후보5회, 대조fold0 80epoch 재현1회, 감마fold0 validation타깃 NaN 교란 재현1회. 기존 대조5개를 재사용했다. 강도탐색·재분할·밝기 재학습0회다.
- 206 당시 소스·대조54개 hash를 검증했다. 대조fold0의 마지막5개state·평균state·OOF·800회 order/D4/LR/loss/gradient norm/CPU·MPS RNG가 exact 재현됐다. 대조5개 state는 run009와 exact이고 전부 재추론했으며206의 ID/타깃/fold/controlOOF도 exact다.
- 양 군은 fold별 초기 state 및 모든800 order/D4/LR/CPU·MPS RNG가 동일하다. 각 train영상의 노출20/20/40을 검증했다.
- 대조·후보10개 모두 독립5state 평균 및 CPU/MPS 재로드 통과. MPS 최대오차0, CPU 최대오차 {v['max_CPU_prediction_error']:.10f}다.
- 강도 산식은 validation이미지 NaN 변경에 불변, 정규화는 validation타깃 NaN 변경에 불변이다. 감마fold0 전체80epoch에서 validation타깃 NaN 교란 후 state/OOF/stream/최종optimizer/scheduler exact를 확인했다.
- 고정 첫seed치환은 `Q=P021+0.081*slope_fold*(new_first_CNN-old_first_CNN)`이다. 계수와 best021을194/177 계보에 대조해 exact임을 확인했다. 나머지2seed·보정·잔차 고정이며 전체 후단 재학습이나 새로운 가중치 탐색이 아니다.
- 동일400 ID paired bootstrap10,000회(seed202610214), CI와 상위5개 제외 및 ID별 이득을 직접 재계산해 검증했다. 소스/plan은 학습 전 잠갔고 학습·평가 정책은 변경하지 않았다.

중복 train1,600관측 RMSE는 대조 {train['control']['train_RMSE_repeated1600']:.6f}, 감마 {train['gamma']['train_RMSE_repeated1600']:.6f}다. 같은400장을 여러 실험에 누적·적응적으로 재사용했으므로 독립검증이 아니며 조건부 CI는 선택편향·seed·재학습 변동을 보정하지 않는다.

## 범위와 다음 상태

{decision} 자동 재분할·다른 강도·추가 증강은 실행하지 않았다. 213 밝기 재분할도 진행하지 않는 최신 지시를 유지한다. 감마214는 이번 고정 정책의 수행 완료이며 모든 감마 설정이나 화질 증강의 결론으로 일반화하지 않는다. Blur는 다음 미수행 항목으로만 남기며 실행 대기열을 만들지 않는다.

기존파일 {len(c.read(R/'preservation.json'))}개와 원본400 hash를 보존했고 best021 SHA256 `{audit['best021_SHA256']}`가 유지됐다. 기존 uv .venv, 무료CPU2 threads/MPS를 사용했으며 측정 최대 RSS+MPS {mem/1e9:.3f}GB, swap증가0, 비용0이다. 설치·다운로드·외부Claude·test·이전holdout·전체500학습·CSV생성/수정·제출은 없었다. 새 관리표는 docs/records/roadmap-audit/214-management.json 및 roadmap-30-after-214.md에 저장하며 과거 스냅샷을 덮어쓰지 않는다. 차단 사항 없음.

근거: plan.json/code-lock.json, preflight.py/preflight.json, engine.py, training-complete.json, validation.json, final-audit.json, summary.json, photometric-diagnostics-summary.json, ID-residuals.json 및 fold별 state/streams.
"""
with (R/'REPORT.md').open('x') as out:out.write(report)
c.write(R/'completion.json',{'study':214,'completed':True,'decision':decision,'best021_preserved':True,'training_runs':7,'candidate_runs':5,'verification_runs':2,'reused_controls':5,'report':'experiments/gamma-augmentation-214/REPORT.md','next':'No automatic resplit/strength search;213 brightness resplit not pursued;blur untested','blockers':[]})
c.check();c.write(R/'artifact-manifest.json',{str(p.relative_to(P)):c.sha(p) for p in R.rglob('*') if p.is_file() and p.name!='finalize.log'});c.emit(audit)
