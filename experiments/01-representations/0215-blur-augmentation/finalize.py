from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ['BLUR215_RAW_EDA']='1'
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
for name,a,b in [('standalone',z['control'],z['blur']),('fixed_ensemble',ref['best021'],ref['best021']+ref['coefficient']*(z['blur']-ref['old_first_CNN']))]:
 ae=(z['y']-a)**2;be=(z['y']-b)**2;gain=float(np.sqrt(np.sum(ae)/400)-np.sqrt(np.sum(be)/400));r=s['comparisons'][name];assert abs(gain-r['gain_baseline_minus_candidate'])<1e-12
 boots=np.array([np.sqrt(ae[ix].mean())-np.sqrt(be[ix].mean()) for ix in draws]);np.testing.assert_allclose(boots,boot[name],rtol=0,atol=1e-12);np.testing.assert_allclose(np.quantile(boots,[.025,.975]),r['conditional_CI95'],rtol=0,atol=1e-12)
 removed=np.argsort(ae-be,kind='stable')[-5:];keep=np.ones(400,bool);keep[removed]=False;assert abs(float(np.sqrt(ae[keep].mean())-np.sqrt(be[keep].mean()))-r['top5_removed_gain'])<1e-12
 for i,row in enumerate(res):assert row['ID']==z['ids'][i] and abs(row[name+'_squared_error_gain']-(ae[i]-be[i]))<1e-12
# Same final optimizer and scheduler on the validation-label poison replay.
pa=torch.load(_repo_io(R/'poison-fold0-blur/final-live-and-optimizer.pt'),map_location='cpu',weights_only=True);pb=torch.load(_repo_io(R/'fold-0-blur/final-live-and-optimizer.pt'),map_location='cpu',weights_only=True);e.equal_tree(pa,pb)
train={}
for arm in ['control','blur']:
 errors=[];grad=[]
 for k in range(5):
  folder=old/f'fold-{k}-control' if arm=='control' else R/f'fold-{k}-blur';p=c.load(folder/'predictions.npz');st=c.load(folder/'streams.npz');errors.extend(((z['y'][p['train_indices']]-p['train_predictions'])**2).tolist());grad.extend(st['values'][:,1].tolist())
 train[arm]={'train_RMSE_repeated1600':float(np.sqrt(np.mean(errors))),'gradient_clip_fraction':float(np.mean(np.array(grad)>5))}
resources=[c.read(p) for p in R.glob('*-resources.json')];samples=[v for r in resources for v in r['trace']];mem=max(v['conservative_RSS_plus_MPS'] for v in samples);swap=max(max(v['swap']-r['initial_swap'] for v in r['trace']) for r in resources);assert mem<6e9 and swap==0
fits=list(R.glob('*/fit.json'));assert len(fits)==7;assert sum(c.read(p)['updates'] for p in fits)==5600;assert not list(R.rglob('*.csv'))
audit={'passed':True,'historical206_source_control_manifest_hashes_verified':len(keys),'original206_ID_target_fold_control_arrays_exact':True,'fixed_ensemble194_and177_exact':True,'poison_final_state_optimizer_scheduler_exact':True,'score_direct10000bootstrap_and_ID_residuals_verified':True,'pretraining_plan_and_fit_code_unchanged':True,'protected_existing_files':len(c.read(R/'preservation.json')),'raw_development400_SHA256_preserved':True,'training_runs':7,'candidate_runs':5,'verification_runs':2,'updates':5600,'reused_control_runs':5,'best021_SHA256':c.sha(P/'artifacts/submission-candidate-021.csv'),'peak_RSS_plus_MPS_bytes':mem,'swap_growth_bytes':swap,'train_diagnostics':train,'no_new_CSV':True,'no_install_download_network_Claude_holdout_test_full500_submission':True,'cost':0}
c.write(R/'final-audit.json',audit)
rows=[]
for name,label in [('standalone','CNN 단독'),('fixed_ensemble','고정 첫 seed 치환')]:
 q=s['comparisons'][name];lo,hi=q['conditional_CI95'];rows.append(f"| {label} | {q['baseline_RMSE']:.8f} | {q['candidate_RMSE']:.8f} | {q['gain_baseline_minus_candidate']:+.8f} | [{lo:+.8f}, {hi:+.8f}] | {q['top5_removed_gain']:+.8f} | {q['positive_folds']}/5 |")
grains=c.read(R/'grain-measurement-deltas.json');diag={}
for scale in ['model128','native256']:
 ds=[f['diagnostics'][scale] for f in pre['folds']];gs=[q for q in grains if q['scale']==scale]
 diag[scale]={'gradient_energy_ratio':[min(d['gradient_energy_ratio']['min'] for d in ds),max(d['gradient_energy_ratio']['max'] for d in ds)],'laplacian_variance_ratio':[min(d['laplacian_variance_ratio']['min'] for d in ds),max(d['laplacian_variance_ratio']['max'] for d in ds)],'max_pixel_shift':max(d['max_pixel_shift'] for d in ds),'max_if_requantized_error':max(d['if_uint8_requantized_max_error'] for d in ds),'max_absolute_dark_fraction_change':{t:max(d['phase'][t]['absolute_fraction_change']['max'] for d in ds) for t in ['fixed_025','fixed_035','fixed_050','train_pooled_Q10']},'max_small_components_completely_disappeared':{t:max(d['phase'][t]['small_components_completely_disappeared']['max'] for d in ds) for t in ['fixed_025','fixed_035','fixed_050','train_pooled_Q10']},'grain_proxy_changes':{name:{'min':min(g['after'][name]-g['before'][name] for g in gs),'max':max(g['after'][name]-g['before'][name] for g in gs),'mean':float(np.mean([g['after'][name]-g['before'][name] for g in gs]))} for name in gs[0]['before']},'watershed_paired_measurements':len(gs)}
c.write(R/'measurement-diagnostics-summary.json',diag)
decision='사전 단독 기준 충족. 전체 모델 개선으로 간주하지 않으며 독립 확인 전 best021 유지.' if s['further_review_eligible'] else '사전 단독 기준 미달. 이번 블러 정책을 채택하지 않고 best021 유지.'
report=f"""# 215: 작은 블러 한 정책의 비교

**{decision} 추가 강도·재분할·다른 증강은 실행하지 않고 이번 비교를 완료했다.**

| 비교 | 대조 RMSE | 블러 RMSE | 이득(대조−블러) | 조건부 paired 95% CI | 상위 5 ID 제외 이득 | 개선 fold |
|---|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(rows)+f"""

사전 기준은 단독 이득≥0.02, 개선fold≥3/5, CI하한>0, 상위5개 squared-error 이득ID 제외 후 이득>0 및 기술검증 통과였다. 결과: `{s['criteria']}`. 상위5개는 각 비교에서 따로 제거했다. ID목록·fold이득은 summary.json, 모든400 ID의 예측·잔차·제곱오차 이득은 ID-residuals.json/OOF.npz/ensemble-OOF.npz에 있다.

## 사전 고정과 화질 EDA

블러 학습 ablation은 기존 로드맵에서 미실시였다. 071/072의 복원·098 NLM 측정 대조와 다른 정책이다. 개발400 원본256×256 grayscale만 읽었고 기존 BOX128 uint8→float32/255를 다시 계산해 캐시와 exact를 확인했다. 원본256과 모델128 해상도, 좌표·종횡비는 그대로이며 추가 crop/resize를 하지 않았다. 실제 블러는 기존 모델128 입력에서 적용한다. native256은 같은3×3픽셀 필터의 별도 진단이며 물리적 길이가 같다는 의미는 아니다.

train320의 gradient energy E를 `mean(diffX²)+mean(diffY²)`로 정의하고, `a=min(0.04,0.1*max(0,1-Q10(E)/Q50(E)))`를 float32로 고정했다. 작은 train 선명도 변동을 근거로 이웃 평균의 강도를 제한한 한 산식이며 결과·validation·타깃을 보지 않았다. 각 축의 필터는 `[a,1-2a,a]`, 경계처리는 reflect, 두 축으로 차례 적용한다. 정규화된 양의 국소 평균이고 Gaussian filter라고 부르지 않는다.

fold별 a는 {[round(f['alpha'],9) for f in pre['folds']]}다. 축별 kernel RMS반경 sqrt(2a)는 약0.273–0.280픽셀이며, 이 작은 공간범위가 측정 영향도 작다는 보장은 없다. ID hash와 epoch의2주기로 원본40회/블러40회, 노출50%를 고정했다. 다른 강도·확률·감마/밝기와의 조합은0회다. 블러는 fold train 고정정규화 이전에만 적용하고 validation은 원본을 사용했다.

## 경계·선명도·결정립/상 측정에 미친 영향

float범위[0,1], finite, shape동일, kernel합1, float64연산 대조 오차<1e-6, 평균오차<1e-6, 추가 clipping0/재양자화0을 확인했다. 고정 train정규화 뒤에도 입력차이는 남았다. 가상uint8반올림 오차는 최대 {diag['model128']['max_if_requantized_error']:.8f}이나 실제 학습에서는 반올림하지 않았다. D4와 블러 순서의 수치차이도1e-6미만이었다.

| 진단 | model128 실제 입력 | native256 진단 |
|---|---|---|
| gradient energy 비율(변환/원본) | {diag['model128']['gradient_energy_ratio']} | {diag['native256']['gradient_energy_ratio']} |
| Laplacian 분산 비율 | {diag['model128']['laplacian_variance_ratio']} | {diag['native256']['laplacian_variance_ratio']} |
| threshold0.50 어두운 분율 최대 절대변화 | {100*diag['model128']['max_absolute_dark_fraction_change']['fixed_050']:.4f}%p | {100*diag['native256']['max_absolute_dark_fraction_change']['fixed_050']:.4f}%p |
| threshold0.50 원래1–4픽셀 연결요소 완전소실 최대수/영상 | {diag['model128']['max_small_components_completely_disappeared']['fixed_050']:.0f} | {diag['native256']['max_small_components_completely_disappeared']['fixed_050']:.0f} |
| watershed 영역수 변화 범위(면적≥16) | {diag['model128']['grain_proxy_changes']['count_area_ge16']['min']:+.0f}~{diag['model128']['grain_proxy_changes']['count_area_ge16']['max']:+.0f} | {diag['native256']['grain_proxy_changes']['count_area_ge16']['min']:+.0f}~{diag['native256']['grain_proxy_changes']['count_area_ge16']['max']:+.0f} |

고정threshold0.25/0.35/0.50와 fold train pooledQ10에서 분율,8연결요소수,1–4픽셀 작은 영역수·완전소실을 전train에서 진단했다. 원래 component의 모든 픽셀이 변환 후 threshold밖으로 벗어나면 완전소실로 정의한다. 합쳐짐/분리의 물리적 추적이나 실제 상 정답은 아니다. 각 fold·해상도·ID별 원본/변환 측정값은 phase-*.npz, sharpness-*.npz, 전체요약은 preflight.json에 있다.

결정립 대리값은 fold마다 라벨과 무관한 IDhash로 train12장을 먼저 선택하고, 기존 `grain_features.segment(image_float*255,sigma=1)`의 watershed를 그대로 사용했다. 총120쌍(60train관측×2해상도)에서 면적≥16 영역수, 평균등가직경, 평균d^−1/2, 경계분율을 비교했다. 재양자화 없이 float를 넘겼으며 기존 측정코드의 자체Gaussian1 평활화는 양쪽에 동일하다. 상세 before/after는 grain-measurement-deltas.json과 measurement-diagnostics-summary.json에 있다. 기존097/098/183에서 검증되지 않은 측정법이므로 실제 결정립 변화를 측정했다고 주장하지 않는다.

**블러는 좌표와 해상도를 유지해도 선명도·경계와 작은 어두운 영역·결정립/상 대리 측정값을 바꿨다. 구조나 hardness 라벨 보존은 입증되지 않았다.** 이번 실험은 원래타깃을 유지한 작은 증강 정책의 경험적 효과를 평가한 것이며, 진단을 보고 강도를 다시 선택하지 않았다.

## 학습과 검증

213/214와 같은 원본400의5fold(320/80), seed20261003+fold, TinyCNN91,905파라미터/128입력, D4, MSE, batch32, AdamW0.002/weight_decay0.01, cosine80epoch/eta_min0.0001, gradient clip5, 마지막5state동일평균이다. 바뀐 학습변수는 입력 블러뿐이다.

- 새 학습 **7회/5,600update**: 후보5회, 대조80epoch재현1회, 블러validation타깃NaN교란80epoch재현1회. 기존 대조5개 재사용. 탐색·재분할0회.
- 206 역사적 소스·대조54개 hash 확인. fold0대조의 state/epoch76–80/OOF 및 모든800 order/D4/LR/loss/gradientnorm/CPU·MPS RNG exact. 대조5개는 run009state와 exact이며 재추론과206OOF/ID/타깃/fold도 exact.
- 후보/대조 초기state 및 모든800 order/D4/LR/RNG 동일. 각 train영상 blur40/original40회 노출 확인. 전fold train통계/강도는 validation변경에 불변.
- 10개모델의5state독립평균 및 CPU/MPS재로드 통과. MPS오차0, CPU최대오차 {v['max_CPU_prediction_error']:.10f}. fold0라벨NaN교란 뒤80epoch state/OOF/stream/optimizer/scheduler exact.
- 고정치환 `Q=P021+0.081*slope_fold*(new_first_CNN-old_first_CNN)`을194/177계보와 대조했다. 나머지2seed·보정·잔차 고정이다. **약한 첫 seed 치환 이득을 전체 모델 개선으로 취급하지 않는다.**
- 400ID paired bootstrap10,000회(seed202610215), CI/상위5제외/ID잔차를 직접재검산했다. 같은400을 반복선택한 조건부CI로서 누적선택편향·seed·재학습 불확실성을 보정하지 않는다.

중복train1,600관측 RMSE는 대조 {train['control']['train_RMSE_repeated1600']:.6f}, 블러 {train['blur']['train_RMSE_repeated1600']:.6f}. 측정 최대 RSS+MPS {mem/1e9:.3f}GB, swap증가0, CPU2threads/MPS, 기존uv.venv, 비용0이다.

기존 {len(c.read(R/'preservation.json'))}파일·원본400hash·best021 SHA256 `{audit['best021_SHA256']}` 보존. 설치/다운로드/AI/test/holdout/전체500/CSV/제출 없음. 비교 후 정지하며213–215요약·남은부분8개·다음우선가설1개는 docs/records/roadmap-audit/handoff-after-215/HANDOFF.md에 정리한다. 미실시블러를 완료로 바꾼 최신범위는 완료22·부분8·미실시0이다. 차단사항 없음.
"""
with (R/'REPORT.md').open('x') as f:f.write(report)
c.write(R/'completion.json',{'study':215,'completed':True,'decision':decision,'best021_preserved':True,'training_runs':7,'candidate_runs':5,'verification_runs':2,'reused_controls':5,'report':'experiments/blur-augmentation-215/REPORT.md','handoff':'docs/records/roadmap-audit/handoff-after-215/HANDOFF.md','next':'one hypothesis proposed only; no automatic resplit or strength exploration','blockers':[]})
c.check();c.write(R/'artifact-manifest.json',{str(p.relative_to(P)):c.sha(p) for p in R.rglob('*') if p.is_file() and p.name!='finalize.log'});c.emit(audit)
