import os
os.environ['RES216_RAW_EDA']='1'
import common as c,engine as e
from common import R,P,np,torch
c.check();s=c.read(R/'summary.json');v=c.read(R/'validation.json');pre=c.read(R/'preflight.json');info=c.read(R/'training-complete.json');z=c.load(R/'OOF.npz');ref=c.load(R/'frozen-reference.npz');boot=c.load(R/'bootstrap.npz');draws=boot['draws'];y=z['y'];fold=z['fold'];features=c.read(R/'image-features.json');res=c.read(R/'ID-residuals.json');old=P/'experiments/contrast-augmentation-206'
# Correct inherited215 narrative only; retain the generated original and all numeric results.
c.write(R/'summary-as-generated.json',s)
s['limitations']=['같은 개발400의 반복·적응적 사용에 따른 선택 편향. 독립 검증이 아님.','고정fit의 조건부 CI로 seed·재학습·누적선택 불확실성 제외.','동일 업데이트 예산이며 동일 연산량·시간이 아님. 원본 시야 기준 receptive field도 달라짐.','단독 CNN 신호와 고정 첫seed 치환을 구분하며 전체 파이프라인 개선을 주장하지 않음.']
s['decision']='단독 screening 통과; 같은400의 분할 민감도 확인을 위한 사전고정 재분할 검토만 제안. 고정 치환 기준 미달로 best021 유지; 자동후속학습 없음.' if s['further_review_eligible'] and not s['ensemble_further_review_eligible'] else s['decision']
with (R/'summary.json').open('w') as f:c.json.dump(s,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
c.write(R/'narrative-correction.json',{'reason':'Removed inherited215 blur/failure wording from216 summary. No metrics, comparisons, gates, plan or learning code changed.','original':'summary-as-generated.json','corrected':'summary.json'})
with c.Monitor('final-audit',600):
 assert len(features)==len(res)==400 and draws.shape==(10000,400)
 np.testing.assert_array_equal(draws,np.random.default_rng(202610256).integers(0,400,size=(10000,400)))
 a,b=z['control'],z['native'];qa=ref['best021']+ref['coefficient']*(a-ref['old_first_CNN']);qb=ref['best021']+ref['coefficient']*(b-ref['old_first_CNN'])
 if info['reused_controls']:np.testing.assert_array_equal(a,ref['old_first_CNN']);np.testing.assert_array_equal(qa,ref['best021'])
 comparisons=[('standalone',a,b),('fixed_ensemble',qa,qb),('versus_best021',ref['best021'],qb)]
 for name,aa,bb in comparisons:
  ae=(y-aa)**2;be=(y-bb)**2;r=s['comparisons'][name];calc=np.array([np.sqrt(np.mean(ae[ix]))-np.sqrt(np.mean(be[ix])) for ix in draws]);np.testing.assert_allclose(calc,boot[name],rtol=0,atol=1e-12);np.testing.assert_allclose(np.quantile(calc,[.025,.975]),r['conditional_CI95'],rtol=0,atol=1e-12);ix=np.argsort(ae-be,kind='stable')[-5:];keep=np.ones(400,bool);keep[ix]=False;assert abs(np.sqrt(ae[keep].mean())-np.sqrt(be[keep].mean())-r['top5_removed_gain'])<1e-12
  for i,row in enumerate(res):
   assert row['ID']==z['ids'][i]==features[i]['ID']
   if name!='versus_best021':assert abs(row[name+'_squared_error_gain']-(ae[i]-be[i]))<1e-12
 for n,h in c.read(R/'raw-image-hashes.json').items():assert c.sha(P/n)==h
 manifest=c.read(old/'artifact-manifest.json');keys=[n for n in manifest if ('/fold-' in n and '-control/' in n) or n.endswith(('/engine.py','/common.py','/frozen-reference.npz','/training-code-lock.json'))]
 for n in keys:assert c.sha(P/n)==manifest[n]
 diagnostics={};sg=(y-a)**2-(y-b)**2;eg=(y-qa)**2-(y-qb)**2
 fnames=['BOX_reconstruction_RMS','native_small_components_1to4px','native_gradient_energy','native_laplacian_variance','native_dark_fraction','dark_fraction_delta','area_matched_components_delta','mean_delta']
 for row in features:
  for dest,key in [('dark_fraction_delta','dark_fraction'),('area_matched_components_delta','components_area_matched'),('mean_delta','mean')]:row[dest]=row['native_'+key]-row['BOX_'+key]
 for name in fnames:
  x=np.array([r[name] for r in features]);order=np.argsort(x,kind='stable');bins=np.array_split(order,3);corr=lambda t:float(np.corrcoef(x,t)[0,1]) if x.std()>0 and t.std()>0 else None
  diagnostics[name]={'quantiles':np.quantile(x,[0,.1,.5,.9,1]).tolist(),'Pearson_with_CNN_SE_gain':corr(sg),'Pearson_with_fixed_SE_gain':corr(eg),'descriptive_equal_count_bins':[]}
  for j,ix in enumerate(bins):diagnostics[name]['descriptive_equal_count_bins'].append({'bin':j,'n':len(ix),'range':[float(x[ix].min()),float(x[ix].max())],'CNN128_RMSE':float(np.sqrt(np.mean((y[ix]-a[ix])**2))),'CNN256_RMSE':float(np.sqrt(np.mean((y[ix]-b[ix])**2))),'CNN_gain':float(np.sqrt(np.mean((y[ix]-a[ix])**2))-np.sqrt(np.mean((y[ix]-b[ix])**2))),'fixed_gain':float(np.sqrt(np.mean((y[ix]-qa[ix])**2))-np.sqrt(np.mean((y[ix]-qb[ix])**2)))})
 c.write(R/'resolution-error-diagnostics.json',{'features':diagnostics,'scope':'Descriptive same400 analysis; equal-count stable ordering includes ties; no causal attribution or threshold selection; pixel gradients/small1-4pixel objects are different physical scale across128/256; area-matched counts use native16 vs BOX4pixels; no physical ground truth','per_ID':features})
 train={}
 for arm in ['control','native']:
  errors=[];grad=[];seconds=[]
  for k in range(5):
   folder=P/info['folders'][f'{arm}-{k}'];pr=c.load(folder/'predictions.npz');st=c.load(folder/'streams.npz');errors.extend(((y[pr['train_indices']]-pr['train_predictions'])**2).tolist());grad.extend(st['values'][:,1].tolist());seconds.append(c.read(folder/'fit.json')['seconds'])
  train[arm]={'train_RMSE_repeated1600':float(np.sqrt(np.mean(errors))),'gradient_clip_fraction':float(np.mean(np.array(grad)>5)),'fit_seconds':seconds}
 fits=list(R.glob('*/fit.json'));expected=7 if info['reused_controls'] else 12;assert len(fits)==expected;updates=sum(c.read(p)['updates'] for p in fits);assert updates==800*expected;assert c.read(R/'probe.json')['completed_updates']==8;assert not list(R.rglob('*.csv'));assert not list(R.glob('*-failure.json'));assert not list(R.glob('*-STOP.json'))
 c.check()
resources=[c.read(p) for p in R.glob('*-resources.json')];samples=[v for r in resources for v in r['trace']];mem=max(v['conservative_RSS_plus_MPS'] for v in samples);swap=max(max(v['swap']-r['initial_swap'] for v in r['trace']) for r in resources);assert mem<6e9 and swap==0
besthash=c.sha(P/'artifacts/submission-candidate-021.csv');assert besthash=='7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b'
audit={'passed':True,'historical206_source_control_manifest_hashes_verified':len(keys),'reused_control_fits':5 if info['reused_controls'] else 0,'full_training_runs':len(fits),'full_fit_updates':updates,'discarded_probe_updates':8,'total_training_updates':updates+8,'failed_or_interrupted_fits':0,'bootstrap_all10000_direct_verified':True,'paired_all800_order_D4_LR_RNG_initial_state_verified':True,'all10_model_reload_and_last5state_average_passed':True,'poison_state_stream_OOF_optimizer_scheduler_exact':True,'pretraining_code_plan_unchanged':True,'raw400_preserved':True,'protected_files':len(c.read(R/'preservation.json')),'best021_SHA256':besthash,'peak_RSS_plus_MPS_bytes':mem,'swap_growth_bytes':swap,'CPU_prediction_max_error':v['max_CPU_prediction_error'],'MPS_prediction_max_error':v['max_MPS_prediction_error'],'train_diagnostics':train,'new_install_download_network_holdout_test_full500_CSV_submission':False,'cost':0}
c.write(R/'final-audit.json',audit)
rows=[]
for name,label in [('standalone','CNN 단독'),('fixed_ensemble','고정 첫seed 대응 치환'),('versus_best021','best021 대비 치환')]:
 q=s['comparisons'][name];lo,hi=q['conditional_CI95'];rows.append(f"| {label} | {q['baseline_RMSE']:.6f} | {q['candidate_RMSE']:.6f} | {q['gain_baseline_minus_candidate']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {q['top5_removed_gain']:+.6f} | {q['positive_folds']}/5 |")
decision='단독 screening을 통과해 같은400의 분할 민감도를 확인하는 사전 고정 재분할의 후속 검토를 제안한다.' if s['further_review_eligible'] else '단독 screening 기준을 충족하지 못해 이번 입력 정책 비교를 종료한다.'
ensemble='고정 치환도 전체 앙상블 후보의 후속 검토 기준을 충족했다.' if s['ensemble_further_review_eligible'] else '고정 치환은 전체 앙상블 후보의 후속 검토 기준을 충족하지 못했다.'
report=f"""# 216 — TinyCNN128/80 대 native256/80

**{decision} {ensemble} best021은 유지하며 재분할·추가학습은 실행하지 않았다.**

| 비교 | 대조 RMSE | native256 RMSE | 이득 | 조건부 paired95%CI | 상위5제외 이득 | 개선fold |
|---|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(rows)+f"""

## 비교와 재사용 검증

개발400의 동일 original5fold(train320/validation80), seed20261003+fold, TinyCNN91,905parameters, batch32, D4, MSE, AdamW lr0.002/weight_decay0.01, cosine80/eta_min0.0001, clip5, 800updates 및 epoch76–80 state 동일평균을 고정했다. A는 PIL BOX128 uint8→float32/255, B는 원본native256→float32/255다. 각 군의 fold train 픽셀 전체 mean/population std로 정규화한다. validation 통계와 사진증강을 사용하지 않았다.

**동일 업데이트 예산이지 동일 연산량/FLOPs/시간은 아니다.** 같은 픽셀 receptive field가 차지하는 원본 시야 비율도 달라진다. 정보량 하나만 분리한 인과실험으로 해석하지 않는다. 기존128/80 vs256/40은 epoch와 cosine 길이가 달랐으며 이번 결과가 과거 비교를 공정한 해상도 비교로 소급해 바꾸지 않는다.

TinyCNN.forward에 resize가 없음을 코드로 확인했고 첫Conv가 실제 [2,1,256,256] 입력을 받는 것을 hook으로 확인했다. raw개발400 SHA256은215기록과 같고, BOX128 재구성은 기존 캐시와 exact다. CSV로더 load_dev는 호출하지 않았다. 원시train.csv/test/holdout을 열지 않았다.

기존 대조 재사용: {info['reused_controls']}. 206소스/대조 {len(keys)}개 manifest hash를 검증했다. fold0 80epoch 재현의 마지막5state/평균state/OOF 및800회 order/D4/LR/loss/gradientnorm/CPU·MPS RNG를 exact 대조했다. 나머지4개도 run009state·정규화·재추론·OOF 일치를 확인했다. Q_arm=P021+0.081*slope_fold*(C_arm−C_old_first)를 적용했고, 재사용 시 Q_A=P021 exact다. 현재 대응 치환 효과와best021대비 효과를 각각 저장했다. 나머지seed와후단보정은 고정이며 전체 파이프라인 재학습 효과가 아니다.

## 실행과 검증

새 완전fit **{len(fits)}회/{updates}updates**, 버린 native256 자원probe **8updates**, 합계 **{updates+8}updates**. 새 후보5,128전체재현1,256validation라벨NaN전체재현1이며 기존대조5는 재사용이다(해당 실행 경로 기준). 실패/중단0,재분할0,추가튜닝0. 모델10개 CPU/MPS 재로드, 마지막5state 독립재평균,800노출/order/D4/LR/초기state/RNG 일치,전fold train통계의validation변경 불변성,NaN교란 뒤 state/OOF/stream/최종optimizer/scheduler exact를 통과했다. CPU최대오차 {v['max_CPU_prediction_error']:.9f}, MPS오차 {v['max_MPS_prediction_error']:.9f}.

paired400ID bootstrap10,000회 seed202610256으로 CI를 계산하고 전10,000회 직접인덱싱으로 재검산했다. 각 비교에서 SE대조−SE후보가 가장 큰5개를 stable sort로 따로 제거했다. 단독 기준은 이득≥0.02,개선fold≥3/5,CI하한>0,top5제외>0 및 기술검증 통과다. 단독 결과 `{s['criteria']}`. 앙상블 후속기준은 best021대비 치환에도 동일조건을 요구하며 대응 치환의양수/CI하한/top5조건도 확인했다.

fold별 이득·제외ID는 summary.json,400ID 예측·잔차·SE이득은 ID-residuals.json과 OOF.npz/ensemble-OOF.npz, bootstrap draw와결과는 bootstrap.npz에 보존했다.

## 해상도 민감 특징·오차

원본256과 BOX128의 평균/표준편차/threshold0.5어두운분율,8연결요소수,작은1–4픽셀요소수,gradient energy,Laplacian variance,128을2배반복한 원본복원RMS를 진단했다. 면적기준 연결요소수는 원본16픽셀 대BOX4픽셀로 대응시켰다. 작은1–4픽셀 및 픽셀gradient는 양해상도에서 물리길이가 다르므로 직접 동등한 물리측정으로 부르지 않는다. 실제상/결정립정답은 없으며 threshold측정은 대리값이다.

이미지특징과 CNN/고정치환SE이득의 Pearson상관,라벨과 무관한 특징순서의3분할별RMSE를 resolution-error-diagnostics.json에 저장했다. 이는 반복사용400에서의 사후기술진단이며 원인증명·새threshold선택·튜닝근거로 사용하지 않았다.

"""
for name in fnames:
 q=diagnostics[name];report+=f"- {name}: Q0/10/50/90/100={q['quantiles']}; CNN SE이득 상관={q['Pearson_with_CNN_SE_gain']}; 낮음/중간/높음3구간 CNN이득={[r['CNN_gain'] for r in q['descriptive_equal_count_bins']]}\n"
report+=f"""

중복train1600관측 RMSE:128 {train['control']['train_RMSE_repeated1600']:.6f},256 {train['native']['train_RMSE_repeated1600']:.6f}. gradientclip 비율:128 {train['control']['gradient_clip_fraction']:.4f},256 {train['native']['gradient_clip_fraction']:.4f}. 학습시간은 하드웨어상태와검증계측 영향을 받으며 동등연산량 지표가 아니다.

## 한계와 보존

같은개발400을 반복·적응적으로 선택한 편향이 남는다. 위 CI는 고정fit에 조건부이며 누적선택편향·seed변동·재학습불확실성을 포함하지 않는다. 독립holdout검증 또는 전체앙상블 개선으로 주장하지 않는다. 같은400의 재분할도 독립자료 검증은 아니다. 좋은신호가 있어도 다음사전고정재분할 제안까지만 하고자동후속학습하지 않는다.

측정 최대RSS+MPS **{mem/1e9:.3f}GB**, swap증가0,기존uv CPU2threads/MPS,비용0. 원본400·보호{audit['protected_files']}파일·best021보존. best021 SHA256 `{besthash}`. 새설치/다운로드/네트워크/test/holdout/전체500/CSV생성수정/제출 없음. 별도 diagrams/스킬 파일 수정 없음. 차단사항 없음.
"""
with (R/'REPORT.md').open('x') as f:f.write(report)
c.write(R/'completion.json',{'study':216,'completed':True,'decision':decision,'ensemble_decision':ensemble,'full_fits':len(fits),'full_updates':updates,'probe_updates':8,'total_updates':updates+8,'reused_controls':5 if info['reused_controls'] else 0,'best021_preserved':True,'automatic_followup_training':False,'report':'experiments/resolution-216/REPORT.md','blockers':[]})
c.check();c.write(R/'artifact-manifest.json',{str(p.relative_to(P)):c.sha(p) for p in R.rglob('*') if p.is_file() and p.name!='finalize.log'});c.emit(audit)
