from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,math,json,datetime
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[4];R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import diagnose as d
d.protect();d.guard();assert d.read(_repo_path(R, 'verification.json'))['passed'];legacy=d.read(_repo_path(R, 'L4-legacy.json'));lm=d.read(_repo_path(R, 'L4-legacy-measurements.json'));plan=d.read(_repo_path(R, 'L4-legacy-extension-plan-v2.json'))
for p,h in plan['source'].items():assert d.sha(_repo_path(P, p))==h
assert d.sha(_repo_path(R, 'legacy_grain_v2.py'))==plan['script_sha256']
for row in lm['rows']:
 for c in row['configs']:
  assert c['count']==c['interior_count']+c['border_count']
  assert c['empty0']==(c['interior_count']==0)
  assert 0<=c['border_area_fraction']+c['interior_area_fraction']<=1
  if c['empty0']:assert c['interior_diam_mean']==c['interior_hall_petch']==0
assert lm['cached_descriptor_maxerror']==0
# Additional posthoc descriptive check: magnitude clustering need not predict sign.
kn=d.load(_repo_path(R, 'L2-neighbors.npz'))['indices'][:,:5];signed={}
for split in ['original','20261031','20261101']:
 z=d.data(split);r=z['y']-z['blend_L_MLP'];signed[split]={'residual_vs_k5_mean_residual_Pearson':float(np.corrcoef(r,r[kn].mean(1))[0,1]),'directed_edge_residual_sign_agreement':float(np.mean(np.sign(r[:,None])==np.sign(r[kn]))),'cross_fold_edge_fraction':float(np.mean(z['fold'][:,None]!=z['fold'][kn]))}
d.write(_repo_path(R, 'L3-signed-neighbor-posthoc.json'),{'posthoc_descriptive':True,'results':signed,'not_a_fitted_knn_predictor':True,'OOF_dependency_caveat':'Neighbor OOF residuals share training labels/models and are not an independent noise estimate; negative correlation is not a recipe to invert or use neighbors at deployment.'})
l1=d.read(_repo_path(R, 'L1.json'));l2=d.read(_repo_path(R, 'L2.json'));l3=d.read(_repo_path(R, 'L3.json'));l4=d.read(_repo_path(R, 'L4.json'));ex=l1['example216']
summary={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':['L1','L2','L3','L4'],'best021_preserved':True,'best021_sha256':d.sha(_repo_path(P, 'artifacts/submission-candidate-021.csv')),'new_training_fits':0,'new_candidates':0,'installation_network_cost':0,'holdout_test_access':False,'L1_scope':{'studies':list(range(205,218)),'OOF_blocks':len(l1['coverage']),'reported_comparisons':len(l1['candidates']),'note':'78 comparisons include controls and duplicated role views, not78 independent models. Earlier001-204 not exhaustively reanalysed.'},'L1_example216':{k:ex[k] for k in ['A_gain','P','lambda_nonnegative_oracle','oracle_RMSE_gain']},'L1_example216_maxMSEreduction':ex['A_gain']**2/(4*ex['P']),'L1_repartition_optima':[{'split':c['split'],'A_gain':c['A_gain'],'P':c['P'],'lambda_optimum':c['lambda_nonnegative_oracle']} for c in l1['candidates'] if c['name']=='217:fixed_native'],'L2':{'knn':'Frozen L finalCLS1024 cosine; k1/5/10,12nearest pairs inspected','exact_D4_duplicates_in_12':sum(c['best_raw_D4']['exact'] for c in l2['pairs']),'automatic_exclusions':0,'image_audit':'Visual overview shows similar grain elongation/texture classes with distinct arrangements and darkphase/porelike spots. Not proof of generator families or mislabeled pairs. D4raw metrics cover12 only; no universal duplicate absence claim.'},'L3':{'top40_SSEshares':{s:v['deciles'][-1]['SSE_share'] for s,v in l3['results'].items()},'pooled_top40_internal_k5_fraction':l3['results']['pooled_meanSE']['top_decile_k5_internal_edge_fraction'],'random_fixed40_expectation':39/399,'signed_local_descriptive':signed},'L4':{'TV_four_configs_empty0':l4['empty0'],'Gaussian_sigma2_empty0':legacy['sigma2']['groups']['empty0'],'TV_border_fraction_quantiles':l4['feature_quantiles']['border_area_fraction'],'Gaussian_sigma2_border_fraction_quantiles':legacy['sigma2']['feature_quantiles']['border_area_fraction'],'actual_code_issue':legacy['active_zero_encoding'],'TV_diameter_instability_vs_pooled_SE_spearman':l4['descriptive_spearman']['diameter_range']['pooled_SE'],'TV_low_high_instability_quartile_RMSE':[l4['groups']['diameter_range_quartile1']['pooled_RMSE'],l4['groups']['diameter_range_quartile4']['pooled_RMSE']],'geometry_limit':'Ideal disk complete-object retention falls with size: radius4 .93848, radius64 .25. Real grains are noncircular, masks uncertain and border sizes truncated; no calibrated bias-correction formula is established.','visual_audit':'Eight largest border-fraction cases show large visible frame-touching structures excluded in orange, with within-region fragmentation and some porelike splits in green interior. Mask validity is not groundtruth-confirmed.','all_recomputed_configs':2400,'descriptor_max_error':0},'followup':{'preferred_next':'D3 measurement-validity/encoding control first, not broadCNN variants or a claimed major improvement.','evidence':'Observed sentinel0 enters a real active descriptor; border censoring affects many images and configuration sensitivity relates to error magnitude. No verified physicalgrain truth and weak signed-residual associations, so expected improvement is unknown.','bounded_proposal':'Before anymodel fit, lock twoarms: current compact57 vs same fixed features with an explicit sigma2-validity indicator and valid-scale-only HallPetch aggregation (diagnostic thirdarm flag-only separates newmissingness indicator from changedaggregation). No label-defined thresholds, image exclusions or outcome-derived corrections. Verify grain masks/geometry and exact feature differences first. If later authorized, identical folds/train-only preprocessing/nested existingKRR settings, report entire finalrecipe including necessary residual recalibration, not just frozen firstseed substitution; original then unchangedrepeat protocol, noPublic tuning.','not_executed':True,'D2_requirements':'Rawimage residualCNN is not established by magnitude clustering; signedneighbor correlations ~-.10 do not show smooth residualdirection. If considered, fixedsame anchorrecipe at every node, wholeouterV excluded from anchor/residualtarget/tuning. A clean pairedbranch builds anchor_A onA and residualCNN_B on yB-anchor_A(B); outerprediction must be anchor_A(V)+CNN_B(V), averagepairs, and deployment must reproduce identical pairedpolicy. Compare pairedanchor-only andbest021 separately. Do not add corrections trainedagainst a differentanchor tobest021. Any residualhyperparameterCV must rebuild anchors excluding itsinner validation, not naively split pooledcrossfit targets. Cache/anchor graph and compute must be audited before authorizingfits.','novelty_limit':'Existing113/115 already train linear/nonlinear feature residuals; residualarchitectureCNN is not the same as directresidualtargetCNN. Limitedsourceaudit found no explicit completed exactcurrent021 rawimage residualtarget study, but no exhaustive noveltyclaim.','resources':'Current diagnostics: TV400189.6s extraction (190.6swithsave), Gaussian40036s; noMPS used. Followup anchorrefit cost unmeasured; do not promise shorttotal or launchlarge graph automatically.'},'limitations':['Allsame400 with historical adaptive reuse; repartitions are not independent holdout.','Oraclelambda and posthocgroups cannot select deployed weights or establishgeneralization.','78 pairedcomparisons are correlated; no multiplecomparison significanceclaim.','No94percenterrorcoverage claim, noPublicnoise/SE estimate, no prizeguarantee.','No1percentfrequency or.98correlation terminationrule.','The failedslow legacy diagnostic wasinterrupted beforeoutput; v2bbox-equivalent implementationcompleted, oldsource/plan/logretained.']}
d.write(_repo_path(R, 'summary.json'),summary)
text=f'''L1–L4 진단 완료 — best021 보존 / 새 학습 0회

범위와 검증
개발400, 원본 및 두 기존 재분할만 사용했다. 새 학습·패키지·전송·결제·test/holdout 접근은 없다. 후보021 SHA256={summary['best021_sha256']}.
L1은 최근205–217의 저장 OOF16블록/30파일을 빠짐없이 확인하고 대조 포함78개 비교를 보고했다. 이는78개 독립 모델이 아니며001–204 전체를 재분석했다는 뜻이 아니다. ID/정답/fold/400행coverage/기존manifest·validation을 확인했다. 새 재학습으로 provenance를 재증명하지는 않았다. verification.json 및 L1-paired-losses.npz 참조.

L1 — 부호와 회고적 상한
r=y−기준예측, δ=후보−기준, A=2mean(rδ), P=mean(δ²), MSE변화=−λA+λ²P. 최초 전달된 A=−2mean(rδ)는 이 보고서 A의 음수다.
고정 범위λ∈[0,1], grid0/.25/.5/.75/1; λ*=clip(A/(2P),0,1)은 회고 진단만이다. 후보216: A={ex['A_gain']:.9f}, P={ex['P']:.9f}, λ*={ex['lambda_nonnegative_oracle']:.9f}, 최대MSE감소={summary['L1_example216_maxMSEreduction']:.9f}, 개발RMSE12.786200776 기준 개선={ex['oracle_RMSE_gain']:.9f}. Public숫자는 계산에 사용하지 않았다.
217 재분할A는 정렬항−1.227550, B는−0.569572로 둘 다 λ≥0 최적0이다. EMA·Huber도 원본의 양의 정렬이 재분할에서 유지되지 않았다. 비중을 줄여 구제하거나 작은 차이를 제출 근거로 삼지 않았다.

L2 — 특징 이웃과 원본 이미지
기존 frozenDINOv2L의 마지막CLS1024, perimage L2정규화 cosine; label이나학습metric없음. k1/5/10, 가장 가까운고유12쌍을 결과보기전 고정했다. D4완전중복0/12, native픽셀상관은 낮고 비슷한텍스처/늘어진형상 내에서 입자배치는 다르다. 이웃은 morphology유사성이며 생성family·같은물리입자·오라벨을 입증하지 않는다. 자동제외0. L2-neighbor-pairs.png와 전체수치L2.json 참조.

L3 — 큰 오차의 집중과 방향은 다르다
상위40장SSE비율: original50.0475%, repartition49.0697%/49.6739%, ID별3분할평균49.1920%. 모두 관측오차로 정한 설명집단이다.
고정label-blind k5에서 pooled오차상위40 내부이웃비율19.0%; 임의40개고정크기 기대9.774%, 단순재배치95%범위약5.0–15.5%. 선택편향보정/유의성검정/일반화근거가 아니다.
오차크기는 모여있지만 signed residual과 이웃평균residual 상관은−0.102/−0.099/−0.116이다. 이를 노이즈하한이나 잔차CNN성공증거로 삼거나 부호반전이웃모델을 제안하지 않는다. OOF잔차끼리도 학습자료 의존성이 있다.

L4 — active두 추출기 구분
TV추출4설정×400에서empty0없음. 초기Gaussianσ1은0/400, σ2는3/400(TRAIN_000206/223/482). 유효한 내부입자가없다는sentinel0를 실제크기0로 해석하면안된다. compact_regression.py의HallPetch대리량은σ1/σ2를단순평균하므로3장에서는유효σ1값이절반이된다. 이코딩효과는확인했지만성능인과는아직검증안됐다.
경계제외면적 median: TV14.47%, Gaussianσ1 24.47%, σ2 33.63%. σ2빈3장은median99.29%가경계영역이다. 이3장의pooledRMSE22.644이지만표본3개이고사후그룹이다.
TV설정간직경range와pooledSE Spearman0.2874; range하위/상위사분위RMSE8.7953/16.6977. signed잔차와의단순상관은약−0.0244에불과해직접bias보정공식을정할근거는없다.
모든400장×6설정=2400분할을재계산했고기존사용descriptor정확일치(최대오차0). 경계선·경계접촉·내부·small지지집합을검산했다. 실제물리정답은없다. 큰border8장시각검사에서도크고길쭉한경계구조제외와영역내분절이함께보였다.
이상적인disk중심이256사각내균일할때complete-object포함확률은((256−2r)/256)². r4는.93848,r64는.25다. 이는samplinggeometry단위진단일뿐,비원형입자/잘못된분할/시야밖중심을모르는실제자료에그대로역확률보정하지않는다. 합성disk는학습자료로사용안했다.

후속 제안(아직 미실행)
우선D3의측정유효성·sentinel처리대조를추천한다. 현행특징,유효성flag만추가,flag+valid-scale평균의작고고정된대조로코딩과추가정보효과를구분할수있다. 단0.75%빈도만수정해큰개선을기대한다는뜻은아니다. morphology정답없는border편향공식이나tail확장부터적용하지않는다. 추후학습승인시fold내전처리/기존nested회귀를유지하고,실제전체파이프라인영향을평가해야한다.
D2직접잔차CNN은추가증거가부족하다. 시행한다면anchor와residualCNN의정보집합을명확히고정해야한다. anchor_A로만든 yB−anchor_A(B)를학습한CNN_B는anchor_A(V)에더하는동일pairedpolicy로검증·배포한다. 다른anchor잔차를best021에그대로더하지않는다. outerV는모든anchor/잔차target/tuning에서제외하고,inner tuning도그innerV를본anchor를재사용하지않는다. pairedanchor-only대조와best021대조를따로보고해야한다. 이미113/115의특징잔차학습이있으므로'최초잔차학습'으로주장하지않는다.

실행기록
TV재측정190.64초, 수정후Gaussian400재측정약36초. 최초legacy진단에서regionprops객체동등비교병목을발견해해당프로세스만중단했다(exit130). 이전script/plan/log를보존하고동일bbox조건의v2로수정하여한번완료했다. 모델fit0회,최고모델변경0. 새훈련/후보파일/제출은없다.
같은400의누적선택편향은그대로남는다. '오차94%해결',PublicSE.03/노이즈하한,수상보장은확인된사실이아니다.
'''
with (_repo_path(R, 'REPORT.txt')).open('x') as f:f.write(text)
files={str(p.relative_to(P)):d.sha(p) for p in R.iterdir() if p.is_file() and p.name!='MANIFEST.json'};d.write(_repo_path(R, 'MANIFEST.json'),{'files':files,'protected_originals':d.read(_repo_path(R, 'plan.json'))['protected'],'final_summary_sha256':d.sha(_repo_path(R, 'summary.json'))});d.protect();print(json.dumps({'complete':True,'summary':str(_repo_path(R, 'summary.json')),'report':str(_repo_path(R, 'REPORT.txt')),'legacy_cache_error':lm['cached_descriptor_maxerror'],'new_training_fits':0},ensure_ascii=False))
