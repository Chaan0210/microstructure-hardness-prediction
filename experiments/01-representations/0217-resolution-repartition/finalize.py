from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();s=c.read(R/'summary.json');validation=c.read(R/'validation.json');info=c.read(R/'training-complete.json')['splits'];boot=c.load(R/'bootstrap.npz');err=c.load(R/'comparison-squared-errors.npz');ids=err['ids'];draws=boot['draws'];checks=[];train={};diagn=c.read(R/'complementarity-diagnostics.json');oldsummary=c.read(P/'experiments/resolution-216/summary.json')
with c.Monitor('final-audit',600):
 np.testing.assert_array_equal(draws,np.random.default_rng(202610256).integers(0,400,size=(10000,400)))
 for label,parts in [('individual_partitions',['original']+c.SPLITS),('ID_cluster_aggregates',['two_repartitions','all_three_partitions'])]:
  for part in parts:
   for name in ['standalone','fixed_ensemble','versus_best021']:
    tag=part+'_'+name;ae=err[tag+'_A'];be=err[tag+'_B'];A=ae.mean(0) if ae.ndim==2 else ae;B=be.mean(0) if be.ndim==2 else be;q=s[label][part][name];direct=np.array([np.sqrt(ae[...,ix].mean())-np.sqrt(be[...,ix].mean()) for ix in draws]);np.testing.assert_allclose(direct,boot[tag],rtol=0,atol=1e-12);np.testing.assert_allclose(np.quantile(direct,[.025,.975]),q['conditional_CI95'],rtol=0,atol=1e-12);assert abs(np.sqrt(ae.mean())-np.sqrt(be.mean())-q['gain'])<1e-12;top=np.argsort(A-B,kind='stable')[-5:];assert ids[top].tolist()==q['top5_removed_IDs'];keep=np.ones(400,bool);keep[top]=False;assert abs(np.sqrt(ae[...,keep].mean())-np.sqrt(be[...,keep].mean())-q['top5_removed_gain'])<1e-12;keep=~np.isin(ids,oldsummary['comparisons'][name]['top5_removed_IDs']);assert abs(np.sqrt(ae[...,keep].mean())-np.sqrt(be[...,keep].mean())-q['fixed216_top5_excluded_gain'])<1e-12;checks.append({'comparison':tag,'all10000_ID_cluster_draws_direct_verified':True,'top5_all_repetitions_excluded':True})
 for split in c.SPLITS:
  z=c.load(R/split/'OOF.npz');ref=c.load(R/split/'frozen-reference.npz');ep=c.load(R/split/'ensemble-OOF.npz');rows=c.read(R/split/'ID-residuals.json');np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(ep['fixed_native'],ref['best021']+ref['coefficient']*(z['native']-ref['old_first_CNN']));train[split]={}
  if info[split]['reused_controls']:np.testing.assert_array_equal(ep['fixed_control'],ref['best021'])
  for i,row in enumerate(rows):
   assert row['ID']==ids[i]
   for name,a,b in [('standalone','control','native'),('fixed_ensemble','fixed_control','fixed_native'),('versus_best021','best021','fixed_native')]:assert abs(row[name+'_SEgain']-((z['y'][i]-ep[a][i])**2-(z['y'][i]-ep[b][i])**2))<1e-12
  for arm in ['control','native']:
   ses=[];grad=[];times=[]
   for k in range(5):
    folder=P/info[split]['folders'][f'{arm}-{k}'];p=c.load(folder/'predictions.npz');st=c.load(folder/'streams.npz');f=c.read(folder/'fit.json');ses.extend((z['y'][p['train_indices']]-p['train_predictions'])**2);grad.extend(st['values'][:,1]);times.append(f['seconds']);assert f['updates']==800
   train[split][arm]={'repeated1600_train_RMSE':float(np.sqrt(np.mean(ses))),'gradient_clip_fraction':float(np.mean(np.array(grad)>5)),'fit_seconds':times}
  # independently recompute fixed substitution MSE identity and correlation values
  a=z['control'];b=z['native'];y=z['y'];delta=ep['fixed_native']-ep['fixed_control'];res=ep['fixed_control']-y;stored=diagn['partitions'][split]['fixed_MSE_gain_decomposition'];alignment=-2*np.dot(res,delta)/400;penalty=np.dot(delta,delta)/400;assert abs(alignment-stored['alignment_minus2mean_r_delta'])<1e-12;assert abs(penalty-stored['delta_squared_penalty'])<1e-12;assert abs(alignment-penalty-stored['net_MSEgain'])<1e-12
  original=P/info[split]['folders']['native-0'];poison=R/split/'poison-fold0-native';e.equal_tree(torch.load(_repo_io(original/'final-live-and-optimizer.pt'),map_location='cpu',weights_only=True),torch.load(_repo_io(poison/'final-live-and-optimizer.pt'),map_location='cpu',weights_only=True))
 assert len(validation['checks'])==20;oldvalidation=c.read(P/'experiments/huber-repartition-212/validation.json');assert oldvalidation['passed'];assert sum(p['arm']=='control' for p in oldvalidation['poison_replays'])==2
 fits=list(R.glob('*/*/fit.json'));expected=sum(7 if i['reused_controls'] else 12 for i in info.values());assert len(fits)==expected;updates=sum(c.read(p)['updates'] for p in fits);assert updates==800*expected;assert c.read(R/'probe.json')['completed_updates']==8;assert not list(R.glob('*-failure.json')) and not list(R.glob('*-STOP.json')) and not list(R.rglob('*.csv'))
 # Original216 numerical and decision evidence remains exact and no originalfit is rerun.
 assert s['original216_ensemble_further_review_eligible_preserved']==oldsummary['ensemble_further_review_eligible']==False
 c.check()
resources=[c.read(p) for p in R.glob('*-resources.json')];samples=[x for r in resources for x in r['trace']];mem=max(x['conservative_RSS_plus_MPS'] for x in samples);swap=max(max(x['swap']-r['initial_swap'] for x in r['trace']) for r in resources);assert mem<6e9 and swap==0
besthash=c.sha(P/'artifacts/submission-candidate-021.csv');assert besthash=='7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b'
audit={'passed':True,'new_full_fits':expected,'candidate_fits':10,'control_replay_fits':2,'candidate_poison_fits':2,'fallback_extra_fits':expected-14,'full_fit_updates':updates,'discarded_probe_updates':8,'total_updates':updates+8,'failed_or_interrupted_fits':0,'reused_controls':5*sum(i['reused_controls'] for i in info.values()),'historical_controls_hashes_OOF_normalization_and_full_replay_exact':all(i['reused_controls'] for i in info.values()),'model_reload_checks':20,'CPU_max_prediction_error':validation['max_CPU_prediction_error'],'MPS_max_prediction_error':validation['max_MPS_prediction_error'],'poison_replay_optimizer_state_stream_OOF_exact':True,'all10000_direct_cluster_bootstrap_checks':checks,'216_all_files_and_decisions_preserved':True,'protected_existing_files':len(c.read(R/'preservation.json')),'best021_SHA256':besthash,'peak_RSS_plus_MPS_bytes':mem,'swap_growth_bytes':swap,'train_diagnostics':train,'rawdata_reads':0,'outside217_writes':0,'new_install_download_test_holdout_full500_CSV_submission':False,'downstream_refits':0,'cost':0}
c.write(R/'final-audit.json',audit)
rows=[]
for part in ['original']+c.SPLITS+['two_repartitions','all_three_partitions']:
 qs=s['individual_partitions'].get(part,s['ID_cluster_aggregates'].get(part))
 for name,label in [('standalone','CNN단독'),('fixed_ensemble','고정첫seed치환')]:
  q=qs[name];lo,hi=q['conditional_CI95'];rows.append(f"| {part} | {label} | {q['baseline_RMSE']:.6f}→{q['candidate_RMSE']:.6f} | {q['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {q['top5_removed_gain']:+.6f} | {str(q.get('positive_folds','—'))} |")
report=f"""# 217 — 두 재분할에서 TinyCNN128/256 비교

**{s['decision']} best021을 유지한다. {s['downstream_assessment']} 후단 재학습과 제출 승격은 실행하지 않았다.**

사용자가 기존 두 분할의 추가 비교·결과 저장 질문에 ‘저장해’라고 승인한 범위로 실행했다. 216에서 단독 CNN은 개선됐지만 고정 첫 seed 치환 효과는 없었다는 원래 결론을 보존했다.

| 분할/합산 | 비교 | RMSE 128→256 | 이득 | 조건부 95% CI | 자체 top5 제외 이득 | 개선 fold/5 |
|---|---|---|---:|---|---:|---:|
"""+'\n'.join(rows)+f"""

이득은 대조 RMSE−후보 RMSE다. 대조 재사용은 {[info[t]['reused_controls'] for t in c.SPLITS]}이며, 재사용 시 고정 대조는 각 분할의 best021과 exact 일치한다. 따라서 대응 치환 효과와 best021 대비 효과가 같지만 summary.json에 각각 저장했다. 모든 fold 이득·제외 ID와 원본216 top5 고정 제외 감도도 이 파일에 있다.

## 실험과 통계

기존 20261031/20261101의 각각 5fold, 동일 개발400을 사용했다. 각 fold는 train320/validation80이다. 216과 동일한 TinyCNN(91,905 parameters), BOX128 대 native256, 80 epoch, batch32, seed20261003+fold, D4, MSE, AdamW(lr0.002, decay0.01), cosine80(eta_min0.0001), clip5, 800 updates 및 epoch76–80 state 동일 평균을 유지했다. 각 군 입력의 해당 train 픽셀 전체 mean/population std와 같은 train 타깃 통계만 사용했다. 추가 튜닝·증강·seed 탐색은 없다.

**동일 업데이트 예산이며 동일 연산량·시간은 아니다.** 원본 시야 기준 receptive field도 달라진다. native256 캐시는 216 해시와 일치하며 TinyCNN 내부에 128 강제 리사이즈가 없다. 기존128/80 대256/40의 예산이 달랐다는 기록을 소급 변경하지 않았다.

두 새 재분할 합산이 주된 안정성 결과다. 원본까지 포함한 3분할 합산은 선택에 사용된216을 포함하는 기술적 요약이다. 합산은 ID별 반복 분할의 제곱오차 평균을 전체 ID에 평균한 뒤 제곱근을 취한다. 예측 평균 앙상블이 아니다. 400 ID를 10,000회(seed202610256) bootstrap하고, 뽑힌 ID의 모든 분할 관측을 함께 유지했다. 800/1200개의 독립 표본으로 계산하지 않았다. top5도 ID별 평균 SE 이득을 stable 정렬한 뒤 해당 ID의 모든 반복 관측을 제거한다.

각 새 분할의 단독 기준은 이득≥0.02, 개선 fold≥3/5, CI 하한>0, top5 제외 이득>0 및 기술검증이다. 두 새 분할 합산은 이득>0, CI 하한>0, top5 제외>0이다. 고정 치환 기준은 별도로 평가했다. 결과: `{s['stability_passed']}`. 원본 포함 합산이 좋아도216의 앙상블 효과 없음이나 새 분할의 기준 미달을 취소하지 않는다.

## 단독 개선과 보완성

고정 치환은 `Q_arm=P021+w*(C_arm−C_old_first)`, `w=0.081*기존 split/fold calibrator slope`다. 나머지 두 seed·보정·잔차 모델은 그대로 둔다. ‘첫 seed를 뺀 나머지’는 `(Q_A−w*C_A)/(1−w)`로 정의한 대수적 잔여 예측이며 CNN 없이 독립 학습한 모델은 아니다.

| 분할 | CNN128/256 잔차상관 | CNN128↔나머지 | CNN256↔나머지 | 정렬 이득 | 변화제곱 페널티 | 순 MSE 이득 |
|---|---:|---:|---:|---:|---:|---:|
"""
for split,d in diagn['partitions'].items():
 x=d['fixed_MSE_gain_decomposition'];report+=f"| {split} | {d['CNN128_vs256_residual_correlation']:.6f} | {d['CNN128_vs_removed_firstseed_remainder_residual_correlation']:.6f} | {d['CNN256_vs_removed_firstseed_remainder_residual_correlation']:.6f} | {x['alignment_minus2mean_r_delta']:+.6f} | {x['delta_squared_penalty']:.6f} | {x['net_MSEgain']:+.6f} |\n"
report+=f"""
고정 치환의 MSE 이득은 `−2*mean(r_A*delta)−mean(delta²)`와 exact 일치한다. `r_A=Q_A−y`, `delta=Q_B−Q_A`다. 단독 CNN 오차가 감소해도 기존 앙상블 잔차를 줄이는 방향의 수정인지는 별개다. 정렬 이득이 변화제곱 페널티를 넘어야 순이득이 난다. 상관은 평균을 제거하고 정렬 항은 bias도 포함하므로 둘을 동일시하지 않는다. 다른 두 CNN seed 평균과의 상관, 단독 MSE 분해, 예측 변화 RMS는 complementarity-diagnostics.json에 저장했다. 진단용 계수나 가중치는 맞추지 않았다.

**후단 판단:** {s['downstream_assessment']} 이는 이번 범위를 넘어서는 별도 판단이며 실제 후단 재학습은 0회다.

## 실제 실행과 검증

- 새 완전 fit **{expected}회·{updates} updates**: 후보10, 대조 전체 재현2, 후보 NaN 교란 재현2, fallback 추가{expected-14}회. 버린 native256 smoke8 updates를 더해 총 **{updates+8} updates**다. 기존 대조{audit['reused_controls']}개를 재사용했으며 실패·중단0회다.
- 대조 fold0 두 개의 800 update order/D4/LR/loss/gradient norm/input hash/CPU·MPS RNG, epoch76–80/평균 state/OOF/최종 optimizer·scheduler를 exact 재현했다. 나머지 대조도 hash·ID·정규화·재추론을 확인했다.
- 양 군의 초기 state 및800 order/D4/LR/RNG 일치, 모델20개 마지막5 state 재평균·CPU/MPS 재로드, 전 fold train 통계의 validation 교란 불변성, 두 native fold0의 NaN 라벨 교란 state/stream/OOF/optimizer exact를 통과했다. 기존 대조 NaN 교란은212 증거와 이번 전체 재현으로 확인했다.
- CPU 최대 재추론 오차 {validation['max_CPU_prediction_error']:.9f}(허용0.001), MPS 오차 {validation['max_MPS_prediction_error']:.9f}. 15개 비교의 bootstrap 각10,000회를 직접 인덱싱으로 독립 재계산했다. top5/고정216 top5 제외와 새분할800개 ID별 SE 이득도 검산했다.
- 관측 최대 RSS+MPS **{mem/1e9:.3f}GB**, swap 증가0, 기존 uv CPU2threads/MPS, 비용0이다.

보호 기존파일{audit['protected_existing_files']}개·216 전파일·best021 해시를 보존했다. best021 SHA256은 `{besthash}`다. 217은 원시 data 파일을 열지 않았고 개발400 캐시만 사용했다. 새 설치/다운로드/네트워크/test/holdout/전체500/CSV 생성수정/제출/스킬·diagram 수정은 없다. 차단사항 없다.

계획의 상속된 `fixed_ensemble.coef_by_fold/calibrators`는 원본216 메타데이터다. 217의 실제 고정계수는 시작부터 최상위 `calibrators[split][fold]`와 common.lineage를 사용했고, preflight에서212 고정 계보와 exact 확인했다. 이 구분은 성능 확인 전에 plan-interpretation-correction.json에 기록했으며 잠긴 원래 계획·계산 코드·결과 수치는 바꾸지 않았다.

같은400의 반복·적응적 사용과 누적 선택편향이 남는다. CI는 고정 fit에 조건부이며 seed·재학습·선택 불확실성을 보정하지 않는다. 재분할은 독립 자료 검증이 아니다. 단독 개선으로 제출 승격하지 않았으며, 전체 후단 재학습 효과는 이번 결과로 알 수 없다.
"""
with (R/'REPORT.md').open('x') as f:f.write(report)
c.write(R/'completion.json',{'study':217,'completed':True,'decision':s['decision'],'downstream_assessment':s['downstream_assessment'],'new_full_fits':expected,'full_updates':updates,'probe_updates':8,'total_updates':updates+8,'reused_controls':audit['reused_controls'],'best021_preserved':True,'original216_preserved':True,'downstream_retraining':False,'report':'experiments/resolution-repartition-217/REPORT.md','blockers':[]})
c.check();c.write(R/'artifact-manifest.json',{str(p.relative_to(P)):c.sha(p) for p in R.rglob('*') if p.is_file() and p.name!='finalize.log'});c.emit({k:v for k,v in audit.items() if k not in ['all10000_direct_cluster_bootstrap_checks','train_diagnostics']})
