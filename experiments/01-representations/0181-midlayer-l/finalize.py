from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
from datetime import datetime,timezone
import numpy as np,torch
import runtime as f
from readout import load,predict,check
R=f.R;P=f.P

def main():
 plan=check();first=f.read(R/'original/report.json');splits=['original','20261031','20261101'] if first['advance'] else ['original'];features=load(R/'features.npz');reports={};count=0;errors={k:[] for k in ['021','LAST_blend','MID_blend']}
 for split in splits:
  rr=f.read(R/split/'report.json');assert rr['passed_validation'];assert f.sha(R/split/'predictions.npz')==rr['predictions_sha256'];z=load(R/split/'predictions.npz');assert z['ids'].tolist()==plan['IDs'];assert rr['plan_sha256']==f.sha(R/'plan.json')
  for arm in ['MID','LAST']:
   np.testing.assert_array_equal(z[arm],z[arm+'_seeds'].mean(0));np.testing.assert_array_equal(z[arm+'_blend'],.9*z['018']+.1*z[arm])
  for name,hh in rr['fold_report_hashes'].items():
   path=R/split/name;assert f.sha(path)==hh;br=f.read(path);T=np.array(br['T']);V=np.array(br['V']);arm=br['arm'];assert not set(z['g'][T])&set(z['g'][V])
   for j,(seed,rec) in enumerate(zip([20261128,20261205,20261219],br['heads'])):
    file=path.parent/f'{arm}-{seed}.pt';assert f.sha(file)==rec['checkpoint_sha256'];obj=torch.load(_repo_io(file),map_location='cpu',weights_only=True);np.testing.assert_array_equal(obj['T'].numpy(),T);np.testing.assert_array_equal(obj['V'].numpy(),V);assert obj['ym']==float(z['y'][T].mean()) and obj['ys']==float(z['y'][T].std());np.testing.assert_array_equal(predict(obj,features[arm][V]),z[arm+'_seeds'][j,V]);count+=1
  for rec in rr['poison_replays']:
   file=R/split/f'{rec["arm"]}-poison-{rec["seed"]}.pt';assert f.sha(file)==rec['checkpoint_sha256']
  for k in errors:errors[k].append((z['y']-z[k])**2)
  reports[split]=rr
 eligible=len(splits)==3 and all(r['RMSE']['MID_blend']<min(r['RMSE']['021'],r['RMSE']['LAST_blend']) for r in reports.values());resources={}
 for path in R.glob('*-resources.json'):
  a=f.read(path);t=a['trace'];resources[path.name]={'seconds':a['seconds'],'peak_RSS_bytes':max(v['peak_RSS'] for v in t),'peak_MPS_bytes':max(v['MPS_driver'] for v in t),'max_conservative_RSS_plus_MPS':max(v['conservative_RSS_plus_MPS'] for v in t),'swap_growth':max(v['swap_bytes'] for v in t)-a['initial_swap_bytes'],'min_memory_free_percent':min(v['system_memory_free_percent'] for v in t)}
 rec={'study':181,'UTC':datetime.now(timezone.utc).isoformat(),'outcome':'validated_proposal_only' if eligible else 'rejected_close_L_representation_variants','propose_candidate':eligible,'L_representation_variants_closed':not eligible,'current_best':'021','duplicate_audit':f.read(R/'duplicate-audit.json'),'plan':{k:plan[k] for k in ['layers_zero_based','features','head','formula','gate']},'partitions_run':splits,'original_gate':first['advance'],'RMSE':{k:v['RMSE'] for k,v in reports.items()},'comparisons':{k:v['comparisons'] for k,v in reports.items()},'correlations':{k:v['correlations'] for k,v in reports.items()},'conditional_group_bootstrap_gain_CI95':{k:v['conditional_group_bootstrap_gain_CI95'] for k,v in reports.items()},'historical_subgroups_diagnostic_only':{k:v['historical_subgroups_diagnostic_only'] for k,v in reports.items()},'pooled_RMSE':{k:float(np.sqrt(np.mean(v))) for k,v in errors.items()},'feature_validation':f.read(R/'feature-validation.json'),'validation':{k:{n:v[n] for n in ['main_head_fits','wholefold0_poison_refits','poison_replays','equal_predictive_parameters_perhead','paired_initialization_order_exact','same_device_reload_exact','manual_forward_max_error','CPU_MPS_max_error','all_train_scaler_target_scopes_verified','train_RMSE_byarm','seconds']} for k,v in reports.items()},'second_process_heads_reloaded':count,'second_process_prediction_error':0.,'resources':resources,'protected_files_unchanged':plan['protected'],'evidence_SHA256':{n:f.sha(R/n) for n in ['plan.json','duplicate-audit.json','code-lock.json','readout-code-lock.json','feature-validation.json','finalize.py']},'scope':'Freshdevelopment400layerfeatures only; noholdout/test/full500/newCSV/upload/install/download/payment. ExistingofficialLweightandtorchonly.','limitations':['L12information alreadyin021joint12/18/24features; thisisreadoutisolation,notnewphysicalinformation.','S044midpointRidgefailedblend;181doesnotrerunS orlayersearch.','Tokenrepresentations arenotproventexture/physicalgrainmeasurements.','Same400adaptivehistory; conditionalbootstrap notindependentvalidation.','Familyclosure is a researchdecision under authorizedscope, not proofallpossibleLarchitectures fail.'],'remaining_blockers':[],'cost':0};f.write(P/'reports/midlayer-L-181-summary.json',rec)
 if not eligible:f.write(P/'reports/L-representation-closure-181.json',{'closed':True,'best':'021','reason':'179spatialreplacement,180jointPCAreadout and181midpointisolationfailedtheirlockedgates. No furtherLlayer/feature/readout variants orrescueweights are scheduled.','studies':[179,180,181],'summary_sha256':f.sha(P/'reports/midlayer-L-181-summary.json'),'preserve_globalL021':True,'UTC':rec['UTC']})
 with (R/'README.md').open('x') as fp:
  fp.write('#181 L midpoint versus final readout\n\n'+rec['outcome']+'; currentbest021.\n\n')
  fp.write('044already comparedSblock6/12 withsameRidge; midblend13.5572628869 wasworsefinal13.5224522318.177usedL12/18/24combined,notmidpoint-onlyMLP. L12 informationalreadyexistsin021. Thisoneexperiment isolatesL12 fromL24 withsame3072CLS/patchmean/stdfeatures andsameexisting64hidden3seedMLP. No layer sweep.\n\n')
  for k in ['head','formula','gate']:fp.write(plan[k]+'\n\n')
  fp.write('Fresh400rawimagefeaturesmatched177slicesbitexact. CPU/MPS3smoke andbackbonestate/hashchecks passed. Everyhead saved/reloaded,independentGELUforward/CPU-MPS/train-scaler statistics validated. Fold0botharms3seedtrainingreplayedwithoutside-trainfeatures/labelsperturbed; evaluateoriginalqueriesandrequireexactweights/statistics/predictions. Pre-fitvalidationreview sourceandlocks retained. Finalizer separatelyreplayedall30mainheads.\n\nResults:\n\n```json\n'+json.dumps(rec['RMSE'],indent=2)+'\n```\n\n')
  fp.write('Commands fromprojectroot (completedpaths areexclusive;preserveoutputs):\n\n```sh\n.venv/bin/python -B experiments/midlayer-L-181/prepare.py\n.venv/bin/python -B experiments/midlayer-L-181/extract.py\n.venv/bin/python -B experiments/midlayer-L-181/readout.py original\n')
  for split in splits[1:]:fp.write(f'.venv/bin/python -B experiments/midlayer-L-181/readout.py {split}\n')
  fp.write('.venv/bin/python -B experiments/midlayer-L-181/finalize.py\n```\n\nNo newCSV/full500/holdout/test/install/download/cost/upload. Featurecache,checkpoints,plan/codehashes,validation andresourcetraces retained. Summary:reports/midlayer-L-181-summary.json.\n')
 with (P/'reports/README-before-181.md').open('xb') as fp:fp.write((P/'README.md').read_bytes())
 with (P/'README.md').open('a') as fp:
  fp.write('\n\n##181 L중간12층 대 최종24층 동일readout 완료\n\n')
  fp.write(f'기존S044중간층Ridge실패와L177다층입력의차이를감사하고,층sweep없이L12/L24각3072차원에같은MLP64/3seed/100epochs를적용했습니다.021에12층정보가이미포함돼있으므로새정보추가가아닌readout분리실험입니다. 원본021={first["RMSE"]["021"]:.12f},최종층결합={first["RMSE"]["LAST_blend"]:.12f},중간층결합={first["RMSE"]["MID_blend"]:.12f}. 결과={rec["outcome"]},분할={splits}.\n\n')
  fp.write('개발400새추출은177캐시와정확일치,CPU/MPS·모든head재로딩·독립수식·fold0라벨/특징교란재학습검증통과.018/021과보호파일보존. holdout/test/전체500/새CSV/설치/다운로드/비용/업로드없음. 보고서 `reports/midlayer-L-181-summary.json`,재현 `experiments/midlayer-L-181/README.md`.\n')
  if not eligible:fp.write('\n179/180/181의고정대조실패에따라추가L표현변형을종료합니다.021의기존globalL성분은유지합니다. 종료기록 `reports/L-representation-closure-181.json`.\n')
 print(json.dumps({'outcome':rec['outcome'],'RMSE':rec['RMSE'],'resources':resources,'reloaded_heads':count,'summary_SHA256':f.sha(P/'reports/midlayer-L-181-summary.json')},indent=2))
if __name__=='__main__':main()
