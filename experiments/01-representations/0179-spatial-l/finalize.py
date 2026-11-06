"""179 independent output reconstruction and durable result, no model selection changes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json,hashlib
from datetime import datetime,timezone
import numpy as np
import runtime as f
R=f.R;P=f.P

def main():
 plan=f.check();original=f.read(R/'original/report.json');splits=['original','20261031','20261101'] if original['advance'] else ['original'];reports={};errs={k:[] for k in ['018','021','primary']};allgroups=None
 for split in splits:
  rep=f.read(R/split/'report.json');assert rep['passed_validation'] and rep['plan_sha256']==f.sha(R/'plan.json');assert f.sha(R/split/'predictions.npz')==rep['predictions_sha256']
  with np.load(_repo_io(R/split/'predictions.npz'),allow_pickle=False) as z:
   assert z['ids'].tolist()==plan['IDs'];np.testing.assert_array_equal(.9*z['L_swap018']+.1*z['globalL'],z['primary']);np.testing.assert_array_equal(.9*z['018']+.1*z['globalL'],z['021']);np.testing.assert_allclose(z['primary']-z['021'],.9*(z['L_corrections']-z['B_corrections']).mean(0),atol=1e-12,rtol=0)
   for k in errs:errs[k].append((z['y']-z[k])**2)
   for name,hh in rep['block_report_hashes'].items():
    file=R/split/name;assert f.sha(file)==hh;br=f.read(file);assert f.sha(file.parent/'predictions.npz')==br['predictions_sha256']
    for m,h in br['models_sha256'].items():assert f.sha(file.parent/m)==h
   allgroups=z['g'].copy()
  reports[split]=rep
 errs={k:np.stack(v) for k,v in errs.items()};pooled={k:float(np.sqrt(v.mean())) for k,v in errs.items()};unique,inv=np.unique(allgroups,return_inverse=True);rng=np.random.default_rng(202710179);boot={k:[] for k in ['018','021']}
 for _ in range(3000):
  counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inv]
  for k in boot:boot[k].append(float(np.sqrt(np.sum(errs[k]*counts)/(len(splits)*counts.sum()))-np.sqrt(np.sum(errs['primary']*counts)/(len(splits)*counts.sum()))))
 propose=len(splits)==3 and all(r['RMSE']['primary']<min(r['RMSE']['018'],r['RMSE']['021']) for r in reports.values());resources={}
 for file in R.glob('*-resources.json'):
  obj=f.read(file);z=obj['trace'];resources[file.name]={'seconds':obj['seconds'],'peak_RSS':max(x['peak_RSS'] for x in z),'max_MPS':max(x['MPS_driver'] for x in z),'max_conservative_RSS_plus_MPS':max(x['conservative_RSS_plus_MPS'] for x in z),'swap_growth':max(x['swap_bytes'] for x in z)-obj['initial_swap_bytes'],'min_free_percent':min(x['system_memory_free_percent'] for x in z)}
 rec={'study':179,'UTC':datetime.now(timezone.utc).isoformat(),'outcome':'validated_proposal_only' if propose else 'rejected_no_new_candidate','propose_new_candidate':propose,'original_gate_passed':original['advance'],'partitions_run':splits,'RMSE':{s:r['RMSE'] for s,r in reports.items()},'comparisons':{s:r['comparison'] for s,r in reports.items()},'correlations':{s:r['correlations'] for s,r in reports.items()},'pooled_RMSE':pooled,'conditional_group_bootstrap_gain_CI95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},'feature_validation':f.read(R/'feature-validation.json'),'validation':{s:{k:r[k] for k in ['passed_validation','main_model_objects','main_nonnull_fits','outer0_poison_model_objects','inner_heldout_poison_refits','independent_max_error','all_B_inner_full_weights_predictions_exact','018_and021_control_exact','all_reload_exact','all_train_local_scaling_and_group_scopes_verified','outer_fold0_all3_Lblocks_poison_exact','upstream_neural_anchors_reused_with_exclusion_certificates','seconds']} for s,r in reports.items()},'resource_summaries':resources,'source_SHA256':{n:f.sha(R/n) for n in ['plan.json','code-lock.json','readout-code-lock.json','feature-validation.json','finalize.py']},'protected_files_unchanged':plan['protected'],'scope':'Development400, original5fold and gatedexistingresplits only. No holdout/test/full500/CSV/upload/install/download/cost. Existing raw400/Lweights/Btokens/physics/anchors reused as declared.','limitations':['Repeated same400 adaptive development; bootstrap is conditional, not independent generalization evidence.','Random projection rule/rank is matched; B andLchannelcoordinates are different.','Spatial graph energy measures contextual token differences, not proven physical grain boundaries.','No concatenation or ratio/layer/rank/seed tuning; one replacement contrast only.','021 browser submission/result remains parent-managed; no browser query here.'],'remaining_blockers':[],'cost':0};dest=P/'reports/spatial-L-179-summary.json';f.write(dest,rec)
 lines=['# Study179: DINO-L spatial replacement', '',plan['hypothesis'],'',plan['statistic'],'',plan['projection'],'',plan['recipe'],'',plan['primary'],'','## Result','',f'Outcome: {rec["outcome"]}; partitions: {splits}.','',json.dumps(rec['RMSE'],indent=2),'','## Validation and reuse','', 'All B andL residual fits use the same honest blocks and inner folds. Existing certified upstream anchors and015base are reused; no training against global OOF residuals. B inner predictions/full predictions/512-bootstrap weights and018/021 controls must match exactly. All main residual objects are reloaded and checked with independent scaling and dual Ridge/eigen RBF calculations. All three fold0 L blocks are refitted after outer labels are poisoned; existing upstream exclusion certificates are reused. The scope of this replay is explicit, not a complete upstream neural retraining.','', 'Feature checks include three cached B samples against the old projection/statistic, independent2520edge enumeration, three L CPU/MPS samples, token-grid D4 invariance, spatial shuffling with unchanged global mean/std, exact400 globalL final-layer summaries and image hashes. Token-grid invariance does not imply whole-image DINO invariance.','', 'No concatenation arm, alternative rank/layer/seed/ratio, holdout/test access, full500 fit, new CSV, install, download, cost or upload. Existing018/021 preserved. Scores remain adaptive development evidence.','', '## Commands','', 'Run from the project root using the existing .venv. Completed output paths are exclusive; preserve them. The original setup and code locks are recorded before extraction/fitting.','', '```sh','.venv/bin/python -B experiments/spatial-L-179/prepare.py','.venv/bin/python -B experiments/spatial-L-179/extract.py','.venv/bin/python -B experiments/spatial-L-179/readout.py original',*(['.venv/bin/python -B experiments/spatial-L-179/readout.py 20261031','.venv/bin/python -B experiments/spatial-L-179/readout.py 20261101'] if len(splits)==3 else []),'.venv/bin/python -B experiments/spatial-L-179/finalize.py','```','', 'Evidence: plan.json, code locks, feature-validation.json, each partition report.json/predictions.npz and all block models/reports, resource traces; project reports/spatial-L-179-summary.json.']
 with (R/'README.md').open('x') as fp:fp.write('\n'.join(lines)+'\n')
 backup=P/'reports/README-before-179.md'
 with backup.open('xb') as fp:fp.write((P/'README.md').read_bytes())
 with (P/'README.md').open('a') as fp:
  fp.write('\n\n## 179 DINO-L 공간 특징 교체 대조 완료\n\n')
  fp.write('018의 B 마지막층 이웃patch 공동변화528을 동일 정의·QR32·seed·정규화·nested 잔차학습의 L528로 교체했습니다. 물리57 및 upstream anchor는 유지하고,021 globalL10%도 고정했습니다. 단순추가·비율탐색은 하지 않았습니다.\n\n')
  fp.write(f'결과: {rec["outcome"]}, 실행분할={splits}. 원본5fold RMSE018={original["RMSE"]["018"]:.12f}, 고정021={original["RMSE"]["021"]:.12f}, 교체primary={original["RMSE"]["primary"]:.12f}. 재분할 gate={original["advance"]}. 상세실패·상관·검증·자원 기록 `reports/spatial-L-179-summary.json`, 재현자료 `experiments/spatial-L-179/README.md`.\n\n')
  fp.write('개발400 새 L공간 추출·B대조 정확재현·전체잔차모델 독립수식/재로딩·fold0 3블록 라벨제외 검증 통과. 기존018/021 및 보호파일 보존. holdout/test/전체500/새CSV/업로드/설치/다운로드/비용 없음.021제출과결과는 부모관리이며 여기서 조회하지 않았습니다.\n')
 print(json.dumps({'outcome':rec['outcome'],'RMSE':rec['RMSE'],'summary':str(dest),'SHA256':f.sha(dest),'resources':resources},indent=2))
if __name__=='__main__':main()
