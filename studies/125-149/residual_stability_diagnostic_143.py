"""Development-only current017 residual/selector stability diagnostics; no candidate fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import spatial_joint_140 as q
import numpy as np
q.torch.set_num_threads(2);q.check();e=q.e;a=q.a;records=[];partitions={};kinds=a.n.KINDS
for split in e.SPLITS:
 rr=e.read(q.OUT/split/'report.json');z=a.readnp(q.OUT/split/'predictions.npz',rr['predictions_sha256']);err=z['y']-z['adjacent'];base=np.full(400,np.nan);cor=np.full(400,np.nan)
 for k in range(5):
  root=q.OUT/split/f'adjacent-fold-{k}';fr=e.read(root/'report.json');fz=a.readnp(root/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];base[V]=fz['base'];cor[V]=fz['corrections'].mean(0)
  for j in range(3):
   path=root/f'block-{j}';r=e.read(path/'report.json');b=a.readnp(path/'predictions.npz',r['predictions_sha256']);loss=(b['inner_predictions']-b['residual'])**2;scores=loss.mean(1);order=np.argsort(scores,kind='stable');win,second=order[:2];delta=loss[second]-loss[win];se=float(delta.std(ddof=1)/np.sqrt(len(delta)));fold=np.empty(len(delta),int);lookup={int(i):ii for ii,i in enumerate(b['block_indices'])}
   for t,f in enumerate(r['splits']):fold[[lookup[int(i)] for i in f['validation_indices']]]=t
   leave=[kinds[int(np.argmin(loss[:,fold!=t].mean(1)))] for t in range(3)];E=b['residual'][None,:]-b['inner_predictions'];corr=np.corrcoef(E);records.append({'split':split,'outer':k,'block':j,'chosen':r['chosen'],'runnerup':kinds[second],'best_vs_second_MSE_gap':float(scores[second]-scores[win]),'paired_loss_SE_descriptive':se,'gap_over_SE':float((scores[second]-scores[win])/se) if se else None,'leave_one_inner_fold_out_choices':leave,'choice_changed':any(v!=r['chosen'] for v in leave),'linear_nonlinear_error_correlation_ridge01_rbf01':float(corr[2,4]),'linear_nonlinear_prediction_difference_RMS':float(np.sqrt(np.mean((b['inner_predictions'][2]-b['inner_predictions'][4])**2))),'residual_mean':float(b['residual'].mean())})
 np.testing.assert_allclose(base+cor,z['adjacent'],atol=1e-12,rtol=0);partitions[split]={'017_RMSE':float(np.sqrt(np.mean(err**2))),'remaining_residual_mean':float(err.mean()),'remaining_residual_SD':float(err.std()),'residual_correction_correlation':float(np.corrcoef(err,cor)[0,1]),'residual_raw_prediction_correlation':float(np.corrcoef(err,base)[0,1]),'base_MSE_minus017_MSE':float(np.mean((z['y']-base)**2)-np.mean(err**2))}
out={'study':'143 current017 residual and hardselector diagnostics','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Only existing400 development predictions and innerOOF residuals; no newfits/holdout/test; no candidate score or fittedblend.','partitions':partitions,'blocks':records,'summary':{'blocks':len(records),'choice_changes_when_one_inner_fold_omitted':sum(v['choice_changed'] for v in records),'gap_less_than_one_paired_loss_SE':sum(v['gap_over_SE'] is not None and v['gap_over_SE']<1 for v in records),'median_gap_over_SE':float(np.median([v['gap_over_SE'] for v in records if v['gap_over_SE'] is not None])),'median_ridge01_rbf01_error_correlation':float(np.median([v['linear_nonlinear_error_correlation_ridge01_rbf01'] for v in records])),'median_ridge01_rbf01_prediction_difference_RMS':float(np.median([v['linear_nonlinear_prediction_difference_RMS'] for v in records]))},'limits':'Foldomission descriptiveonly,notnewCV;pairedSE assumesIIDrows forscale illustration and isnotconfidence/effectiveindependentn. Reused400researchhistory; evidence motivatesstabilityhypothesis,notguaranteedgain.','source_sha256':e.sha(Path(__file__)),'cost':0};e.write(_repo_path(ROOT, 'reports/residual-stability-143-diagnostic.json'),out);e.emit({k:v for k,v in out.items() if k!='blocks'})
