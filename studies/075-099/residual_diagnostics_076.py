"""Post009 descriptive diagnostics; no new corrected-score calculation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_calibration_074 as base
import numpy as np
from scipy.stats import spearmanr
fv=base.fv;OUT=_repo_path(ROOT, 'artifacts/residual-diagnostics-076')

def load_outer(split):
 d=base.inputs(split);v=json.loads((_repo_path(ROOT, 'reports/nested-calibration-074-validation.json')).read_text());path=base.OUT/split/'development_predictions.npz';assert fv.sha(path)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);d['reference009']=z['primary'].copy()
 old=json.loads((_repo_path(ROOT, 'reports/dinob-complement-064-validation.json')).read_text());path=base.prior.OUT/split/'development_predictions.npz';assert fv.sha(path)==old['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];d['legacy']=z['reference003'].copy();d['Smean']=z['Smean'].copy();d['Bmean']=z['Bmean'].copy()
 d['Dmean']=(d['Smean']+d['Bmean'])/2;d['contrast']=d['Dmean']-d['legacy'];np.testing.assert_allclose(d['reference008'],.75*d['legacy']+.25*d['Dmean'],atol=1e-12,rtol=0);return d

def inner_matrix(d,split,k):
 tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);fd=base.OUT/split/f'fold-{k}';fr=json.loads((fd/'report.json').read_text());splitreport=json.loads((base.OUT/split/'report.json').read_text());assert fv.sha(fd/'report.json')==splitreport['fold_report_sha256'][f'fold-{k}/report.json'];assert fv.sha(fd/'calibration.npz')==fr['calibration_sha256'];A=np.full((len(tr),2),np.nan);components=np.full((len(tr),3),np.nan);seen=[]
 for j in range(3):
  cell=fd/f'inner-{j}';cr=json.loads((cell/'report.json').read_text());assert fv.sha(cell/'report.json')==fr['inner_report_sha256'][f'inner-{j}/report.json'];assert fv.sha(cell/'predictions.npz')==cr['files_sha256']['predictions.npz'];ti=np.array(cr['train_indices']);vi=np.array(cr['validation_indices']);assert not set(ti)&set(va) and not set(vi)&set(va) and set(ti)|set(vi)==set(tr) and not set(ti)&set(vi);assert not set(d['g'][ti])&set(d['g'][vi]);seen.extend(vi.tolist());pos=np.array([list(tr).index(i) for i in vi])
  with np.load(_repo_io(cell/'predictions.npz'),allow_pickle=False) as z:
   np.testing.assert_array_equal(z['train_indices'],ti);np.testing.assert_array_equal(z['validation_indices'],vi);legacy=.5*z['baseline']+.25*z['kernel']+.25*z['cnn'];D=(z['Smean']+z['Bmean'])/2;A[pos,0]=z['ensemble008'];A[pos,1]=D-legacy;components[pos]=np.column_stack([legacy,z['Smean'],z['Bmean']]);np.testing.assert_allclose(A[pos,0],.75*legacy+.25*D,atol=1e-12,rtol=0)
 assert sorted(seen)==tr.tolist() and np.isfinite(A).all()
 with np.load(_repo_io(fd/'calibration.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['inner_oof'],A[:,0]);np.testing.assert_array_equal(z['train_indices'],tr)
 return tr,va,A,components

def main():
 base.check();OUT.mkdir(exist_ok=False);pub=json.loads((_repo_path(ROOT, 'reports/candidate009-public-submission.json')).read_text());results={};matrices={}
 for split in ('original','20261031','20261101'):
  d=load_outer(split);y=d['y'];p=d['reference009'];err=y-p;q=(p-p.mean())/p.std();features={'prediction':p,'prediction_squared':q*q,'DINO_minus_003':d['contrast'],'B_minus_S':d['Bmean']-d['Smean'],'DINO_disagreement':abs(d['Bmean']-d['Smean']),'legacy_DINO_disagreement':abs(d['contrast'])};folds=[]
  for k in range(5):
   tr,va,A,C=inner_matrix(d,split,k);B=np.column_stack([np.ones(len(tr)),A[:,0]]);target_res=y[tr]-B@np.linalg.lstsq(B,y[tr],rcond=None)[0];contrast_res=A[:,1]-B@np.linalg.lstsq(B,A[:,1],rcond=None)[0];partial=float(np.corrcoef(target_res,contrast_res)[0,1]);cuts=np.quantile(A[:,0],[.2,.4,.6,.8]);bins=np.searchsorted(cuts,d['reference008'][va]);folds.append({'fold':k,'training_only_partial_correlation':partial,'training_only_partial_correlation_scope':'Descriptive association after removing linear raw-ensemble prediction; same320pairs used to estimate nuisance trends. Not cross-validated extension performance. All upstream predictions generated strictly insideoutertrain.','prediction_bin_cuts_inner_train':cuts.tolist(),'outer_bin_bias':[{'bin':j,'n':int(sum(bins==j)),'mean_residual009':float(err[va[bins==j]].mean()) if np.any(bins==j) else None} for j in range(5)]});matrices[f'{split}_fold{k}']=A
  results[split]={'residual_correlations_descriptive':{n:{'pearson':float(np.corrcoef(x,err)[0,1]),'spearman':float(spearmanr(x,err).statistic)} for n,x in features.items()},'folds':folds,'prediction_quintile_bias_descriptive':[float(err[ix].mean()) for ix in np.array_split(np.argsort(p),5)]}
 with (_repo_path(OUT, 'inner_training_matrices.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),**matrices)
 report={'scope':'Development400 only. Initial exploratory calculations preceded this written report; no claim that diagnostics were preregistered. No new-model corrected OOF performance calculated here.','public':'009 outcome recorded separately; never used for feature/weight/coefficient choice.','findings':'Prediction-conditioned linear residual trend mostly removed by009. DINOmean minus003 has the clearest consistent signed association among inspected reliability contrasts and is positive after adjusting for rawensemble prediction in14/15 train-onlyfolds. B-minus-S and unsigned disagreement show weaker/inconsistent evidence. Middle-bin residual patterns also exist but do not justify selecting multiple nonlinear calibrators.','decision':'One extension only: add the signed DINOmean-minus003 contrast to the existing affine calibrator. Fit3parameters total by unregularized OLS inside eachoutertrain. PreserveequalS/Bweights; no ratio grid, knots, nonlinear gates, regularization orPublic tuning.','nonduplication':'074 used one rawensemble prediction. 047 combined imagefeatures throughadditive/productkernels, not train-only component-prediction contrast calibration. 064 fixed S/Bmean; no nested learned legacy/DINO contrast coefficient.','results':results,'matrix_sha256':fv.sha(_repo_path(OUT, 'inner_training_matrices.npz')),'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['residual_diagnostics_076.py','nested_calibration_074.py','validate_nested_calibration_074_v2.py']},'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/nested-calibration-074-validation.json','reports/candidate009-public-submission.json','reports/joint-047-summary.json']},'protected_best009_sha256':pub['sha256'],'limits':fv.CAVEAT+' Diagnostic correlations and the choice of extension are adaptive; not independent evidence.'};fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'decision':report['decision'],'training_positive_partial_correlations':sum(f['training_only_partial_correlation']>0 for r in results.values() for f in r['folds']),'contrasts':{s:r['residual_correlations_descriptive']['DINO_minus_003'] for s,r in results.items()}}),flush=True)
if __name__=='__main__':main()
