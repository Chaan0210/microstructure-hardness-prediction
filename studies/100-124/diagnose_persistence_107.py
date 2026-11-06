"""Explain rejected107 extrapolation; no model changes or new selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib
import persistence_study_107 as s

def main():
 d=s.inputs('original');records=[]
 for arm in s.ARMS:
  X=d[arm]
  for k in range(5):
   a=joblib.load(_repo_io(s.OUT/'original'/f'{arm}-fold{k}.joblib'));m=a['model'];va=np.flatnonzero(d['fid']==k);Z=m.scaler.transform(X[va]);i,j=np.unravel_index(abs(Z).argmax(),Z.shape);pred=m.predict(X[va]);records.append({'arm':arm,'fold':k,'max_abs_standardized_validation_feature':float(abs(Z[i,j])),'most_extreme_ID':d['ids'][va[i]],'most_extreme_column':int(j),'train_feature_mean':float(m.scaler.mean_[j]),'train_feature_std':float(m.scaler.scale_[j]),'raw_validation_feature':float(X[va[i],j]),'prediction_for_extreme_row':float(pred[i]),'single_column_prediction_contribution':float(Z[i,j]*m.ridge.coef_[j]*m.ys),'prediction_range':[float(pred.min()),float(pred.max())],'train_columns_sd_below1e_20':int(((m.scaler.scale_<1e-20)&(m.scaler.var_>0)).sum())})
 out={'study':'107 posthoc failure diagnosis only','records':records,'finding':'Standardizing extremely small Gaussian tail cells separately makes rare validation features enormous in standardized units. Finite raw arrays and exact training/reload do not ensure stable extrapolation. Column237 is highbirth/highlifetime squarecell; validbirth+life<=1 makes itsGaussian spillover verysmall. This is a feature scaling/design failure for this fixedpipeline, not a corrupted cache or changed model.','independent_Ridge_validation':s.read(_repo_path(ROOT, 'reports/persistence-107-original-validation.json')),'no_clipping_or_refit_or_threshold_tuning':True,'feature_model_hashes_preserved':True,'source_sha256':s.sha(Path(__file__))};s.write(s.OUT/'failure-diagnosis.json',out);s.check();s.emit({'diagnosis_saved':True,'largest_standardized_value':max(r['max_abs_standardized_validation_feature'] for r in records)})
if __name__=='__main__':main()
