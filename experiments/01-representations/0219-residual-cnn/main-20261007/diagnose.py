"""Post-fixed-comparison descriptive diagnostics. No tuning, fits or lambda search."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
O=Path(__file__).resolve().parent;S=O.parent;R=_RepoPath(__file__).resolve().parents[4];sys.path.insert(0,str(S))
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits
from contracts import sha,require,lock_manifest
from data import load_development
from bundle import load_bundle
threadpool_limits(1);torch.set_num_threads(1);require(json.loads((_repo_path(O, 'scores.json')).read_text())['passed'],'Need completed comparison first');f,labels,contract=load_development();rows=[];rp={k:[] for k in ['r','dtrue','dperm']}
for k,g in enumerate(contract['graphs']):
 folder=_repo_path(O, f'fold-{k}');B,V=map(np.array,(g['B'],g['V']));anchor=joblib.load(_repo_io(folder/'frozen-anchor.joblib'));rb=labels.take(B)-anchor.predict(f,B);rv=labels.take(V)-anchor.predict(f,V);record={'fold':k,'residual_train_n':len(B),'train_residual_RMS':float(np.sqrt(np.mean(rb**2))),'validation_anchor_residual_RMS':float(np.sqrt(np.mean(rv**2))),'arms':{}}
 for arm in ['true','permuted']:
  p=load_bundle(folder/arm,sha(_repo_path(S, 'protocol.json')),sha(_repo_path(S, 'data-contract.json')));target=rb if arm=='true' else rb[np.random.default_rng(20264050+k).permutation(len(B))];tb=p.residual.predict(f,B);tv=p.residual.predict(f,V);record['arms'][arm]={'training_target_RMSE':float(np.sqrt(np.mean((target-tb)**2))),'validation_RMSE':float(np.sqrt(np.mean((rv-tv)**2))),'validation_correction_mean':float(tv.mean()),'validation_correction_std':float(tv.std()),'validation_residual_correlation':float(np.corrcoef(rv,tv)[0,1])};rp['dtrue' if arm=='true' else 'dperm'].extend(tv)
 rp['r'].extend(rv);rows.append(record)
arrays={k:np.asarray(v) for k,v in rp.items()};decomp={}
for name in ['dtrue','dperm']:
 r,d=arrays['r'],arrays[name];A=float(2*np.mean(r*d));P=float(np.mean(d*d));decomp[name]={'alignment_twice_mean_r_times_d':A,'correction_mean_square':P,'MSE_change_P_minus_A':P-A,'residual_correction_correlation':float(np.corrcoef(r,d)[0,1])}
result={'passed':True,'folds':rows,'MSE_decomposition_at_fixed_lambda1':decomp,'no_lambda_optimization':True,'no_new_model_fit':True,'scope':'Descriptive after all fixed models scored; not used to select epochs,seeds,weights or further arms','scores_sha256':sha(_repo_path(O, 'scores.json'))};lock_manifest(_repo_path(O, 'diagnostics.json'),result);print(json.dumps(result,indent=2))
