"""Read-only replay of diagnostics; outputs verification evidence inside own folder."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
import diagnose as d
R=d.R;P=d.P
p=d.load(R/'plan.json');s=d.load(R/'summary.json');pro=d.load(R/'provenance.json')
assert d.sha(R/'diagnose.py')==p['script_sha256']
for f,h in {**p['inputs_sha256'],**p['source_sha256']}.items():assert d.sha(P/f)==h,f
with np.load(_repo_io(R/'features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'];assert ids==p['IDs']
maxerr=0.;HHerr=0.;phaseerr=0.;count=0
for i,ident in enumerate(ids):
    path=P/'data/train'/f'{ident}.png';assert d.sha(path)==pro['image_sha256'][ident]
    with Image.open(_repo_io(path)) as im:a=np.asarray(im).copy()
    v,_=d.features(a);maxerr=max(maxerr,float(abs(v-X[i]).max()))
    x=a.astype(float)/255
    win=sliding_window_view(x,(2,2));hh=(win*np.array([[.5,-.5],[-.5,.5]])).sum(axis=(-1,-2))
    independent=(x[:-1,:-1]-x[:-1,1:]-x[1:,:-1]+x[1:,1:])/2
    HHerr=max(HHerr,float(abs(hh-independent).max()))
    ph=np.array([np.mean(hh[oy::2,ox::2]**2) for oy in range(2) for ox in range(2)])
    phaseerr=max(phaseerr,float(abs(np.log((ph.max()+1e-12)/(ph.min()+1e-12))-X[i,4])))
    count+=1
assert maxerr==0 and HHerr<1e-15 and phaseerr<1e-12
score_err=0;equal_targets=True
with np.load(_repo_io(R/'comparison-predictions.npz'),allow_pickle=False) as z:
    for row in s['comparisons_retrospective']:
        split=row['split'];pred=z[split];y=z['y']
        with np.load(_repo_io(P/f'experiments/dinol-177/{split}/predictions.npz'),allow_pickle=False) as inp:assert np.array_equal(y,inp['y'])
        vals=[(sum((float(a)-float(b))**2 for a,b in zip(y,q))/len(y))**.5 for q in pred]
        score_err=max(score_err,abs(vals[0]-row['base57_RMSE']),abs(vals[1]-row['plus6_RMSE']))
assert score_err<1e-12
result={'pass':True,'images_replayed':count,'feature_replay_max_abs_error':maxerr,'independent_sliding_window_HH_max_abs_error':HHerr,'independent_phase_descriptor_max_abs_error':phaseerr,'scalar_RMSE_recalculation_max_abs_error':score_err,'all_source_and_input_hashes_unchanged':True,'same_400_targets_across_repartitions':equal_targets,'D4_first_image_all8_max_abs_error':s['D4_max_abs_error'],'fold0_validation_label_poison_coefficients_exact':s['validation_label_poison_max_coef_difference']==0,'fit_count_clarification':'35 main fits =30 hardness+5 multioutput feature recovery; 2 additional fold0 poison replay fits, total37. No new fits during this verification.','scope':'All data reads development IDs and development-only cache entries. No best021/test/holdout/full500 file read or changed.'}
d.write('verification.json',result)
print(json.dumps(result,indent=2))
