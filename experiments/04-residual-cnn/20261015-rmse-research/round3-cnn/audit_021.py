"""Read-only disaggregation of 021; weight curves are diagnostics, never fit choices."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(_repo_path(HERE.parent, 'round2')))
from models import inputs
import research as r

d=inputs();sid=d['ids'][d['dev']];y=d['y'][d['dev']]
path=r.EXP/'simplification-202/predictions.npz';z=np.load(_repo_io(path));pos={i:j for j,i in enumerate(z['ids'])};ix=np.array([pos[i]for i in sid])
assert np.array_equal(z['y'][ix],y)and np.array_equal(z['fold'][:,ix],d['fold'])
old=z['R'][:,ix];assert np.max(np.abs(old-d['C73_cnn']))<1e-8
cnn=z['CNN_seeds'].mean(1)[:,ix];s=z['S_seeds'].mean(1)[:,ix];b=z['B_seeds'].mean(1)[:,ix];large=z['L_seeds'].mean(1)[:,ix]
raw=z['raw'][:,ix];short=z['short'][:,ix];affine=z['affine'][:,ix];correction=z['correction'][:,ix];slopes=z['slopes'][:,ix]
krr=(raw-.3*cnn-.2*s-.2*b)/.3
reconstructed=.9*(.9*affine+.1*short+correction)+.1*large
assert np.max(np.abs(reconstructed-old))<1e-8
base=d['strict_base'];tab=(base-.16*old)/.84;valid=~d['fail']
components={'021_composite':old,'TinyCNN_mean3':cnn,'DINO_S_MLP_mean3':s,'DINO_B_MLP_mean3':b,'DINO_L_MLP_mean3':large,'old_compact_KRR':krr,'short_B':short}
stats={name:dict(oof_rmse=r.rmse(p,y),error_correlation_with_non021=float(np.corrcoef((p-y)[valid],(tab-y)[valid])[0,1]))for name,p in components.items()}
weights=[]
for weight in (0.,.08,.16,.24):
    pred=np.where(d['fail'],base,(1-weight)*tab+weight*old)
    weights.append(dict(weight=weight,**r.metrics(pred,y,d['fold'])))
result=dict(recipe_021='.9*(.9*affine(.3*KRR+.3*CNN_mean3+.2*S_mean3+.2*B_mean3)+.1*shortB+classical_residual)+.1*L_mean3',
    current_021_weight=.16,TinyCNN_linear_coefficient_before_affine_slope=.03888,
    TinyCNN_actual_linear_coefficient_range=[float((.03888*slopes).min()),float((.03888*slopes).max())],
    TinyCNN_actual_linear_coefficient_median=float(np.median(.03888*slopes)),components=stats,
    strict073_rmse=r.rmse(base,y),non021_renormalized_rmse=r.rmse(tab,y),diagnostic_weight_curve=weights,
    reconstruction_maxdiff=float(np.max(np.abs(reconstructed-old))),source_sha256=r.digest(path),
    limitations=['Weight curve uses repeatedly explored 400-ID OOF; it is not nested weight selection or independent evidence to delete 021.',
                'The .16 share is a historical composite weight, not a proven optimum and not the standalone TinyCNN share.',
                'New CNN complement training uses separate honest all500 folds and cannot be scored as direct replacement of this entire composite.'])
r.save_json(_repo_path(HERE, 'architecture-audit.json'),result)
print(json.dumps(result,indent=2))
