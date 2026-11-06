"""Read-only development diagnostics: no fitting, holdout, or test access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, hashlib, json, sys
from pathlib import Path
ROOT = _RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve() != (_repo_path(ROOT, '.venv')).resolve():
    raise SystemExit('Use project .venv')
import numpy as np
from scipy.stats import spearmanr

def read_json(p): return json.loads((_repo_path(ROOT, p)).read_text())
def digest(p): return hashlib.sha256((_repo_path(ROOT, p)).read_bytes()).hexdigest()
def rmse(y, p): return float(np.sqrt(np.mean((y-p)**2)))
def rho(a,b): return float(spearmanr(a,b).statistic)
def summary(a):
    return dict(zip(('min','p10','p25','median','p75','p90','max'), map(float,np.percentile(a,[0,10,25,50,75,90,100]))))

def main():
    split=read_json('artifacts/run-001/split_manifest.json')
    ids=sorted(split['development_folds']); assert len(ids)==400
    files=['artifacts/run-022/development_oof.csv','artifacts/run-022/development_features.npz','artifacts/run-025/development_features.npz','artifacts/run-022/report.json','artifacts/run-025/report.json']
    with (_repo_path(ROOT, files[0])).open() as f: rows=list(csv.DictReader(f))
    byid={r['ID']:r for r in rows}; assert len(rows)==len(byid)==400 and set(byid)==set(ids)
    y=np.array([float(byid[s]['hardness']) for s in ids])
    preds={name:np.array([float(byid[s][col]) for s in ids]) for name,col in [('candidate003','candidate003'),('candidate004','combined_kernel_multiseed_blend')]}
    data=[]
    for run in ('run-022','run-025'):
        with np.load(_repo_path(ROOT, f'artifacts/{run}/development_features.npz'),allow_pickle=False) as z:
            zi=list(z['ids']); assert len(set(zi))==400 and set(zi)==set(ids)
            X=z['X'][[zi.index(s) for s in ids]].copy()
        names=read_json(f'artifacts/{run}/report.json')['features']; assert len(names)==57
        data.append((X,names))
    X,names=data[0]; strong,snames=data[1]
    queries=['denoised_noise_sigma','denoised_grain_median_grain_count','denoised_grain_range_grain_count','denoised_grain_median_interior_diam_mean','denoised_grain_range_interior_diam_mean','denoised_grain_median_small_region_area_fraction','inverse_sqrt_grain_diameter','grain_scale_sensitivity','image_mean','image_std']
    relationships={}
    for name in queries:
        v=X[:,names.index(name)]
        relationships[name]={'summary':summary(v),'rho_label':rho(v,y),**{model:{'rho_signed_residual_y_minus_pred':rho(v,y-p),'rho_absolute_residual':rho(v,abs(y-p))} for model,p in preds.items()}}
    label_bins=np.quantile(y,[.1,.9]); labels=[]
    for label,mask in [('bottom10',y<=label_bins[0]),('middle80',(y>label_bins[0])&(y<label_bins[1])),('top10',y>=label_bins[1])]:
        labels.append({'stratum':label,'n':int(mask.sum()),'label_mean':float(y[mask].mean()),'label_range':[float(y[mask].min()),float(y[mask].max())],**{model:{'rmse':rmse(y[mask],p[mask]),'mean_residual_y_minus_pred':float((y-p)[mask].mean()),'prediction_mean':float(p[mask].mean())} for model,p in preds.items()}})
    paired={}
    for suffix in ('median_grain_count','range_grain_count','median_interior_diam_mean','range_interior_diam_mean'):
        a=X[:,names.index('denoised_grain_'+suffix)]; b=strong[:,snames.index('strong_denoised_grain_'+suffix)]
        paired[suffix]={'weak':summary(a),'strong':summary(b),'spearman_weak_strong':rho(a,b),'median_strong_over_weak':float(np.median(b/np.maximum(a,1e-12)))}
    folds=[]
    for k in range(5):
        mask=np.array([split['development_folds'][s]==k for s in ids])
        folds.append({'fold':k,'n':int(mask.sum()),'label_mean':float(y[mask].mean()),'label_std':float(y[mask].std()),**{m:rmse(y[mask],p[mask]) for m,p in preds.items()}})
    result={'scope':'development400 cached features and OOF only; no new fits or holdout/test reads','inputs_sha256':{p:digest(p) for p in files},'label_summary':summary(y),'label_mean':float(y.mean()),'label_std':float(y.std()),'label_unique_count':len(np.unique(y)),'negative_label_count':int((y<0).sum()),'prediction_summary':{m:{'summary':summary(p),'std':float(p.std()),'rmse':rmse(y,p),'mean_residual':float((y-p).mean())} for m,p in preds.items()},'label_strata':labels,'feature_relationships':relationships,'segmentation_parameter_sensitivity':paired,'folds':folds,'limits':['No ground truth grain segmentation/physical grain size is provided: parameter sensitivity is not actual estimation error.','Correlations are descriptive, not causal; repeated diagnostics and model selection on these 400 rows cause optimism.','Residuals conditioned on labels naturally show regression-to-mean; no target expansion/calibration is recommended from this alone.','Weak/strong prominence changes alter grain identities and counts, so more stable counts do not establish more accurate morphology.']}
    (_repo_path(ROOT, 'reports/development-audit-026.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
