"""Preregistered label-blind, train-fold-only feasibility gate. No regressors."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,time,hashlib,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk
import torch
import formation_model_049 as m
OUT=_repo_path(ROOT, 'artifacts/formation-pilot-049')
SEED=20261126

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def read_image(i):
    assert i in json.loads((_repo_path(OUT, 'plan.json')).read_text())['ids']
    return np.asarray(Image.open(_repo_path(ROOT, 'data/train', f'{i}.png')).convert('L'))
def setup():
    manifest=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())
    ids=sorted(k for k,v in manifest['development_folds'].items() if v!=0)
    ids=sorted(np.random.default_rng(SEED).choice(ids,40,replace=False).tolist())
    OUT.mkdir(exist_ok=False)
    plan={'scope':'40 fixed original fold0 training images; no hardness labels, holdout/test, model fitting, upload or external data. Synthetic fixtures only algorithm checks.','ids':ids,'seed':SEED,'selection':'Sorted original development fold!=0 IDs, numpy default_rng choice40 without replacement; no label file opened.','hypothesis':'Joint piecewise-constant latent region field + sparse dark layer, bounded isotropic Gaussian blur, affine contrast/offset and bounded separable AR1 residual likelihood may separate imaging nuisance from morphology better than sequential TV watershed.','model':'Own approximate continuous Potts-like log-gradient prior, not a reproduction of either referenced paper, not a known competition generator. Two blur initializations,100 fixed Adam steps, summarize median; no objective-based start selection.','nonidentifiability':'Unknown regions/blur/contrast/noise trade off; constraints and two starts do not identify true generator. Grain/phase/pore/alignment are proxies. Isotropic blur and separable Gaussian AR noise may be wrong.','references':['https://arxiv.org/abs/1405.5850','https://www.jmlr.org/beta/papers/v15/wipf14a.html'],'duplicate_audit':{'denoised_grain_features.py':'TV .5/1 noise followed by gradient watershed; no joint rendering likelihood','strong_denoised_grain_features.py':'Same denoise with larger marker prominence','connected_grain_features.py':'TV + local dark ridge + closing/connected complement','autocov_shape_features.py':'Nonzero-lag covariance and separate noise indicators; no inverse model','simple_noise_variance_subtraction':'Excluded; previous audit found harmful shortcut'},'nuisance_calibration':'Same40 train images only: robust HH noise, q98-q2 contrast; blur indicator gradient-energy ratio sigma1/sigma0. No claim identifies true blur. Fixed perturbations: affine span0.9 +offset0.05; extra blur0.5px; Gaussian-correlated noise sigma0.5px with std=.5*image HH noise. Record train quantiles and added-noise bounds. These are mild synthetic probes, not estimated true generator distribution.','variants':['identity','affine','blur','correlated_noise'],'stability_gate':{'min_reduction_each_descriptor':0.30,'definition':'Per descriptor median over40 images of RMS difference over3 perturbations vs identity, normalized by baseline identity cross-image IQR; candidate/baseline ratio<=.70 for all4. Nearzero baseline sensitivity has no evidence of30% improvement and fails.','floor_scale':[1.0,0.02,0.001,0.02]},'preservation_gate':{'fixture_boundary_recall_min':0.80,'fixture_boundary_precision_min':0.60,'boundary_tolerance_px':2,'fixture_pore_recall_min':0.75,'fixture_pore_precision_min':0.50,'fixture_max_relative_grain_error':0.30,'fixture_max_abs_phase_error':0.10,'fixture_max_abs_alignment_error':0.15,'require_all_six_fixtures':True,'real_image_interquartile_range_ratio_min':0.5,'real_boundary_anchor_recall_median_min':0.80,'real_pore_anchor_recall_median_min':0.75,'real_start_disagreement_normalized_median_max':0.5,'real_degenerate_fraction_max':0.10,'degenerate_definition':'latent std<.02 OR noise near upperbound>1.95*n0 OR blur>1.72 OR boundary fraction<.002 OR >.35','no_anchor_policy':'Empty anchors uninformative, never scored perfect; require>=10 images with pore anchors and >=30 with boundary anchors.'},'fixtures':'Three analytic non-Voronoi rectangular/elongated/wavy region layouts at256px, alternating intensity phases, known thin1px boundaries and disk radius3/5/7 isolated+attached pores; two fixed noise/blur severities. Deterministic fixtures are not training data.','resource_gate':{'seconds_per_image_two_starts_max':10,'rss_bytes_max':3000000000,'estimated_full_minutes_max':30},'failure_policy':'Any gate failure rejects this family without residual regression, CV, parameter tuning or candidate creation. Only all-gates pass permits separately implemented inner-crossfit residual study.','sources':{n:sha(_repo_path(ROOT, n)) for n in ['formation_model_049.py','formation_pilot_049.py','denoised_grain_features.py','phase_features.py']},'protected':{n:sha(_repo_path(ROOT, n)) for n in ['artifacts/submission-candidate-006.csv','artifacts/dino-multilayer-044/candidate-design.json','artifacts/final-model-045/dino-regressor.joblib']}}
    save(_repo_path(OUT, 'plan.json'),plan);print(json.dumps({'plan':str(_repo_path(OUT, 'plan.json')),'images':40,'labels_read':False}),flush=True)

def nuisance(x):
    a=x.astype(float)/255; lo,hi=np.percentile(a,(2,98)); g0=np.mean(np.diff(a,axis=0)**2)+np.mean(np.diff(a,axis=1)**2); b=ndi.gaussian_filter(a,1)
    g1=np.mean(np.diff(b,axis=0)**2)+np.mean(np.diff(b,axis=1)**2)
    return {'noise':m.old._noise_sigma(a),'contrast':float(hi-lo),'blur_indicator':float(g1/max(g0,1e-12))}

def perturb(x,kind,index):
    a=x.astype(float)/255
    if kind=='affine': a=.9*a+.05
    elif kind=='blur':a=ndi.gaussian_filter(a,.5)
    elif kind=='correlated_noise':
        noise=ndi.gaussian_filter(np.random.default_rng(SEED+index).normal(size=a.shape),.5)
        noise=noise/max(noise.std(),1e-9)*.5*nuisance(x)['noise'];a=a+noise
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)

def benchmark():
    torch.set_num_threads(2); p=json.loads((_repo_path(OUT, 'plan.json')).read_text());x=read_image(p['ids'][0]);t=time.perf_counter();result=m.estimate(x);elapsed=time.perf_counter()-t
    report={'device':'cpu','seconds_per_image_two_starts':elapsed,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'projected_minutes_160_images_plus6fixtures':elapsed*166/60,'features':result['features'].tolist(),'diagnostics':result['diagnostics']}
    gate=p['resource_gate'];report['resource_pass']=elapsed<=gate['seconds_per_image_two_starts_max'] and report['rss_bytes']<=gate['rss_bytes_max'] and report['projected_minutes_160_images_plus6fixtures']<=gate['estimated_full_minutes_max']
    np.savez_compressed(_repo_path(OUT, 'benchmark.npz'),features=result['features'],u=result['u'],d=result['d'],render=result['render'])
    save(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['setup','benchmark']);a=ap.parse_args();globals()[a.action]()
