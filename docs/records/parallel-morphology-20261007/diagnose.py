"""Bounded development-only image generation diagnostics; no physical identification."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,time,csv,resource,socket
from pathlib import Path
R=Path(__file__).resolve().parent
P=R.parents[2]
assert Path(sys.prefix).resolve()==(_repo_path(P, '.venv')).resolve()
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from PIL import Image
from threadpoolctl import threadpool_limits,threadpool_info
threadpool_limits(2)
def blocked(*a,**k): raise RuntimeError('network forbidden')
socket.socket.connect=blocked
socket.create_connection=blocked

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
    with (_repo_path(R, name)).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):return json.loads(Path(p).read_text())
NAMES=['smooth_bright_dark_log_energy_ratio','smooth_intensity_energy_cv','edge_smooth_log_energy_ratio','tile_HH_energy_cv','phase_log_energy_range','phase_mean_over_RMS_max']
SPLITS=['original','20261031','20261101']

def features(raw):
    x=np.asarray(raw,dtype=float)/255
    # Undecimated Haar diagonal difference: all 255x255 valid 2x2 supports.
    hh=(x[:-1,:-1]-x[:-1,1:]-x[1:,:-1]+x[1:,1:])/2
    guide=ndi.gaussian_filter(x,2,mode='reflect')
    grad=ndi.gaussian_gradient_magnitude(x,2,mode='reflect')
    box=lambda a:(a[:-1,:-1]+a[:-1,1:]+a[1:,:-1]+a[1:,1:])/4
    guide=box(guide);grad=box(grad)
    # Use every valid support for conditioning. Quantile cuts are image-local.
    smooth=grad<=np.quantile(grad,.5)
    edge=grad>=np.quantile(grad,.75)
    bins=np.digitize(guide,np.quantile(guide[smooth],[.25,.5,.75]))
    h2=hh**2; eps=1e-12
    energies=np.array([h2[smooth&(bins==b)].mean() if np.any(smooth&(bins==b)) else 0 for b in range(4)])
    se=float(h2[smooth].mean());ee=float(h2[edge].mean())
    phase=np.array([h2[a::2,b::2].mean() for a in range(2) for b in range(2)])
    pm=np.array([abs(hh[a::2,b::2].mean())/(np.sqrt(phase[2*a+b])+eps) for a in range(2) for b in range(2)])
    # D4-symmetric central crop of undecimated field, 15x15 cells each17px.
    tiles=h2.reshape(15,17,15,17).mean((1,3))
    v=np.array([np.log((energies[3]+eps)/(energies[0]+eps)),energies.std()/(energies.mean()+eps),np.log((ee+eps)/(se+eps)),tiles.std()/(tiles.mean()+eps),np.log((phase.max()+eps)/(phase.min()+eps)),pm.max()])
    assert np.isfinite(v).all()
    return v,{'bin_energy':energies.tolist(),'phase_energy':phase.tolist(),'smooth_energy':se,'edge_energy':ee,'HH_energy':float(h2.mean()),'saturated_fraction':float(((x==0)|(x==1)).mean()),'sigma_HH':float(np.median(abs(hh))/.6744897501960817)}

def prepare():
    m=load(_repo_path(P, 'artifacts/run-001/split_manifest.json'));ids=sorted(m['development_folds']);assert len(ids)==400
    sources=['grain_features.py','denoised_grain_features.py','autocov_shape_features.py','native_texture_features.py','blindspot_preflight_102.py','formation_model_049.py','candidate_features.py']
    plan={'created_UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'IDs':ids,'diagnoses':['Conditional HH energy by image-local smoothed brightness, weak/strong gradient and spatial tiles.','HH energy/mean dependence on four raster parity phases; no new segmentation.'],'novelty':'102 already tested HH lag correlations; 094 global photometry/noise and tile image contrast; native28 LoG .7/2.8 tile co-location; 183 watershed threshold/graph stability; 218 missing grain validity. Here conditional Haar energy and four-phase energy/mean imbalance, not those completed studies. Limited code review, not exhaustive novelty proof.','descriptors':NAMES,'definition':'See pure features() implementation. Native256 no resampling/denoising/segmentation. Gaussian sigma2 only conditioning guide; q50 low-gradient, q75 high-gradient; brightness quartiles within low-gradient. Haar energy is high-frequency structure+noise, not identified observation-noise variance.','comparison':'Fixed Ridge alpha100 with train-only StandardScaler, intercept, raw hardness targets. Compare compact57 vs compact57+all six descriptors on original and BOTH existing repeats, 30 fits. No tuning/feature selection/residual learning/ensemble change. Additional fivefold Ridge alpha100 compact57->six features assesses limited linear recoverability, not proof of information absence.','outcomes':'No outcome values used to pick descriptors. All target associations/comparisons retrospective on heavily reused400; no adoption/p-values. No residualCNN or best021 output modification.','fixtures':'32 deterministic iid Gaussian grayscale fields; additional fixed intensity-dependent variance, parity-dependent variance, and noiseless tessellation fixture. These are engineering controls, not generator priors. D4 invariance checks first image and fixture.','limits':'CPU threads<=2, runtime600s, peakRSS<2GB, no external data/network/install/GPU. Only exact development IDs opened as images. Existing files and caches read-only; all outputs in this folder.','source_sha256':{n:sha(_repo_path(P, n)) for n in sources},'script_sha256':sha(__file__),'inputs_sha256':{n:sha(_repo_path(P, n)) for n in ['artifacts/run-001/split_manifest.json','artifacts/run-022/development_features.npz']}}
    write('plan.json',plan)
    print('Plan locked; six features, alpha100, 3 existing partitions.',flush=True)

def run():
    t=time.perf_counter();p=load(_repo_path(R, 'plan.json'));assert sha(__file__)==p['script_sha256'];ids=p['IDs']
    for f,h in p['inputs_sha256'].items():assert sha(_repo_path(P, f))==h
    X=[];details=[];hashes={};d4err=0
    for i,ident in enumerate(ids):
        path=_repo_path(P, 'data/train', f'{ident}.png')
        with Image.open(_repo_io(path)) as im:
            assert im.mode=='L' and im.size==(256,256);raw=np.asarray(im).copy()
        v,detail=features(raw);X.append(v);details.append({'ID':ident,**detail});hashes[ident]=sha(path)
        if i==0:
            for k in range(4):
                for q in [np.rot90(raw,k),np.fliplr(np.rot90(raw,k))]:d4err=max(d4err,float(abs(features(q)[0]-v).max()))
        if (i+1)%100==0:print(f'Extracted {i+1}/400 in {time.perf_counter()-t:.1f}s',flush=True)
        assert time.perf_counter()-t<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9
    X=np.array(X);assert d4err<1e-9
    np.savez_compressed(_repo_path(R, 'features.npz'),ids=ids,names=NAMES,X=X)
    write('image-measurements.json',details)
    rng=np.random.default_rng(20261007);null=[]
    for _ in range(32):null.append(features(128+10*rng.normal(size=(256,256)))[0])
    null=np.array(null)
    yy,xx=np.indices((256,256));brightness=90+70*xx/255
    fixtures={'iid':128+10*rng.normal(size=(256,256)), 'heteroscedastic':brightness+(4+20*xx/255)*rng.normal(size=(256,256)),'parity_variance':128+(5+15*(xx%2))*rng.normal(size=(256,256))}
    # Deterministic Voronoi illustration solely demonstrates structural contamination.
    seeds=rng.uniform(0,256,(30,2));d=(yy[...,None]-seeds[:,0])**2+(xx[...,None]-seeds[:,1])**2;labels=d.argmin(2);tess=rng.uniform(60,190,30)[labels];boundary=(ndi.maximum_filter(labels,3)!=ndi.minimum_filter(labels,3));tess[boundary]=20;fixtures['noiseless_tessellation']=tess
    fv={n:features(a)[0].tolist() for n,a in fixtures.items()}
    # Column-only phase noise may cancel its variance in every 2x2 Haar support.
    # Explicit positive control with correlated 2x2 blocks tests the phase statistic.
    block=np.repeat(np.repeat(rng.normal(size=(128,128)),2,axis=0),2,axis=1)
    fixtures['block2_correlated']=128+10*block
    fv['block2_correlated']=features(fixtures['block2_correlated'])[0].tolist()
    assert fv['heteroscedastic'][0]>1 and fv['block2_correlated'][4]>1
    write('fixtures.json',{'names':NAMES,'iid32_quantiles':np.quantile(null,[.05,.5,.95],axis=0).tolist(),'fixed_cases':fv,'D4_max_abs_error':d4err,'warning':'Noiseless tessellation yields nonzero HH. Parity-dependent independent pixel variance can cancel in Haar energy; negative phase finding does not exclude raster variance modulation.'})
    summary={'n':400,'features':{},'extraction_seconds':time.perf_counter()-t}
    with np.load(_repo_path(P, 'artifacts/run-022/development_features.npz'),allow_pickle=False) as z:
        assert z['ids'].tolist()==ids;B=z['X'];groups=z['groups']
    assert B.shape==(400,57)
    predictions={};comparisons=[];recover=np.zeros_like(X);fold_records=[];poison=0.
    for split in SPLITS:
        fn=_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz')
        with np.load(_repo_io(fn),allow_pickle=False) as z:
            assert z['ids'].tolist()==ids;y=z['y'];fold=z['fold'];g=z['g']
        assert len(y)==400 and np.isfinite(y).all()
        ps=np.empty((2,400))
        for k in sorted(set(fold)):
            tr=fold!=k;va=~tr;assert not set(g[tr])&set(g[va])
            for arm,A in enumerate([B,np.column_stack([B,X])]):
                sc=StandardScaler().fit(A[tr]);model=Ridge(alpha=100).fit(sc.transform(A[tr]),y[tr]);ps[arm,va]=model.predict(sc.transform(A[va]))
                if split=='original' and k==0:
                    poisoned=y.copy();poisoned[va]+=99999
                    mm=Ridge(alpha=100).fit(sc.transform(A[tr]),poisoned[tr]);poison=max(poison,float(abs(mm.coef_-model.coef_).max()))
            if split=='original':
                sc=StandardScaler().fit(B[tr]);recover[va]=Ridge(alpha=100).fit(sc.transform(B[tr]),X[tr]).predict(sc.transform(B[va]))
            fold_records.append({'split':split,'fold':int(k),'base_RMSE':float(np.sqrt(np.mean((y[va]-ps[0,va])**2))),'plus_RMSE':float(np.sqrt(np.mean((y[va]-ps[1,va])**2)))})
        scores=np.sqrt(np.mean((ps-y)**2,axis=1));gain=float(scores[0]-scores[1]);comparisons.append({'split':split,'base57_RMSE':float(scores[0]),'plus6_RMSE':float(scores[1]),'gain':gain});predictions[split]=ps
        if split=='original':original_y=y.copy()
    for j,n in enumerate(NAMES):
        sr=float(spearmanr(X[:,j],original_y).statistic)
        cor=[abs(float(spearmanr(X[:,j],B[:,k]).statistic)) for k in range(57) if np.ptp(B[:,k])>0]
        summary['features'][n]={'q05_q50_q95':np.quantile(X[:,j],[.05,.5,.95]).tolist(),'iid_q95':float(np.quantile(null[:,j],.95)),'fraction_above_iid_q95_descriptive':float(np.mean(X[:,j]>np.quantile(null[:,j],.95))),'compact57_max_abs_Spearman':max(cor),'compact57_fixed_Ridge_OOF_R2':float(r2_score(X[:,j],recover[:,j])),'hardness_Spearman_retrospective':sr}
    summary.update({'comparisons_retrospective':comparisons,'fold_results':fold_records,'fit_count':35,'D4_max_abs_error':d4err,'validation_label_poison_max_coef_difference':poison,'hardness_labels_used':True,'holdout_test_full500_access':False,'scope':'400 development images only; manifest contains held-out ID metadata but no holdout data loaded. No test data/cache/submission read.','elapsed_seconds':time.perf_counter()-t,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'threadpools':threadpool_info(),'decision':'Diagnostic only, best021 unchanged. No hyperparameter, feature, weight, threshold or seed rescue.','limitations':['Retrospective reuse of 400; repeats are not independent1200.','Six correlations and two image diagnoses not multiplicity-corrected; no significance claim.','Fixed Ridge comparison is not complete KRR/DINO/CNN ensemble or deployment result.','Conditional HH energy mixes true high-frequency structure, blur, clipping and noise.','Smoothed guide is derived from same image, so conditioning can itself induce dependence.','Fixed linear recovery failure does not prove descriptor absent from nonlinear compact57 models or DINO/CNN.']})
    write('summary.json',summary);write('provenance.json',{'image_sha256':hashes,'plan_sha256':sha(_repo_path(R, 'plan.json')),'source_sha256':sha(__file__),'inputs_sha256':p['inputs_sha256'],'prediction_input_sha256':{s:sha(_repo_path(P, f'experiments/dinol-177/{s}/predictions.npz')) for s in SPLITS}})
    np.savez_compressed(_repo_path(R, 'comparison-predictions.npz'),ids=ids,y=original_y,recover=recover,**predictions)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
