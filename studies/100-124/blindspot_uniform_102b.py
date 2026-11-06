"""24-image train-only uniform-region/iid-control diagnostic, never model training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,hashlib,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,gaussian_gradient_magnitude
from skimage.restoration import denoise_tv_chambolle
OUT=_repo_path(ROOT, 'artifacts/blindspot-preflight-102/revised24');REPS=16

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 assert not (_repo_path(OUT.parent, 'report.json')).exists();OUT.mkdir(exist_ok=False)
 m=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'))
 with (_repo_path(ROOT, 'artifacts/grain-visual-audit-097/all400_noise_contrast.csv')).open() as f:rows=[r for r in csv.DictReader(f) if m['development_folds'][r['ID']]!=0]
 assert len(rows)==320;nq=np.quantile([float(r['noise']) for r in rows],[1/3,2/3]);cq=float(np.median([float(r['contrast_q95_q05']) for r in rows]));chosen=[];strata=[]
 for n in range(3):
  for c in range(2):
   pool=[r for r in rows if np.searchsorted(nq,float(r['noise']),side='right')==n and int(float(r['contrast_q95_q05'])>=cq)==c];pool.sort(key=lambda r:hashlib.sha256(('102b-24:'+r['ID']).encode()).hexdigest());assert len(pool)>=4
   selected=pool[:4];chosen.extend(selected);strata.append({'noise_bin':n,'contrast_bin':c,'available':len(pool),'ids':[r['ID'] for r in selected]})
 ids=[r['ID'] for r in chosen];assert len(set(ids))==24 and not set(ids)&set(m['holdout'])
 p={'study':'102 revised24 label-free uniform-region noise diagnostic','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supersedes':'Prior32 random train-only plan preserved but never run. The revised plan uses 24 noise-by-contrast selected images and iid controls before measurement.','ids':ids,'strata':strata,'noise_cutpoints':nq.tolist(),'contrast_cutpoint':cq,'selection':'Noise tertiles x mediancontrast within original outertrain320;4 smallest sha256(102b-24:+ID) ineachcell. Reuse097 label-free metadata. No outervalidation image/label access.',
 'uniform_regions':'Native256 grayscale /255. Smooth guide G3 (Gaussian sigma3 reflect) only locates low-gradient regions; rank64 nonoverlapping32px tiles by p90 gradient ofG3, select four smallest (stable row-major ties). Reporteach selected tile p90-p10 guide span/noise proxy. Atleast2 selected tiles withspan<=1*HHsigma needed forimage tobe provisionally flat; not groundtruthflatness.',
 'metrics':'For eachimage and fixedraw/TVweight1*HHsigma/GaussianSigma1 arms: sixcorrelations on selectedraw-guide tiles:3 horizontal/vertical/diagonal correlations ofnonoverlapping2x2HH (raw displacement2px), and3 corresponding correlations ofarm-minusGaussianSigma1 residual at1px. Pairvalues centered per tile then pooled; no pairs crossingtiles. Independently check np.corrcoef against dotproduct formula. Report signed values; correlate fine signal aswellasnoise.',
 'iid_control':'16 fixed seed20271003+imageindex*100+rep draws: guideG3 + Normal(0,rawHHsigma) per pixel,clip[0,1],round8bit. Diagnostic only, never training. Rerunidenticalflat-region selection, noiseestimation,TV/smoothing andcorrelations. Guide isestimated,notcleantruth; controls calibrate artifacts underthissurrogate,notidentifyactualnoiselaw.',
 'warning_gate':'Do not fitrestoration if fewerthan18/24 images have>=2 provisionallyflattiles OR >=6/24 images have anyraw HHdirection |rho| exceedingmaximum same-direction |rho| among16iidcontrols +.05. These are engineering safeguards,notpvalues. Even if bothclear, assesswhetherfine morphology andnoise remainconfounded; if unresolved stop peruser. Do notpick maskingradius orretune thresholds fromresults.',
 'source':{'paper':'https://proceedings.mlr.press/v97/batson19a.html','conditions':'Conditional unbiasedness plus noiseindependence betweenmasked coordinates andcomplement; predictor outputatmaskedJ independent oforiginalxJ.'},
 'scope':'No fit,hardnesslabels,outervalidationimages,holdout/test,externalrawtransmission,newpackages,cost. Syntheticdata only iid diagnostics,notlearning.',
 'sources':{'blindspot_uniform_102b.py':sha(Path(__file__))},'inputs':{n:sha(_repo_path(ROOT, n)) for n in ['artifacts/run-001/split_manifest.json','artifacts/grain-visual-audit-097/all400_noise_contrast.csv','artifacts/blindspot-preflight-102/plan.json','reports/frame-features-101-summary.json']},'protected011':read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256'],'image_sha256':{r['ID']:r['image_sha256'] for r in chosen}}
 write(_repo_path(OUT, 'plan.json'),p);check();print(json.dumps({'prepared':True,'images':24,'strata':strata,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))}),flush=True)

def hh(a):return (a[::2,::2]-a[::2,1::2]-a[1::2,::2]+a[1::2,1::2])/2

def inspect(x):
 noise=float(np.median(np.abs(hh(x)))/.6744897501960817);guide=gaussian_filter(x,3,mode='reflect');grad=gaussian_gradient_magnitude(guide,1,mode='reflect');tiles=[(y,z) for y in range(0,256,32) for z in range(0,256,32)];order=sorted(range(64),key=lambda i:float(np.quantile(grad[tiles[i][0]:tiles[i][0]+32,tiles[i][1]:tiles[i][1]+32],.9)))[:4];locations=[tiles[i] for i in order]
 spans=[float(np.diff(np.quantile(guide[y:y+32,z:z+32],[.1,.9]))[0]/max(noise,1/255)) for y,z in locations]
 tv=denoise_tv_chambolle(x,weight=noise,eps=1e-4,max_num_iter=50,channel_axis=None) if noise>1e-12 else x.copy();arms={'raw':x,'TV':tv,'smoothing':gaussian_filter(x,1,mode='reflect')};metrics={};maxerr=0.
 for name,a in arms.items():
  residual=a-gaussian_filter(a,1,mode='reflect');v=[]
  for kind in ('HH','residual'):
   for dy,dx in ((0,1),(1,0),(1,1)):
    us=[];vs=[]
    for y,z in locations:
     t=hh(a[y:y+32,z:z+32]) if kind=='HH' else residual[y:y+32,z:z+32];H,W=t.shape;u=t[:H-dy,:W-dx].ravel();w=t[dy:,dx:].ravel();us.append(u-u.mean());vs.append(w-w.mean())
    u=np.concatenate(us);w=np.concatenate(vs);den=float(np.sqrt((u@u)*(w@w)))
    val=float((u@w)/den) if den>1e-20 else None
    if val is not None:ind=float(np.corrcoef(u,w)[0,1]);maxerr=max(maxerr,abs(ind-val));assert abs(ind-val)<1e-12
    v.append(val)
  metrics[name]=v
 return {'noise_proxy':noise,'guide':guide,'tile_origins_yx':locations,'tile_span_over_noise':spans,'provisional_flat_tiles':sum(z<=1 for z in spans),'metrics':metrics,'independent_error':maxerr}

def run():
 p=check();start=time.perf_counter();results=[];maxerr=0.
 for i,ident in enumerate(p['ids']):
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');assert sha(path)==p['image_sha256'][ident]
  with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);x=np.asarray(im,dtype=float)/255
  real=inspect(x);controls=[];maxerr=max(maxerr,real['independent_error'])
  for rep in range(REPS):
   rng=np.random.default_rng(20271003+i*100+rep);control=np.rint(np.clip(real['guide']+rng.normal(0,real['noise_proxy'],x.shape),0,1)*255)/255;z=inspect(control);controls.append({'metrics':z['metrics'],'provisional_flat_tiles':z['provisional_flat_tiles']});maxerr=max(maxerr,z['independent_error'])
  null=np.array([z['metrics']['raw'][:3] for z in controls],dtype=float);raw=np.array(real['metrics']['raw'][:3],dtype=float);assert np.isfinite(null).all() and np.isfinite(raw).all();flags=(abs(raw)>np.max(abs(null),axis=0)+.05)
  del real['guide'];results.append({'ID':ident,'real':real,'iid_controls':controls,'raw_HH_exceeds_iid_plus_005':flags.tolist(),'any_direction_warning':bool(flags.any()),'saturation_fraction':float(np.mean((x==0)|(x==1)))})
  print(json.dumps({'images_complete':i+1,'seconds':time.perf_counter()-start}),flush=True)
 informative=sum(z['real']['provisional_flat_tiles']>=2 for z in results);warnings=sum(z['any_direction_warning'] for z in results);quant={}
 for arm in ('raw','TV','smoothing'):
  real=np.array([z['real']['metrics'][arm] for z in results]);iid=np.array([c['metrics'][arm] for z in results for c in z['iid_controls']]);quant[arm]={'real_signed_correlation_median':np.median(real,axis=0).tolist(),'iid_signed_correlation_median':np.median(iid,axis=0).tolist(),'real_abs_correlation_p90':np.quantile(abs(real),.9,axis=0).tolist(),'iid_abs_correlation_p90':np.quantile(abs(iid),.9,axis=0).tolist()}
 report={'images':24,'iid_replicates_per_image':REPS,'original_outer_fold_excluded':0,'metric_order':['HH_horizontal','HH_vertical','HH_diagonal','residual_horizontal','residual_vertical','residual_diagonal'],'provisionally_flat_images':informative,'raw_HH_warning_images':warnings,'engineering_gate_clear':bool(informative>=18 and warnings<6),'aggregate':quant,'results':results,'independent_formula_max_error':maxerr,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'source_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-start,'restoration_training':False,'hardness_labels_read':False,'outer_validation_images_read':False,'holdout_test_read':False,'cost':0}
 write(_repo_path(OUT, 'report.json'),report);check();print(json.dumps({k:v for k,v in report.items() if k not in ('results','aggregate')}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args();{'prepare':prepare,'run':run}[a.mode]()
