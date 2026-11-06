"""Label-free train-only correlation warning screen; no restoration training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,csv,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
OUT=_repo_path(ROOT, 'artifacts/blindspot-preflight-102')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 OUT.mkdir(exist_ok=False);m=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));eligible=[i for i,f in m['development_folds'].items() if f!=0]
 ids=sorted(eligible,key=lambda i:hashlib.sha256(('102-train-only:'+i).encode()).hexdigest())[:32];assert len(ids)==32 and not set(ids)&set(m['holdout'])
 inp=['artifacts/run-001/split_manifest.json','artifacts/frame-features-101/extraction.json','reports/frame-features-101-summary.json','reports/research-synthesis-091.json','reports/formation-049-summary.json','reports/grain-visual-audit-097-nlm098-summary.json','reports/frame-censoring-100-summary.json']
 p={'study':'102 blindspot feasibility preflight only','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ids':ids,'selection':'32 smallest sha256(102-train-only:+ID) among original fold!=0; no labels or original outer-validation images decoded. Manifest ID metadata only.',
 'duplicate_audit':'No Noise2Self/Noise2Void/blindspot learned restorer in project source. TV022/071 and NLM098 are fixed image-local filters. Formation049 is per-image inverse Potts/blur/AR-noise optimization, not a learned masked-pixel predictor. Same high-level denoising objective, different estimator; not a guarantee of incremental signal.',
 'measurement':'Native uint8/255, no resize, no D4 canonicalization. Nonoverlapping 2x2 HH coefficients (a-b-c+d)/2 at origin offsets(0,0),(1,1). Pearson autocorrelation on coefficient grids at lags1,2,4 (2,4,8 raw pixels), horizontal/vertical/diagonal. Coefficient pairs share no raw pixels within each phase. Raw versus fixed TVweight1*HHMAD/.67448975,eps1e-4,50iterations. No fitting or tuning.',
 'warning_rule':'For raw only, take maximum absolute nearest-lag correlation over3directions and2phases per image. Flag ordinary independent-pixel Noise2Self rationale if median across32 >.10 OR p90 >.20. This is a conservative engineering warning, not an independence test or noise-identification theorem. No selecting exclusion radius from measured lags. TV is descriptive control only; its residual structure cannot identify true noise.',
 'known_limitations':['Raw HH may retain true fine boundaries/texture; measured dependence is not necessarily observation noise.','Nearzero HH correlation does not prove independence or zero conditional noise mean.','Clipping/quantization can violate unbiased-target assumptions.','TV-induced correlations cannot establish raw-noise independence or performance.','Only32 fixed outertrain images and overlapping diagnostic choices across prior work; no population inference.'],
 'sources':{'blindspot_preflight_102.py':sha(Path(__file__))},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inp},'protected011':read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256'],
 'paper':{'title':'Noise2Self: Blind Denoising by Self-Supervision','url':'https://proceedings.mlr.press/v97/batson19a.html','pdf':'https://proceedings.mlr.press/v97/batson19a/batson19a.pdf','relevant':'Proposition1: conditional unbiasedness and conditional independence between masked coordinates and their complement; J-invariant predictor.'},
 'execution_policy':'Only this label-free preflight; no restoration fit, hardnessCV, synthetic-image training, outervalidation/holdout/test image use, network transmission or install. Report warning outcome before deciding any pilot. No broader masks/radius search if flagged.','cost':0}
 write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'ids':len(ids),'plan_sha256':sha(_repo_path(OUT, 'plan.json'))}),flush=True)

def hh(x,phase):
 a=x[phase:,phase:];a=a[:(a.shape[0]//2)*2,:(a.shape[1]//2)*2];return (a[::2,::2]-a[::2,1::2]-a[1::2,::2]+a[1::2,1::2])/2

def corr(a,dy,dx):
 h,w=a.shape;b=a[:h-dy,:w-dx].ravel();c=a[dy:,dx:].ravel();return float(np.corrcoef(b,c)[0,1]) if b.std()>1e-12 and c.std()>1e-12 else None

def run():
 p=check();start=time.perf_counter();rows=[];image_rows=[];maxerror=0.;hashes={};prov=read(_repo_path(ROOT, 'artifacts/frame-features-101/extraction.json'))['image_sha256']
 for ident in p['ids']:
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');hashes[ident]=sha(path);assert hashes[ident]==prov[ident]
  with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);x=np.asarray(im,dtype=np.float64)/255.
  noise=float(np.median(np.abs(hh(x,0)))/.6744897501960817);tv=denoise_tv_chambolle(x,weight=noise,eps=1e-4,max_num_iter=50,channel_axis=None) if noise>1e-12 else x.copy();nearest={}
  for arm,a in [('raw',x),('TV',tv)]:
   ns=[]
   for phase in (0,1):
    h=hh(a,phase)
    # Independent coefficient computation by explicit 2x2 reshaping.
    b=a[phase:,phase:];b=b[:2*(b.shape[0]//2),:2*(b.shape[1]//2)];tiles=b.reshape(b.shape[0]//2,2,b.shape[1]//2,2);manual=(tiles[:,0,:,0]-tiles[:,0,:,1]-tiles[:,1,:,0]+tiles[:,1,:,1])/2;np.testing.assert_array_equal(h,manual)
    for lag in (1,2,4):
     for direction,dy,dx in [('horizontal',0,lag),('vertical',lag,0),('diagonal',lag,lag)]:
      value=corr(h,dy,dx);H,W=h.shape;u=h[:H-dy,:W-dx].ravel();v=h[dy:,dx:].ravel();u=u-u.mean();v=v-v.mean();den=float(np.sqrt((u@u)*(v@v)));ind=float((u@v)/den) if den>1e-20 else None
      if value is not None:assert ind is not None;maxerror=max(maxerror,abs(ind-value));assert abs(ind-value)<1e-12
      rows.append({'ID':ident,'arm':arm,'phase':phase,'raw_pixel_lag':2*lag,'direction':direction,'HH_correlation':value})
      if lag==1 and value is not None:ns.append(abs(value))
   nearest[arm]=max(ns) if ns else None
  image_rows.append({'ID':ident,'noise_proxy':noise,'saturated_pixel_fraction':float(np.mean((x==0)|(x==1))),'raw_nearest_max_abs_HH_correlation':nearest['raw'],'TV_nearest_max_abs_HH_correlation':nearest['TV']})
 summary={}
 for arm in ('raw','TV'):
  vals=[z[f'{arm}_nearest_max_abs_HH_correlation'] for z in image_rows if z[f'{arm}_nearest_max_abs_HH_correlation'] is not None];summary[arm]={'images_informative':len(vals),'median':float(np.median(vals)),'p90':float(np.quantile(vals,.9)),'min':min(vals),'max':max(vals)}
 flagged=summary['raw']['median']>.10 or summary['raw']['p90']>.20
 for name,rr in [('correlations.csv',rows),('image-summary.csv',image_rows)]:
  with (_repo_path(OUT, name)).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 result={'status':'preflight_complete_no_training','summary':summary,'independent_pixel_rationale_warning':flagged,'warning_is_not_proof_of_correlated_noise':True,'independent_computation_max_error':maxerror,'images':len(image_rows),'rows':len(rows),'source_sha256':sha(Path(__file__)),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'image_sha256':hashes,'correlations_sha256':sha(_repo_path(OUT, 'correlations.csv')),'image_summary_sha256':sha(_repo_path(OUT, 'image-summary.csv')),'saturation_fraction_median':float(np.median([z['saturated_pixel_fraction'] for z in image_rows])),'seconds':time.perf_counter()-start,'restoration_training':False,'hardness_labels_read':False,'outer_validation_images_read':False,'holdout_test_read':False,'cost':0}
 write(_repo_path(OUT, 'report.json'),result);check();print(json.dumps({k:v for k,v in result.items() if k!='image_sha256'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args();{'prepare':prepare,'run':run}[a.mode]()
