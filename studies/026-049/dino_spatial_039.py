"""Frozen original DINOv2: one enlarged frame versus four overlapping native crops."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
import dino_frozen_037 as base
OUT=_repo_path(ROOT, 'artifacts/dino-spatial-039')
ARMS=('full504','native196x4')
def inputs(ident,arm):
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:
  assert im.mode=='L' and im.size==(256,256)
  if arm=='full504':imgs=[im.resize((504,504),Image.Resampling.BICUBIC)]
  elif arm=='native196x4':imgs=[im.crop((x,y,x+196,y+196)) for y in (0,60) for x in (0,60)]
  else:raise ValueError(arm)
  return np.stack([np.ascontiguousarray((np.repeat((np.asarray(a,dtype=np.float32)/255)[None],3,axis=0)-base.MEAN)/base.STD) for a in imgs])
def describe(m,x):
 with torch.inference_mode():
  f=m.forward_features(x);p=f['x_norm_patchtokens'];assert p.shape[1]==(x.shape[-1]//14)**2 and p.shape[2]==384
  a=torch.cat((f['x_norm_clstoken'],p.mean(1),p.std(1,correction=0)),1)
 assert torch.isfinite(a).all();return a

def image_features(m,ident,arm,device):
 x=torch.from_numpy(inputs(ident,arm)).to(device);a=describe(m,x)
 out=a[0] if arm=='full504' else torch.cat((a.mean(0),a.std(0,correction=0)))
 return out.cpu().numpy()
def smoke():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);assert torch.backends.mps.is_available();ids,sp=base.scope();ident=next(i for i in ids if sp['development_folds'][i]!=0);m=base.model();results={}
 # Verify native crop coverage and exactly preserved source intensities before normalization.
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:raw=np.asarray(im,dtype=np.float32)/255
 coverage=np.zeros((256,256),int);crops=inputs(ident,'native196x4')
 for j,(yy,xx) in enumerate(( (y,x) for y in (0,60) for x in (0,60))):
  coverage[yy:yy+196,xx:xx+196]+=1;np.testing.assert_allclose(crops[j,0]*base.STD[0]+base.MEAN[0],raw[yy:yy+196,xx:xx+196],atol=1e-7,rtol=0)
 assert coverage.min()>=1 and coverage.max()==4
 for arm in ARMS:
  outputs={};timings={}
  for device in ('cpu','mps'):
   m=m.to(device);times=[]
   for k in range(3):
    if device=='mps':torch.mps.synchronize()
    t=time.perf_counter();v=image_features(m,ident,arm,device)
    if device=='mps':torch.mps.synchronize()
    if k:times.append(time.perf_counter()-t)
   outputs[device]=v;timings[device]={'median_image_seconds':float(np.median(times)),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
   if device=='mps':timings[device]['driver_bytes']=torch.mps.driver_allocated_memory()
  np.testing.assert_allclose(outputs['cpu'],outputs['mps'],atol=2e-3,rtol=2e-3);results[arm]={'devices':timings,'cpu_mps_max_abs_diff':float(abs(outputs['cpu']-outputs['mps']).max()),'feature_width':len(v)}
 plan={'arms':{'full504':'Whole256imagebicubicresize504;36x36patchtokens,CLS+mean/std1152;grain structures occupy about twice as many pixels as252,interpolation addsnoindependentinformation','native196x4':'4unaltered196cropsat(0,0),(60,0),(0,60),(60,60),coverallpixels;percropCLS+patchmean/std1152;acrosscropmean+populationstd2304;targets localcontext/heterogeneity preservingnativeintensities'},'control':'037frozenwhole252descriptor1152, sameRidgeprotocol','orientation':'No rotation/reflectionaugmentation; no unverifieddirectionlabelinvariance assumed','normalization':'ExistingImageNetRGBmean/std,grayscaleRGBrepeat;backbonefrozen/eval/inference_mode','regression':'StandardScaler→Ridge(alpha=lambda*featurecount);lambda(.1,1,10) innerGroupKFold3same037seeds. Standardizedtarget pertrain. Feature-count scaling normalizes aggregate feature magnitude acrosspanels. NoPCA.','ensemble':'0.75verifiedsamefold003+0.25spatialDINO,fixed','screen':'OriginalCV two newpanels plusmatched252control; selectatmostone newpanel onlyifblendbeats003andmatched252control,with>=3/5foldwinsagainst003. LowestblendRMSEwins;thenconfirmbothrepartitions. Ifnoneeligible, stop.','scope':'Development400only. Reusedsplitsandhistoricalselectionarenotindependentvalidation.','resources':'Stopbefore400extractifMPSimage>1secondorRSS>3GBorMPSdriver>3GB','source_sha256':{n:base.sha(_repo_path(ROOT, n)) for n in ('dino_spatial_039.py','dino_frozen_037.py')},'backbone_provenance':base.PROVENANCE,'protected':{n:base.sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')}}
 base.write(_repo_path(OUT, 'plan.json'),plan);base.write(_repo_path(OUT, 'smoke.json'),{'arms':results,'smoke_training_id':ident,'native_coverage_minmax':[int(coverage.min()),int(coverage.max())],'native_values_preserved':True});print(json.dumps(results),flush=True)

def extract():
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text());smoke=json.loads((_repo_path(OUT, 'smoke.json')).read_text())
 for n,h in plan['source_sha256'].items():assert base.sha(_repo_path(ROOT, n))==h
 prior=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,h in prior['official_source_sha256'].items():assert base.sha(base.SOURCE/n)==h
 for a in smoke['arms'].values():
  d=a['devices']['mps'];assert d['median_image_seconds']<1 and d['rss_peak_bytes']<3_000_000_000 and d['driver_bytes']<3_000_000_000
 torch.set_num_threads(2);ids,sp=base.scope();m=base.model().to('mps');initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};features={};timings={}
 for arm in ARMS:
  values=[];start=time.perf_counter()
  for j,i in enumerate(ids):
   values.append(image_features(m,i,arm,'mps'))
   if (j+1)%80==0:print(json.dumps({'arm':arm,'completed':j+1,'seconds':time.perf_counter()-start}),flush=True)
  X=np.stack(values);assert len(X)==400 and np.isfinite(X).all();features[arm]=X;timings[arm]=time.perf_counter()-start
 assert all(torch.equal(initial[k],v.detach().cpu()) for k,v in m.state_dict().items()) and not m.training and all(p.grad is None for p in m.parameters())
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),**features)
 for n,h in plan['protected'].items():assert base.sha(_repo_path(ROOT, n))==h
 report={'seconds':timings,'shapes':{a:list(x.shape) for a,x in features.items()},'weights_unchanged':True,'feature_sha256':base.sha(_repo_path(OUT, 'development_features.npz')),'image_sha256':{i:base.sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids},'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};base.write(_repo_path(OUT, 'extraction.json'),report);print(json.dumps(report|{'image_sha256':'saved in artifact'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','extract'));a=p.parse_args();smoke() if a.mode=='smoke' else extract()
