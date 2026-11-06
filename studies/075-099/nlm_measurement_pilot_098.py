"""Fixed label-free fidelity gate for a single TV -> NLM substitution."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,resource,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from skimage.segmentation import watershed,find_boundaries
from skimage.morphology import h_minima
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.restoration import denoise_nl_means,denoise_tv_chambolle
import denoised_grain_features as old
OUT=_repo_path(ROOT, 'artifacts/nlm-measurement-098')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def prepare():
 OUT.mkdir(exist_ok=False)
 p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'basis':'097 label-blind12image panels show within-region fragmentation with weak watershed and disappearance of visible interfaces at high markerprominence. 99.586% of lostinterior pixels move into frame-touching regions in48matchedconfig comparisons. Stronger h does not fix denoising.','hypothesis':'Existing skimage noise-aware nonlocal means can reduce stochastic within-region extrema while preserving narrow actual grain boundaries better than currentTV, without raising h. Unproven; requires fidelity gate first.','duplicate_audit':'No nonlocal means implementation/trial in project source. 025 changes markerprominence only; connected-grain changes topology extraction;049 inverseformation;071/072 TV preprocessing beforeDINO. This is one denoiser substitution with unchanged watershed/markers.','baseline':'CanonicalD4 raw[0,1]; TV weight=HH-noise,eps1e-4,maxiter50. Weak watershed h=.1/.2 of gradientq90,GaussianDerivativeSigma.7,7x7blackhat Otsu seed exclusion,area>=16.','candidate':'CanonicalD4 raw[0,1]; skimage denoise_nl_means h=.8*HHnoise,sigma=HHnoise,patch_size5,patch_distance6,fast_modeTrue,preserve_rangeTrue,channel_axisNone. Same exact weakwatershed and summaries. No otherparameter choices.','fixtures':'3 deterministic jittered rectangular Voronoi layouts:4x4,8x8,4x12 seedgrid,30% cellwidthjitter, RNG20270321+layoutindex. Scalar region intensities .62+/-.06,20%darkphase.4;truefind_boundaries thick1pixel raster replaced with.22;Gaussianblur.6 then independentGaussian noise sigma.025/.075,clipandquantizeuint8. Sixfixtures sharetruegeometries and modelcounts. No realhardness labels.','gate':'Across6 cases, mean absolute relative count error(candidate)<=.75*baseline. Candidate maynot worsen percase error by>.10. For BOTH h settings in EACHcase, precision>=.60 andrecall>=.80 forknownboundary tolerance2px. Allmustpass; otherwise rejectthisfixedNLMmeasurementhypothesis withoutCV or tuning. Boundariesexclude2pximageframe.','count_rule':'Eachcase median count overh.1/.2 versus actualnumber ofVoronoiregions. BoundaryPR evaluatedseparatelyeachh; not chosen byoutcome.','resources':'CPU2/noinstall/cost. MaxRSS2GB,onecandidateimage runtime<=2sec (excluding imports); no model fitting, noholdout/testimages.','if_pass':'Then unchangedNLM versus matchedTV31features (median/range acrossh.1/.2 only plusnoise) in independentfixed nestedKRR dev400 control,01190%+branch10%; require>=.02 originalgain,>=3positivefolds,beatmatchedTVstandaloneandblend,fixed395gain>0,bothresplits same. Separate plan before anyhardness scores.','source_sha256':{'nlm_measurement_pilot_098.py':sha(Path(__file__)),'denoised_grain_features.py':sha(_repo_path(ROOT, 'denoised_grain_features.py'))},'protected011':json.loads((_repo_path(ROOT, 'reports/candidate011-public-submission.json')).read_text())['protected_sha256'],'limitations':'Synthetic visualfidelity gate is a resourceallocation rule, not validation of actualgrains. Image-specific unknownnoise/blur/generator may differ. No parameter changes onfailure.'}
 write(_repo_path(OUT, 'plan.json'),p)
def segment(raw,method):
 c=old._canonical_d4(raw);x=c.astype(float)/255.;noise=old._noise_sigma(x);t=time.perf_counter()
 if method=='tv':d=denoise_tv_chambolle(x,weight=noise,eps=1e-4,max_num_iter=50,channel_axis=None) if noise>1e-12 else x.copy()
 else:d=denoise_nl_means(x,h=.8*noise,sigma=noise,patch_size=5,patch_distance=6,fast_mode=True,preserve_range=True,channel_axis=None) if noise>1e-12 else x.copy()
 ridge=ndi.grey_closing(d,size=(7,7),mode='reflect')-d;rm=ridge>threshold_otsu(ridge) if np.ptp(ridge)>1e-12 else np.zeros_like(x,bool);grad=ndi.gaussian_gradient_magnitude(d,.7,mode='reflect');out=[]
 for h in [.1,.2]:
  if np.ptp(grad)<=1e-12:seg=np.ones(x.shape,np.int32)
  else:
   mins=h_minima(grad,max(h*float(np.percentile(grad,90)),1e-12)).astype(bool)&~rm;marks=label(mins,connectivity=2)
   if not marks.any():marks.flat[int(np.argmin(np.where(~rm,grad,np.inf)))]=1
   seg=watershed(grad,marks,watershed_line=True)
  out.append((seg,old._summarize(seg,d)))
 return c,d,out,time.perf_counter()-t

def fixture(ni):
 ny,nx=[(4,4),(8,8),(4,12)][ni];rng=np.random.default_rng(20270321+ni);yy,xx=np.indices((256,256));seeds=[]
 for iy in range(ny):
  for ix in range(nx):seeds.append(((iy+.5+rng.uniform(-.15,.15))*256/ny,(ix+.5+rng.uniform(-.15,.15))*256/nx))
 distance=np.stack([(yy-y)**2+(xx-x)**2 for y,x in seeds]);truth=distance.argmin(0)+1;levels=.62+rng.uniform(-.06,.06,len(seeds));levels[rng.random(len(seeds))<.2]=.4;clean=levels[truth-1];bound=find_boundaries(truth,mode='thick');clean[bound]=.22;blur=ndi.gaussian_filter(clean,.6,mode='reflect');return truth,bound,blur

def transform_truth(raw,truth,bound):
 c=old._canonical_d4(raw)
 for k in range(4):
  for flip in [False,True]:
   apply=lambda x:np.fliplr(np.rot90(x,k)) if flip else np.rot90(x,k)
   if np.array_equal(c,apply(raw)):return apply(truth),apply(bound)
 raise AssertionError('canonical map')
def pr(pred,truth):
 roi=np.ones(truth.shape,bool);roi[:2]=False;roi[-2:]=False;roi[:,:2]=False;roi[:,-2:]=False;p=pred&roi;t=truth&roi
 precision=float((ndi.distance_transform_edt(~t)[p]<=2).mean()) if p.any() else 0.;recall=float((ndi.distance_transform_edt(~p)[t]<=2).mean()) if t.any() else 0.;return precision,recall

def run():
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in plan['source_sha256'].items():assert sha(_repo_path(ROOT, n))==h,n
 results=[];font=ImageFont.load_default(size=14);canvas=Image.new('RGB',(1024,6*298),'white');start=time.perf_counter()
 for layout in range(3):
  truth,bound,blur=fixture(layout)
  for severity,sigma in enumerate([.025,.075]):
   seed=20270401+layout*10+severity;rng=np.random.default_rng(seed);raw=np.clip(np.rint(255*(blur+rng.normal(0,sigma,blur.shape))),0,255).astype(np.uint8);ct,cb=transform_truth(raw,truth,bound);row=layout*2+severity;record={'layout':layout,'noise_sigma':sigma,'true_count':int(np.unique(truth).size),'arms':{}};pictures=[old._canonical_d4(raw)]
   z=np.repeat(pictures[0][:,:,None],3,2);z[cb]=[30,220,70];pictures.append(z)
   for method in ['tv','nlm']:
    c,d,outs,seconds=segment(raw,method);metrics=[]
    for seg,stats in outs:
     precision,recall=pr(seg==0,cb);metrics.append({'count':stats[0],'boundary_precision':precision,'boundary_recall':recall,'interior_support':stats[13]})
    count=float(np.median([m['count'] for m in metrics]));record['arms'][method]={'configs':metrics,'median_count':count,'relative_count_error':abs(count-record['true_count'])/record['true_count'],'seconds':seconds};im=np.repeat(c[:,:,None],3,2);im[outs[1][0]==0]=[255,40,30];pictures.append(im)
    assert np.isfinite(d).all() and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9 and seconds<2
   for j,pic in enumerate(pictures):
    title=['raw','knownboundary','TV h.2','NLM h.2'][j];ImageDraw.Draw(canvas).text((j*256+3,row*298+3),f'{title} | trueN{record["true_count"]} sigma{sigma}',fill='black',font=font);canvas.paste(Image.fromarray(pic).convert('RGB'),(j*256,row*298+40))
   results.append(record);print(json.dumps(record),flush=True)
 mean={m:float(np.mean([r['arms'][m]['relative_count_error'] for r in results])) for m in ['tv','nlm']};countgate=mean['nlm']<=.75*mean['tv'];noninferior=all(r['arms']['nlm']['relative_count_error']<=r['arms']['tv']['relative_count_error']+.10 for r in results);prgate=all(c['boundary_precision']>=.6 and c['boundary_recall']>=.8 for r in results for c in r['arms']['nlm']['configs']);passed=countgate and noninferior and prgate
 canvas.save(_repo_path(OUT, 'fixture-panels.png'));report={'fidelity_gate_pass':bool(passed),'mean_relative_count_error':mean,'count_improvement_gate':bool(countgate),'percase_count_guardrail':bool(noninferior),'boundary_preservation_gate':bool(prgate),'cases':results,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.perf_counter()-start,'no_real_images_or_labels_used':True,'decision':'Eligible for separately locked dev400 control' if passed else 'Reject fixed NLM substitution before hardness CV; no parameter changes or fallback.','plan_sha256':sha(_repo_path(OUT, 'plan.json'))};write(_repo_path(OUT, 'report.json'),report)
 for n,h in plan['protected011'].items():assert sha(_repo_path(ROOT, n))==h,n
 print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
