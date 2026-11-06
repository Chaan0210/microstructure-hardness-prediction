"""Independent selection, image scope and metric replay of fixed102b."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,hashlib,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,gaussian_gradient_magnitude
from skimage.restoration import denoise_tv_chambolle
import blindspot_uniform_102b as s

def independent(x):
 blocks=x.reshape(128,2,128,2);HH=np.einsum('aibj,ij->ab',blocks,np.array([[1,-1],[-1,1]])/2);noise=float(np.median(abs(HH))/.6744897501960817);guide=gaussian_filter(x,3,mode='reflect');grad=gaussian_gradient_magnitude(guide,1,mode='reflect');coords=[(i,j) for i in range(0,256,32) for j in range(0,256,32)];scores=[np.percentile(grad[i:i+32,j:j+32],90) for i,j in coords];idx=np.argsort(scores,kind='stable')[:4];coords=[coords[i] for i in idx];spans=[float((np.percentile(guide[i:i+32,j:j+32],90)-np.percentile(guide[i:i+32,j:j+32],10))/max(noise,1/255)) for i,j in coords]
 arms={'raw':x,'TV':denoise_tv_chambolle(x,weight=noise,eps=1e-4,max_num_iter=50,channel_axis=None) if noise>1e-12 else x.copy(),'smoothing':gaussian_filter(x,1,mode='reflect')};metrics={}
 for arm,a in arms.items():
  vals=[];res=a-gaussian_filter(a,1,mode='reflect')
  for kind in range(2):
   for dy,dx in ((0,1),(1,0),(1,1)):
    pairs=[]
    for i,j in coords:
     z=a[i:i+32,j:j+32];b=np.einsum('aibj,ij->ab',z.reshape(16,2,16,2),np.array([[1,-1],[-1,1]])/2) if kind==0 else res[i:i+32,j:j+32];H,W=b.shape;u=b[0:H-dy,0:W-dx].flatten();v=b[dy:H,dx:W].flatten();pairs.append(np.stack([u-u.mean(),v-v.mean()]))
    vals.append(float(np.corrcoef(np.concatenate(pairs,axis=1))[0,1]))
  metrics[arm]=vals
 return guide,noise,coords,spans,metrics

def main():
 start=time.perf_counter();p=s.check();r=s.read(s.OUT/'report.json');m=s.read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'))
 with (_repo_path(ROOT, 'artifacts/grain-visual-audit-097/all400_noise_contrast.csv')).open() as f:rows=list(csv.DictReader(f))
 rows=[z for z in rows if m['development_folds'][z['ID']]!=0];n=np.percentile([float(z['noise']) for z in rows],[100/3,200/3]);c=np.median([float(z['contrast_q95_q05']) for z in rows]);ids=[]
 for a in range(3):
  for b in range(2):
   pool=[z['ID'] for z in rows if int(float(z['noise'])>=n[0])+int(float(z['noise'])>=n[1])==a and int(float(z['contrast_q95_q05'])>=c)==b];ids+=sorted(pool,key=lambda i:hashlib.sha256(('102b-24:'+i).encode()).hexdigest())[:4]
 assert ids==p['ids']==[z['ID'] for z in r['results']];assert len(set(ids))==24
 for ident in ids:assert m['development_folds'][ident]!=0 and ident not in m['holdout'];assert s.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==p['image_sha256'][ident]
 maxerror=0.;checks=0
 for i in (0,11,23):
  with Image.open(_repo_path(ROOT, 'data/train', f'{ids[i]}.png')) as im:x=np.asarray(im,dtype=float)/255
  guide,noise,coords,spans,metrics=independent(x);real=r['results'][i]['real'];assert [list(c) for c in coords]==real['tile_origins_yx'];np.testing.assert_allclose(spans,real['tile_span_over_noise'],rtol=0,atol=1e-12)
  for arm in metrics:err=float(np.max(abs(np.array(metrics[arm])-real['metrics'][arm])));maxerror=max(maxerror,err);assert err<1e-12
  checks+=1
  for rep in range(16):
   noise_img=np.random.default_rng(20271003+i*100+rep).normal(0,noise,(256,256));x0=np.round(np.minimum(np.maximum(guide+noise_img,0),1)*255)/255;*_,mm=independent(x0)
   for arm in mm:err=float(np.max(abs(np.array(mm[arm])-r['results'][i]['iid_controls'][rep]['metrics'][arm])));maxerror=max(maxerror,err);assert err<1e-12
   checks+=1
 informative=0;warnings=0
 for z in r['results']:
  informative+=sum(a<=1 for a in z['real']['tile_span_over_noise'])>=2
  null=np.array([q['metrics']['raw'][:3] for q in z['iid_controls']]);flags=abs(np.array(z['real']['metrics']['raw'][:3]))>abs(null).max(0)+.05;assert flags.tolist()==z['raw_HH_exceeds_iid_plus_005'];warnings+=bool(flags.any())
 assert informative==r['provisionally_flat_images'] and warnings==r['raw_HH_warning_images'];assert (informative>=18 and warnings<6)==r['engineering_gate_clear']
 assert not (s.OUT.parent/'report.json').exists();s.check();out={'passed':True,'selection24_reconstructed_without_labels':True,'original_outer_fold0_images_excluded':True,'raw_image_hashes_verified':24,'independent_full_pipeline_replays':checks,'real_images_replayed':3,'iid_controls_replayed':48,'independent_max_error':maxerror,'gate_recomputed':r['engineering_gate_clear'],'prior32_plan_never_run':True,'protected011_files_verified':20,'model_training':False,'hardness_labels_read':False,'holdout_test_read':False,'report_sha256':s.sha(s.OUT/'report.json'),'source_sha256':s.sha(Path(__file__)),'seconds':time.perf_counter()-start,'cost':0};s.write(_repo_path(ROOT, 'reports/blindspot-uniform-102b-validation.json'),out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
