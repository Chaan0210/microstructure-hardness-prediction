from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ['BLUR215_RAW_EDA']='1'
import common as c
from common import R,P,np,Image
from transform import blur,calibration,energy
from scipy import ndimage as ndi
import sys
sys.path.insert(0,str(P/'studies'));import grain_features as gf;sys.path.remove(str(P/'studies'))
c.check();d=c.data();hashes={};native=[];folds=[];grainrows=[]
def describe(a):return {'min':float(np.min(a)),'max':float(np.max(a)),'mean':float(np.mean(a)),'quantiles':np.quantile(a,[0,.1,.5,.9,1]).tolist()}
def lapvar(x):
 l=x[:,:,:-2,1:-1]+x[:,:,2:,1:-1]+x[:,:,1:-1,:-2]+x[:,:,1:-1,2:]-4*x[:,:,1:-1,1:-1];return l.var((1,2,3))
def measurements(x,b,threshold):
 before=[];after=[];lost=[]
 for a,z in zip(x[:,0],b[:,0]):
  la,na=ndi.label(a<threshold,structure=np.ones((3,3)));lb,nb=ndi.label(z<threshold,structure=np.ones((3,3)));aa=np.bincount(la.ravel());ab=np.bincount(lb.ravel());small=np.flatnonzero((aa>=1)&(aa<=4));small=small[small>0];survive=np.bincount(la[z<threshold].ravel(),minlength=len(aa))
  before.append([float((a<threshold).mean()),na,len(small)]);after.append([float((z<threshold).mean()),nb,int(np.sum((ab[1:]>=1)&(ab[1:]<=4)))]);lost.append(int(np.sum(survive[small]==0)))
 return np.array(before),np.array(after),np.array(lost)
def grainproxy(a):
 # Reuse the documented legacy sigma1 watershed exactly, passing float*255 to avoid added quantization.
 x,ridge,edge,lab=gf.segment(a.astype(np.float64)*255.,1.);areas=np.bincount(lab.ravel())[1:];areas=areas[areas>=16];diam=2*np.sqrt(areas/np.pi)
 return {'count_area_ge16':int(len(areas)),'mean_equivalent_diameter_px':float(diam.mean()) if len(diam) else 0.,'mean_inv_sqrt_diameter':float(np.mean(1/np.sqrt(diam))) if len(diam) else 0.,'boundary_fraction':float(edge.mean())}
with c.Monitor('preflight',900):
 for i,ID in enumerate(d['ids']):
  p=P/'data/train'/f'{ID}.png';hashes[str(p.relative_to(P))]=c.sha(p)
  with Image.open(_repo_io(p)) as im:
   assert im.mode=='L' and im.size==(256,256);a=np.asarray(im,dtype=np.float32)[None]/255.;x=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)[None]/255.
  np.testing.assert_array_equal(x,d['images'][i]);native.append(a)
 native=np.stack(native)
 for k in range(5):
  T,V=c.indices(k);params=calibration(d['images'][T]);pd=d['images'].copy();pd[V]=np.nan;assert calibration(pd[T])==params;n=c.norm(d,T);by_scale={};selected=sorted(range(len(T)),key=lambda j:__import__('hashlib').sha256((f'blur215-grain:{k}:'+d['ids'][T[j]]).encode()).hexdigest())[:12]
  for scale,x in [('model128',d['images'][T]),('native256',native[T])]:
   b=blur(x,params['alpha']);assert b.shape==x.shape and b.dtype==np.float32 and np.isfinite(b).all() and b.min()>=0 and b.max()<=1
   err=float(abs(b.astype(np.float64)-blur(x.astype(np.float64),params['alpha'])).max());meanerr=float(abs(b.mean((1,2,3))-x.mean((1,2,3))).max());assert max(err,meanerr)<1e-6;nrms=float(np.sqrt(np.mean((((b-n['x_mean'])/n['x_std'])-((x-n['x_mean'])/n['x_std']))**2)));assert nrms>1e-5
   en0,en1=energy(x),energy(b);lv0,lv1=lapvar(x),lapvar(b);assert np.all(en1<=en0+1e-6) and np.all(lv1<=lv0+1e-6);q=np.rint(b*255)/255.;phase={}
   for name,t in [('fixed_025',.25),('fixed_035',.35),('fixed_050',.5),('train_pooled_Q10',float(np.quantile(x,.1)))]:
    a0,a1,lost=measurements(x,b,t);c.arrays(R/f'phase-fold{k}-{scale}-{name}.npz',ids=d['ids'][T],threshold=np.array(t),columns=np.array(['fraction','components8','small_components_area1to4']),before=a0,after=a1,small_components_completely_disappeared=lost)
    phase[name]={'threshold':t,'fraction_change':describe(a1[:,0]-a0[:,0]),'absolute_fraction_change':describe(abs(a1[:,0]-a0[:,0])),'component_count_change':describe(a1[:,1]-a0[:,1]),'small_component_count_change':describe(a1[:,2]-a0[:,2]),'small_components_completely_disappeared':describe(lost),'images_with_disappearance':int(np.sum(lost>0))}
   for j in selected:grainrows.append({'fold':k,'scale':scale,'ID':str(d['ids'][T[j]]),'before':grainproxy(x[j,0]),'after':grainproxy(b[j,0])})
   d4err=float(abs(blur(np.rot90(x[:2],1,axes=(2,3)),params['alpha'])-np.rot90(b[:2],1,axes=(2,3))).max());assert d4err<1e-6
   c.arrays(R/f'sharpness-fold{k}-{scale}.npz',ids=d['ids'][T],gradient_before=en0,gradient_after=en1,laplacian_variance_before=lv0,laplacian_variance_after=lv1)
   by_scale[scale]={'range':[float(b.min()),float(b.max())],'float64_reference_max_error':err,'mean_max_error':meanerr,'normalized_difference_RMS':nrms,'max_pixel_shift':float(abs(b-x).max()),'gradient_energy_ratio':describe(en1/en0),'laplacian_variance_ratio':describe(lv1/lv0),'phase':phase,'if_uint8_requantized_max_error':float(abs(q-b).max()),'if_uint8_requantized_RMS_error':float(np.sqrt(np.mean((q-b)**2))),'actual_clipping_pixels':0,'actual_requantization':False,'D4_roundoff_max_error':d4err}
  folds.append({'fold':k,**params,'diagnostics':by_scale,'validation_image_poison_calibration_exact':True});c.emit({'preflight_fold':k,'alpha':params['alpha'],'grain_proxies_completed':len(grainrows)})
 c.write(R/'grain-measurement-deltas.json',grainrows);c.write(R/'raw-image-hashes.json',hashes);c.write(R/'preflight.json',{'passed':True,'device':c.device,'torch':c.torch.__version__,'numpy':np.__version__,'raw_development_IDs':400,'native_shape':[256,256],'existing_model_input':[128,128],'no_additional_resize':True,'cache_reconstruction_exact':True,'folds':folds,'grain_scope':'12 trainIDs per fold per scale, preselected IDhash;legacygrain_features.segment sigma1;notvalidatedgraintruth','physical_label_invariance_claimed':False,'training_runs':0});c.emit({'preflight_passed':True,'policy':'single small separable3tap blur'})
c.check()
