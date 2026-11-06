from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ['GAMMA214_RAW_EDA']='1'
import common as c
from common import R,P,np,Image
c.check();d=c.data();hashes={};native=[];folds=[]
def calibration(x):
 means=x.mean((1,2,3));iqr=float(np.quantile(means,.75)-np.quantile(means,.25));sens=-(x.astype(np.float64)*np.log(np.maximum(x.astype(np.float64),1e-100))).mean((1,2,3));m=float(np.median(sens));tau=min(.02,.1*iqr/m);assert tau>1e-5
 return {'train_mean_IQR':iqr,'train_median_gamma_mean_sensitivity':m,'tau':tau,'gamma_low':float(np.float32(np.exp(-tau))),'gamma_high':float(np.float32(np.exp(tau)))}
def describe(a):
 return {'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'quantiles':np.quantile(a,[0,.1,.5,.9,1]).tolist()}
def diagnostics(x,g,norm,cut):
 b=np.power(x,np.float32(g));assert b.dtype==np.float32 and b.shape==x.shape and np.isfinite(b).all() and b.min()>=0 and b.max()<=1
 err=float(abs(b.astype(np.float64)-np.power(x.astype(np.float64),np.float64(np.float32(g)))).max());assert err<1e-6
 diff=b-x;maxshift=float(abs(diff).max());assert 0<maxshift<.008
 normdiff=(b-norm['x_mean'])/norm['x_std']-(x-norm['x_mean'])/norm['x_std'];assert float(abs(normdiff).max())>.001
 ranks=[];ties=[];qloss=[]
 for a,z in zip(x,b):
  u,idx,inv=np.unique(a,return_index=True,return_inverse=True);v=z.ravel()[idx];assert np.all(np.diff(v)>0);np.testing.assert_array_equal(v[inv.ravel()],z.ravel())
  for q in [.1,.5,.9]:assert np.array_equal(a<=np.quantile(a,q),z<=np.quantile(z,q))
  quant=np.rint(z*255)/255;qloss.append(len(np.unique(z))-len(np.unique(quant)))
 dark={}
 for name,t in [('fixed_025',.25),('fixed_035',.35),('fixed_050',.5),('train_pooled_Q10',cut)]:
  before=(x<t).mean((1,2,3));after=(b<t).mean((1,2,3));change=after-before
  dark[name]={'threshold':float(t),'mean_fraction_before':float(before.mean()),'mean_fraction_after':float(after.mean()),'signed_fraction_change':describe(change),'abs_fraction_change':describe(abs(change))}
 before_energy=np.square(np.diff(x,axis=2)).mean((1,2,3))+np.square(np.diff(x,axis=3)).mean((1,2,3));after_energy=np.square(np.diff(b,axis=2)).mean((1,2,3))+np.square(np.diff(b,axis=3)).mean((1,2,3));q=np.rint(b*255)/255
 return {'gamma':g,'range':[float(b.min()),float(b.max())],'float64_reference_max_error':err,'max_pixel_shift':maxshift,'per_image_mean_shift':describe(diff.mean((1,2,3))),'normalized_difference_RMS':float(np.sqrt(np.mean(normdiff**2))),'strict_rank_ties_and_rank_quantile_masks_exact':True,'gradient_energy_ratio':describe(after_energy/before_energy),'dark_phase_proxies':dark,'clipping_pixels':0,'requantization_applied':False,'if_uint8_requantized_max_error':float(abs(q-b).max()),'if_uint8_requantized_RMS_error':float(np.sqrt(np.mean((q-b)**2))),'if_uint8_unique_levels_lost':describe(np.array(qloss))}
with c.Monitor('preflight',600):
 for i,ID in enumerate(d['ids']):
  p=P/'data/train'/f'{ID}.png';hashes[str(p.relative_to(P))]=c.sha(p)
  with Image.open(_repo_io(p)) as im:
   assert im.mode=='L' and im.size==(256,256);a=np.asarray(im,dtype=np.float32)[None]/255.;x=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)[None]/255.
  np.testing.assert_array_equal(x,d['images'][i]);native.append(a)
 native=np.stack(native);assert native.shape==(400,1,256,256)
 for k in range(5):
  T,V=c.indices(k);x=d['images'][T];n=c.norm(d,T);params=calibration(x);pd=d['images'].copy();pd[V]=np.nan;assert calibration(pd[T])==params
  # This checks the effective native-vs-existing128 photometric action without changing model resolution.
  diagnostics_by_scale={}
  for label,images in [('model128',x),('native256',native[T])]:
   cut=float(np.quantile(images,.1));diagnostics_by_scale[label]=[diagnostics(images,params[g],n,cut) for g in ['gamma_high','gamma_low']]
  folds.append({'fold':k,**params,'validation_image_poison_calibration_exact':True,'diagnostics':diagnostics_by_scale});c.emit({'preflight_fold':k,'gamma':[params['gamma_low'],params['gamma_high']]})
 metrics=np.stack([native.mean((1,2,3)),native.std((1,2,3)),np.quantile(native,.95,axis=(1,2,3))-np.quantile(native,.05,axis=(1,2,3))],1);c.arrays(R/'raw-quality-EDA.npz',ids=d['ids'],metrics=metrics,columns=np.array(['mean','std','q95-q05']))
 c.write(R/'raw-image-hashes.json',hashes);c.write(R/'preflight.json',{'passed':True,'device':c.device,'torch':c.torch.__version__,'numpy':np.__version__,'raw_development_IDs':400,'native_shape':[256,256],'existing_model_input':[128,128],'no_additional_resize':True,'cache_reconstruction_exact':True,'raw_metric_quantiles':np.quantile(metrics,[0,.1,.5,.9,1],axis=0).tolist(),'folds':folds,'comparison':'206 changes contrast around image mean;213 additive brightness;214 nonlinear monotone power transform changes local contrasts and fixed-threshold fractions','physical_label_invariance_claimed':False,'training_runs':0});c.emit({'preflight_passed':True,'policy':'single train-calibrated symmetric loggamma'})
c.check()
