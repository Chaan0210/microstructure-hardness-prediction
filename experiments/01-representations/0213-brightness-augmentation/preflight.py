from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ['BRIGHTNESS213_RAW_EDA']='1'
import common as c
from common import R,P,np,Image
c.check();d=c.data();raw=[];hashes={};folds=[];maxcache=0.;rawdelta={}
with c.Monitor('preflight',300):
 for i,ID in enumerate(d['ids']):
  p=P/'data/train'/f'{ID}.png';hashes[str(p.relative_to(P))]=c.sha(p)
  with Image.open(_repo_io(p)) as im:
   assert im.mode=='L' and im.size==(256,256)
   a=np.asarray(im,dtype=np.float32)/255.;cached=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)[None]/255.
  np.testing.assert_array_equal(cached,d['images'][i]);maxcache=max(maxcache,float(abs(cached-d['images'][i]).max()))
  dx=np.diff(a,axis=1);dy=np.diff(a,axis=0);lap=a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:]-4*a[1:-1,1:-1]
  raw.append([a.min(),a.max(),a.mean(),a.std(),np.quantile(a,.95)-np.quantile(a,.05),np.mean(a==0),np.mean(a==1),np.mean(dx*dx)+np.mean(dy*dy),np.std(lap)])
 raw=np.array(raw);c.arrays(R/'raw-quality-EDA.npz',ids=d['ids'],metrics=raw,columns=np.array(['min','max','mean','std','q95-q05','zero_fraction','one_fraction','gradient_energy','laplacian_std']))
 for k in range(5):
  T,V=c.indices(k);x=d['images'][T];means=x.mean((1,2,3));iqr=float(np.quantile(means,.75)-np.quantile(means,.25));delta=float(np.float32(min(2/255,.1*iqr)));assert delta>1e-5;n=c.norm(d,T);errors=[];quant=[]
  for sign in [-1,1]:
   b=x+np.float32(sign*delta);assert b.min()>=0 and b.max()<=1
   for a,z in zip(x,b):
    u,inv=np.unique(a,return_inverse=True);v=z.ravel()[np.unique(a,return_index=True)[1]];assert np.all(np.diff(v)>0);np.testing.assert_array_equal(v[inv.ravel()],z.ravel())
   gradient=max(float(abs(np.diff(b,axis=2)-np.diff(x,axis=2)).max()),float(abs(np.diff(b,axis=3)-np.diff(x,axis=3)).max()));contrast=float(abs((np.quantile(b,.95,axis=(1,2,3))-np.quantile(b,.05,axis=(1,2,3)))-(np.quantile(x,.95,axis=(1,2,3))-np.quantile(x,.05,axis=(1,2,3)))).max());shift=(b-n['x_mean'])/n['x_std']-(x-n['x_mean'])/n['x_std'];err=float(abs(shift-sign*delta/n['x_std']).max());assert max(gradient,contrast,err)<1e-6;assert np.max(abs(shift))>.001
   q=np.rint(b*255)/255;quant.append({'sign':sign,'if_requantized_max_error':float(abs(q-b).max()),'if_requantized_RMS_error':float(np.sqrt(np.mean((q-b)**2))),'actual_requantization':False});errors.append({'sign':sign,'range':[float(b.min()),float(b.max())],'gradient_max_error':gradient,'contrast_max_error':contrast,'normalized_shift':sign*delta/n['x_std'],'normalization_identity_error':err,'rank_ties_preserved':True,'clipped_pixels':0})
  # Training-only: replacing validation images and targets cannot affect strength or normalization.
  pd=dict(d);pd['images']=d['images'].copy();pd['images'][V]=np.nan;pd['y']=d['y'].copy();pd['y'][V]=np.nan;assert c.norm(pd,T)==n;assert np.array_equal(pd['images'][T],x)
  folds.append({'fold':k,'delta':delta,'equivalent_8bit_levels':delta*255,'train_mean_IQR':iqr,'train_raw_metrics_quantiles':np.quantile(raw[T],[0,.1,.5,.9,1],axis=0).tolist(),'checks':errors,'quantization_diagnostic':quant,'validation_poison_invariance':True})
 # Native source endpoints explain why the operation is explicitly applied after the existing BOX resize.
 c.write(R/'raw-image-hashes.json',hashes);c.write(R/'preflight.json',{'passed':True,'device':c.device,'torch':c.torch.__version__,'numpy':np.__version__,'raw_count':400,'raw_shape':[256,256],'cache_reconstruction_exact':True,'cache_max_error':maxcache,'raw_quantiles':np.quantile(raw,[0,.1,.5,.9,1],axis=0).tolist(),'folds':folds,'206_difference':'206 mu+c*(x-mu) preserves mean and reduces contrast;213 x+b changes mean and preserves contrast/gradients. No per-image centering; shift survives fixed train normalization and explicit input mean feature.','geometry':'identity coordinates, unchanged dimensions/aspect ratio; rank/gradient checks are not physical label proof','model_training_runs':0});c.emit({'preflight_passed':True,'delta_by_fold':[f['delta'] for f in folds]})
c.check()
