from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ['RES216_RAW_EDA']='1'
import common as c,engine as e
from common import R,P,np,torch,Image
from scipy import ndimage as ndi
c.check();d=c.data();old=P/'experiments/contrast-augmentation-206';manifest=c.read(old/'artifact-manifest.json');keys=[n for n in manifest if ('/fold-' in n and '-control/' in n) or n.endswith(('/engine.py','/common.py','/frozen-reference.npz','/training-code-lock.json'))]
for n in keys:assert c.sha(P/n)==manifest[n],n
priorhash=c.read(P/'experiments/blur-augmentation-215/raw-image-hashes.json');native=[];hashes={};rows=[]
with c.Monitor('preflight',600) as mon:
 for i,ID in enumerate(d['ids']):
  file=P/'data/train'/f'{ID}.png';key=str(file.relative_to(P));h=c.sha(file);assert h==priorhash[key];hashes[key]=h
  with Image.open(_repo_io(file)) as im:
   assert im.mode=='L' and im.size==(256,256);a=np.asarray(im,dtype=np.float32)[None]/255.;b=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)[None]/255.
  np.testing.assert_array_equal(b,d['images'][i]);native.append(a)
  row={'ID':str(ID)}
  for name,x,area in [('native',a[0],16),('BOX',b[0],4)]:
   lab,num=ndi.label(x<.5,structure=np.ones((3,3)));sizes=np.bincount(lab.ravel())[1:];lap=x[:-2,1:-1]+x[2:,1:-1]+x[1:-1,:-2]+x[1:-1,2:]-4*x[1:-1,1:-1]
   row.update({name+'_mean':float(x.mean()),name+'_std':float(x.std()),name+'_dark_fraction':float((x<.5).mean()),name+'_components_area_matched':int(np.sum(sizes>=area)),name+'_small_components_1to4px':int(np.sum((sizes>=1)&(sizes<=4))),name+'_gradient_energy':float(np.diff(x,axis=0).__pow__(2).mean()+np.diff(x,axis=1).__pow__(2).mean()),name+'_laplacian_variance':float(lap.var())})
  row['BOX_reconstruction_RMS']=float(np.sqrt(np.mean((a[0]-np.repeat(np.repeat(b[0],2,0),2,1))**2)));rows.append(row)
 x=np.stack(native);c.arrays(R/'native256.npz',ids=d['ids'],images=x);c.write(R/'raw-image-hashes.json',hashes);c.write(R/'image-features.json',rows)
 nd=c.data('native');norms=[]
 for k in range(5):
  T,V=c.indices(k)
  for arm,dd in [('control',d),('native',nd)]:
   changed=dict(dd);changed['y']=dd['y'].copy();changed['y'][V]=np.nan;changed['images']=dd['images'].copy();changed['images'][V]=np.nan;assert c.norm(changed,T)==c.norm(dd,T);norms.append({'fold':k,'arm':arm,**c.norm(dd,T)})
 m=e.initialize(0);shapes=[];hook=m.features[0].register_forward_pre_hook(lambda mod,args:shapes.append(list(args[0].shape)))
 with torch.no_grad():m(torch.zeros((2,1,256,256),device=c.device))
 hook.remove();assert shapes==[[2,1,256,256]];del m
 ref=c.load(R/'frozen-reference.npz');original=c.load(old/'OOF.npz');f=c.load(P/'experiments/matched-replacement-194/frozen-components.npz');b=c.load(P/'experiments/dinol-177/original/predictions.npz')
 for key in ['ids','y','fold']:np.testing.assert_array_equal(d[key],original[key])
 np.testing.assert_array_equal(ref['best021'],b['blend_L_MLP']);np.testing.assert_array_equal(ref['best021'],f['best021']);np.testing.assert_array_equal(ref['coefficient'],.081*f['slope']);np.testing.assert_array_equal(ref['old_first_CNN'],original['control'])
 # One discarded native probe, exactly8updates. No validation inference or score.
 T,V=c.indices(0);ix=T[:32];n=c.norm(nd,T);m=e.initialize(0);o=e.optimize(m);xb=torch.from_numpy((x[ix]-n['x_mean'])/max(n['x_std'],1e-6)).to(c.device);yb=torch.tensor((d['y'][ix]-n['y_mean'])/n['y_std'],dtype=torch.float32,device=c.device);timings=[]
 c.write(R/'probe-start.json',{'planned_updates':8,'shape':[32,1,256,256],'seed':20261003})
 for j in range(8):
  start=c.time.perf_counter();e.step(m,o,xb,yb,'native');c.sync();timings.append(c.time.perf_counter()-start);mon.sample();c.emit({'probe_update':j+1,'seconds':timings[-1]})
 del m,o,xb,yb;c.gc.collect()
 if c.device=='mps':torch.mps.empty_cache()
 c.write(R/'probe.json',{'completed_updates':8,'discarded_weights':True,'timings':timings,'median_step_seconds':float(np.median(timings)),'scores_computed':False})
 c.write(R/'preflight.json',{'passed':True,'device':c.device,'torch':torch.__version__,'numpy':np.__version__,'cache_reconstruction_exact':True,'raw400_hashes_match215':True,'historical206_hashes_verified':len(keys),'first_conv_input_shape':shapes,'normalization_train_only_validation_poison_invariant':norms,'native_cache_SHA256':c.sha(R/'native256.npz'),'probe_updates':8})
c.check();c.emit({'preflight_passed':True})
