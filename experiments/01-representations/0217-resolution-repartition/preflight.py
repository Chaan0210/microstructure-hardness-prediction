from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.device==c.read(P/'experiments/resolution-216/preflight.json')['device'];records=[]
with c.Monitor('preflight',600) as mon:
 old=P/'experiments/huber-repartition-212';manifest=c.read(old/'artifact-manifest.json');keys=[n for n in manifest if '-control/' in n or n.endswith(('/common.py','/engine.py','/plan.json','/code-lock.json','/frozen-reference.npz','/OOF.npz'))]
 for n in keys:assert c.sha(P/n)==manifest[n],n
 for split in c.SPLITS:
  d=c.data(split);nd=c.data(split,'native');root=R/split;root.mkdir();f=c.lineage(d);ref=c.load(old/split/'frozen-reference.npz');np.testing.assert_array_equal(ref['ids'],d['ids']);np.testing.assert_array_equal(ref['y'],d['y']);np.testing.assert_array_equal(ref['fold'],d['fold']);np.testing.assert_array_equal(ref['old_first_CNN'],d['old_CNN']);np.testing.assert_array_equal(ref['best021'],d['best021'])
  for name in f:np.testing.assert_array_equal(f[name],ref[name])
  c.arrays(root/'frozen-reference.npz',**ref)
  for k in range(5):
   T,V=c.indices(d,k)
   for arm,dd in [('control',d),('native',nd)]:
    altered=dict(dd);altered['y']=dd['y'].copy();altered['y'][V]=np.nan;altered['images']=dd['images'].copy();altered['images'][V]=np.nan;assert c.norm(altered,T)==c.norm(dd,T)
   folder=old/split/f'fold-{k}-control';sd=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);original=torch.load(_repo_io(c.old_checkpoint(split,k)),map_location='cpu',weights_only=True);e.equal_tree(sd['state_dict'],original['state_dict']);assert sd['normalization']==c.norm(d,T);assert sd['train_ids']==d['ids'][T].tolist() and sd['validation_ids']==d['ids'][V].tolist();m=e.initialize(k);m.load_state_dict(sd['state_dict']);pred,_=e.predict(m,d['images'],V,sd['normalization']);np.testing.assert_array_equal(pred,ref['old_first_CNN'][V]);np.testing.assert_array_equal(pred,c.load(folder/'predictions.npz')['validation_predictions']);del m
   records.append({'split':split,'fold':k,'old_checkpoint_state_normalization_IDs_OOF_exact':True,'train_only_normalization_botharms_poison_invariant':True})
  c.emit({'preflight_split':split,'controls5_reloaded_exact':True,'frozen_lineage_exact':True})
 # One discarded8update resource probe; same image/optimizer geometry as216.
 d=c.data(c.SPLITS[0],'native');T,V=c.indices(d,0);ix=T[:32];norm=c.norm(d,T);m=e.initialize(0);o=e.optimize(m);x=torch.from_numpy((d['images'][ix]-norm['x_mean'])/max(norm['x_std'],1e-6)).to(c.device);y=torch.tensor((d['y'][ix]-norm['y_mean'])/norm['y_std'],dtype=torch.float32,device=c.device);times=[]
 c.write(R/'probe-start.json',{'planned_updates':8,'shape':[32,1,256,256]})
 for i in range(8):
  start=c.time.perf_counter();e.step(m,o,x,y,'native');c.sync();times.append(c.time.perf_counter()-start);mon.sample();c.emit({'probe_update':i+1})
 del m,o,x,y;c.gc.collect()
 if c.device=='mps':torch.mps.empty_cache()
 c.write(R/'probe.json',{'completed_updates':8,'discarded_weights':True,'seconds_per_update':times,'scores_computed':False});c.write(R/'preflight.json',{'passed':True,'device':c.device,'torch':torch.__version__,'numpy':np.__version__,'historical212_hashes_verified':len(keys),'records':records,'native_cache_matches216':True,'native_no_resize_evidence':'resolution-216/preflight.json first_conv_input_shape','gradient_probe_updates':8,'full_fits':0})
c.check()
