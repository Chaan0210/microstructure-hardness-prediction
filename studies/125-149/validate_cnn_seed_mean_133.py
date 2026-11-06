"""Verify the fixed seed mean and nested014 equations; no candidate selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse,types
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_seed_mean_133 as x
import validate_mse_readout_132 as prior
import numpy as np,torch
s=x.s;e=x.e;a=x.a
# Reuse independently derived OLS/eigensystem validation with only destination binding.
prior.m=types.SimpleNamespace(OUT=x.OUT,ARM=x.ARM,q=x.q)

def nodecheck(d,path):
 r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);T=z['train'];V=z['validation'];assert r['train']==T.tolist() and r['validation']==V.tolist() and not set(d['g'][T])&set(d['g'][V]);checks=[]
 for si,seed in enumerate(x.SEEDS[1:],1):
  file=path/f'seed-{seed}.pt';rr=r['models'][si-1];assert e.sha(file)==rr['checkpoint_sha256'];m=torch.load(_repo_io(file),map_location='cpu',weights_only=True);assert m['seed']==seed+r['offset'] and m['epochs']==80 and m['average_last_epochs']==5;assert m['train_ids']==[d['ids'][i] for i in T] and m['validation_ids']==[d['ids'][i] for i in V];norm=m['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};errs={device:float(abs(x.predict(m,d,V,device)-z['seeds'][si]).max()) for device in ['mps','cpu']};assert max(errs.values())<1e-3;lr=.0001+.5*(.002-.0001)*(1+np.cos(np.pi*np.arange(1,81)/80));np.testing.assert_allclose([t['lr_after_step'] for t in rr['curve']],lr,atol=1e-15,rtol=0);checks.append({'seed':seed,'reload_error':errs,'train_rows':len(T),'updates':80*int(np.ceil(len(T)/32)),'image_visits':80*len(T)})
 manual=(z['seeds'][0]+z['seeds'][1]+z['seeds'][2])/3;np.testing.assert_array_equal(manual,z['mean3']);return {'path':str(path.relative_to(ROOT)),'models':checks,'equal_mean_verified':True}

def same(left,right):
 for seed in x.SEEDS[1:]:
  p=torch.load(_repo_io(left/f'seed-{seed}.pt'),map_location='cpu',weights_only=True);q=torch.load(_repo_io(right/f'seed-{seed}.pt'),map_location='cpu',weights_only=True);assert s.cal.h.state_hash(p['state_dict'])==s.cal.h.state_hash(q['state_dict']);assert p['normalization']==q['normalization']
 with np.load(_repo_io(left/'predictions.npz')) as p,np.load(_repo_io(right/'predictions.npz')) as q:
  for key in ['train','validation','seeds','mean3']:np.testing.assert_array_equal(p[key],q[key])

def main(split):
 start=time.perf_counter();x.check();d=x.data(split);r=e.read(x.OUT/split/'report.json');p=a.readnp(x.OUT/split/'predictions.npz',r['predictions_sha256']);assert p['ids'].tolist()==d['ids'];np.testing.assert_array_equal(p['fold'],d['fid']);np.testing.assert_array_equal(p['y'],d['y']);nn=[];pipes=[]
 for path in r['nodes']:nn.append(nodecheck(d,_repo_path(ROOT, path)))
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);pipes.append(prior.pipeline(d,dd,split,k))
 pert=dict(d);pert['y']=d['y'].copy();V=np.flatnonzero(d['fid']==0);pert['y'][V]+=10000;root=x.OUT/f'poison-{split}';root.mkdir();dd,paths=x.replace(pert,split,0,root,True);pp,_,rr=a.fold(dd,0,x.ARM,root/'pipeline');np.testing.assert_array_equal(pp,p['primary'][V]);orig=e.read(x.OUT/split/'pipeline-fold-0/report.json');assert rr['outer_calibrator']==orig['outer_calibrator'] and rr['anchor_calibrators']==orig['anchor_calibrators'] and rr['selected']==orig['selected']
 for path in paths:same(path,x.OUT/split/'cnn-fold-0'/path.name)
 A,B,_,_,_=d['cells'][0,0];pert=dict(d);pert['y']=d['y'].copy();pert['y'][B]+=50000;root2=x.OUT/f'inner-poison-{split}';root2.mkdir()
 for name in ['middle-0','deep-0-0','deep-0-1','deep-0-2']:
  old=x.OUT/split/'cnn-fold-0'/name;rec=e.read(old/'report.json');z=a.readnp(old/'predictions.npz',rec['predictions_sha256']);x.node(pert,z['train'],z['validation'],rec['offset'],z['seeds'][0],root2/name);same(root2/name,old)
 # Cached old outer trainers use onlyT labels; verify both extra seeds with a direct poison replay.
 T=np.flatnonzero(d['fid']!=0);pert=dict(d);pert['y']=d['y'].copy();pert['y'][V]+=10000;outerroot=x.OUT/f'outer-cache-poison-{split}';outerroot.mkdir();prev=x.fv.CNN_SEED
 for seed in x.SEEDS[1:]:
  x.fv.CNN_SEED=seed
  try:pred,_=x.fv.cnn_fold(d['images'],pert['y'],T,V,d['ids'],0,'mps',outerroot/f'seed-{seed}.pt')
  finally:x.fv.CNN_SEED=prev
  saved=torch.load(_repo_io(outerroot/f'seed-{seed}.pt'),map_location='cpu',weights_only=True);old=torch.load(_repo_io(x.ckpath(split,0,seed)),map_location='cpu',weights_only=True);assert s.cal.h.state_hash(saved['state_dict'])==s.cal.h.state_hash(old['state_dict'])
  with np.load(_repo_io(x.OUT/'audit'/f'{split}.npz')) as z:np.testing.assert_array_equal(pred,z['seeds'][x.SEEDS.index(seed),V])
 x.check();out={'passed':True,'node_checks':nn,'pipeline_checks':pipes,'new_inner_models_checked':2*len(nn),'outer80_full_nested_label_poison_exact':True,'innerB_anchor_models_label_poison_exact':True,'cached_outer_additional_seeds_label_poison_exact':True,'new_poison_CNN_fits':34,'source_sha256':e.sha(Path(__file__)),'independent_pipeline_validator_sha256':e.sha(_repo_path(ROOT, 'validate_mse_readout_132.py')),'seconds':time.perf_counter()-start};s.write(x.OUT/f'validation-{split}.json',out);e.emit({'validation_passed':True,'split':split,'models_checked':2*len(nn),'poison_fits':34,'seconds':out['seconds'],'max_CPU_error':max(z['reload_error']['cpu'] for n in nn for z in n['models'])})
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('split',choices=e.SPLITS);main(p.parse_args().split)
