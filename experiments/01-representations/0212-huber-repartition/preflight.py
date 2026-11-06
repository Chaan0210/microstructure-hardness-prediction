from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.device=='mps';a=c.read(P/'experiments/huber-loss-211/plan.json');assert c.plan['loss_policy']==a['loss_policy'] and c.plan['recipe']==a['recipe'];records=[]
with c.Monitor('preflight',300):
 for split in c.SPLITS:
  d=c.data(split);root=R/split;root.mkdir();f=c.lineage(d);saved=c.load(P/f'experiments/auxiliary-repartition-201/{split}/frozen-components.npz')
  for name in f:np.testing.assert_array_equal(f[name],saved[name])
  c.arrays(root/'frozen-reference.npz',ids=d['ids'],y=d['y'],fold=d['fold'],best021=d['best021'],old_first_CNN=d['old_CNN'],**f)
  pp=np.full(400,np.nan)
  for k in range(5):
   T,V=c.indices(d,k);norm=c.norm(d,T);oldpath=c.old_checkpoint(split,k);old=torch.load(_repo_io(oldpath),map_location='cpu',weights_only=True);folder=P/f'experiments/ema-repartition-209/{split}/fold-{k}';obj=torch.load(_repo_io(folder/'last5_uniform.pt'),map_location='cpu',weights_only=True)
   for z in [old,obj]:assert z['seed']==20261003+k and z['normalization']==norm and z['train_ids']==d['ids'][T].tolist() and z['validation_ids']==d['ids'][V].tolist()
   assert old['epochs']==80 and old['average_last_epochs']==5;e.equal_tree(old['state_dict'],obj['state_dict']);m=c.legacy.TinyCNN().to(c.device);m.load_state_dict(old['state_dict']);pred,_=e.predict(m,d['images'],V,norm,c.device);np.testing.assert_array_equal(pred,d['old_CNN'][V]);pp[V]=pred
   canon=c.load(folder/'predictions.npz');np.testing.assert_array_equal(canon['last5_uniform'],pred);np.testing.assert_array_equal(canon['validation_indices'],V);op=torch.load(_repo_io(folder/'optimizer-verification.pt'),map_location='cpu',weights_only=True);assert all(int(v['step'])==800 for v in op['optimizer']['state'].values());records.append({'split':split,'fold':k,'old_checkpoint_and209_state_exact':True,'legacy_MPS_OOF_exact':True,'normalization_IDs_seed_verified':True,'all800_RNG_and_final_optimizer_available209':True});del m
  np.testing.assert_array_equal(pp,c.load(P/f'experiments/ema-repartition-209/{split}/OOF.npz')['last5_uniform']);c.emit({'preflight_split':split,'all5_controls_exact':True,'frozen_lineage_exact':True})
 c.write(R/'preflight.json',{'passed':True,'models':10,'records':records,'both_frozen_lineages_exact':True,'policy_exact211':True,'device':c.device,'torch':torch.__version__,'historical_per_update_RNG_source':209,'gradient_fits':0,'performance_scores_computed':False,'plan_SHA256':c.sha(R/'plan.json')})
c.check()
