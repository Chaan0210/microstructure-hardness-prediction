from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch,nn
from study import e,D
import gc,time,math,hashlib

def main():
 c.check();d=c.data('original');X=c.load(R/'resnet50/features.npz')['X'].astype(float);cache=np.load(_repo_io(R/'resnet50/prefix.npy'),mmap_mode='r');root=R/'resnet50';checks=[];start=time.monotonic()
 with c.Monitor('validation',600):
  prefix,base=c.model('resnet50');ph=c.statehash(prefix);th=c.statehash(base);bn=[];base_state=base.state_dict();del prefix
  for n,m in base.named_modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):bn.extend(n+'.'+k for k in m.state_dict())
  O=c.load(root/'OOF.npz');np.testing.assert_array_equal(O['ids'],d['ids']);np.testing.assert_array_equal(O['y'],d['y']);np.testing.assert_array_equal(O['fold'],d['fold'])
  for k in range(5):
   path=root/f'fold-{k}';o=e.ck(path/'model.pt');rec=c.read(path/'report.json');assert c.sha(path/'model.pt')==rec['model_sha256'];T=np.array(o['train']);V=np.array(o['validation']);fold=c.plan['folds'][str(k)];assert o['train']==fold['train'] and o['validation']==fold['validation'];assert not set(d['g'][T])&set(d['g'][V]);expected=e.floor_stats(X,d['y'],T,V,202710185+k)
   for key in ['feature_mean','feature_var','feature_scale']:np.testing.assert_array_equal(o[key],expected[key])
   for key in ['feature_RMS_floor','target_mean','target_std','seed','scaler_policy']:assert o[key]==expected[key]
   assert o['recipe']==c.plan['locked_outer_recipes'][str(k)] and o['adaptation_seed']==202710186+100*k+99;initial=e.ck(path/'initial-head.pt');prior=e.ck(D/'resnet50'/f'fold-{k}/initial-head.pt');assert o['head_initial']==initial['final_head_sha256'] and initial['initial_head_sha256']==prior['initial_head_sha256'];assert o['prefix_initial']==ph and o['tail_initial']==th
   _,tail=c.model('resnet50');tail.load_state_dict(o['tail']);tail.eval();h=e.old.head(X.shape[1]);h.load_state_dict(o['head']);h.eval();assert c.statehash(tail)==o['tail_final'] and c.statehash(h)==o['final_head_sha256']
   for key in bn:assert torch.equal(o['tail'][key],base_state[key])
   saved=c.load(path/'predictions.npz');np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['prediction'],O['native256'][V]);pp=e.old.predict(h,tail,cache,X,V,o,'cpu');error=float(np.max(abs(pp-saved['prediction'])));assert np.isfinite(pp).all()
   updates=c.load(path/'updates.npz')['values'];ref=c.load(D/'resnet50'/f'fold-{k}'/('outer-'+o['recipe']['optimizer'])/'updates.npz')['values'];np.testing.assert_array_equal(updates[:,:5],ref[:,:5]);assert len(updates)==o['update_count']==240 and np.isfinite(updates).all();rng=np.random.default_rng(o['adaptation_seed']+1);orders=np.stack([rng.permutation(T) for _ in range(12)]);assert hashlib.sha256(orders.tobytes()).hexdigest()==o['order_adaptation_sha256'];checks.append({'fold':k,'CPU_maxabs':error,'CPU_tolerance':.002,'passed':error<.002,'train_stats_LR_seed_orders_BN_prefix_ID_finite_passed':True});c.emit(checks[-1]);del h,tail,o;gc.collect()
  assert c.sha(root/'OOF.npz')==c.read(root/'report.json')['OOF_sha256'];assert e.old.evaluate(d,O['native256'])==c.read(root/'report.json')['scores256'];np.testing.assert_array_equal(O['control224'],c.load(D/'resnet50/OOF.npz')['selected'])
  T=np.array(c.plan['folds']['0']['train']);V=np.array(c.plan['folds']['0']['validation']);poison=d['y'].copy();poison[V]=1e6-1000*poison[V];hh,rebuild=e.old.headfit(X,poison,T,V,202710185);initial=e.ck(root/'fold-0/initial-head.pt');assert rebuild['final_head_sha256']==initial['final_head_sha256'];assert torch.equal(rebuild['feature_scale'],initial['feature_scale']);assert rebuild['target_mean']==initial['target_mean'] and rebuild['target_std']==initial['target_std'];del hh;out=root/'audit-label-poison';out.mkdir();pred,_=e.train(rebuild,cache,X,poison[T].copy(),c.plan['locked_outer_recipes']['0'],202710186+99,out,V);ref=e.ck(root/'fold-0/model.pt');replay=e.ck(out/'model.pt')
  for key in ['head','tail']:
   for n in ref[key]:assert torch.equal(ref[key][n],replay[key][n]),(key,n)
  np.testing.assert_array_equal(pred,O['native256'][V]);np.testing.assert_array_equal(c.load(out/'updates.npz')['values'],c.load(root/'fold-0/updates.npz')['values']);c.check()
 report={'passed':all(x['passed'] for x in checks),'checks':checks,'CPU_maxabs':max(x['CPU_maxabs'] for x in checks),'CPU_absolute_tolerance':.002,'full_head100_tail12_label_poison_replay_exact':True,'raw_development_only_audit_hook_active':True,'network_attempts':len(c.network_attempts),'seconds':time.monotonic()-start};assert report['network_attempts']==0;c.write(R/'validation.json',report);c.emit(report)
if __name__=='__main__':main()
