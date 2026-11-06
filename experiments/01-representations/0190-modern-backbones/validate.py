from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch,nn
from study import e,RECIPE,inputs,check
import argparse,time,gc,math,hashlib

def validate(name):
 check();e.CURRENT_MODEL=name;X,d=inputs(name);root=R/name;cache=np.load(_repo_io(root/'prefix.npy'),mmap_mode='r');checks=[];warmchecks=[];start=time.monotonic()
 for n,h in c.read(R/'validation-code-lock.json').items():assert c.sha(R/n)==h,n
 with c.Monitor(name+'-validation',900):
  prefix,base=c.model(name);ph=c.statehash(prefix);th=c.statehash(base);base_state=base.state_dict();buffer_names=set(dict(base.named_buffers()));bn=[]
  for n,m in base.named_modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):bn.extend(n+'.'+k for k in m.state_dict())
  tail_count=sum(p.numel() for n,p in base.named_parameters() if n not in bn);del prefix
  F=c.load(root/'frozen100-OOF.npz');O=c.load(root/'adapt-OOF.npz')
  for arr in [F,O]:
   for key,refkey in [('ids','ids'),('y','y'),('fold','fold'),('groups','g')]:np.testing.assert_array_equal(arr[key],d[refkey])
  for k in range(5):
   fold=c.plan['folds'][str(k)];T=np.array(fold['train']);V=np.array(fold['validation']);assert not set(d['g'][T])&set(d['g'][V]);path=root/f'fold-{k}';o=e.ck(path/'initial-head.pt');expected=e.floor_stats(X,d['y'],T,V,202710185+k)
   for key in ['feature_mean','feature_var','feature_scale']:assert torch.equal(o[key],expected[key])
   for key in ['train','validation','seed','dim','target_mean','target_std','feature_RMS_floor','scaler_policy']:assert o[key]==expected[key]
   xp=X.copy();yp=d['y'].copy();xp[V]=1e7-1e5*xp[V];yp[V]=1e6-1000*yp[V];po=e.floor_stats(xp,yp,T,V,o['seed'])
   for key in ['feature_mean','feature_var','feature_scale']:assert torch.equal(po[key],o[key])
   assert po['target_mean']==o['target_mean'] and po['target_std']==o['target_std'];torch.manual_seed(o['seed']);h=e.old.head(768);assert c.statehash(h)==o['initial_head_sha256'];h.load_state_dict(o['head']);assert c.statehash(h)==o['final_head_sha256'];cpu=e.old.predict(h,None,None,X,V,o,'cpu');saved=c.load(path/'frozen100-predictions.npz');np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(cpu,saved['prediction']);np.testing.assert_array_equal(cpu,F['prediction'][V]);h.to('mps');mps=e.old.predict(h,None,None,X,V,o,'mps');error=float(abs(cpu-mps).max());warmchecks.append({'fold':k,'CPU_reload_exact':True,'CPU_MPS_maxabs':error,'passed':error<.002,'poisoned_validation_features_labels_train_stats_exact':True});del h;gc.collect();torch.mps.empty_cache()
   records=[]
   for arm in ['head_only112','last_stage112']:
    out=path/arm;z=e.ck(out/'model.pt');rec=c.read(out/'report.json');assert c.sha(out/'model.pt')==rec['model_sha256'];train_tail=arm=='last_stage112';assert z['tail_trainable']==train_tail and z['model_name']==name;assert z['tail_trainable_parameters']==(tail_count if train_tail else 0);assert z['head_initial']==o['final_head_sha256'] and z['prefix_initial']==ph and z['tail_initial']==th;assert z['train']==T.tolist() and z['validation']==V.tolist() and z['recipe']==RECIPE and z['adaptation_seed']==202710186+100*k+99
    for key in ['feature_mean','feature_var','feature_scale']:assert torch.equal(z[key],o[key])
    assert z['target_mean']==o['target_mean'] and z['target_std']==o['target_std'];_,tail=c.model(name);tail.load_state_dict(z['tail']);tail.eval();h=e.old.head(768);h.load_state_dict(z['head']);h.eval();assert c.statehash(tail)==z['tail_final'] and c.statehash(h)==z['final_head_sha256'];assert (z['tail_final']!=th)==train_tail
    for key in buffer_names|set(bn):assert torch.equal(z['tail'][key],base_state[key]),key
    saved=c.load(out/'predictions.npz');np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['prediction'],O[arm][V]);pp=e.old.predict(h,tail,cache,X,V,z,'cpu');error=float(abs(pp-saved['prediction']).max());assert np.isfinite(pp).all();h.to('mps');tail.to('mps');again=e.old.predict(h,tail,cache,X,V,z,'mps');np.testing.assert_array_equal(again,saved['prediction'])
    u=c.load(out/'updates.npz')['values'];assert c.sha(out/'updates.npz')==rec['updates_sha256'];rng=np.random.default_rng(z['adaptation_seed']+1);orders=[];expected_rows=[]
    for epoch in range(12):
     order=rng.permutation(T);orders.append(order);factor=.1+.9*(1+math.cos(math.pi*epoch/12))/2
     for j in range(0,len(T),16):expected_rows.append([epoch,j//16,len(order[j:j+16]),RECIPE['head_lr']*factor,RECIPE['tail_lr']*factor if train_tail else 0.])
    np.testing.assert_array_equal(u[:,:5],np.array(expected_rows));assert len(u)==z['update_count']==240 and np.isfinite(u).all();assert hashlib.sha256(np.stack(orders).tobytes()).hexdigest()==z['order_adaptation_sha256'];st=c.read(out/'stability.json');assert st['passed'] and st['tail_trainable']==train_tail and st['tail_changed']==train_tail;assert len(st['gradient_trace'])==240;np.testing.assert_allclose([q['total'] for q in st['gradient_trace']],u[:,5],rtol=1e-6,atol=1e-5);assert st['max_gradient_ratio']<=100 and st['max_feature_RMS_ratio']<=100 and st['max_feature_maxabs_ratio']<=100;records.append(rec);checks.append({'fold':k,'arm':arm,'CPU_maxabs':error,'CPU_tolerance':.002,'passed':error<.002,'fresh_MPS_reload_exact':True,'train_stats_seed_orders_LR_updates_scope_frozen_buffers_verified':True});c.emit({'model':name,**checks[-1]});del h,tail,z;gc.collect();torch.mps.empty_cache()
   assert records[0]['head_initial']==records[1]['head_initial'] and records[0]['order_sha256']==records[1]['order_sha256']
  assert c.sha(root/'frozen100-OOF.npz')==c.read(root/'frozen100-report.json')['OOF_SHA256'];assert c.sha(root/'adapt-OOF.npz')==c.read(root/'adapt-report.json')['OOF_SHA256'];assert e.old.evaluate(d,F['prediction'])==c.read(root/'frozen100-report.json')['score']
  for arm in ['head_only112','last_stage112']:assert e.old.evaluate(d,O[arm])==c.read(root/'adapt-report.json')['scores'][arm]
  # Label poisoning before rebuilding the entire 100-epoch head and both 12-epoch arms.
  T=np.array(c.plan['folds']['0']['train']);V=np.array(c.plan['folds']['0']['validation']);poison=d['y'].copy();poison[V]=1e6-1000*poison[V];hh,rebuild=e.old.headfit(X,poison,T,V,202710185);initial=e.ck(root/'fold-0/initial-head.pt');assert rebuild['final_head_sha256']==initial['final_head_sha256'];assert torch.equal(rebuild['feature_scale'],initial['feature_scale']);assert rebuild['target_mean']==initial['target_mean'] and rebuild['target_std']==initial['target_std'];del hh;audits=[]
  for arm in ['head_only112','last_stage112']:
   out=root/('audit-label-poison-'+arm);out.mkdir();pred,_=e.train(rebuild,cache,X,poison[T].copy(),RECIPE,202710186+99,out,V,train_tail=arm=='last_stage112');ref=e.ck(root/f'fold-0/{arm}/model.pt');replay=e.ck(out/'model.pt')
   for key in ['head','tail']:
    for n in ref[key]:assert torch.equal(ref[key][n],replay[key][n]),(name,arm,key,n)
   np.testing.assert_array_equal(pred,O[arm][V]);np.testing.assert_array_equal(c.load(out/'updates.npz')['values'],c.load(root/f'fold-0/{arm}/updates.npz')['values']);assert c.read(out/'stability.json')==c.read(root/f'fold-0/{arm}/stability.json');audits.append({'arm':arm,'head100_rebuilt_exact':True,'full12_epoch_weights_predictions_gradient_LR_feature_traces_exact':True});del ref,replay;gc.collect();torch.mps.empty_cache()
  check()
 report={'model':name,'passed':all(q['passed'] for q in checks+warmchecks),'CPU_absolute_tolerance':.002,'warmhead_checks':warmchecks,'adapted_checks':checks,'CPU_maxabs':max(q['CPU_maxabs'] for q in checks),'warmhead_CPU_MPS_maxabs':max(q['CPU_MPS_maxabs'] for q in warmchecks),'CPU_failures':[q for q in checks+warmchecks if not q['passed']],'tail_parameters':tail_count,'batchnorm_modules':len(bn),'label_poison_replays':audits,'audit_complete':True,'structural_checks_passed':True,'network_attempts':len(c.network_attempts),'seconds':time.monotonic()-start};assert report['network_attempts']==0;c.write(root/'validation.json',report);c.emit({k:v for k,v in report.items() if k not in ['warmhead_checks','adapted_checks']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=['convnext_tiny','swin_v2_t']);a=p.parse_args();validate(a.model)
