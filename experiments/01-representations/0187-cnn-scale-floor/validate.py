from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch,nn
import run as engine
import argparse,time,gc,math

def bnkeys(model):
 out=[]
 for n,m in model.named_modules():
  if isinstance(m,nn.modules.batchnorm._BatchNorm):out.extend(n+'.'+k for k in m.state_dict())
 return out

def validate(name):
 c.check();
 for record in c.read(B/'provenance.json')['weights'].values():assert c.sha(P/record['path'])==record['sha256']
 engine.CURRENT_MODEL=name;d=c.data('original');X=c.load(B/name/'features.npz')['X'].astype(float);cache=np.load(_repo_io(B/name/'prefix.npy'),mmap_mode='r');root=R/name;checks=[];start=time.monotonic();paths=[]
 for k in range(5):
  paths.extend(root/f'fold-{k}'/f'inner-{j}'/recipe['name'] for j in range(3) for recipe in engine.RECIPES);paths.extend(root/f'fold-{k}'/('outer-'+fam) for fam in ['AdamW','SGD'])
 with c.Monitor(name+'-validation',900):
  prefix,base=c.model(name);ph=c.statehash(prefix);th=c.statehash(base);base_state=base.state_dict();bn=bnkeys(base);del prefix
  for path in paths:
   o=engine.ck(path/'model.pt');rec=c.read(path/'report.json');assert c.sha(path/'model.pt')==rec['model_sha256'];T=np.array(o['train']);V=np.array(o['validation']);assert not set(d['g'][T])&set(d['g'][V]);np.testing.assert_array_equal(o['feature_mean'].numpy(),X[T].mean(0));np.testing.assert_array_equal(o['feature_var'].numpy(),X[T].var(0));var=X[T].var(0);rms=float(np.sqrt(var.mean()));scale=np.maximum(np.sqrt(var),rms) if rms>0 else np.ones_like(var);assert o['feature_RMS_floor']==rms and o['scaler_policy']=='train_RMS_floor';np.testing.assert_array_equal(o['feature_scale'].numpy(),scale);assert o['target_mean']==float(d['y'][T].mean()) and o['target_std']==float(d['y'][T].std());assert o['prefix_initial']==ph and o['tail_initial']==th
   outer=int(path.relative_to(root).parts[0].split('-')[1]);foldplan=c.plan['folds'][str(outer)]
   if path.parent.name.startswith('inner-'):
    inner=int(path.parent.name.split('-')[1]);expected=foldplan['inner'][inner];recipe=next(x for x in engine.RECIPES if x['name']==path.name);seed=202710186+100*outer+inner;initial=engine.ck(path.parent/'initial-head.pt')
   else:
    expected=foldplan;family=path.name.removeprefix('outer-');choice=c.read(path.parent/'selection.json')['family_winners'][family];recipe=engine.RECIPES[choice];seed=202710186+100*outer+99;initial=engine.ck(root/f'fold-{outer}/initial-head.pt')
   assert o['train']==expected['train'] and o['validation']==expected['validation'];assert o['recipe']==recipe and o['adaptation_seed']==seed and o['head_initial']==initial['final_head_sha256']
   _,tail=c.model(name);tail.load_state_dict(o['tail']);tail.eval();h=engine.old.head(X.shape[1]);h.load_state_dict(o['head']);h.eval();assert c.statehash(tail)==o['tail_final'] and c.statehash(h)==o['final_head_sha256']
   for n in bn:assert torch.equal(o['tail'][n],base_state[n]),n
   saved=c.load(path/'predictions.npz');np.testing.assert_array_equal(saved['indices'],V);pp=engine.old.predict(h,tail,cache,X,V,o,'cpu');err=float(abs(pp-saved['prediction']).max());assert np.isfinite(pp).all();relative_error=float(np.max(abs(pp-saved['prediction'])/np.maximum(1.,abs(saved['prediction']))))
   updates=c.load(path/'updates.npz')['values'];assert c.sha(path/'updates.npz')==rec['updates_sha256'];expected=[];rng=np.random.default_rng(o['adaptation_seed']+1);orders=[]
   for ep in range(12):
    order=rng.permutation(T);orders.append(order);factor=.1+.9*(1+math.cos(math.pi*ep/12))/2
    for j in range(0,len(T),16):expected.append([ep,j//16,len(order[j:j+16]),o['recipe']['head_lr']*factor,o['recipe']['tail_lr']*factor])
   np.testing.assert_array_equal(updates[:,:5],np.array(expected));assert updates.shape[0]==o['update_count']==12*math.ceil(len(T)/16);assert np.isfinite(updates).all() and (updates[:,5:]>=0).all();assert __import__('hashlib').sha256(np.stack(orders).tobytes()).hexdigest()==o['order_adaptation_sha256'];checks.append({'path':str(path.relative_to(R)),'CPU_reload_maxabs':err,'CPU_reload_max_relative_to_max1_prediction':relative_error,'CPU_absolute_tolerance_pass':err<.002,'updates':len(updates),'trainstats_exact':True});del tail,h,o;gc.collect()
   if len(checks)%14==0:c.emit({'validation':name,'models_checked':len(checks),'max_CPU_error':max(z['CPU_reload_maxabs'] for z in checks)})
  O=c.load(root/'OOF.npz');report=c.read(root/'report.json');assert c.sha(root/'OOF.npz')==report['OOF_sha256'];np.testing.assert_array_equal(O['ids'],d['ids']);np.testing.assert_array_equal(O['y'],d['y']);np.testing.assert_array_equal(O['fold'],d['fold'])
  for k in range(5):
   fold=c.plan['folds'][str(k)];T=np.array(fold['train']);V=np.array(fold['validation']);path=root/f'fold-{k}';inner=c.load(path/'inner-OOF.npz');chosen=c.read(path/'selection.json');ip=inner['predictions'];np.testing.assert_array_equal(inner['train'],T);winner,families,mse=engine.selection(d['y'],T,ip);assert winner==chosen['winner_index'] and families==chosen['family_winners'];np.testing.assert_array_equal(np.sqrt(mse),chosen['inner_RMSE'])
   for j,cell in enumerate(fold['inner']):
    H=cell['validation'];positions=[T.tolist().index(i) for i in H];hashes=[];initials=[]
    for ri,recipe in enumerate(engine.RECIPES):
     sub=path/f'inner-{j}'/recipe['name'];pred=c.load(sub/'predictions.npz')['prediction'];np.testing.assert_array_equal(ip[ri,positions],pred);r=c.read(sub/'report.json');hashes.append(r['order_sha256']);initials.append(r['head_initial'])
    assert len(set(hashes))==len(set(initials))==1
   for fam,ri in families.items():
    pred=c.load(path/('outer-'+fam)/'predictions.npz')['prediction'];np.testing.assert_array_equal(O[fam][V],pred)
    if ri==winner:np.testing.assert_array_equal(O['selected'][V],pred)
   poisoned=d['y'].copy();poisoned[V]=1e6-poisoned[V]*1000;w2,f2,m2=engine.selection(poisoned,T,ip);assert w2==winner and f2==families;np.testing.assert_array_equal(mse,m2)
  for arm in ['selected','AdamW','SGD']:
   assert np.isfinite(O[arm]).all();calc=engine.old.evaluate(d,O[arm]);assert calc==report['scores'][arm]
  # Full trajectory replays; heldout labels changed before rebuilding train-only head/scalers.
  audits=[];k=0;fold=c.plan['folds']['0'];path=root/'fold-0'
  for mode in ['inner','outer']:
   if mode=='inner':T=np.array(fold['inner'][0]['train']);V=np.array(fold['inner'][0]['validation']);original=path/'inner-0/adamw_low';initial=engine.ck(path/'inner-0/initial-head.pt');recipe=engine.RECIPES[0];seed=202710186
   else:
    T=np.array(fold['train']);V=np.array(fold['validation']);choice=c.read(path/'selection.json');recipe=engine.RECIPES[choice['winner_index']];original=path/('outer-'+recipe['optimizer']);initial=engine.ck(root/'fold-0/initial-head.pt');seed=202710186+99
   poison=d['y'].copy();poison[V]=1e6-poison[V]*1000;hh,rebuilt=engine.old.headfit(X,poison,T,V,202710185);assert rebuilt['final_head_sha256']==initial['final_head_sha256'];np.testing.assert_array_equal(rebuilt['feature_mean'],initial['feature_mean']);np.testing.assert_array_equal(rebuilt['feature_scale'],initial['feature_scale']);assert rebuilt['feature_RMS_floor']==initial['feature_RMS_floor'];assert rebuilt['target_mean']==initial['target_mean'] and rebuilt['target_std']==initial['target_std'];del hh
   out=root/('audit-'+mode+'-label-poison');out.mkdir();pred,r=engine.train(rebuilt,cache,X,poison[T].copy(),recipe,seed,out,V);ref=engine.ck(original/'model.pt');replay=engine.ck(out/'model.pt')
   for key in ['head','tail']:
    for n in ref[key]:assert torch.equal(ref[key][n],replay[key][n]),(mode,key,n)
   np.testing.assert_array_equal(pred,c.load(original/'predictions.npz')['prediction']);np.testing.assert_array_equal(c.load(out/'updates.npz')['values'],c.load(original/'updates.npz')['values']);audits.append({'scope':mode,'altered_validation_rows':len(V),'full_frozen100_warmstart_rebuilt_exact':True,'full12epoch_refit_state_prediction_gradient_LR_exact':True,'updates':r['updates']});del ref,replay;gc.collect()
 c.check();out={'passed':all(x['CPU_absolute_tolerance_pass'] for x in checks),'structural_and_label_poison_checks_passed':True,'CPU_absolute_tolerance':.002,'CPU_absolute_tolerance_failures':[x for x in checks if not x['CPU_absolute_tolerance_pass']],'audit_completed_all_models':True,'model':name,'checked_models':len(checks),'max_CPU_reload_error':max(z['CPU_reload_maxabs'] for z in checks),'checks':checks,'label_poison_replays':audits,'all_trainlocal_stats_BN_prefix_LR_updates_selection_ID_order_finite_verified':True,'network_attempts':len(c.network_attempts),'seconds':time.monotonic()-start};assert out['network_attempts']==0;c.write(root/'validation.json',out);c.emit({k:v for k,v in out.items() if k!='checks'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=['resnet50','efficientnet_b0']);a=p.parse_args();validate(a.model)
